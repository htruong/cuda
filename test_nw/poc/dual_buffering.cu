#define LIMIT -999
#define BLOCK_SIZE 16
#define MAX_SEQ_LEN 2100
#define MAX_SEQ_NUM 1024
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

inline void cudaCheckError(int line, cudaError_t ce)
{
    if (ce != cudaSuccess) {
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        exit(1);
    }
}

// HACK Huan's hack
// this is not the updated validation code
int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length)
{
    unsigned int i = 0;
    while (i!=length) {
        if ( (score_matrix_cpu[i]) == (score_matrix[i] >> 2) ) {
            ++i;
            continue;
        }
        else {
            printf("i = %d, expected %d, got %d.\n",i, score_matrix_cpu[i], score_matrix[i] >> 2);
            return 0;
        }
    }
    return 1;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}

__global__ void dummy_function(int * array, unsigned int howlarge)
{
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
		for (int i = 0; i < 9; i++) {
			for (int delta=0; delta<howlarge; delta=delta+blockDim.x*gridDim.x) {
				if (tid+delta < howlarge)
		        		array[tid+delta] = array[tid+delta] + tid;	
			}
		}
}


void runTest()
{
    double start, end, now;
    unsigned int nints = 500 * 1024 * 1024;
    unsigned int sz = nints * sizeof(int);

		unsigned int nints_small = 1 * 1024 * 1024;
		unsigned int sz_small = nints_small * sizeof(int);

		#ifdef _LP64
		printf ("Running on a 64-bit platform!\n", 0);
		#else
		#endif
    
    int * dummy_cpu, * dummy_cpu2, * dummy_small_cpu, * dummy_small_cpu2;
    cudaMallocHost( (void**) &dummy_cpu, sz );
    cudaMallocHost( (void**) &dummy_cpu2, sz );
	cudaMallocHost ( (void**) &dummy_small_cpu, sz_small);
	cudaMallocHost ( (void**) &dummy_small_cpu2, sz_small);


		int * dummy_gpu, * dummy_gpu2, * dummy_small_gpu, * dummy_small_gpu2;
		cudaMalloc( (void**) &dummy_gpu, sz );
		cudaMalloc( (void**) &dummy_gpu2, sz );
		cudaMalloc( (void**) &dummy_small_gpu, sz_small );
		cudaMalloc( (void**) &dummy_small_gpu2, sz_small );

		double kernelt = 0, memcpyt = 0, st = 0, ast = 0;


#define TIMES 5	
    start = gettime();
    dummy_function<<<100,512>>>(dummy_gpu, nints);
    cudaDeviceSynchronize();
    end = gettime();
    printf("time for kernel call = %f\n", end-start);
	
    start = gettime();
    cudaMemcpy(dummy_cpu, dummy_gpu, sz, cudaMemcpyDeviceToHost );
    cudaDeviceSynchronize();
    end = gettime();
    printf("time for memcopy D-H = %f\n", end-start);

    start = gettime();
    cudaMemcpy(dummy_small_gpu, dummy_small_cpu, sz_small, cudaMemcpyHostToDevice);
    dummy_function<<<100,512>>>(dummy_gpu, nints);
    cudaMemcpy(dummy_cpu, dummy_gpu, sz, cudaMemcpyDeviceToHost );
    cudaDeviceSynchronize();
    end = gettime();
    printf("time for one iteration = %f\n", end-start);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

#define DEBUG 1
	
    for (int sync=0; sync<2; sync++){

        start = gettime();
	for (int i = 0; i< TIMES; i++) {
	        // small sync copy H->D	
		cudaMemcpyAsync(dummy_small_gpu, dummy_small_cpu, sz_small, cudaMemcpyHostToDevice, stream1);
                //kernel function
	        dummy_function<<<100,512, 0, stream1>>>(dummy_gpu, nints);
    	        cudaDeviceSynchronize();
                //large copy D->H can be sync or async
#ifdef DEBUG		
                now = gettime();
#endif
		if (sync){
		    cudaMemcpy(dummy_cpu, dummy_gpu, sz, cudaMemcpyDeviceToHost);
                }else{
		    cudaMemcpyAsync(dummy_cpu, dummy_gpu, sz, cudaMemcpyDeviceToHost, stream1 );
                }
#ifdef DEBUG		
		printf("(A)sync call took %f\n", gettime() - now);
#endif
		// small sync copy H->D 
                cudaMemcpyAsync(dummy_small_gpu2, dummy_small_cpu2, sz_small, cudaMemcpyHostToDevice, stream2);
                //kernel function
                dummy_function<<<100,512, 0, stream2>>>(dummy_gpu2, nints);
                cudaDeviceSynchronize();
                //large copy D->H can be sync or async
                if (sync){
                    cudaMemcpy(dummy_cpu2, dummy_gpu2, sz, cudaMemcpyDeviceToHost);
                }else{
                    cudaMemcpyAsync(dummy_cpu2, dummy_gpu2, sz, cudaMemcpyDeviceToHost , stream2);
                }

	}
    	cudaDeviceSynchronize();
        end = gettime();
	if (!sync)
		printf("%d iterations: time for ASYNC calls = %f\n",TIMES,end-start);
	else
		printf("%d iterations: time for SYNC calls = %f\n",TIMES,end-start);

   }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    runTest();
    return EXIT_SUCCESS;
}

