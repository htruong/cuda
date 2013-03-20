#ifndef VERSION
#define VERSION 0
#endif

#include <stdio.h>

#include <sys/time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

inline void handleReturnCode(cudaError_t rc, char *message){
	 if (rc != cudaSuccess)
	    {
	        fprintf(stderr, "%s (error code %s)!\n", message, cudaGetErrorString(rc));
	        exit(EXIT_FAILURE);
	    }
}

double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}


/**
 * CUDA Kernel Device code
 */
__global__ void
kernel(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int delta;

    for (delta = 0; delta < numElements; delta+=blockDim.x*gridDim.x){
		if (i+delta < numElements)
		{
			C[i+delta] = A[i+delta] + B[i+delta];
			C[i+delta] = C[i+delta] + A[i+delta];
			C[i+delta] = C[i+delta] + B[i+delta];
		}
    }
}

/**
 * Host main routine
 */
int
main( int argc, char* argv[] )
{

    printf("Running version %d\n", VERSION);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    err = cudaSetDevice(0);
    handleReturnCode(err,"Failed to set device");
    
    double start = gettime();

    // Print the vector length to be used, and compute its size
    int numElements;

    if (argc < 2){
    	printf("usage: ./memory%d SIZE_OF_VECTOR\n", VERSION);
    	exit(-1);
    }else{
    	numElements = atoi(argv[1]);
    }

    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A, *h_B, *h_C;

#if (VERSION == 1)
   unsigned int flag = cudaHostAllocDefault;
#elif (VERSION == 2)
   cudaSetDeviceFlags(cudaDeviceMapHost);
   unsigned int flag = cudaHostAllocMapped;
#endif

#if (VERSION == 0)

    // Allocate the host input vector A
    h_A = (float *)malloc(size);

    // Allocate the host input vector B
    h_B = (float *)malloc(size);

    // Allocate the host output vector C
    h_C = (float *)malloc(size);

#else

    err = cudaHostAlloc( &h_A, size, flag);
    handleReturnCode(err,"Failed to allocate host vector A");

    err = cudaHostAlloc( &h_B, size, flag);
    handleReturnCode(err,"Failed to allocate host vector B");

    err = cudaHostAlloc( &h_C, size, flag);
    handleReturnCode(err,"Failed to allocate host vector C");

#endif

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

#if (VERSION == 0 || VERSION == 1)
    // Allocate the device input vector A
    err = cudaMalloc((void **)&d_A, size);
    handleReturnCode(err,"Failed to allocate device vector A");

    // Allocate the device input vector B
    err = cudaMalloc((void **)&d_B, size);
    handleReturnCode(err,"Failed to allocate device vector B");

    // Allocate the device output vector C
    err = cudaMalloc((void **)&d_C, size);
    handleReturnCode(err,"Failed to allocate device vector C");
#elif (VERSION == 2)

    err = cudaHostGetDevicePointer((void **)&d_A, h_A, 0);
	handleReturnCode(err,"Failed to get pointer to device vector A");

    err = cudaHostGetDevicePointer((void **)&d_B, h_B, 0);
	handleReturnCode(err,"Failed to get pointer to device vector B");

    err = cudaHostGetDevicePointer((void **)&d_C, h_C, 0);
	handleReturnCode(err,"Failed to get pointer to device vector C");

 #endif

printf("h_A=%p, h_B=%p, h_C=%p\n", h_A, h_B, h_C);
printf("d_A=%p, d_B=%p, d_C=%p\n", d_A, d_B, d_C);

#if (VERSION !=2)
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    handleReturnCode(err,"Failed to copy vector A from host to device");

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    handleReturnCode(err,"Failed to copy vector B from host to device");
#endif

    // Launch CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    handleReturnCode(err,"Failed to launch kernel kernel");

#if (VERSION != 2)
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    handleReturnCode(err,"Failed to copy vector C from device to host");
#else
    cudaDeviceSynchronize();
#endif

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(2*h_A[i] + 2*h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

#if (VERSION == 0 || VERSION ==1)
    // Free device global memory
    err = cudaFree(d_A);
    handleReturnCode(err, "Failed to free device vector A");

    err = cudaFree(d_B);
    handleReturnCode(err, "Failed to free device vector B");

    err = cudaFree(d_C);
    handleReturnCode(err, "Failed to free device vector C");
#endif

#if (VERSION ==0)
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
#else
    err = cudaFreeHost(h_A);
    handleReturnCode(err, "Failed to free host vector A");

    err = cudaFreeHost(h_B);
    handleReturnCode(err, "Failed to free host vector B");

    err = cudaFreeHost(h_C);
    handleReturnCode(err, "Failed to free host vector C");

#endif

    // Reset the device and exit
    err = cudaDeviceReset();
    handleReturnCode(err, "Failed to deinitialize the device");

    printf ("total time = %f\n", gettime()-start);

    printf("Done\n");
    return 0;
}

