#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <needle.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include <needle_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	runTest( argc, argv);
    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
	double time;
	double start_time;
	double end_time;
	    
	int max_rows, max_cols, penalty;
    int *input_itemsets[NUM_SEQ], *output_itemsets[NUM_SEQ], *referrence[NUM_SEQ];
	int *matrix_cuda[NUM_SEQ], *referrence_cuda[NUM_SEQ];
	int size;
	FILE *fptr = fopen("profiling.csv","wa+");
    //fprintf(fptr,"start,%lf\n",time);
    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 3)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
		usage(argc, argv);
    }
	
	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}
	
	time = gettime();
	cudaSetDevice(0);
	start_time = gettime();
	fprintf(stdout,"First API,%lf\n",start_time-time);
	time = start_time;
	
	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	for (int i=0; i<NUM_SEQ; ++i){
		referrence[i] = (int *)malloc( max_rows * max_cols * sizeof(int) );
    	input_itemsets[i] = (int *)malloc( max_rows * max_cols * sizeof(int) );
		output_itemsets[i] = (int *)malloc( max_rows * max_cols * sizeof(int) );
	}

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );
	
	for (int n = 0; n<NUM_SEQ; ++n){
    	for (int i = 0 ; i < max_cols; i++){
			for (int j = 0 ; j < max_rows; j++){
				input_itemsets[n][i*max_cols+j] = 0;
			}
		}
	}
	//printf("Start Needleman-Wunsch\n");
	for (int n = 0; n<NUM_SEQ; ++n){
		for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
       		input_itemsets[n][i*max_cols] = rand() % 10 + 1;
		}
	    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
    	   input_itemsets[n][j] = rand() % 10 + 1;
		}
		for (int i = 1 ; i < max_cols; i++){
			for (int j = 1 ; j < max_rows; j++){
				referrence[n][i*max_cols+j] = blosum62[input_itemsets[n][i*max_cols]][input_itemsets[n][j]];
			}
		}
	
    	for( int i = 1; i< max_rows ; i++)
       		input_itemsets[n][i*max_cols] = -i * penalty;
		for( int j = 1; j< max_cols ; j++)
       		input_itemsets[n][j] = -j * penalty;
	}

    size = max_cols * max_rows;
	
    dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;
	
	cudaStream_t *streams = (cudaStream_t *)malloc(NUM_SEQ*sizeof(cudaStream_t));
	for (int n=0; n<NUM_SEQ; ++n)
		cudaStreamCreate(&streams[n]);
	
	start_time = gettime();
	for (int n=0; n<NUM_SEQ; ++n){
		cudaMalloc((void**)& referrence_cuda[n], sizeof(int)*size);
		cudaMalloc((void**)& matrix_cuda[n], sizeof(int)*size);
	}
	end_time = gettime();
	
	start_time = gettime();
	fprintf(stdout,"CPU,%lf\n",start_time-time);
	time = start_time;
	for (int n=0; n<NUM_SEQ; ++n){
		cudaMemcpy(referrence_cuda[n], referrence[n], sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(matrix_cuda[n], input_itemsets[n], sizeof(int) * size, cudaMemcpyHostToDevice);
	}
	start_time = gettime();
	fprintf(stdout,"Memcpy to device,%lf\n",start_time-time);
	time = start_time;

	// Concurrent compare multiple sequences
	//printf("Processing top-left matrix\n");
	//process top-left matrix
	for( int i = 1 ; i <= block_width ; i++){
		dimGrid.x = i;
		dimGrid.y = 1;
		for (int n=0; n<NUM_SEQ; ++n){
			needle_cuda_shared_1<<<dimGrid, dimBlock, 0, streams[n]>>>(referrence_cuda[n], matrix_cuda[n]
	                                      ,max_cols, penalty, i, block_width); 
		}
	}
	//printf("Processing bottom-right matrix\n");
   	//process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){
		dimGrid.x = i;
		dimGrid.y = 1;
		for (int n=0; n<NUM_SEQ; ++n){
			needle_cuda_shared_2<<<dimGrid, dimBlock, 0, streams[n]>>>(referrence_cuda[n], matrix_cuda[n]
	                                      ,max_cols, penalty, i, block_width); 
		}
	}
	cudaDeviceSynchronize();
	start_time = gettime();
	fprintf(stdout,"kernel,%lf\n",start_time-time);
	time = start_time;
	for (int n=0; n<NUM_SEQ; ++n){
		cudaMemcpy(output_itemsets[n], matrix_cuda[n], sizeof(int) * size, cudaMemcpyDeviceToHost);	
	}
	//cudaDeviceSynchronize();
	start_time = gettime();
	fprintf(stdout,"Memcpy to host,%lf\n",start_time-time);

#ifdef TRACEBACK
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");
	
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			//fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = output_itemsets[(i - 1) * max_cols + j - 1];
		    w  = output_itemsets[ i * max_cols + j - 1 ];
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = output_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
			
		//fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;};
	}
	
//	fclose(fpo);
#endif
	for (int n=0; n<NUM_SEQ; ++n){
		cudaFree(referrence_cuda[n]);
		cudaFree(matrix_cuda[n]);
		free(referrence[n]);
		free(input_itemsets[n]);
		free(output_itemsets[n]);
		cudaStreamDestroy(streams[n]);
	}
	end_time = gettime();
	fprintf(fptr,"End,%lf",end_time-time);
}

