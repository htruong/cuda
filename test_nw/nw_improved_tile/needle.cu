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
#include "needle.h"
#include "needle_cpu.h"

// includes, kernels
#include "needle_cpu.c"
#include "needle_kernel_tile.cu"

inline void cudaCheckError(int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
		exit(1);
	}
}

int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length) 
{
	unsigned int i = 0;
	while (i!=length){
		if ( score_matrix_cpu[i]==score_matrix[i] ){
			++i;
			continue;
		}
		else {
			printf("i = %d\n",i);	
			return 0;
		}	
	}
	return 1;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

double gettime(){
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
	fprintf(stderr, "Usage: %s <pair number> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<pair number>  - times of comparison\n");
	fprintf(stderr, "\t<penalty> - penalty (negative integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
	double time, end_time;
	int pair_num;
	short penalty;
    char sequence_set1[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0}, sequence_set2[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0};
	unsigned int pos1[MAX_SEQ_NUM] = {0}, pos2[MAX_SEQ_NUM] = {0}, max_len[MAX_SEQ_NUM] = {0}, pos_matrix[MAX_SEQ_NUM] = {0};
	unsigned int dim_matrix[MAX_SEQ_NUM] = {0};
	unsigned int maxLength = 0;
	int *score_matrix;
	int *score_matrix_cpu;
	char *d_sequence_set1, *d_sequence_set2;
	unsigned int *d_pos1, *d_pos2, *d_pos_matrix, *d_dim_matrix;
	int *d_score_matrix;
	int seq1_len, seq2_len;

	if (argc == 3)
	{
		pair_num = atoi(argv[1]);
		penalty = atoi(argv[2]);
		if (pair_num>MAX_SEQ_NUM){
			fprintf(stderr, "\t<pair number>  - times of comparison should be less than %d\n",MAX_SEQ_NUM);
			exit(1);
		}
	}
    else{
		usage(argc, argv);
    }

	// first API
	time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(0) );
	
	end_time = gettime();
	fprintf(stdout,"First API,%lf\n",end_time-time);
	time = end_time;
	
	// Get input data
	  
	srand ( 7 );
	pos_matrix[0] = max_len[0] = pos1[0] = pos2[0] = 0;
	for (int i=0; i<pair_num; ++i){
		//please define your own sequence 1
		seq1_len = 2048; //64+rand() % 20;
		//printf("Seq1 length: %d\n", seq1_len);	
		for (int j=0; j<seq1_len; ++j)		
			sequence_set1[ pos1[i] + j ] = rand() % 20 + 1;
		pos1[i+1] = pos1[i] + seq1_len;
		//please define your own sequence 2.
		seq2_len = 2048;//64+rand() % 20;		
		//printf("Seq2 length: %d\n\n", seq2_len);		
		for (int j=0; j<seq2_len; ++j)
			sequence_set2[ pos2[i] +j ] = rand() % 20 + 1;
		pos2[i+1] = pos2[i] + seq2_len;

		max_len[i] = seq1_len >= seq2_len ? seq1_len : seq2_len;
		if ( max_len[i]>maxLength )
			maxLength = max_len[i];
		dim_matrix[i] = ceil( (double)max_len[i]/TILE_SIZE)*TILE_SIZE;
		
		// notice that the matrix has 1 row and 1 column margins
		//printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
		pos_matrix[i+1] = pos_matrix[i] + (dim_matrix[i]+1)*(dim_matrix[i]+1);
	}

	score_matrix = (int *)malloc( pos_matrix[pair_num]*sizeof(int));
	score_matrix_cpu = (int *)malloc( pos_matrix[pair_num]*sizeof(int)); 
	needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);

	// printf("Start Needleman-Wunsch\n");

	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1, sizeof(char)*pos1[pair_num] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2, sizeof(char)*pos2[pair_num] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix, sizeof(int)*pos_matrix[pair_num]) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1, sizeof(unsigned int)*(pair_num+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2, sizeof(unsigned int)*(pair_num+1) ) );
 	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix, sizeof(unsigned int)*(pair_num+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_dim_matrix, sizeof(unsigned int)*(pair_num+1) ) );
	// CPU phases
	end_time = gettime();
	fprintf(stdout,"CPU,%lf\n",end_time-time);
	time = end_time;
	
	// Memcpy to device
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set1, sequence_set1, sizeof(char)*pos1[pair_num], cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set2, sequence_set2, sizeof(char)*pos2[pair_num], cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_pos1, pos1, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_pos2, pos2, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
 	cudaCheckError( __LINE__, cudaMemcpy( d_pos_matrix, pos_matrix, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_dim_matrix, dim_matrix, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
	
	//end_time = gettime();
	//fprintf(stdout,"Memcpy to device,%lf\n",end_time-time);
	//time = end_time;

	// the threads in block should equal to the STRIDE_SIZE

	int tileSize = TILE_SIZE;
	int iteration = maxLength / tileSize + 1;
	if ( maxLength%tileSize==0 )
		iteration--;
	dim3 dimGrid(1,1);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	needleman_cuda_init<<< pair_num, 512>>>(d_score_matrix, d_pos_matrix, d_dim_matrix, penalty);
	//process top-left matrix
	for( int i = 1; i <= iteration; ++i) {
		dimGrid.x = pair_num;
		dimGrid.y = i;
		needleman_cuda_tile_upleft<<<dimGrid, dimBlock>>>(d_sequence_set1, d_sequence_set2, d_pos1, d_pos2,
														d_score_matrix, d_pos_matrix, d_dim_matrix, i, penalty);
	}
	//process bottom-right matrix
	for( int i = iteration - 1; i >= 1 ; i--){
		dimGrid.x = pair_num;
		dimGrid.y = i;
		needleman_cuda_tile_bottomright<<<dimGrid, dimBlock>>>(d_sequence_set1, d_sequence_set2, d_pos1, d_pos2,
															d_score_matrix, d_pos_matrix, d_dim_matrix, i, penalty);	
	}

	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	//end_time = gettime();
	//fprintf(stdout,"kernel,%lf\n",end_time-time);
	//time = end_time;
	// Memcpy to host
	cudaCheckError( __LINE__, cudaMemcpy( score_matrix, d_score_matrix, sizeof(int)*pos_matrix[pair_num], cudaMemcpyDeviceToHost ) );    

	end_time = gettime();
	//fprintf(stdout,"Memcpy to host,%lf\n",end_time-time);
	fprintf(stdout,"Total CUDA implementation time, %lf\n",end_time-time);
	time = end_time;
	
	if ( validation(score_matrix_cpu, score_matrix, pos_matrix[pair_num]) )		
		printf("Validation: PASS\n");
	else
		printf("Validation: FAIL\n");		

#ifdef TRACEBACK
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
#endif
//	fclose(fpo);
	cudaFree(d_sequence_set1);
	cudaFree(d_sequence_set2);
	cudaFree(d_pos1);
	cudaFree(d_pos2);
 	cudaFree(d_pos_matrix);
	cudaFree(d_score_matrix);
	free(score_matrix);
	
}
