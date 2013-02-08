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
//#include "needle_cpu.c"
//#include "needle_kernel_dynamic.cu"
#include "needle_kernel_diagonal.cu"

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
	fprintf(stderr, "\t<penalty> - penalty(negative integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
	double time, end_time;
	int pair_num;
	short penalty;
    char sequence_set1[2][MAX_SEQ_LEN*MAX_SEQ_NUM] = {0}, sequence_set2[2][MAX_SEQ_LEN*MAX_SEQ_NUM] = {0};
	unsigned int pos1[2][MAX_SEQ_NUM] = {0}, pos2[2][MAX_SEQ_NUM] = {0}, pos_matrix[2][MAX_SEQ_NUM] = {0};
	int *score_matrix[2];
	int *score_matrix_cpu[2];
	char *d_sequence_set1[2], *d_sequence_set2[2];
	unsigned int *d_pos1[2], *d_pos2[2], *d_pos_matrix[2];
	int *d_score_matrix[2];
	int seq1_len, seq2_len;

	if (argc == 3)
	{
		pair_num = atoi(argv[1]);
		penalty = atoi(argv[2]);
		if (pair_num>MAX_SEQ_NUM){
			fprintf(stderr, "\t<pair number>  - times of comparison should be less than %d\n",MAX_SEQ_NUM);
			exit(1);
		}
		if (pair_num<=1){
			fprintf(stderr, "\t<pair number>  - times of comparison should be bigger than 1\n");
			exit(1);
		}
		if (pair_num%2!=0){
			fprintf(stderr, "\t<pair number>  - times of comparison should be even number\n");
			exit(1);
		}
	}
    else{
		usage(argc, argv);
    }

	int pair_num0 = pair_num / 2;
	int pair_num1 = pair_num / 2;
	// first API
	time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(0) );
	
	end_time = gettime();
	fprintf(stdout,"First API,%lf\n",end_time-time);
	time = end_time;
	
	// Get input data
	  
	srand ( 7 );
	// 1st half
	pos_matrix[0][0] = pos1[0][0] = pos2[0][0] = 0;
	for (int i=0; i<pair_num0; ++i){	// first half
		//please define your own sequence 1
		seq1_len = 2048; //64+rand() % 20;
		//printf("Seq1 length: %d\n", seq1_len);	
		for (int j=0; j<seq1_len; ++j)		
			sequence_set1[0][ pos1[0][i] + j ] = rand() % 20 + 1;
		pos1[0][i+1] = pos1[0][i] + seq1_len;
		//please define your own sequence 2.
		seq2_len = 2048;//64+rand(2) % 20;		
		//printf("Seq2 length: %d\n\n", seq2_len);		
		for (int j=0; j<seq2_len; ++j)		
			sequence_set2[0][ pos2[0][i] +j ] = rand() % 20 + 1;
		pos2[0][i+1] = pos2[0][i] + seq2_len;
		//printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
		pos_matrix[0][i+1] = pos_matrix[0][i] + (seq1_len+1) * (seq2_len+1);
	}
	// 2nd half
	pos_matrix[1][0] = pos1[1][0] = pos2[1][0] = 0;
	for (int i=0; i<pair_num1; ++i){	// second half
		seq1_len = 2048; //64+rand() % 20;
		for (int j=0; j<seq1_len; ++j)		
			sequence_set1[1][ pos1[1][i] + j ] = rand() % 20 + 1;
		pos1[1][i+1] = pos1[1][i] + seq1_len;
		seq2_len = 2048;//64+rand() % 20;		
		for (int j=0; j<seq2_len; ++j)		
			sequence_set2[1][ pos2[1][i] +j ] = rand() % 20 + 1;
		pos2[1][i+1] = pos2[1][i] + seq2_len;
		pos_matrix[1][i+1] = pos_matrix[1][i] + (seq1_len+1) * (seq2_len+1);
	}
	score_matrix[0] = (int *)malloc( pos_matrix[0][pair_num0]*sizeof(int));
	score_matrix[1] = (int *)malloc( pos_matrix[1][pair_num1]*sizeof(int));
	score_matrix_cpu = (int *)malloc( pos_matrix[pair_num]*sizeof(int));
	needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);

	// printf("Start Needleman-Wunsch\n");
	// 1st half
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1[0], sizeof(char)*pos1[0][pair_num0] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2[0], sizeof(char)*pos2[0][pair_num0] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix[0], sizeof(int)*pos_matrix[0][pair_num0]) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1[0], sizeof(unsigned int)*(pair_num0+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2[0], sizeof(unsigned int)*(pair_num0+1) ) );
 	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix[0], sizeof(unsigned int)*(pair_num0+1) ) );

	// 2nd half
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1[1], sizeof(char)*pos1[1][pair_num1] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2[1], sizeof(char)*pos2[1][pair_num1] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix[1], sizeof(int)*pos_matrix[1][pair_num1]) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1[1], sizeof(unsigned int)*(pair_num1+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2[1], sizeof(unsigned int)*(pair_num1+1) ) );
 	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix[1], sizeof(unsigned int)*(pair_num1+1) ) );
	
	// CPU phases
	end_time = gettime();
	fprintf(stdout,"CPU,%lf\n",end_time-time);
	time = end_time;
	
	// Memcpy to device
	// 1st half
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set1[0], sequence_set1[0], sizeof(char)*pos1[0][pair_num0], cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set2[0], sequence_set2[0], sizeof(char)*pos2[0][pair_num0], cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_pos1[0], pos1[0], sizeof(unsigned int)*(pair_num0+1), cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_pos2[0], pos2[0], sizeof(unsigned int)*(pair_num0+1), cudaMemcpyHostToDevice ) );
 	cudaCheckError( __LINE__, cudaMemcpy( d_pos_matrix[0], pos_matrix[0], sizeof(unsigned int)*(pair_num0+1), cudaMemcpyHostToDevice ) );
	
	//end_time = gettime();
	//fprintf(stdout,"Memcpy to device,%lf\n",end_time-time);
	//time = end_time;

	needleman_cuda_diagonal<<<pair_num0,512>>>(d_sequence_set1[0], d_sequence_set2[0], 
									   d_pos1[0], d_pos2[0],
									   d_score_matrix[0], d_pos_matrix[0],
									   pair_num0, penalty);
	
	// Memcpy to device
	// 2nd half
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set1[1], sequence_set1[1], sizeof(char)*pos1[1][pair_num1], cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set2[1], sequence_set2[1], sizeof(char)*pos2[1][pair_num1], cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_pos1[1], pos1[1], sizeof(unsigned int)*(pair_num1+1), cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_pos2[1], pos2[1], sizeof(unsigned int)*(pair_num1+1), cudaMemcpyHostToDevice ) );
 	cudaCheckError( __LINE__, cudaMemcpy( d_pos_matrix[1], pos_matrix[1], sizeof(unsigned int)*(pair_num1+1), cudaMemcpyHostToDevice ) );
	
	needleman_cuda_diagonal<<<pair_num1,512>>>(d_sequence_set1[1], d_sequence_set2[1], 
									   d_pos1[1], d_pos2[1],
									   d_score_matrix[1], d_pos_matrix[1],
									   pair_num1, penalty);

	// 1st half
	cudaCheckError( __LINE__, cudaMemcpy(score_matrix[0],d_score_matrix[0],sizeof(int)*pos_matrix[0][pair_num0],cudaMemcpyDeviceToHost));
	// 2nd half	
	cudaCheckError( __LINE__, cudaMemcpy(score_matrix[1],d_score_matrix[1],sizeof(int)*pos_matrix[1][pair_num1], cudaMemcpyDeviceToHost ) );

	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
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
	cudaFree(d_sequence_set1[0]);cudaFree(d_sequence_set1[1]);
	cudaFree(d_sequence_set2[0]);cudaFree(d_sequence_set2[1]);
	cudaFree(d_pos1[0]);cudaFree(d_pos1[1]);
	cudaFree(d_pos2[0]);cudaFree(d_pos2[1]);
 	cudaFree(d_pos_matrix[0]);cudaFree(d_pos_matrix[1]);
	cudaFree(d_score_matrix[0]);cudaFree(d_score_matrix[1]);
	free(score_matrix[0]);free(score_matrix[1]);
	
}
