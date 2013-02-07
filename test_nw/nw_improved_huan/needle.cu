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


#define LENGTH 2000
#define TRACEBACK
// includes, kernels
#include "needle_cpu.c"
//#include "needle_kernel_dynamic.cu"
#include "needle_kernel_diagonal.cu"

inline void cudaCheckError(int line, cudaError_t ce)
{
    if (ce != cudaSuccess) {
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        exit(1);
    }
}

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
    char sequence_set1[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0}, sequence_set2[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0};
    unsigned int pos1[MAX_SEQ_NUM] = {0}, pos2[MAX_SEQ_NUM] = {0}, pos_matrix[MAX_SEQ_NUM] = {0};
    int *score_matrix;
    int *trace_matrix;
    int *score_matrix_cpu;
    int *trace_matrix_cpu;
    char *d_sequence_set1, *d_sequence_set2;
    unsigned int *d_pos1, *d_pos2, *d_pos_matrix;
    int *d_score_matrix;
    int seq1_len, seq2_len;

    if (argc == 3)
    {
        pair_num = atoi(argv[1]);
        penalty = atoi(argv[2]);
        if (pair_num>MAX_SEQ_NUM) {
            fprintf(stderr, "\t<number of pairs>  - number of pairs, must be less than %d\n",MAX_SEQ_NUM);
            exit(1);
        }
    }
    else {
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
    pos_matrix[0] = pos1[0] = pos2[0] = 0;
    for (int i=0; i<pair_num; ++i) {
        //please define your own sequence 1
        seq1_len = LENGTH; //64+rand() % 20;
        //printf("Seq1 length: %d\n", seq1_len);
        for (int j=0; j<seq1_len; ++j)
            sequence_set1[ pos1[i] + j ] = rand() % 20 + 'A';
        pos1[i+1] = pos1[i] + seq1_len;
        //please define your own sequence 2.
        seq2_len = LENGTH;//64+rand() % 20;
        //printf("Seq2 length: %d\n\n", seq2_len);
        for (int j=0; j<seq2_len; ++j)
            sequence_set2[ pos2[i] +j ] = rand() % 20 + 'A';
        pos2[i+1] = pos2[i] + seq2_len;
        //printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
        pos_matrix[i+1] = pos_matrix[i] + (seq1_len+1) * (seq2_len+1);
    }
    score_matrix = (int *)malloc( pos_matrix[pair_num]*sizeof(int));
    
    score_matrix_cpu = (int *)malloc( pos_matrix[pair_num]*sizeof(int));	
    
	#ifdef _LP64
	printf ("Running on a 64-bit platform!\n");
	#else
	#endif

	/*
	short M = -1;
	printf("M: "BYTETOBINARYPATTERN" "BYTETOBINARYPATTERN"\n",
		BYTETOBINARY(M>>8), BYTETOBINARY(M));
    */
    
    printf ("Allocating %dMB of memory... \
		(sizeof int=%d bytes, sizeof short=%d bytes)\n",
		pos_matrix[pair_num]*sizeof(int)/1024/1024,
		sizeof(int),
		sizeof(short)
	);
	
    needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);

    // printf("Start Needleman-Wunsch\n");

    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1, sizeof(char)*pos1[pair_num] ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2, sizeof(char)*pos2[pair_num] ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix, sizeof(int)*pos_matrix[pair_num]) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1, sizeof(unsigned int)*(pair_num+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2, sizeof(unsigned int)*(pair_num+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix, sizeof(unsigned int)*(pair_num+1) ) );

    // CPU phases
    end_time = gettime();
    fprintf(stdout,"CPU,%lf\n",end_time-time);
    time = end_time;

    // Memcpy to device
    cudaCheckError( __LINE__,
		cudaMemcpy( d_sequence_set1, sequence_set1, sizeof(char)*pos1[pair_num], cudaMemcpyHostToDevice )
	);
	
    cudaCheckError( __LINE__,
		cudaMemcpy( d_sequence_set2, sequence_set2, sizeof(char)*pos2[pair_num], cudaMemcpyHostToDevice )
	);
	
    cudaCheckError( __LINE__,
		cudaMemcpy( d_pos1, pos1, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice )
	);
	
    cudaCheckError( __LINE__,
		cudaMemcpy( d_pos2, pos2, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice )
	);
	
    cudaCheckError( __LINE__,
		cudaMemcpy( d_pos_matrix, pos_matrix, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice )
	);

    //end_time = gettime();
    //fprintf(stdout,"Memcpy to device,%lf\n",end_time-time);
    //time = end_time;

    needleman_cuda_diagonal<<<pair_num,512>>>(d_sequence_set1, d_sequence_set2,
            d_pos1, d_pos2,
            d_score_matrix, d_pos_matrix,
            pair_num, penalty);
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
		printf("Here comes the result of the first pair...\n");
		int seq1_begin = pos1[0];
		int seq1_end = pos1[1];
		int seq2_begin = pos2[0];
		int seq2_end = pos2[1];
		int *current_matrix = score_matrix + pos_matrix[0];
		printf("1st seq len = %d =\n%.*s\n", seq1_end - seq1_begin, seq1_end - seq1_begin, sequence_set1 + seq1_begin);
		printf("2nd seq len = %d =\n%.*s\n", seq2_end - seq2_begin, seq2_end - seq2_begin, sequence_set2 + seq2_begin);
		printf("traceback = \n");
		bool done = false;
		int current_pos = ((seq1_end - seq1_begin)+1) * ((seq2_end - seq2_begin)+1) -1; // start at the last cell of the matrix

		for (int i = 0; i < LENGTH + 1; i++) {
			for (int j = 0; j < LENGTH + 1; j++) {
				int dir = current_matrix[i*(LENGTH+1)+j];
				if ((dir & 0x03) == TRACE_UL) {
					printf("\\");
				} else if ((dir & 0x03) == TRACE_U) {
					printf("^");
				} else if ((dir & 0x03) == TRACE_L) {
					printf("<");
				} else {
					printf("-");
				}
			}
			printf("\n");
		}


		for (int i = 0; i < LENGTH + 1; i++) {
			for (int j = 0; j < LENGTH + 1; j++) {
				int dir = current_matrix[i*(LENGTH+1)+j] >> 2;
				printf("%4d ", dir);
			}
			printf("\n");
		}
		
		printf("Actual traceback:\n");
		while (!done) {
			int dir = current_matrix[current_pos];
//			printf("current_pos = %d, dir = %x, score = %d\n", current_pos, dir & 0x03, dir >> 2);
			
			if ((dir & 0x03) == TRACE_UL) {
				printf("\\");
				current_pos = current_pos - (seq1_end - seq1_begin + 1) - 1;
			} else if ((dir & 0x03) == TRACE_U) {
				printf("^");
				current_pos = current_pos - (seq1_end - seq1_begin + 1);
			} else if ((dir & 0x03) == TRACE_L) {
				printf("<");
				current_pos = current_pos - 1;
			} else {
				printf("seems to have reached the origin...");
				done = true;
			}
		}
		printf("traceback done!\n");
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
