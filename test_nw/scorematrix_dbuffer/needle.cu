#define LIMIT -999
#define BLOCK_SIZE 16
#define MAX_SEQ_LEN 2100
#define MAX_SEQ_NUM 1024
#define LENGTH 1600

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
#include "needle_kernel_dynamic.cu"
#include "needle_kernel_diagonal.cu"

inline void cudaCheckError(int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
		cudaDeviceReset();
		exit(1);
	}
	
}

int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length)
{
    unsigned int i = 0;
    while (i!=length) {
        if ( (score_matrix_cpu[i]) == (score_matrix[i]) ) {
            ++i;
            continue;
        }
        else {
            printf("i = %d, expected %d, got %d.\n",i, score_matrix_cpu[i], score_matrix[i]);
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
	fprintf(stderr, "Usage: %s <pairs count> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<pairs count>  - how many pairs you want to do at once\n");
	fprintf(stderr, "\t<penalty> - penalty(negative integer)\n");
	exit(1);
}


void memcpy_and_run (bool async,
				unsigned int begin,
				unsigned int end,
				cudaStream_t * stream,
				char *sequence_set1,
				char *sequence_set2,
				char *d_sequence_set1,
				char *d_sequence_set2,
				unsigned int *pos1,
				unsigned int *pos2,
				unsigned int *d_pos1,
				unsigned int *d_pos2,
				int *score_matrix,
				unsigned int *pos_matrix,
				int *d_score_matrix,
				unsigned int *d_pos_matrix,
				short penalty) {
				
		unsigned int batch_size = end-begin;
		// Memcpy to device

		#ifdef VERBOSE
		double start_marker = 0;
		start_marker = gettime();
		printf("-- Start calculation from %d to %d --\n", begin, end);
		#endif
		if (async) {
			cudaMemcpyAsync( d_sequence_set1, sequence_set1 + pos1[begin], sizeof(char)*(pos1[end] - pos1[begin]), cudaMemcpyHostToDevice, *stream);
			cudaMemcpyAsync( d_sequence_set2, sequence_set2 + pos2[begin], sizeof(char)*(pos2[end] - pos2[begin]), cudaMemcpyHostToDevice, *stream);
			cudaMemcpyAsync( d_pos1, pos1 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice, *stream );
			cudaMemcpyAsync( d_pos2, pos2 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice, *stream );
			cudaMemcpyAsync( d_pos_matrix, pos_matrix /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice, *stream );
		} else {
			cudaMemcpy( d_sequence_set1, sequence_set1 + pos1[begin], sizeof(char)*(pos1[end] - pos1[begin]), cudaMemcpyHostToDevice );
			cudaMemcpy( d_sequence_set2, sequence_set2 + pos2[begin], sizeof(char)*(pos2[end] - pos2[begin]), cudaMemcpyHostToDevice );
			cudaMemcpy( d_pos1, pos1 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice );
			cudaMemcpy( d_pos2, pos2 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice );
			cudaMemcpy( d_pos_matrix, pos_matrix /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice );
		}

		#ifdef VERBOSE
		printf("\t [%d - %d] Memcpy CPU-GPU: %f\n", begin, end, gettime() - start_marker);
		start_marker = gettime();
		#endif
		
		if (async) {
			needleman_cuda_diagonal<<<batch_size,512, 0, *stream>>>(d_sequence_set1, d_sequence_set2,
					d_pos1, d_pos2,
					d_score_matrix, d_pos_matrix,
					batch_size, penalty);
		} else {
			needleman_cuda_diagonal<<<batch_size,512>>>(d_sequence_set1, d_sequence_set2,
					d_pos1, d_pos2,
					d_score_matrix, d_pos_matrix,
					batch_size, penalty);
		}
		//cudaCheckError( __LINE__, cudaDeviceSynchronize() );
		
		#ifdef VERBOSE
		printf("\t [%d - %d] Kernel: %f\n", begin, end, gettime() - start_marker);
		start_marker = gettime();
		#endif
		if (async) {
			cudaMemcpyAsync( score_matrix + pos_matrix[begin], d_score_matrix, sizeof(int)*(pos_matrix[end] - pos_matrix[begin]), cudaMemcpyDeviceToHost, *stream );
		} else {
			cudaMemcpy( score_matrix + pos_matrix[begin], d_score_matrix, sizeof(int)*(pos_matrix[end] - pos_matrix[begin]), cudaMemcpyDeviceToHost );
		}
		#ifdef VERBOSE
		printf("\t [%d - %d] Memcpy GPU-CPU: %f\n", begin, end, gettime() - start_marker);
		#endif
}

void needleman_gpu_sync(char *sequence_set1,
				char *sequence_set2,
				unsigned int *pos1,
				unsigned int *pos2,
				int *score_matrix,
				unsigned int *pos_matrix,
				unsigned int max_pair_no,
				short penalty)
{
}

void needleman_gpu(char *sequence_set1,
				char *sequence_set2,
				unsigned int *pos1,
				unsigned int *pos2,
				int *score_matrix,
				unsigned int *pos_matrix,
				unsigned int max_pair_no,
				short penalty)
{

	double start_marker; // Start time marker
	
	// First we need to see how to devide the memory...
	// Query the device capabilities to see how much we can allocate for this problem

	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("GPU Memory avaliable: Free: %lu, Total: %lu\n",freeMem/1024/1024, totalMem/1024/1024);

	unsigned int eachSeqMem = sizeof(char)*LENGTH*2
					+ sizeof(int)*(LENGTH+1)*(LENGTH+1)
					+ sizeof(unsigned int)*3;
	unsigned int batch_size = freeMem * 0.8 / eachSeqMem; // Safety reasons...
	unsigned int half_b = batch_size / 2;
	unsigned int other_half_b = batch_size - half_b;

	printf("Each batch will be doing this many pairs: %d\n", batch_size);

	////////////////////////////////////////////////////////////////////////////

	// This implementation comes with the assumption that all sequences will be
	// having the same size.

	char *d_sequence_set1_h1, *d_sequence_set2_h1, *d_sequence_set1_h2, *d_sequence_set2_h2;
	unsigned int *d_pos1_h1, *d_pos2_h1, *d_pos_matrix_h1, *d_pos1_h2, *d_pos2_h2, *d_pos_matrix_h2;
	int *d_score_matrix_h1, *d_score_matrix_h2;
	
	cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    
	start_marker = gettime();
	// Allocating memory for both halves
	
	// First half
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1_h1, sizeof(char)*(pos1[half_b]) ));
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2_h1, sizeof(char)*(pos2[half_b])) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix_h1, sizeof(int)*(pos_matrix[half_b])) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1_h1, sizeof(unsigned int)*(half_b+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2_h1, sizeof(unsigned int)*(half_b+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix_h1, sizeof(unsigned int)*(half_b+1) ) );

    // Second half
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1_h2, sizeof(char)*(pos1[other_half_b]) ));
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2_h2, sizeof(char)*(pos2[other_half_b])) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix_h2, sizeof(int)*(pos_matrix[other_half_b])) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1_h2, sizeof(unsigned int)*(other_half_b+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2_h2, sizeof(unsigned int)*(other_half_b+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix_h2, sizeof(unsigned int)*(other_half_b+1) ) );
    
	fprintf(stdout,"cudaMalloc = %f\n", gettime()-start_marker);
	
	bool done = false;

	unsigned int start = 0;
	unsigned int end = 0;
	
	while (!done) {
		if (start + half_b > max_pair_no) {
			end = max_pair_no;
			
			memcpy_and_run (true,
				start,
				end,
				&stream1,
				sequence_set1,
				sequence_set2,
				d_sequence_set1_h1,
				d_sequence_set2_h1,
				pos1,
				pos2,
				d_pos1_h1,
				d_pos2_h2,
				score_matrix,
				pos_matrix,
				d_score_matrix_h1,
				d_pos_matrix_h1,
				penalty);
				
			done = true;
		} else {
			end = start + half_b;

			memcpy_and_run (true,
				start,
				end,
				&stream1,
				sequence_set1,
				sequence_set2,
				d_sequence_set1_h1,
				d_sequence_set2_h1,
				pos1,
				pos2,
				d_pos1_h1,
				d_pos2_h2,
				score_matrix,
				pos_matrix,
				d_score_matrix_h1,
				d_pos_matrix_h1,
				penalty);

			start = end;
			
				if (start + other_half_b > max_pair_no) {
					end = max_pair_no;
					
					memcpy_and_run (true,
						start,
						end,
						&stream2,
						sequence_set1,
						sequence_set2,
						d_sequence_set1_h1,
						d_sequence_set2_h1,
						pos1,
						pos2,
						d_pos1_h1,
						d_pos2_h2,
						score_matrix,
						pos_matrix,
						d_score_matrix_h1,
						d_pos_matrix_h1,
						penalty);

					done = true;
				} else {
					end = start + other_half_b;

					memcpy_and_run (true,
						start,
						end,
						&stream2,
						sequence_set1,
						sequence_set2,
						d_sequence_set1_h1,
						d_sequence_set2_h1,
						pos1,
						pos2,
						d_pos1_h1,
						d_pos2_h2,
						score_matrix,
						pos_matrix,
						d_score_matrix_h1,
						d_pos_matrix_h1,
						penalty);
					
				}
		}
		
		start = end;
		
	}
	cudaDeviceSynchronize();
	
	cudaFree(d_sequence_set1_h1);
	cudaFree(d_sequence_set1_h2);
	cudaFree(d_sequence_set2_h1);
	cudaFree(d_sequence_set2_h2);
	cudaFree(d_pos1_h1);
	cudaFree(d_pos1_h2);
	cudaFree(d_pos2_h2);
	cudaFree(d_pos2_h2);
	cudaFree(d_pos_matrix_h1);
	cudaFree(d_pos_matrix_h2);
	cudaFree(d_score_matrix_h1);
	cudaFree(d_score_matrix_h2);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}


void runTest( int argc, char** argv)
{
	double time, end_time;
	int pair_num;
	short penalty;
	char sequence_set1[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0}, sequence_set2[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0};
	unsigned int pos1[MAX_SEQ_NUM] = {0}, pos2[MAX_SEQ_NUM] = {0}, pos_matrix[MAX_SEQ_NUM] = {0};
	int *score_matrix;
	int *score_matrix_cpu;
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

	// Get input data

	srand ( 7 );
	pos_matrix[0] = pos1[0] = pos2[0] = 0;
	for (int i=0; i<pair_num; ++i) {
		//please define your own sequence 1
		seq1_len = LENGTH; //64+rand() % 20;
		//printf("Seq1 length: %d\n", seq1_len);
		for (int j=0; j<seq1_len; ++j)
			sequence_set1[ pos1[i] + j ] = rand() % 20;
		pos1[i+1] = pos1[i] + seq1_len;
		//please define your own sequence 2.
		seq2_len = LENGTH;//64+rand() % 20;
		//printf("Seq2 length: %d\n\n", seq2_len);
		for (int j=0; j<seq2_len; ++j)
			sequence_set2[ pos2[i] +j ] = rand() % 20;
		pos2[i+1] = pos2[i] + seq2_len;
		//printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
		pos_matrix[i+1] = pos_matrix[i] + (seq1_len+1) * (seq2_len+1);
	}

	score_matrix_cpu = (int *)malloc( pos_matrix[pair_num]*sizeof(int));

	#ifdef _LP64
	printf ("Running on a 64-bit platform!\n", 0);
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

	time = gettime();
	needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);

	// CPU phases
	end_time = gettime();
	fprintf(stdout,"CPU calc: %lf\n",end_time-time);
	// We need to free the score matrix for the cpu to prevent biasness against the scoring for the GPU calc
	free(score_matrix_cpu);

	cudaMallocHost((void **) &score_matrix, pos_matrix[pair_num]*sizeof(int));

	time = gettime();
	needleman_gpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix, pos_matrix, pair_num, penalty);

	// GPU phases
	end_time = gettime();
	fprintf(stdout,"GPU calc: %lf\n",end_time-time);

	/////////////////
	fprintf(stdout,"Recalculating the score matrix to verify correctness...\n",0);

	score_matrix_cpu = (int *)malloc( pos_matrix[pair_num]*sizeof(int));
	needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);


	if ( validation(score_matrix_cpu, score_matrix, pos_matrix[pair_num]) )
		printf("Validation: PASS\n", 0);
	else
		printf("Validation: FAIL\n", 0);

	cudaDeviceReset();
	
	#ifdef TRACEBACK
		printf("Here comes the result of the first pair...\n", 0);
		int seq1_begin = pos1[0];
		int seq1_end = pos1[1];
		int seq2_begin = pos2[0];
		int seq2_end = pos2[1];
		int *current_matrix = score_matrix + pos_matrix[0];
		printf("1st seq len = %d =\n%.*s\n", seq1_end - seq1_begin, seq1_end - seq1_begin, sequence_set1 + seq1_begin);
		printf("2nd seq len = %d =\n%.*s\n", seq2_end - seq2_begin, seq2_end - seq2_begin, sequence_set2 + seq2_begin);
		printf("traceback = \n", 0);
		bool done = false;
		int current_pos = ((seq1_end - seq1_begin)+1) * ((seq2_end - seq2_begin)+1) -1; // start at the last cell of the matrix

		// Fix LENGTH, so that it takes more than just square... this is not important
		for (int i = 0; i < LENGTH + 1; i++) {
			for (int j = 0; j < LENGTH + 1; j++) {
				int dir = current_matrix[i*(LENGTH+1)+j];
				if ((dir & 0x03) == TRACE_UL) {
					printf("\\", 0);
				} else if ((dir & 0x03) == TRACE_U) {
					printf("^", 0);
				} else if ((dir & 0x03) == TRACE_L) {
					printf("<", 0);
				} else {
					printf("-", 0);
				}
			}
			printf("\n", 0);
		}

		// Fix LENGTH, so that it takes more than just square... this is not important
		for (int i = 0; i < LENGTH + 1; i++) {
			for (int j = 0; j < LENGTH + 1; j++) {
				int dir = current_matrix[i*(LENGTH+1)+j] >> 2;
				printf("%4d ", dir);
			}
			printf("\n", 0);
		}

		printf("Actual traceback:\n", 0);
		while (!done) {
			int dir = current_matrix[current_pos];
//			printf("current_pos = %d, dir = %x, score = %d\n", current_pos, dir & 0x03, dir >> 2);

			if ((dir & 0x03) == TRACE_UL) {
				printf("\\", 0);
				current_pos = current_pos - (seq1_end - seq1_begin + 1) - 1;
			} else if ((dir & 0x03) == TRACE_U) {
				printf("^", 0);
				current_pos = current_pos - (seq1_end - seq1_begin + 1);
			} else if ((dir & 0x03) == TRACE_L) {
				printf("<", 0);
				current_pos = current_pos - 1;
			} else {
				printf("*", 0);
				done = true;
			}
		}
		printf("traceback done!\n", 0);
	#endif

	//	fclose(fpo);
	//free(score_matrix);


}
