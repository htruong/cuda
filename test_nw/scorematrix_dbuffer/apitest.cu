#define MAX_SEQ_NUM 1024
#define MAX_SEQ_LEN 2100
#define LENGTH 1536


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

#include "needle_gpu.h"
#include "needle_cpu.h"



void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <pairs count> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<pairs count>  - how many pairs you want to do at once\n");
	fprintf(stderr, "\t<penalty> - penalty(negative integer)\n");
	exit(1);
}

double get_time(){
	struct timeval t;
	gettimeofday(&t,NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}


int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length)
{
    for (unsigned int i = 0; i < length; i++) {
		if ( (score_matrix_cpu[i]) != (score_matrix[i]) ) {
			printf("i = %d, seq pair=%d, element %d, expected %d, got %d.\n",i, i / (LENGTH*LENGTH), i % (LENGTH*LENGTH), score_matrix_cpu[i], score_matrix[i]);
			return 0;
		}
	}
    return 1;
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

	// Get input data
	srand ( 7 );
	pos_matrix[0] = pos1[0] = pos2[0] = 0;
	for (int i=0; i<pair_num; i++) {
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

	printf ("Allocating %dMB of memory... \
		(sizeof int=%d bytes, sizeof short=%d bytes)\n",
		pos_matrix[pair_num]*sizeof(int)/1024/1024,
		sizeof(int),
		sizeof(short)
	);

	time = get_time();
	needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);

	// CPU phases
	end_time = get_time();
	fprintf(stdout,"CPU calc: %f\n",end_time-time);
	// We need to free the score matrix for the cpu to prevent biasness against the scoring for the GPU calc
	free(score_matrix_cpu);






	/* NOTE --- ACTUAL CUDA API CODE BEGINS HERE --- */
	
	#ifdef DUAL_BUFFERING
	cudaMallocHost((void **) &score_matrix, pos_matrix[pair_num]*sizeof(int));
	#else
	score_matrix = (int *)malloc(pos_matrix[pair_num]*sizeof(int));
	#endif

	void * ctx;
	
	time = get_time();
	ctx = needle_init(
		0, // GPU number I should work on
		LENGTH, // Max length per sequence
		sequence_set1, // Pointer to sequence set 1
		sequence_set2, // Pointer to sequence set 2
		pos1, // Pointer arrays to set 1
		pos2, // Pointer arrays to set 2
		score_matrix, // Score matrix to store
		pos_matrix // Position matrix to store back
		);
	end_time = get_time();
	fprintf(stdout,"GPU init: %f\n", end_time-time);

	// ---- GPU should have be initialized now ----
	// You can call this as many times as you like in the loop
	
	time = get_time();
	needle_align(ctx, pair_num);
	end_time = get_time();
	fprintf(stdout,"GPU calc: %f\n",end_time-time);

	// ---- Finalize ---
	time = get_time();
	needle_finalize(ctx);
	end_time = get_time();
	fprintf(stdout,"GPU cleanup: %f\n",end_time-time);


	/* NOTE --- ACTUAL CUDA API CODE ENDS HERE --- */

	
	/////////////////
	fprintf(stdout,"Recalculating the score matrix to verify correctness...\n",0);

	score_matrix_cpu = (int *)malloc( pos_matrix[pair_num]*sizeof(int));
	needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);

	if ( validation(score_matrix_cpu, score_matrix, pos_matrix[pair_num]) )
		printf("Validation: PASS\n", 0);
	else
		printf("Validation: FAIL\n", 0);


	//	fclose(fpo);
	free(score_matrix_cpu);
	cudaFreeHost(score_matrix);

	cudaDeviceReset();
}

////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
	runTest( argc, argv);
	return EXIT_SUCCESS;
}



