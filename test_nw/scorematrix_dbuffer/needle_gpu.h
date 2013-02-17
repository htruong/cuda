#ifndef __NEEDLE_H__
#define __NEEDLE_H__

// Huan and Daniel's Needleman-Wunch's API

void * needle_prepare(
	const int gpu_num,
	unsigned int max_length_per_seq,
	unsigned int * optimal_batch_size,
	char * gpu_name
);	

void needle_allocate(
    void *ctx,
	char *sequence_set1,
	char *sequence_set2,
	unsigned int *pos1,
	unsigned int *pos2,
	int *score_matrix,
	unsigned int *pos_matrix
	);

void needle_align(void * ctx, int num_pairs);

void needle_finalize(void * ctx);

#endif	//__NEEDLE_H__
