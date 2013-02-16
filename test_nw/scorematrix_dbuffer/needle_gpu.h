#ifndef __NEEDLE_H__
#define __NEEDLE_H__

// Huan and Daniel's Needleman-Wunch's API

void * needle_init(
	const int gpu_num, // GPU number I should work on
	unsigned int max_length_per_seq,
	char *sequence_set1, // Pointer to sequence set 1
	char *sequence_set2, // Pointer to sequence set 2
	unsigned int *pos1, // Pointer arrays to set 1
	unsigned int *pos2, // Pointer arrays to set 2
	int *score_matrix, // Score matrix to store
	unsigned int *pos_matrix // Position matrix to store back
);

void needle_align(void * needle_ctx, int num_pairs);

void needle_finalize(void * needle_ctx);

#endif	//__NEEDLE_H__
