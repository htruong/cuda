#ifndef __NEEDLE_CPU_H__
#define __NEEDLE_CPU_H__

void needleman_cpu(char *sequence_set1, 
				   char *sequence_set2, 
				   unsigned int *pos1, 
				   unsigned int *pos2,
				   int *score_matrix,
				   unsigned int *pos_matrix,
				   unsigned int max_pair_no, 
				   short penalty);


#endif	// __NEEDLE_CPU_H__
