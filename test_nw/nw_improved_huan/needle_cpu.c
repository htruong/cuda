#include "needle_cpu.h"

short max_3( short a, short b, short c )
{
	short temp = a>b ? a: b; 
	return c>temp ? c: temp; 
}

void needleman_cpu(char *sequence_set1, 
				   char *sequence_set2, 
				   unsigned int *pos1, 
				   unsigned int *pos2,
				   int *score_matrix, 
				   unsigned int *pos_matrix,
				   unsigned int max_pair_no, 
				   short penalty)
{
		for (int i=0; i<max_pair_no; ++i){
			
			char *seq1 = sequence_set1+pos1[i];
			char *seq2 = sequence_set2+pos2[i];
			int seq1_len = pos1[i+1] - pos1[i];
			int seq2_len = pos2[i+1] - pos2[i];
			int *matrix = score_matrix + pos_matrix[i];
			int dia, up, left;
			for(int j = 1; j <= seq1_len; ++j)
				matrix[j] = penalty * j;
			for(int j = 1; j <= seq2_len; ++j)
				matrix[j*(seq1_len+1)+0] = penalty * j;

			//fill the score matrix
			for(int k = 1; k <= seq2_len; ++k){           
				for(int j = 1; j <= seq1_len; ++j){						
					dia = matrix[(k-1)*(seq1_len+1)+j-1] + ((seq2[k-1] == seq1[j-1]) ? 1 : -1);
				//	dia = matrix[(k-1)*(seq1_len+1)+j-1]+blosum62_cpu[ seq2[k-1] ][ seq1[j-1] ];
					up	= matrix[(k-1)*(seq1_len+1)+j] + penalty;
					left= matrix[k*(seq1_len+1)+j-1] + penalty;
					matrix[k*(seq1_len+1)+j] = max_3(left, dia, up);
				}
			}
		}
}
