#include <needle.h>
#include <stdio.h>
#include <cuda.h>

// 1:dia 2:up 4:left

#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

/*******************************************************************************
	pos1 and pos2 are the array store the position 
********************************************************************************/
__global__ void needleman_cuda_traceback(char *direction_matrix, unsigned int *pos_matrix, 
									     unsigned int *pos1, unsigned int *pos2,
										 char *trace_vector, unsigned int max_pair_no)
{
	int pair_no = blockIdx.x;	// for each block, caculate one pair
	char *matrix = direction_matrix+pos_matrix[pair_no];
	int seq1_len = pos1[pair_no+1] - pos1[pair_no]; 
	int seq2_len = pos2[pair_no+1] - pos2[pair_no];
	int index_x, index_y, trace_len = 0;
	char direct;
	int trace_max_len = MAX_TRACE_LEN;
	index_x = seq2_len;
	index_y = seq1_len;
	if (pair_no<max_pair_no){
	while ( index_x!=0 || index_y!=0 ) {
		direct = matrix[index_x*(seq1_len+1)+index_y];
		switch (direct) {
			case 1:
				index_x = index_x - 1;				
				index_y = index_y - 1;
				trace_vector[pair_no*trace_max_len+trace_len] = direct;
				break;
			case 2:
				index_x = index_x - 1;
				trace_vector[pair_no*trace_max_len+trace_len] = direct;				
				break;
			case 4:
				index_y = index_y - 1;
				trace_vector[pair_no*trace_max_len+trace_len] = direct;				
				break;
			default: break;
		}
		trace_len++;	
	}
	}
}
