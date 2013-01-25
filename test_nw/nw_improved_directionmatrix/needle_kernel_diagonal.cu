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
__global__ void needleman_cuda_diagonal(char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   char *direction_matrix, unsigned int *pos_matrix,
									   unsigned int max_pair_no, short penalty)
{
	int pair_no, seq1_len, seq2_len;
	int tid = threadIdx.x;
	// 48 KB/4 = 12KB, seq1+sqe2, diagonal1, diagonal2, diagonal3
	__shared__ char s_seq1[MAX_SEQ_LEN];
	__shared__ char s_seq2[MAX_SEQ_LEN];
	__shared__ short s_dia1[MAX_SEQ_LEN];
	__shared__ short s_dia2[MAX_SEQ_LEN];
	__shared__ short s_dia3[MAX_SEQ_LEN];
	short *p_dia1, *p_dia2, *p_dia3, *p_tmp;
	pair_no = blockIdx.x;	// for each block, caculate one pair
	char *seq1 = sequence_set1 + pos1[pair_no];
	char *seq2 = sequence_set2 + pos2[pair_no];	
	char *matrix = direction_matrix+pos_matrix[pair_no];
	seq1_len = pos1[pair_no+1] - pos1[pair_no]; 
	seq2_len = pos2[pair_no+1] - pos2[pair_no];

	char direct;

	// load the two sequences
	unsigned int stride_length = blockDim.x;
	for (int i=0; i<seq1_len/stride_length+1; ++i){
		if ( tid+i*stride_length<seq1_len )
			s_seq1[tid+i*stride_length+1] = seq1[tid+i*stride_length];
	}	
	for (int i=0; i<seq2_len/stride_length+1; ++i){
		if ( tid+i*stride_length<seq2_len )
		s_seq2[tid+i*stride_length+1] = seq2[tid+i*stride_length];
	}
	__syncthreads();

	/*dia_upperleft( s_seq1, seq1_len, s_seq2, seq2_len, matrix, seq2_len, 
						s_dia1, s_dia2, s_dia3, penalty);*/
	
	/*dia_lowerright( s_seq1, seq1_len, s_seq2, seq2_len, matrix, seq2_len, 
						s_dia1, s_dia2, s_dia3, 1, penalty);*/

	int stripe = blockDim.x;
	int index_x;
	int index_y;
	int iteration;	
	// process the left-up triangle
	s_dia1[0] = matrix[0] = 0;
	s_dia2[0] = penalty * 1;  matrix[1*(seq1_len+1)] = 2;// 1:dia 2:up 4:left
	s_dia2[1] = penalty * 1;  matrix[1] = 4;
	
	p_dia1 = s_dia1;
	p_dia2 = s_dia2;
	p_dia3 = s_dia3;	
	for (int i=2; i<=seq2_len; ++i){	// ith diagonal line		
		iteration = (i+1)/blockDim.x;
		if ( (i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			if ( tid+stripe*j<=i ) {	// ith diagonal has i+1 elements
				index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;
				if ( index_y==0 )	{	// 2; up
					p_dia3[ index_y ] =  penalty * i;
					direct = 2;
				}
				else if ( index_y==i ) {	// 4: left
					p_dia3[ index_y ] =  penalty * i;
					direct = 4;
				}
				else {
					p_dia3[ index_y ] = 		\
						maximum(p_dia2[ index_y-1 ] + penalty,	// left
								p_dia2[ index_y ] + penalty,	// up
								p_dia1[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ],\
								&direct );
				}
				// store to global memory
				matrix[ index_x*(seq1_len+1)+index_y ] = direct;//p_dia3[ index_y ];								
			}
		}
		p_tmp = p_dia1;
		p_dia1 = p_dia2;
		p_dia2 = p_dia3;
		p_dia3 = p_tmp;		
		__syncthreads();
	}

	//int tid = threadIdx.x;
	stripe = blockDim.x;
	//int index_x, index_y;
	iteration = (seq1_len+1)/blockDim.x;
	if ( (seq1_len+1)%blockDim.x!=0 ) iteration++; 
	// calculate the 1th diagonal
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<seq1_len ) {
			index_x = seq2_len - (tid+stripe*i);	index_y = 1 + (tid+stripe*i);
			p_dia3[ tid+stripe*i ] = \
						maximum(p_dia2[ tid+stripe*i ] + penalty,	// left
								p_dia2[ tid+stripe*i+1 ] + penalty,	// up							    
								p_dia1[ tid+stripe*i ] + blosum62[s_seq2[index_x]][s_seq1[index_y]], // dia
						 		&direct);
			matrix[ index_x*(seq1_len+1)+index_y ] = direct;
		}
	}
	p_tmp = p_dia1;
	p_dia1 = p_dia2;
	p_dia2 = p_dia3;
	p_dia3 = p_tmp;		
	__syncthreads();
	for (int i=2; i<=seq1_len; ++i){	// ith diagonal line, start from 2
		iteration = (seq1_len-i+1)/blockDim.x;
		if ( (seq1_len-i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			index_x = seq2_len - (tid+stripe*j);
			index_y =  i + (tid+stripe*j);
			if ( tid+stripe*j +i <seq1_len+1 ) {	
				p_dia3[ tid+stripe*j ] = 	\
					maximum(p_dia2[ tid+stripe*j ] + penalty,	// left
							p_dia2[ tid+stripe*j+1 ] + penalty,	// up
							p_dia1[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ], // dia
							&direct);
				// store to global memory
				matrix[ index_x*(seq1_len+1)+index_y ] = direct;
			}
		}
		p_tmp = p_dia1;
		p_dia1 = p_dia2;
		p_dia2 = p_dia3;
		p_dia3 = p_tmp;
		__syncthreads();
	}

}
