//#include <needle_gpu.h>
//#include <stdio.h>
#include <cuda.h>

#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

#define MAX_SEQ_LEN 2100

__constant__ char blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  7, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

__device__ __host__ short maximum( short a, short b, short c)
{
	int k;
	if( a <= b )	k = b;
	else	k = a;
	if( k <=c )	return(c);
	else	return(k);
}

__device__ void dia_upperleft(char *s_seq1, unsigned int seq1_len, 
								char *s_seq2, unsigned int seq2_len,
								int *matrix, unsigned int dia_len,									
								int *s_dia1, int *s_dia2, int *s_dia3,
								int penalty)
{
	int tid = threadIdx.x;
	int stripe = blockDim.x;
	int index_x;
	int index_y;
	int iteration;
	int *p_dia1, *p_dia2, *p_dia3, *p_tmp;
	// process the left-up triangle
	s_dia1[0] = matrix[0] = 0;
	s_dia2[0] = matrix[1] = penalty * 1; 
	s_dia2[1] = matrix[1*(seq1_len+1)] = penalty * 1;
	
	p_dia1 = s_dia1;
	p_dia2 = s_dia2;
	p_dia3 = s_dia3;	
	for (int i=2; i<=seq2_len; ++i){	// ith diagonal line		
		iteration = (i+1)/blockDim.x;
		if ( (i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			if ( tid+stripe*j<=i ) {	// ith diagonal has i+1 elements
				index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;
				if ( index_y==0 || index_y==i )	p_dia3[ index_y ] =  penalty * i;
				else {
					p_dia3[ index_y ] = 		\
						maximum(p_dia2[ index_y ] + penalty,	// up
						    p_dia2[ index_y-1 ] + penalty,	// left
							p_dia1[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
				}
				// store to global memory
				matrix[ index_x*(seq1_len+1)+index_y ] = p_dia3[ index_y ];
			}
		}
		p_tmp = p_dia1;
		p_dia1 = p_dia2;
		p_dia2 = p_dia3;
		p_dia3 = p_tmp;		
		__syncthreads();
	}
}

__device__ void dia_lowerright( char *s_seq1, unsigned int seq1_len, 
								char *s_seq2, unsigned int seq2_len,
								int *matrix, unsigned int dia_len,									
								int *s_dia1, int *s_dia2, int *s_dia3,
								unsigned int start, int penalty)
{
	int tid = threadIdx.x;
	int stripe = blockDim.x;
	int index_x, index_y;
	int iteration = dia_len/blockDim.x;
	int *p_dia1, *p_dia2, *p_dia3, *p_tmp;
	if ( dia_len%blockDim.x!=0 ) iteration++; 
	// initial, load from shared memory
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<seq1_len+1 ) {
			index_x = seq2_len - (tid+stripe*i);	index_y = (tid+stripe*i);
			s_dia1[ tid+stripe*i ] = matrix[ index_x*(seq1_len+1)+index_y ];
		}
	}	
	__syncthreads();
	p_dia1 = s_dia1;
	p_dia2 = s_dia2;
	p_dia3 = s_dia3;
	// calculate the 1th diagonal
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<seq1_len ) {
			index_x = seq2_len - (tid+stripe*i);	index_y = 1 + (tid+stripe*i);
			p_dia2[ tid+stripe*i ] = \
						maximum(p_dia1[ tid+stripe*i+1 ] + penalty,	// up
							    p_dia1[ tid+stripe*i ] + penalty,	// left
								matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x]][s_seq1[index_y]] );
			matrix[ index_x*(seq1_len+1)+index_y ] = p_dia2[ tid+stripe*i ];
		}
	}
	__syncthreads();
	for (int i=2; i<=seq1_len; ++i){	// ith diagonal line, start from 2
		iteration = (seq1_len-i+1)/blockDim.x;
		if ( (seq1_len-i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			index_x = seq2_len - (tid+stripe*j);
			index_y =  i + (tid+stripe*j);
			if ( tid+stripe*j +i <seq1_len+1 ) {	
				p_dia3[ tid+stripe*j ] = 	\
					maximum(p_dia2[ tid+stripe*j+1 ] + penalty,	// up
						    p_dia2[ tid+stripe*j ] + penalty,	// left
							p_dia1[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
				// store to global memory
				matrix[ index_x*(seq1_len+1)+index_y ] = p_dia3[ tid+stripe*j ];
			}
		}
		p_tmp = p_dia1;
		p_dia1 = p_dia2;
		p_dia2 = p_dia3;
		p_dia3 = p_tmp;
		__syncthreads();
	}
}

/*******************************************************************************
	pos1 and pos2 are the array store the position 
********************************************************************************/
__global__ void needleman_cuda_diagonal(char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   int *score_matrix, unsigned int *pos_matrix,
									   unsigned int max_pair_no, int penalty)
{
	int pair_no, seq1_len, seq2_len;
	int tid = threadIdx.x;
	// 48 KB/4 = 12KB, seq1+sqe2, diagonal1, diagonal2, diagonal3
	__shared__ char s_seq1[MAX_SEQ_LEN];
	__shared__ char s_seq2[MAX_SEQ_LEN];
	__shared__ int s_dia1[MAX_SEQ_LEN];
	__shared__ int s_dia2[MAX_SEQ_LEN];
	__shared__ int s_dia3[MAX_SEQ_LEN];
	int *p_dia1, *p_dia2, *p_dia3, *p_tmp;
	pair_no = blockIdx.x;	// for each block, caculate one pair
	char *seq1 = sequence_set1 + pos1[pair_no];
	char *seq2 = sequence_set2 + pos2[pair_no];	
	int *matrix = score_matrix+pos_matrix[pair_no];
	seq1_len = pos1[pair_no+1] - pos1[pair_no]; 
	seq2_len = pos2[pair_no+1] - pos2[pair_no];

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
	s_dia2[0] = matrix[1] = penalty * 1; 
	s_dia2[1] = matrix[1*(seq1_len+1)] = penalty * 1;
	
	p_dia1 = s_dia1;
	p_dia2 = s_dia2;
	p_dia3 = s_dia3;	
	for (int i=2; i<=seq2_len; ++i){	// ith diagonal line		
		iteration = (i+1)/blockDim.x;
		if ( (i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			if ( tid+stripe*j<=i ) {	// ith diagonal has i+1 elements
				index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;
				if ( index_y==0 || index_y==i )	p_dia3[ index_y ] =  penalty * i;
				else {
					p_dia3[ index_y ] = 		\
						maximum(p_dia2[ index_y ] + penalty,	// up
						    p_dia2[ index_y-1 ] + penalty,	// left
							p_dia1[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
				}
				// store to global memory
				matrix[ index_x*(seq1_len+1)+index_y ] = p_dia3[ index_y ];
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
	// initial, load from shared memory
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<seq1_len+1 ) {
			index_x = seq2_len - (tid+stripe*i);	index_y = (tid+stripe*i);
			s_dia1[ tid+stripe*i ] = matrix[ index_x*(seq1_len+1)+index_y ];
		}
	}	
	__syncthreads();
	p_dia1 = s_dia1;
	p_dia2 = s_dia2;
	p_dia3 = s_dia3;
	// calculate the 1th diagonal
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<seq1_len ) {
			index_x = seq2_len - (tid+stripe*i);	index_y = 1 + (tid+stripe*i);
			p_dia2[ tid+stripe*i ] = \
						maximum(p_dia1[ tid+stripe*i+1 ] + penalty,	// up
							    p_dia1[ tid+stripe*i ] + penalty,	// left
								matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x]][s_seq1[index_y]] );
			matrix[ index_x*(seq1_len+1)+index_y ] = p_dia2[ tid+stripe*i ];
		}
	}
	__syncthreads();
	for (int i=2; i<=seq1_len; ++i){	// ith diagonal line, start from 2
		iteration = (seq1_len-i+1)/blockDim.x;
		if ( (seq1_len-i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			index_x = seq2_len - (tid+stripe*j);
			index_y =  i + (tid+stripe*j);
			if ( tid+stripe*j +i <seq1_len+1 ) {	
				p_dia3[ tid+stripe*j ] = 	\
					maximum(p_dia2[ tid+stripe*j+1 ] + penalty,	// up
						    p_dia2[ tid+stripe*j ] + penalty,	// left
							p_dia1[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
				// store to global memory
				matrix[ index_x*(seq1_len+1)+index_y ] = p_dia3[ tid+stripe*j ];
			}
		}
		p_tmp = p_dia1;
		p_dia1 = p_dia2;
		p_dia2 = p_dia3;
		p_dia3 = p_tmp;
		__syncthreads();
	}

}
