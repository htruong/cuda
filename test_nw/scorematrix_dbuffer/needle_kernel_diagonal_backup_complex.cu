#include <needle.h>
#include <stdio.h>
#include <cuda.h>

#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif


__device__ void dia_upperleft(char *s_seq1, unsigned int seq1_len, 
								char *s_seq2, unsigned int seq2_len,
								short *matrix, unsigned int dia_len,									
								short *s_dia1, short *s_dia2, short *s_dia3,
								int penalty)
{
	int tid = threadIdx.x;
	int stripe = blockDim.x;
	int index_x;
	int index_y;
	int iteration;	
	// process the left-up triangle
	s_dia1[0] = matrix[0] = 0;
	s_dia2[0] = matrix[1] = penalty * 1; 
	s_dia2[1] = matrix[1*(seq1_len+1)] = penalty * 1;
	for (int i=2; i<=seq2_len; ++i){	// ith diagonal line		
		iteration = (i+1)/blockDim.x;
		if ( (i+1)%blockDim.x != 0 ) 	iteration++;		
		if (i%3==2) { 
			for (int j=0; j<iteration; ++j) {
				if ( tid+stripe*j<=i ) {	// ith diagonal has i+1 elements
					index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;
					if ( index_y==0 || index_y==i )	s_dia3[ index_y ] =  penalty * i;
					else {
						s_dia3[ index_y ] = 		\
						maximum(s_dia2[ index_y ] + penalty,	// up
							    s_dia2[ index_y-1 ] + penalty,	// left
								s_dia1[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					}
					// store to global memory
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia3[ index_y ];
				}
			}
		}
		else if (i%3==0) {
			for (int j=0; j<iteration; ++j) {
				if ( tid+stripe*j<=i ) {
					index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;					 	
					if ( index_y==0 || index_y==i )	s_dia1[ index_y ] =  penalty * i;
					else {
						s_dia1[ tid+stripe*j ] = 		\
							maximum(s_dia3[ index_y ] + penalty,	// up
								    s_dia3[ index_y-1 ] + penalty,	// left
									s_dia2[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					}
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia1[ index_y ];
				}			
			}
		}
		else {	//i%3==1
			for (int j=0; j<iteration; ++j) {
				index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;
				if ( tid+stripe*j<=i ) {
					if ( (tid+stripe*j)==0 || (tid+stripe*j)==i )	s_dia2[ tid+stripe*j ] =  penalty * i;
					else {
						s_dia2[ tid+stripe*j ] = 		\
							maximum(s_dia1[ tid+stripe*j ] + penalty,	// up
								    s_dia1[ tid+stripe*j-1 ] + penalty,	// left
									s_dia3[ tid+stripe*j-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					}
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia2[ index_y ];
				}
			}	
		}
		__syncthreads();
	}
}

__device__ void dia_lowerright( char *s_seq1, unsigned int len1, 
								char *s_seq2, unsigned int len2,
								short *matrix, unsigned int dia_len,									
								short *s_dia1, short *s_dia2, short *s_dia3,
								unsigned int start, int penalty)
{
	int tid = threadIdx.x;
	int stripe = blockDim.x;
	int index_x, index_y;
	int iteration = dia_len/blockDim.x;
	if ( dia_len%blockDim.x!=0 ) iteration++; 
	// initial, load from shared memory
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<dia_len ) {
			index_x = len2 - (tid+stripe*i);	index_y = start-1 + (tid+stripe*i);
			s_dia1[ tid+stripe*i ] = matrix[ index_x*(len1+1)+index_y ];
		}
	}
	s_dia1[ dia_len ] = matrix[ (len2-dia_len)*(len1+1)+start-1 + dia_len ];	
	__syncthreads();
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<dia_len ) {
			index_x = len2 - (tid+stripe*i);	index_y = start + (tid+stripe*i);
			s_dia2[ tid+stripe*i ] = \
						maximum(s_dia1[ tid+stripe*i+1 ] + penalty,	// up
							    s_dia1[ tid+stripe*i ] + penalty,	// left
								matrix[(index_x-1)*(len1+1)+index_y-1]+blosum62[s_seq2[index_x]][s_seq1[index_y]] );
			matrix[ index_x*(len1+1)+index_y ] = s_dia2[ tid+stripe*i ];
		}
	}
	__syncthreads();
	for (int i=1; i<dia_len; ++i){	// ith diagonal line
		iteration = (dia_len-i)/blockDim.x;
		if ( (dia_len-i)%blockDim.x != 0 ) 	iteration++;
		if (i%3==1) { 
			for (int j=0; j<iteration; ++j) {
				index_x = len2 - (tid+stripe*j);
				index_y = start + i + (tid+stripe*j);
				if ( tid+stripe*j +i <dia_len ) {	
					s_dia3[ tid+stripe*j ] = 	\
						maximum(s_dia2[ tid+stripe*j+1 ] + penalty,	// up
							    s_dia2[ tid+stripe*j ] + penalty,	// left
								s_dia1[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					// store to global memory
					matrix[ index_x*(len1+1)+index_y ] = s_dia3[ tid+stripe*j ];
				}
			}
		}
		else if (i%3==2) { 
			for (int j=0; j<iteration; ++j) {
				index_x = len2 - (tid+stripe*j);
				index_y = start + i + (tid+stripe*j);
				if ( tid+stripe*j +i <dia_len ) {	
					s_dia1[ tid+stripe*j ] = 		\
						maximum(s_dia3[ tid+stripe*j+1 ] + penalty,	// up
							    s_dia3[ tid+stripe*j ] + penalty,	// left
								s_dia2[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					// store to global memory
					matrix[ index_x*(len1+1)+index_y ] = s_dia1[ tid+stripe*j ];
				}
			}
		}
		else {	// i%3==0
			for (int j=0; j<iteration; ++j) {
				index_x = len2 - (tid+stripe*j);
				index_y = start + i + (tid+stripe*j);
				if ( tid+stripe*j +i <dia_len ) {	
					s_dia2[ tid+stripe*j ] = 		\
						maximum(s_dia1[ tid+stripe*j+1 ] + penalty,	// up
							    s_dia1[ tid+stripe*j ] + penalty,	// left
								s_dia3[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					// store to global memory
					matrix[ index_x*(len1+1)+index_y ] = s_dia2[ tid+stripe*j ];
				}
			}
		}
		__syncthreads();
	}
}

/*******************************************************************************
	pos1 and pos2 are the array store the position 
********************************************************************************/
__global__ void needleman_cuda_diagonal(char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   short *score_matrix, unsigned int *pos_matrix,
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
	pair_no = blockIdx.x;	// for each block, caculate one pair
	char *seq1 = sequence_set1 + pos1[pair_no];
	char *seq2 = sequence_set2 + pos2[pair_no];	
	short *matrix = score_matrix+pos_matrix[pair_no];
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
	for (int i=2; i<=seq2_len; ++i){	// ith diagonal line		
		iteration = (i+1)/blockDim.x;
		if ( (i+1)%blockDim.x != 0 ) 	iteration++;		
		if (i%3==2) { 
			for (int j=0; j<iteration; ++j) {
				if ( tid+stripe*j<=i ) {	// ith diagonal has i+1 elements
					index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;
					if ( index_y==0 || index_y==i )	s_dia3[ index_y ] =  penalty * i;
					else {
						s_dia3[ index_y ] = 		\
						maximum(s_dia2[ index_y ] + penalty,	// up
							    s_dia2[ index_y-1 ] + penalty,	// left
								s_dia1[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					}
					// store to global memory
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia3[ index_y ];
				}
			}
		}
		else if (i%3==0) {
			for (int j=0; j<iteration; ++j) {
				if ( tid+stripe*j<=i ) {
					index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;					 	
					if ( index_y==0 || index_y==i )	s_dia1[ index_y ] =  penalty * i;
					else {
						s_dia1[ tid+stripe*j ] = 		\
							maximum(s_dia3[ index_y ] + penalty,	// up
								    s_dia3[ index_y-1 ] + penalty,	// left
									s_dia2[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					}
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia1[ index_y ];
				}			
			}
		}
		else {	//i%3==1
			for (int j=0; j<iteration; ++j) {
				index_x = i-(tid+stripe*j);	index_y = tid+stripe*j;
				if ( tid+stripe*j<=i ) {
					if ( (tid+stripe*j)==0 || (tid+stripe*j)==i )	s_dia2[ tid+stripe*j ] =  penalty * i;
					else {
						s_dia2[ tid+stripe*j ] = 		\
							maximum(s_dia1[ tid+stripe*j ] + penalty,	// up
								    s_dia1[ tid+stripe*j-1 ] + penalty,	// left
									s_dia3[ tid+stripe*j-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					}
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia2[ index_y ];
				}
			}	
		}
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
	// calculate the 1th diagonal
	for (int i=0; i<iteration; ++i) {
		if ( tid+stripe*i<seq1_len ) {
			index_x = seq2_len - (tid+stripe*i);	index_y = 1 + (tid+stripe*i);
			s_dia2[ tid+stripe*i ] = \
						maximum(s_dia1[ tid+stripe*i+1 ] + penalty,	// up
							    s_dia1[ tid+stripe*i ] + penalty,	// left
								matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x]][s_seq1[index_y]] );
			matrix[ index_x*(seq1_len+1)+index_y ] = s_dia2[ tid+stripe*i ];
		}
	}
	__syncthreads();
	for (int i=2; i<=seq1_len; ++i){	// ith diagonal line, start from 2
		iteration = (seq1_len-i+1)/blockDim.x;
		if ( (seq1_len-i+1)%blockDim.x != 0 ) 	iteration++;
		if (i%3==2) { 
			for (int j=0; j<iteration; ++j) {
				index_x = seq2_len - (tid+stripe*j);
				index_y =  i + (tid+stripe*j);
				if ( tid+stripe*j +i <seq1_len+1 ) {	
					s_dia3[ tid+stripe*j ] = 	\
						maximum(s_dia2[ tid+stripe*j+1 ] + penalty,	// up
							    s_dia2[ tid+stripe*j ] + penalty,	// left
								s_dia1[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					// store to global memory
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia3[ tid+stripe*j ];
				}
			}
		}
		else if (i%3==0) { 
			for (int j=0; j<iteration; ++j) {
				index_x = seq2_len - (tid+stripe*j);
				index_y = i + (tid+stripe*j);
				if ( tid+stripe*j +i <seq1_len+1 ) {	
					s_dia1[ tid+stripe*j ] = 		\
						maximum(s_dia3[ tid+stripe*j+1 ] + penalty,	// up
							    s_dia3[ tid+stripe*j ] + penalty,	// left
								s_dia2[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					// store to global memory
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia1[ tid+stripe*j ];
				}
			}
		}
		else {	// i%3==1
			for (int j=0; j<iteration; ++j) {
				index_x = seq2_len - (tid+stripe*j);
				index_y = i + (tid+stripe*j);
				if ( tid+stripe*j +i <seq1_len+1 ) {	
					s_dia2[ tid+stripe*j ] = 		\
						maximum(s_dia1[ tid+stripe*j+1 ] + penalty,	// up
							    s_dia1[ tid+stripe*j ] + penalty,	// left
								s_dia3[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
					// store to global memory
					matrix[ index_x*(seq1_len+1)+index_y ] = s_dia2[ tid+stripe*j ];
				}
			}
		}
		__syncthreads();
	}

}
