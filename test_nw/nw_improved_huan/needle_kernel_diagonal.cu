#include <needle.h>
#include <stdio.h>
#include <cuda.h>


#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

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

// HACK: Huan's hack
// THIS FUNCTION TAKES 3 PARAMS, original UP, original LEFT, PENALTY and DIAG PENALTY
// Do not calculate the scores beforehand
// It will RETURN the score and the direction, a little bit ugly
// FIRST 2 bits will be the direction, the REST will be the score.
__device__ __host__ int gpu_max_3( int a, int b, int c, short penalty, short diag_penalty/*, short * idx*/)
{
	int tmp;
	int tmp_l;

	// This one discards the last 2 bits for the directional traceback.
	
	int tmp_a = (a >> 2) + penalty;
	int tmp_b = (b >> 2) + penalty;
	int tmp_c = (c >> 2) + diag_penalty;
	
	if (tmp_a > tmp_b) {
		//*idx = 0;
		tmp = tmp_a << 2 | TRACE_U;
		tmp_l = tmp_a;
	} else {
		//*idx = 1;
		tmp = tmp_b << 2 | TRACE_L;
		tmp_l = tmp_b;
	}
	if (tmp_c > tmp_l) {
		//*idx = 3;
		tmp = tmp_c << 2 | TRACE_UL;
		tmp_l = tmp_c;
	}

	//printf("Got a, b, c, ta, tb, tc = %x, %x, %x | %x, %x, %x | max= %d\n", a, b, c, tmp_a, tmp_b, tmp_c, tmp_l >> 2);
	return tmp;
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

	short tmp_ret = 0;

	// load the two sequences
	unsigned int stride_length = blockDim.x;
	for (int i=0; i<seq1_len/stride_length+1; ++i) {
		if ( tid+i*stride_length<seq1_len )
			s_seq1[tid+i*stride_length+1] = seq1[tid+i*stride_length];
	}
	for (int i=0; i<seq2_len/stride_length+1; ++i) {
		if ( tid+i*stride_length<seq2_len )
			s_seq2[tid+i*stride_length+1] = seq2[tid+i*stride_length];
	}
	__syncthreads();

	int stripe = blockDim.x;
	int index_x;
	int index_y;
	int iteration;
	// process the left-up triangle
	s_dia1[0] = matrix[0] = 0;
	s_dia2[0] = matrix[1] = penalty * 1 << 2 | TRACE_L;
	s_dia2[1] = matrix[1*(seq1_len+1)] = penalty * 1 << 2 | TRACE_U;

	p_dia1 = s_dia1;
	p_dia2 = s_dia2;
	p_dia3 = s_dia3;
	for (int i=2; i<=seq2_len; ++i) {	// ith diagonal line
		iteration = (i+1)/blockDim.x;
		if ( (i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			if ( tid+stripe*j<=i ) {	// ith diagonal has i+1 elements
				index_x = i-(tid+stripe*j);
				index_y = tid+stripe*j;

				// HACK Huan's hack
				// We want to calculate all the scores and directions for the
				// ones on the TOP and LEFTmost of the matrix
				if ( index_y==0 ) {
					p_dia3[ index_y ] =  (penalty * i  << 2) | TRACE_U;
				} else if (index_y==i) {
					p_dia3[ index_y ] =  (penalty * i  << 2) | TRACE_L;
				}
					
				//if ( index_y==0 || index_y==i )	p_dia3[ index_y ] =  penalty * i  << 2;
				else {
					p_dia3[ index_y ] = 	
						gpu_max_3(p_dia2[ index_y ],	// up
								p_dia2[ index_y-1 ],	// left
								p_dia1[ index_y-1 ], penalty, (s_seq2[index_x] == s_seq1[index_y]) ? 1 : -1 /*blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] */ /*,
								&tmp_ret */
						);
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
			index_x = seq2_len - (tid+stripe*i);
			index_y = (tid+stripe*i);
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
			index_x = seq2_len - (tid+stripe*i);
			index_y = 1 + (tid+stripe*i);
			p_dia2[ tid+stripe*i ] = 
				gpu_max_3(p_dia1[ tid+stripe*i+1 ],	// up
						p_dia1[ tid+stripe*i ],	// left
						matrix[(index_x-1)*(seq1_len+1)+index_y-1], penalty, (s_seq2[index_x] == s_seq1[index_y]) ? 1 : -1 //blosum62[s_seq2[index_x]][s_seq1[index_y]]/*,&tmp_ret */
				);
			matrix[ index_x*(seq1_len+1)+index_y ] = p_dia2[ tid+stripe*i ];
		}
	}
	__syncthreads();
	for (int i=2; i<=seq1_len; ++i) {	// ith diagonal line, start from 2
		iteration = (seq1_len-i+1)/blockDim.x;
		if ( (seq1_len-i+1)%blockDim.x != 0 ) 	iteration++;
		for (int j=0; j<iteration; ++j) {
			index_x = seq2_len - (tid+stripe*j);
			index_y =  i + (tid+stripe*j);
			if ( tid+stripe*j +i <seq1_len+1 ) {
				p_dia3[ tid+stripe*j ] = 
					gpu_max_3(p_dia2[ tid+stripe*j+1 ],	// up
							p_dia2[ tid+stripe*j ],	// left
							p_dia1[ tid+stripe*j+1 ], penalty, (s_seq2[index_x] == s_seq1[index_y]) ? 1 : -1 //, blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ]/*, &tmp_ret */
					);
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
