#include <needle.h>
#include <stdio.h>
#include <cuda.h>

#define WARP_SIZE 32

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

__device__ __host__ short maximum( short a, short b, short c)
{
	int k;
	if( a <= b )	k = b;
	else	k = a;
	if( k <=c )	return(c);
	else	return(k);
}
/*
__global__ void helloCUDA(float f)
{
		printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}*/

/*******************************************************************************
	32 x 32, default configuration is 1024, 32 threads will cooperate on 1 stride
	pos1 and pos2 are the array store the position 
********************************************************************************/
__global__ void needleman_cuda_dynamic(char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   short *score_matrix, unsigned int *pos_matrix,
									   unsigned int max_pair_no, short penalty)
{ 
	__shared__ char s_seq1[32][WARP_SIZE];
	__shared__ short s_line1[32][WARP_SIZE+1];
	__shared__ short s_line2[32][WARP_SIZE+1];
	char seq2_ch;
	char *seq1, *seq2;
	unsigned int pair_no = threadIdx.x / WARP_SIZE;	// Assume 32 threads(a warp) is a group to finish one pair
	unsigned int tid_in_warp = threadIdx.x % WARP_SIZE;
	unsigned int seq1_len = pos1[pair_no+1] - pos1[pair_no];	// length
	unsigned int seq2_len = pos2[pair_no+1] - pos2[pair_no];	// length
	unsigned int remain_data = seq1_len % WARP_SIZE;
	short * matrix;
	int warp_size;
	unsigned int stride = seq1_len / WARP_SIZE;
	if ( (seq1_len % WARP_SIZE)!=0 ) stride++;
	
	// Caculate for different stride
	matrix = score_matrix + pos_matrix[pair_no];
	seq1 = sequence_set1 + pos1[pair_no];
	seq2 = sequence_set2 + pos2[pair_no];
	
	if ( pair_no<max_pair_no ){
	for (int i=0; i<stride; ++i){	
		if (i<stride-1)
			warp_size = WARP_SIZE;
		else	
			warp_size = remain_data;
		 
		if (tid_in_warp<warp_size){
			// load sequence
			s_seq1[pair_no][tid_in_warp] = seq1[tid_in_warp+i*WARP_SIZE];

			// Initial the first 1 lines in score_matrix		
			// Inital the first 1 column
			if (0==i)
				matrix[0] = s_line1[pair_no][0] = 0;
			else
				s_line1[pair_no][0] = matrix[i*WARP_SIZE];

			s_line1[pair_no][1+tid_in_warp] = penalty * (1+tid_in_warp+i*WARP_SIZE);
			matrix[1+tid_in_warp+i*WARP_SIZE] = s_line1[pair_no][1+tid_in_warp];
			
		}
		__syncthreads();

		// fill the matrix
		for (int k=1; k<= seq2_len; ++k){
			//pair_no = threadIdx.x;			
			seq2_ch = seq2[k-1];
						
			if (tid_in_warp==0){	// for computing			
				//pair_no = threadIdx.x;
				if (k%2==0){
					if (i==0)	// initial the first column
						s_line1[pair_no][0] = matrix[k*(seq1_len+1)] = penalty * k;
					else
						s_line1[pair_no][0] = matrix[k*(seq1_len+1)+i*WARP_SIZE];

					for (int j=0; j<warp_size; ++j){
						s_line1[pair_no][1+j] = maximum(s_line2[pair_no][1+j]+ penalty,	\
														s_line1[pair_no][j] + penalty,	\
														s_line2[pair_no][j]+blosum62[seq2_ch][ s_seq1[pair_no][j] ]);
					}
				}				
				else {
					if (i==0)	// initial the first column
						s_line2[pair_no][0] = matrix[k*(seq1_len+1)] = penalty * k;
					else
						s_line2[pair_no][0] = matrix[k*(seq1_len+1)+i*WARP_SIZE];

					for (int j=0; j<warp_size; ++j){
						s_line2[pair_no][1+j] = maximum(s_line1[pair_no][1+j] + penalty,	\
														s_line2[pair_no][j] + penalty,	\
														s_line1[pair_no][j]+blosum62[seq2_ch][ s_seq1[pair_no][j] ]);
					}
				}
						
			}
			__syncthreads();
			// store data to global memory
			if (tid_in_warp<warp_size){
				if (k%2==0)				
					matrix[k*(seq1_len+1) + 1+tid_in_warp+i*WARP_SIZE] = s_line1[pair_no][1+tid_in_warp];
				else
					matrix[k*(seq1_len+1) + 1+tid_in_warp+i*WARP_SIZE] = s_line2[pair_no][1+tid_in_warp];
			}		
			__syncthreads();
		}
	}
	}
}


__global__ void needleman_cuda_diagonal_global(char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   short *score_matrix, unsigned int *pos_matrix,
									   unsigned int max_pair_no, short penalty)
{
	int pair_no, seq1_len, seq2_len;
	int min_len;
	//int max_len;
	int tid = threadIdx.x;
	
	// 48 KB/3 = 16KB, seq1+sqe2, diagonal1, diagonal2
	__shared__ char s_seq1[MAX_SEQ_LEN];
	__shared__ char s_seq2[MAX_SEQ_LEN];
	//__shared__ short s_diagonal1[MAX_SEQ_LEN];
	//__shared__ short s_diagonal2[MAX_SEQ_LEN];
	pair_no = blockIdx.x;	// for each block, caculate one pair
	char *seq1 = sequence_set1 + pos1[pair_no];
	char *seq2 = sequence_set2 + pos2[pair_no];	
	short *matrix = score_matrix+pos_matrix[pair_no];
	seq1_len = pos1[pair_no+1] - pos1[pair_no]; 
	seq2_len = pos2[pair_no+1] - pos2[pair_no];
	int which_bigger;	
	if ( seq1_len<=seq2_len ){
		min_len = seq1_len;
		//max_len = seq2_len;
		which_bigger = 2;
	}
	else{
		min_len = seq2_len;
		//max_len = seq1_len;
		which_bigger = 1;
	}
	unsigned int stride = min_len / blockDim.x + 1;
	unsigned int stride_length = blockDim.x;
	// load the two sequences
	for (int i=0; i<seq1_len/stride_length+1; ++i){
		if ( tid+i*stride_length<seq1_len )
			s_seq1[tid+i*stride_length] = seq1[tid+i*stride_length];
	}	
	for (int i=0; i<seq2_len/stride_length+1; ++i){
		if ( tid+i*stride_length<seq2_len )
		s_seq2[tid+i*stride_length] = seq2[tid+i*stride_length];
	}
	__syncthreads();
	// process the left-up triangle
	matrix[1] = penalty * 1;		
	matrix[1*(seq1_len+1)] = penalty * 1;
	for (int i=2; i<=min_len; ++i){	// ith diagonal line
		matrix[i] = penalty * i;		
		matrix[i*(seq1_len+1)] = penalty * i;
		stride = (i+1)/blockDim.x + 1;
		for (int j=0; j<stride; ++j){
			if (tid+j*stride_length<i-1){ // m(i,j): i-> i-tid, j->tid. start from m(i-1,1);
				int index_x = ( i-1-(tid+j*stride_length) );
				int index_y = 1+tid+j*stride_length;
				matrix[ index_x*(seq1_len+1) + index_y ] = 		\
					maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw		
			}	
		}
		__syncthreads();
	}
	// process the middle
	if (which_bigger==2){
		for (int i=min_len+1; i<=seq2_len; ++i){
			matrix[i*(seq1_len+1)] = penalty * i;
			stride = seq1_len/blockDim.x + 1;
			for (int j=0; j<stride; ++j){
				if (tid+j*stride_length<seq1_len){
					int index_x = ( i-1-(tid+j*stride_length) );
					int index_y = 1+tid+j*stride_length;
					matrix[ index_x*(seq1_len+1) + index_y ] = 		\
						maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw	
				}
			}
			__syncthreads();
		}
	}
	else{	// seq1 is longer than seq2
		for (int j=1; j<=seq1_len-seq2_len; ++j){
			matrix[j+seq2_len] = penalty * (j+seq2_len);
			stride = seq2_len/blockDim.x + 1;
			for (int i=0; i<stride; ++i){
				if (tid+i*stride_length<seq2_len){
					int index_x = seq2_len - ( tid+i*stride_length );
					int index_y = j + ( tid+i*stride_length );
					matrix[ index_x*(seq1_len+1) + index_y ] = 	\
						maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw
				}
			}
			__syncthreads();
		}
	} 
	// process the right-bottom triangle
	int i;
	if (which_bigger==2)	// sequence2 is longer
		i = seq2_len - seq1_len + 1;
	else
		i = 1;		
	for (  ; i<=seq2_len; ++i){	// ith diagonal line
		stride = (seq2_len-i+1)/blockDim.x + 1;
		for (int j=0; j<stride; ++j){
			if (tid+j*stride_length<seq2_len-i+1){ // m(i,j): i-> i-tid, j->tid. start from m(i-1,1);
				int index_x = (i+tid+j*stride_length);
				int index_y = seq1_len-(tid+j*stride_length);
				matrix[ index_x*(seq1_len+1) + index_y ] = 		
					maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw	
			}	
		}
		__syncthreads();
	}
}


__global__ void needleman_cuda_diagonal_shmem(char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   short *score_matrix, unsigned int *pos_matrix,
									   unsigned int max_pair_no, short penalty)
{
	int pair_no, seq1_len, seq2_len;
	int min_len;
	//int max_len;
	int tid = threadIdx.x;
	
	// 48 KB/3 = 16KB, seq1+sqe2, diagonal1, diagonal2
	__shared__ char s_seq1[MAX_SEQ_LEN];
	__shared__ char s_seq2[MAX_SEQ_LEN];
	__shared__ short s_diagonal1[MAX_SEQ_LEN];
	__shared__ short s_diagonal2[MAX_SEQ_LEN];
	__shared__ short s_diagonal3[MAX_SEQ_LEN];
	pair_no = blockIdx.x;	// for each block, caculate one pair
	char *seq1 = sequence_set1 + pos1[pair_no];
	char *seq2 = sequence_set2 + pos2[pair_no];	
	short *matrix = score_matrix+pos_matrix[pair_no];
	seq1_len = pos1[pair_no+1] - pos1[pair_no]; 
	seq2_len = pos2[pair_no+1] - pos2[pair_no];
	int which_bigger;	
	if ( seq1_len<=seq2_len ){
		min_len = seq1_len;
		//max_len = seq2_len;
		which_bigger = 2;
	}
	else{
		min_len = seq2_len;
		//max_len = seq1_len;
		which_bigger = 1;
	}
	unsigned int stride = min_len / blockDim.x + 1;
	unsigned int stride_length = blockDim.x;
	// load the two sequences
	for (int i=0; i<seq1_len/stride_length+1; ++i){
		if ( tid+i*stride_length<seq1_len )
			s_seq1[tid+i*stride_length] = seq1[tid+i*stride_length];
	}	
	for (int i=0; i<seq2_len/stride_length+1; ++i){
		if ( tid+i*stride_length<seq2_len )
		s_seq2[tid+i*stride_length] = seq2[tid+i*stride_length];
	}
	__syncthreads();
	// process the left-up triangle
	matrix[1] = penalty * 1;		
	matrix[1*(seq1_len+1)] = penalty * 1;
	for (int i=2; i<=min_len; ++i){	// ith diagonal line
		matrix[i] = penalty * i;		
		matrix[i*(seq1_len+1)] = penalty * i;
		stride = (i+1)/blockDim.x + 1;
		for (int j=0; j<stride; ++j){
			if (tid+j*stride_length<i-1){ // m(i,j): i-> i-tid, j->tid. start from m(i-1,1);
				int index_x = ( i-1-(tid+j*stride_length) );
				int index_y = 1+tid+j*stride_length;
				matrix[ index_x*(seq1_len+1) + index_y ] = 		\
					maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw		
			}	
		}
		__syncthreads();
	}
	// process the middle
	if (which_bigger==2){
		for (int i=min_len+1; i<=seq2_len; ++i){
			matrix[i*(seq1_len+1)] = penalty * i;
			stride = seq1_len/blockDim.x + 1;
			for (int j=0; j<stride; ++j){
				if (tid+j*stride_length<seq1_len){
					int index_x = ( i-1-(tid+j*stride_length) );
					int index_y = 1+tid+j*stride_length;
					matrix[ index_x*(seq1_len+1) + index_y ] = 		\
						maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw	
				}
			}
			__syncthreads();
		}
	}
	else{	// seq1 is longer than seq2
		for (int j=1; j<=seq1_len-seq2_len; ++j){
			matrix[j+seq2_len] = penalty * (j+seq2_len);
			stride = seq2_len/blockDim.x + 1;
			for (int i=0; i<stride; ++i){
				if (tid+i*stride_length<seq2_len){
					int index_x = seq2_len - ( tid+i*stride_length );
					int index_y = j + ( tid+i*stride_length );
					matrix[ index_x*(seq1_len+1) + index_y ] = 	\
						maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw
				}
			}
			__syncthreads();
		}
	} 
	// process the right-bottom triangle
	int i;
	if (which_bigger==2)	// sequence2 is longer
		i = seq2_len - seq1_len + 1;
	else
		i = 1;		
	for (  ; i<=seq2_len; ++i){	// ith diagonal line
		stride = (seq2_len-i+1)/blockDim.x + 1;
		for (int j=0; j<stride; ++j){
			if (tid+j*stride_length<seq2_len-i+1){ // m(i,j): i-> i-tid, j->tid. start from m(i-1,1);
				int index_x = (i+tid+j*stride_length);
				int index_y = seq1_len-(tid+j*stride_length);
				matrix[ index_x*(seq1_len+1) + index_y ] = 		
					maximum(matrix[ (index_x-1)*(seq1_len+1) + index_y] + penalty,	// up
						    matrix[ (index_x)*(seq1_len+1) + index_y-1] + penalty,	// left
							matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x-1]][s_seq1[index_y-1]]); //nw	
			}	
		}
		__syncthreads();
	}
}
