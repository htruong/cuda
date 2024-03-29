#include <needle.h>
#include <stdio.h>
#include <cuda.h>

#define STRIDE_SIZE 128		// 
#define PAIR_IN_BLOCK 32

// 1:dia 2:up 4:left

__global__ void needleman_cuda_dynamic(char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   char *direction_matrix, short *stride_overlap,
									   unsigned int *pos_matrix, unsigned int max_pair_no, short penalty)
{ 
	__shared__ char s_seq1[PAIR_IN_BLOCK][STRIDE_SIZE+1];
	__shared__ char s_direct[PAIR_IN_BLOCK][STRIDE_SIZE+1];
	__shared__ short s_line1[PAIR_IN_BLOCK][STRIDE_SIZE+1];
	__shared__ short s_line2[PAIR_IN_BLOCK][STRIDE_SIZE+1];
	short *pl1, *pl2, *ptmp;
	char *matrix;		// direction matrix
	char seq2_ch, *seq1, *seq2;
	unsigned int start_pair_no = blockIdx.x * PAIR_IN_BLOCK;
	unsigned int end_pair_no = (blockIdx.x+1)*PAIR_IN_BLOCK<max_pair_no? (blockIdx.x+1)*PAIR_IN_BLOCK:max_pair_no; 
	unsigned int tid = threadIdx.x;
	unsigned int seq1_len = pos1[start_pair_no+1] - pos1[start_pair_no];
	unsigned int seq2_len;
	unsigned int stride = seq1_len / STRIDE_SIZE;
	int i, j, k, index_x, index_y;
	if ( (seq1_len % STRIDE_SIZE)!=0 ) stride++;
	
	if ( start_pair_no<end_pair_no ){
	// Divide the long array into different stride
	for (i=0; i<stride; ++i) {
		/************** Initialization **************/
		index_x = 0; index_y = tid + i*STRIDE_SIZE+1;
		/* Here, we should add a bunch of assignment statement */		
		for (j=start_pair_no; j<end_pair_no; ++j) {	// iteration between different loops
			if ( i==0 )	s_line1[j-start_pair_no][0] = *(direction_matrix + pos_matrix[j]) = 0;
			else	s_line1[j-start_pair_no][0] = stride_overlap[j*MAX_SEQ_LEN+index_x]; //matrix[i*STRIDE_SIZE];
			unsigned int load_stride = STRIDE_SIZE/blockDim.x;
			for ( k=0; k<load_stride; ++k ) {
				index_y = tid + i*STRIDE_SIZE+1 + k*blockDim.x;			
				if ( index_y<=seq1_len ) {			
					matrix = direction_matrix + pos_matrix[j];
					matrix[index_y] = 4;
					s_line1[j-start_pair_no][tid+1+k*blockDim.x] = penalty * index_y;
					seq1 = sequence_set1 + pos1[j];
					s_seq1[j-start_pair_no][tid+1+k*blockDim.x] = seq1[index_y-1];
					if ( index_y==seq1_len || index_y%STRIDE_SIZE==0 )
						stride_overlap[j*MAX_SEQ_LEN+index_x] = s_line1[j-start_pair_no][tid+1+k*blockDim.x];			
				}
			}
			
		}
		/**************************************/
		
		/************* fill the matrix **************/
		pl1 = s_line1[tid];
		pl2 = s_line2[tid];
		seq2_len = pos2[start_pair_no+1] - pos2[start_pair_no];
		for (index_x=1; index_x<=seq2_len; ++index_x){	// each iteration stands for each row
			if ( start_pair_no+tid<end_pair_no ) {		// seq2_len should be the maximum length
				seq2 = sequence_set2 + pos2[start_pair_no+tid];
				seq1_len = pos1[start_pair_no+tid+1] - pos1[start_pair_no+tid];	// length
				matrix = direction_matrix + pos_matrix[start_pair_no+tid];
				index_y = i*STRIDE_SIZE+1;
				/************* Calculation **************/
				seq2_ch = seq2[index_x-1];
				if (i==0) {
					pl2[0] =  penalty * index_x;	// initialize the first column
					matrix[index_x*(seq1_len+1)] = 2;	// 1:dia 2:up 4:left
				}
				else {
					pl2[0] = stride_overlap[(tid+start_pair_no)*MAX_SEQ_LEN+index_x];
						//matrix[index_x*(seq1_len+1) + i*STRIDE_SIZE];	// load the data from last iteration
				}			
				for (k=0; k<STRIDE_SIZE; ++k) {
					pl2[1+k] = maximum(pl2[k]+penalty,		\
									   pl1[1+k]+penalty,	\
									   pl1[k]+blosum62[seq2_ch][s_seq1[tid][k+1]],	\
									   &s_direct[tid][1+k]);
				}
				stride_overlap[(tid+start_pair_no)*MAX_SEQ_LEN+index_x] = pl2[STRIDE_SIZE];
				/************* swap the pointer ****************/
				ptmp = pl1;	pl1 = pl2; pl2 = ptmp;
			}
			__syncthreads();
			/*********** store data to global memory **********/	
			index_y = tid + i*STRIDE_SIZE+1;		
			for (j=start_pair_no; j<end_pair_no; ++j) {
				unsigned int load_stride = STRIDE_SIZE/blockDim.x;
				for ( k=0; k<load_stride; ++k ) {
					index_y = tid + i*STRIDE_SIZE+1 + k*blockDim.x;	
					if ( index_y<=seq1_len ) {		
						matrix = direction_matrix + pos_matrix[j];				
						seq1_len = pos1[j+1] - pos1[j];		
						if ( index_x&1 == 0)
							matrix[index_x*(seq1_len+1)+index_y] = s_direct[j-start_pair_no][tid+1+k*blockDim.x];
						else
							matrix[index_x*(seq1_len+1)+index_y] = s_direct[j-start_pair_no][tid+1+k*blockDim.x];
					}
				}
			}
			__syncthreads();
		}
	}	//end of for
	}	//end of if
}
