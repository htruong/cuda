#define LIMIT -999
#define BLOCK_SIZE 16

#include <stdio.h>

#include <cuda.h>
#include <sys/time.h>
#include "needle_gpu.h"

// includes, kernels
#include "needle_kernel_diagonal.cu"

/* Private structure, do not bother */
struct needle_context {
	unsigned int gpu_num;
	char *sequence_set1;
	char *sequence_set2;
	unsigned int *pos1;
	unsigned int *pos2;
	int *score_matrix;
	unsigned int *pos_matrix;
	unsigned int max_pair_no;
	short penalty;
	// Grunt work... eww
	char *d_sequence_set1_h1, *d_sequence_set2_h1, *d_sequence_set1_h2, *d_sequence_set2_h2;
	unsigned int *d_pos1_h1, *d_pos2_h1, *d_pos_matrix_h1, *d_pos1_h2, *d_pos2_h2, *d_pos_matrix_h2;
	int *d_score_matrix_h1, *d_score_matrix_h2;
	cudaStream_t *stream1, *stream2;
	unsigned int half_b, other_half_b, max_length_per_seq;
	double total_kernel_time, total_memtransfer_time, total_memtransfer_initial_time;
	char * gpu_name;
	unsigned int optimal_batch_size;
};

int check_mappable_host ( cudaDeviceProp * p )
{
	int support = p->canMapHostMemory;
	if(support == 0) {
		printf("%s does not support mapping host memory.\n", p->name);
	} else {
		printf("%s supports mapping host memory.\n",p->name);
	}
	return support;
}


double gettime(){
	struct timeval t;
	gettimeofday(&t,NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}


void memcpy_and_run (
				needle_context * ctx,
				unsigned int begin,
				unsigned int end,
				cudaStream_t * stream,
				char *sequence_set1,
				char *sequence_set2,
				char *d_sequence_set1,
				char *d_sequence_set2,
				unsigned int *pos1,
				unsigned int *pos2,
				unsigned int *d_pos1,
				unsigned int *d_pos2,
				int *score_matrix,
				unsigned int *pos_matrix,
				int *d_score_matrix,
				unsigned int *d_pos_matrix,
				short penalty) {
				
		unsigned int batch_size = end-begin;
		// Memcpy to device

		double start_marker = 0;
		start_marker = gettime();

		#ifdef VERBOSE
		printf("-- Start calculation from %d to %d --\n", begin, end);
		#endif

		#ifdef DUAL_BUFFERING
		cudaMemcpyAsync( d_sequence_set1, sequence_set1 + pos1[begin], sizeof(char)*(pos1[end] - pos1[begin]), cudaMemcpyHostToDevice, *stream);
		cudaMemcpyAsync( d_sequence_set2, sequence_set2 + pos2[begin], sizeof(char)*(pos2[end] - pos2[begin]), cudaMemcpyHostToDevice, *stream);
		cudaMemcpyAsync( d_pos1, pos1 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice, *stream );
		cudaMemcpyAsync( d_pos2, pos2 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice, *stream );
		cudaMemcpyAsync( d_pos_matrix, pos_matrix /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice, *stream );
		#else
		//printf("-- Start calculation from %d to %d cp1 --\n", begin, end); 
		cudaMemcpy( d_sequence_set1, sequence_set1 + pos1[begin], sizeof(char)*(pos1[end] - pos1[begin]), cudaMemcpyHostToDevice );
		//printf("-- Start calculation from %d to %d cp2 --\n", begin, end);
		cudaMemcpy( d_sequence_set2, sequence_set2 + pos2[begin], sizeof(char)*(pos2[end] - pos2[begin]), cudaMemcpyHostToDevice );
		//printf("-- Start calculation from %d to %d cp3 --\n", begin, end);
		cudaMemcpy( d_pos1, pos1 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice );
		//printf("-- Start calculation from %d to %d cp4 --\n", begin, end);
		cudaMemcpy( d_pos2, pos2 /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice );
		//printf("-- Start calculation from %d to %d cp5 --\n", begin, end);
		cudaMemcpy( d_pos_matrix, pos_matrix /*+ begin*/, sizeof(unsigned int)*(batch_size+1), cudaMemcpyHostToDevice );
		#endif

		ctx->total_memtransfer_initial_time += gettime() - start_marker;

		#ifdef VERBOSE
		printf("\t [%d - %d] Memcpy CPU-GPU: %f\n", begin, end, gettime() - start_marker);
		#endif

		start_marker = gettime();
		
		#ifdef DUAL_BUFFERING
		needleman_cuda_diagonal<<<batch_size,512, 0, *stream>>>(d_sequence_set1, d_sequence_set2,
				d_pos1, d_pos2,
				d_score_matrix, d_pos_matrix,
				batch_size, penalty);
		#else
		needleman_cuda_diagonal<<<batch_size,512>>>(d_sequence_set1, d_sequence_set2,
				d_pos1, d_pos2,
				d_score_matrix, d_pos_matrix,
				batch_size, penalty);
		#endif
		
		cudaDeviceSynchronize();

		ctx->total_kernel_time += gettime() - start_marker;
		
		#ifdef VERBOSE
		printf("\t [%d - %d] Kernel: %f\n", begin, end, gettime() - start_marker);
		#endif

		start_marker = gettime();

	  	
		#ifdef DUAL_BUFFERING
		cudaMemcpyAsync( score_matrix + pos_matrix[begin], d_score_matrix, sizeof(int)*(pos_matrix[end] - pos_matrix[begin]), cudaMemcpyDeviceToHost, *stream );
		#else
		cudaMemcpy( score_matrix + pos_matrix[begin], d_score_matrix, sizeof(int)*(pos_matrix[end] - pos_matrix[begin]), cudaMemcpyDeviceToHost );
		#endif

		ctx->total_memtransfer_time += gettime() - start_marker;

		#ifdef VERBOSE
		printf("\t [%d - %d] Memcpy GPU-CPU: %f\n", begin, end, gettime() - start_marker);
		#endif
		
}

void needleman_gpu(char *sequence_set1,
				char *sequence_set2,
				unsigned int *pos1,
				unsigned int *pos2,
				int *score_matrix,
				unsigned int *pos_matrix,
				unsigned int max_pair_no,
				short penalty,
				char *d_sequence_set1_h1,
				char *d_sequence_set2_h1,
				char *d_sequence_set1_h2,
				char *d_sequence_set2_h2,
				unsigned int *d_pos1_h1,
				unsigned int *d_pos2_h1,
				unsigned int *d_pos_matrix_h1,
				unsigned int *d_pos1_h2,
				unsigned int *d_pos2_h2,
				unsigned int *d_pos_matrix_h2,
				int *d_score_matrix_h1,
				int *d_score_matrix_h2,
				cudaStream_t * stream1,
				cudaStream_t * stream2,
				needle_context * ctx
				)
{
	bool done = false;

	unsigned int start = 0;
	unsigned int end = 0;
	bool turn = true;
	while (!done) {
		int tmp_batch_sz = turn ? ctx->half_b : ctx->other_half_b;
		if (start + tmp_batch_sz >= max_pair_no) {
			end = max_pair_no;
			done = true;
		} else {
			end = start + tmp_batch_sz;
		}
		
		memcpy_and_run (
			ctx,
			start,
			end,
			turn ? stream1 : stream2 ,
			sequence_set1,
			sequence_set2,
			turn ? d_sequence_set1_h1 : d_sequence_set1_h2,
			turn ? d_sequence_set2_h1 : d_sequence_set2_h2,
			pos1,
			pos2,
			turn ? d_pos1_h1 : d_pos1_h2,
			turn ? d_pos2_h1 : d_pos2_h2,
			score_matrix,
			pos_matrix,
			turn ? d_score_matrix_h1 : d_score_matrix_h2,
			turn ? d_pos_matrix_h1 : d_pos_matrix_h2,
			penalty);
				
		start = end;
		#ifdef DUAL_BUFFERING
		turn = !turn;
		#endif
	}
	cudaDeviceSynchronize();
	
	printf("__CSV__,%s,%d,%f,%f,%f\n",
		ctx->gpu_name,
		ctx->optimal_batch_size,
		ctx->total_memtransfer_initial_time,
		ctx->total_kernel_time,
		ctx->total_memtransfer_time);
}

void * needle_prepare(
	const int gpu_num,
	unsigned int max_length_per_seq,
	unsigned int * optimal_batch_size,
	char * gpu_name
	)
{
	printf("NEEDLEMAN MODULE PREPPING CPU...\n", 0);
	cudaSetDevice(gpu_num);
	
	cudaDeviceProp * prop = new cudaDeviceProp;
	cudaGetDeviceProperties(prop, gpu_num);
	check_mappable_host (prop);
	
	strncpy (gpu_name, prop->name, 256);

	delete prop;

	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("GPU Memory avaliable: Free: %lu, Total: %lu\n",freeMem/1024/1024, totalMem/1024/1024);

	unsigned int eachSeqMem = sizeof(char)*max_length_per_seq*2
					+ sizeof(int)*(max_length_per_seq+1)*(max_length_per_seq+1)
					+ sizeof(unsigned int)*3;
	unsigned int batch_size = totalMem * 0.7 / eachSeqMem; // Safety reasons...

	cudaStream_t * stream1 = new cudaStream_t;
	cudaStream_t * stream2 = new cudaStream_t;

	unsigned int half_b, other_half_b;
	#ifdef DUAL_BUFFERING
	half_b = batch_size / 2;
	other_half_b = batch_size - half_b;
    cudaStreamCreate(stream1);
    cudaStreamCreate(stream2);
	#else
	half_b = batch_size;
	#endif

	printf("Each batch will be doing this many pairs: %d\n", batch_size);
	* optimal_batch_size = batch_size;
	
	struct needle_context * internal_ctx = new needle_context;
	
	internal_ctx->gpu_num = gpu_num;
	internal_ctx->half_b = half_b;
	internal_ctx->other_half_b = other_half_b;
	internal_ctx->gpu_name = (char *) malloc(256);
	strncpy (internal_ctx->gpu_name, gpu_name, 256);
	internal_ctx->stream1 = stream1;
	internal_ctx->stream2 = stream2;
	internal_ctx->optimal_batch_size = * optimal_batch_size;
	
	return (void *) internal_ctx;
}

void needle_allocate(
    void *ctx,
	char *sequence_set1,
	char *sequence_set2,
	unsigned int *pos1,
	unsigned int *pos2,
	int *score_matrix, 
	unsigned int *pos_matrix
	)
{
	double start_marker; // Start time marker
	
	struct needle_context *internal_ctx = (needle_context *) ctx;
	
	unsigned int half_b, other_half_b;
	half_b = internal_ctx->half_b;
	other_half_b = internal_ctx->other_half_b;
	// First we need to see how to devide the memory...
	// Query the device capabilities to see how much we can allocate for this problem
	////////////////////////////////////////////////////////////////////////////

	// This implementation comes with the free assumption that 
	// all sequences will be having the same size :'(

	char *d_sequence_set1_h1, *d_sequence_set2_h1, *d_sequence_set1_h2, *d_sequence_set2_h2;
	unsigned int *d_pos1_h1, *d_pos2_h1, *d_pos_matrix_h1, *d_pos1_h2, *d_pos2_h2, *d_pos_matrix_h2;
	int *d_score_matrix_h1, *d_score_matrix_h2;

	start_marker = gettime();
	// Allocating memory for both halves

	// First half
	  cudaMalloc( (void**)&d_sequence_set1_h1, sizeof(char)*(pos1[1]*half_b) );
    cudaMalloc( (void**)&d_sequence_set2_h1, sizeof(char)*(pos1[1]*half_b)) ;
    cudaMalloc( (void**)&d_score_matrix_h1, sizeof(int)*(pos_matrix[1]*half_b)) ;
    //cudaMalloc( (void**)&d_score_matrix_h1, sizeof(int)*(pos_matrix[1]*half_b)) ;
		//cudaHostGetDevicePointer( (void**)&d_score_matrix_h1, (void *) score_matrix, 0);
    cudaMalloc( (void**)&d_pos1_h1, sizeof(unsigned int)*(half_b+1) ) ;
    cudaMalloc( (void**)&d_pos2_h1, sizeof(unsigned int)*(half_b+1) ) ;
    cudaMalloc( (void**)&d_pos_matrix_h1, sizeof(unsigned int)*(half_b+1) ) ;

    #ifdef DUAL_BUFFERING
    // Second half
    cudaMalloc( (void**)&d_sequence_set1_h2, sizeof(char)*(pos1[1]*other_half_b) );
    cudaMalloc( (void**)&d_sequence_set2_h2, sizeof(char)*(pos2[1]*other_half_b)) ;
    cudaMalloc( (void**)&d_score_matrix_h2, sizeof(int)*(pos_matrix[1]*other_half_b)) ;
    //cudaMalloc( (void**)&d_score_matrix_h2, sizeof(int)*(pos_matrix[1]*other_half_b)) ;
		//cudaHostGetDevicePointer( (void**)&d_score_matrix_h2, (void *) (score_matrix + pos_matrix[half_b]), 0);
    cudaMalloc( (void**)&d_pos1_h2, sizeof(unsigned int)*(other_half_b+1) );
    cudaMalloc( (void**)&d_pos2_h2, sizeof(unsigned int)*(other_half_b+1) ) ;
    cudaMalloc( (void**)&d_pos_matrix_h2, sizeof(unsigned int)*(other_half_b+1) ) ;
	#endif


	fprintf(stdout,"cudaMalloc = %f\n", gettime()-start_marker);

	////////////////////////////////////////////////////////////////////////////
	// WARNING BOILERPLATE CODE !
	// Jesus, why I'm doing this? - Huan.

	internal_ctx->sequence_set1 = sequence_set1;
	internal_ctx->sequence_set2 = sequence_set2;
	internal_ctx->pos1 = pos1;
	internal_ctx->pos2 = pos2;
	internal_ctx->score_matrix = score_matrix;
	internal_ctx->pos_matrix = pos_matrix;
	internal_ctx->d_sequence_set1_h1 = d_sequence_set1_h1;
	internal_ctx->d_sequence_set2_h1 = d_sequence_set2_h1;
	internal_ctx->d_sequence_set1_h2 = d_sequence_set1_h2;
	internal_ctx->d_sequence_set2_h2 = d_sequence_set2_h2;
	internal_ctx->d_pos1_h1 = d_pos1_h1;
	internal_ctx->d_pos2_h1 = d_pos2_h1;
	internal_ctx->d_pos_matrix_h1 = d_pos_matrix_h1;
	internal_ctx->d_pos1_h2 = d_pos1_h2;
	internal_ctx->d_pos2_h2 = d_pos2_h2;
	internal_ctx->d_pos_matrix_h2 = d_pos_matrix_h2;
	internal_ctx->d_score_matrix_h1 = d_score_matrix_h1;
	internal_ctx->d_score_matrix_h2 = d_score_matrix_h2;
	internal_ctx->penalty = -10;
	internal_ctx->total_kernel_time = 0;
	internal_ctx->total_memtransfer_time = 0;
	internal_ctx->total_memtransfer_initial_time = 0;


	printf("-- NEEDLEMAN MODULE INITIALIZING DONE --\n", 0);
	
}

void needle_align(void * ctx, int num_pairs) {
	////////////////////////////////////////////////////////////////////////////
	// WARNING BOILERPLATE CODE !

	struct needle_context *internal_ctx = (needle_context *) ctx;

	needleman_gpu(
		internal_ctx->sequence_set1,
		internal_ctx->sequence_set2,
		internal_ctx->pos1,
		internal_ctx->pos2,
		internal_ctx->score_matrix,
		internal_ctx->pos_matrix,
		num_pairs,
		internal_ctx->penalty,
		internal_ctx->d_sequence_set1_h1,
		internal_ctx->d_sequence_set2_h1,
		internal_ctx->d_sequence_set1_h2,
		internal_ctx->d_sequence_set2_h2,
		internal_ctx->d_pos1_h1,
		internal_ctx->d_pos2_h1,
		internal_ctx->d_pos_matrix_h1,
		internal_ctx->d_pos1_h2,
		internal_ctx->d_pos2_h2,
		internal_ctx->d_pos_matrix_h2,
		internal_ctx->d_score_matrix_h1,
		internal_ctx->d_score_matrix_h2,
		internal_ctx->stream1,
		internal_ctx->stream2,
		internal_ctx
	);
}


void needle_finalize(void * ctx)
{
	struct needle_context *internal_ctx = static_cast<struct needle_context *>(ctx);

	
	cudaFree(internal_ctx->d_sequence_set1_h1);
	cudaFree(internal_ctx->d_sequence_set2_h1);
	cudaFree(internal_ctx->d_pos1_h1);
	cudaFree(internal_ctx->d_pos2_h2);
	cudaFree(internal_ctx->d_pos_matrix_h1);
	cudaFree(internal_ctx->d_score_matrix_h1);

	#ifdef DUAL_BUFFERING
	cudaFree(internal_ctx->d_sequence_set1_h2);
	cudaFree(internal_ctx->d_sequence_set2_h2);
	cudaFree(internal_ctx->d_pos1_h2);
	cudaFree(internal_ctx->d_pos2_h2);
	cudaFree(internal_ctx->d_pos_matrix_h2);
	cudaFree(internal_ctx->d_score_matrix_h2);

	cudaStreamDestroy(*(internal_ctx->stream1));
	cudaStreamDestroy(*(internal_ctx->stream2));
	#endif


	delete internal_ctx->gpu_name;
	delete internal_ctx->stream1;
	delete internal_ctx->stream2;

	delete(internal_ctx);
}
