#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define NO_EDGE_FND -1
#define LIST_END -1

#define CUDA_DEVICE 0

#define THREADS_PER_BLOCK 16 

/*
Use these #define in the preprocessor directives to enable/disable your code improvements.
For example:

#ifdef IMP1
  YOUR CODE IMPROVEMENT #1
#else
  ORIGINAL CODE
#endif

-----

#define IMP1	1
#define IMP2	1
#define IMP3	1
*/

/* ADD_COMMENT
 * - Describe all the data structures listed below, along with their fields.
 */

typedef struct VERTEX_t
{
	int num;
	int ei;
	int eo;
	int cyc;
	int max_adj;
	int next_v;
	int proc;
}VERTEX_t;

typedef struct EDGE_t
{
	int vo;
	int vi;
	int w;
	int adj_w;
	int next_o;
	int next_i;
	int dead;
	int rmvd;
	int buried;
}EDGE_t;

typedef struct DIGRAPH_t
{
	VERTEX_t *v;
	EDGE_t *e;
	int num_v;
	int num_e;
	int mst_sum;
}DIGRAPH_t;

typedef struct CYCLEDATA_t
{
	unsigned int curr;
	unsigned int mrkr;
	unsigned int cyc;
	unsigned int state;
	unsigned int self_fnd;
	unsigned int start;
	unsigned int max;
}CYCLEDATA_t;

/* ADD_COMMENT
 * - Describe the high-level functionality provided by the function below, along with their parameters.
 * - Specify which functions run on the host and which on the device.
 */

void cudaCheckError(cudaError_t ce);
void addEdge(DIGRAPH_t *d, int addr);
__global__ void trimSpanningTree(EDGE_t *e, VERTEX_t *v);
__global__ void findCycles(EDGE_t *e, VERTEX_t *v, int num_v);
int restoreSpanningTree(DIGRAPH_t *d);


inline void cudaCheckError(cudaError_t ce) {
    if (ce != cudaSuccess) {
        printf("Cuda error: %s\n\n", cudaGetErrorString(ce));
        exit(1);
    }
}

__global__ void trimSpanningTree(EDGE_t *e,VERTEX_t *v,int num_v)
{
	int max, max_addr, last, last_max, next;
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Check if vertex is in bounds and not the root
	if((id < num_v) && (id != 0))
	{
		max = 0;
		max_addr = -1;
		last = -1;
		last_max = -1;

		// Get head of the linked list
		next = v[id].ei;

		// While the tail is not found
		while(next != -1)
		{
			// Check max and mark
			if(e[next].w > max)
			{
				max = e[next].w;
				if(max_addr != -1)
				{
					// Remove old max
					e[max_addr].rmvd = 1;
				}
				max_addr = next;
				last_max = last;
			}
			// If not max mark removed
			else
			{
				e[next].rmvd = 1;
			}
			// Store last and get next
			last = next;
			next = e[next].next_i;

		}

		// If not already at the front of the list, move it there
		if(last_max != -1)
		{
			next = e[max_addr].next_i;

			e[last_max].next_i = next;
			e[max_addr].next_i = v[id].ei;
			v[id].ei = max_addr;
		}
	}
}

__global__ void findCycles(EDGE_t *e,VERTEX_t *v,int num_v)
{
	int i;
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	int curr = id;
	int start = 0;
	int self_fnd = 0;
	int cyc_found = 1;
	int max = 0;
	
	// Check if vertex is in bounds and not the root
	if((id < num_v) && (id != 0))
	{
		// The edges can be backtracked (# of vertices) times until
		// it is known whether the initial vertex is connected to the root
		for(i = 0;i<num_v;i++)
		{
			// Check if root found
			if(curr == 0)
			{
				// Mark cycle as zero and break
				cyc_found = 0;
				break;
			}
			
			// Get next vertex
			curr = e[v[curr].ei].vo;
		}
		
		// Mark starting point in the cycle
		start = curr;
		
		// If the root was not found within (# of vertices) backtrackings,
		// then a cycle has been found
		if(cyc_found == 1)
		{
			
			max = 0;
			
			// If the initial vertex is found at the start or while scanning for the largest vertex in the cycle
			// than it belongs to the actual cycle and is not a branch off of it
			if(start == id)
			{
				self_fnd = 1;
			}
			
			
			// Scan for the max vertex number in the cycle. This is how we will know what cycle a vertex belongs to
			// later on
			while(curr != max)
			{
				if(curr > max)
				{
					max = curr;
				}
				// Mark self found
				if(curr == id)
				{
					self_fnd = 1;
				}
				// go to next vertex
				curr = e[v[curr].ei].vo;
				/*
				if(curr == start)
				{
					break;
				}*/
				
			}
			
			// If self was found in the scanning, mark the vertex with a cycle number
			// equal to the max found
			if(self_fnd == 1)
			{
				v[id].cyc = max;
			}
			// Otherwise, mark it as not a cycle
			else
			{
				v[id].cyc = 0;
			}
		}
		else
		{
			v[id].cyc = 0;
		}
	}
}

void addEdge(DIGRAPH_t *d,int addr)
{
	// Next list item
	int next;
	
	// Insert edge at head of the outgoing list
	next = d->v[d->e[addr].vo].eo;
	d->v[d->e[addr].vo].eo = addr;
	d->e[addr].next_o = next;
	
	// Insert edge at the head of the incoming list
	next = d->v[d->e[addr].vi].ei;
	d->v[d->e[addr].vi].ei = addr;
	d->e[addr].next_i = next;
}

int restoreSpanningTree(DIGRAPH_t *d)
{

	int i = 0;
	
	int cyc_found = 0;
	
	int max = 0;
	
	int next,num_proc,cycle,cyc_max,cyc_max_addr,prev,after;

	// Find the max adjusted incoming for each vertex and store it in the vertex
	for(i = 0;i<d->num_v;i++)
	{
		if(d->v[i].cyc > 0)
		{
			d->v[i].max_adj = NO_EDGE_FND;
			max = 0;
			
			next = d->v[i].ei;
			while(next != LIST_END)
			{
				if((d->e[next].rmvd == 1) && (d->v[i].cyc != d->v[d->e[next].vo].cyc) && (d->e[next].buried != 1))
				{
					d->e[next].adj_w = d->e[next].w - d->e[d->v[i].ei].w;
					
					if(d->e[next].w > max)
					{
						max = d->e[next].w;
						d->v[i].max_adj = next;
					}
				}
				next = d->e[next].next_i;
			}
		}
		
		d->v[i].proc = 0;
	}

	num_proc = 0;
	
	for(i = 0;i < d->num_v;i++)
	{
		if(d->v[i].cyc == 0)
		{
			d->v[i].proc = 1;
			num_proc++;
		}
		else
		{
			cyc_found = 1;
		}
	}

	while(num_proc != d->num_v)
	{
		cycle = 0;
		cyc_max = -100000;
		cyc_max_addr = -1;
		
		for(i = 0;i < d->num_v;i++)
		{
			
			if(d->v[i].proc==1)	
			{
				continue;
			}
			
			if(cycle == 0)
			{
				cycle = d->v[i].cyc;
			}
			
			if(d->v[i].cyc == cycle)
			{
				
				if(d->v[i].max_adj != -1)
				{
					if((d->e[d->v[i].max_adj].adj_w > cyc_max))
					{
						cyc_max = d->e[d->v[i].max_adj].adj_w;
						cyc_max_addr = d->v[i].max_adj;
					}
				}
				d->v[i].proc = 1;
				
				num_proc++;

				continue;
				
			}
			else
			{
				continue;
			}
			

		}
		
		d->e[d->v[d->e[cyc_max_addr].vi].ei].buried = 1;
		d->e[d->v[d->e[cyc_max_addr].vi].ei].rmvd = 1;
		d->e[cyc_max_addr].rmvd = 0;
		
		after = d->e[cyc_max_addr].next_i;
		
		next = d->v[d->e[cyc_max_addr].vi].ei;
		while(next != -1)
		{
			if(d->e[next].next_i == cyc_max_addr)
			{
				prev = next;
			}
			next = d->e[next].next_i;
		}
		
		d->e[prev].next_i = after;
		
		d->e[cyc_max_addr].next_i = d->v[d->e[cyc_max_addr].vi].ei;
		d->v[d->e[cyc_max_addr].vi].ei = cyc_max_addr;		
	}

	// First contraction is done, some information needs to be reinitialized
	
	for(i = 0;i<d->num_e;i++)
	{
		d->e[i].dead = 0;
		d->e[i].adj_w = -100000000;
	}

	for(i = 0;i<d->num_v;i++)
	{
		d->v[i].max_adj = -1;
	}
	
	return(cyc_found);
	
}

int main(int argc, char **argv)
{

	/* ADD_COMMENT
	 * - Indicate the use of the variable below, and whether they reside on host or on device.
	 */

	FILE *fin;

	int i, fnd_c;

	DIGRAPH_t d;
	
	EDGE_t *e_gpu, *e;
	VERTEX_t *v_gpu, *v;

	if(argc < 2)
	{
		printf("Error: Must have 1 argument (name of the file containing the directed graph\n");
		{
			return(-1);
		}
	}

	fin = fopen(argv[1],"r");
	if(fin == NULL)
	{
		printf("Error: Could not open input file");
		return(-1);
	}

	fscanf(fin,"%u",&d.num_v);
	fscanf(fin,"%u",&d.num_e);

	/* ADD_COMMENT
	 * - instruction below
	 * */
    cudaCheckError(cudaSetDevice(CUDA_DEVICE));

    /* ADD_CODE
     * - compute the size (in Bytes) of the data structures that must be transferred to the device, and save it in dev_mem_req variable
     * - query the device memory availability of the GPU, and save it to dev_mem_ava
     * - print the hardware configuration of the underlying device
     * - if dev_mem_req > 75% of dev_mem_ava, stop and return the message: "too much memory required on device"
     */


    /* ADD_COMMENT
     * - instructions below (please add a comment at each blank line)
     */

	v = (VERTEX_t *)malloc(d.num_v*sizeof(VERTEX_t));
	e = (EDGE_t *)malloc(d.num_e*sizeof(EDGE_t));

	d.v = v;
	d.e = e;

	memset(d.v,sizeof(VERTEX_t)*d.num_v,0);
	memset(d.e,sizeof(EDGE_t)*d.num_e,0);

	for(i = 0;i<d.num_v;i++)
	{
		d.v[i].num = i;
		d.v[i].eo = -1;
		d.v[i].ei = -1;
	}

	for(i = 0;i<d.num_e;i++)
	{
		fscanf(fin,"%i\t%i\t%i",&d.e[i].vo,&d.e[i].vi,&d.e[i].w);
		addEdge(&d,i);
	}

	dim3 threads_per_block(THREADS_PER_BLOCK);
	dim3 blocks_per_grid(d.num_v / THREADS_PER_BLOCK + 2);

	cudaCheckError(cudaMalloc((void **) &e_gpu, d.num_e*sizeof(EDGE_t)));
	cudaCheckError(cudaMalloc((void **) &v_gpu, d.num_v*sizeof(VERTEX_t)));

	cudaCheckError(cudaMemcpy((void *) e_gpu, d.e, d.num_e * sizeof(EDGE_t), cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy((void *) v_gpu, d.v, d.num_v * sizeof(VERTEX_t), cudaMemcpyHostToDevice));

	trimSpanningTree<<<blocks_per_grid, threads_per_block>>> (e_gpu,v_gpu,d.num_v);
	cudaCheckError(cudaGetLastError());

	cudaCheckError(cudaMemcpy((void *)  d.e,(void *)e_gpu, d.num_e * sizeof(EDGE_t), cudaMemcpyDeviceToHost));
	cudaCheckError(cudaMemcpy((void *)  d.v,(void *)v_gpu, d.num_v * sizeof(VERTEX_t), cudaMemcpyDeviceToHost));

	fnd_c = 1;

	while(fnd_c > 0)
	{
		cudaCheckError(cudaMemcpy((void *) e_gpu, d.e, d.num_e * sizeof(EDGE_t), cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemcpy((void *) v_gpu, d.v, d.num_v * sizeof(VERTEX_t), cudaMemcpyHostToDevice));

		findCycles<<<blocks_per_grid, threads_per_block>>> (e_gpu,v_gpu,d.num_v);
		cudaCheckError(cudaGetLastError());

		cudaCheckError(cudaMemcpy((void *)  d.e,(void *)e_gpu, d.num_e * sizeof(EDGE_t), cudaMemcpyDeviceToHost));
		cudaCheckError(cudaMemcpy((void *)  d.v,(void *)v_gpu, d.num_v * sizeof(VERTEX_t), cudaMemcpyDeviceToHost));

		fnd_c = restoreSpanningTree(&d);
	}

    /* ADD_CODE
     * - Check whether the found MST is indeed a directed spanning tree. You can implement this in a separate function and invoke it here.
     * - Print the found MST into file (the file name should be the last argument to the program). You can implement this in a separate function and invoke it here.
     * - Print to stdout the weight of the MST, and the number of iterations needed to find it.
     */

	free(e);
	free(v);
	cudaCheckError(cudaFree(e_gpu));
	cudaCheckError(cudaFree(v_gpu));
	return(0);
}
