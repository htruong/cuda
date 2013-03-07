#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define NO_EDGE_FND -1
#define LIST_END -1

/* The device number we will use */
/* It's better to be able to specify the device 
 * that we'll use at compile time */
#ifndef CUDA_DEVICE
#define CUDA_DEVICE 0
#endif

#define THREADS_PER_BLOCK 16

/* All Huan's defs to improve the code will go here */
#define FREEMEM_ULTILIZATION 0.75

/* ADD_COMMENT
 * - Describe all the data structures listed below, along with their fields.
 */

/* The parts marked with TODO DELETE can be safely removed, 
 *	however I'm keeping it here 
 *	because you did not instruct me to delete them */

/* Type of a vertex aka node */
typedef struct VERTEX_t
{
	int num;			// Node number
	int ei;				// Edge in
	int eo;				// Edge out
	int cyc;			// Is the vertex in a cyclic drirected subgraph?
	int max_adj;		// Adjusted incoming cost
	int next_v;			// No need TODO DELETE
	int proc;			// Flag: Node is processed?
} VERTEX_t;

/* Type of an edge */
typedef struct EDGE_t
{
	int vo;				// Node out
	int vi;				// Node in
	int w;				// Weight of the edge
	int adj_w;			// Adjusted weight
	int next_o;			// Next node out
	int next_i;			// Next node in
	int dead;			// Flag: Dead? 
	int rmvd;			// Flag: Removed?
	int buried;			// Flag: Buried?
} EDGE_t;

/* Type of a directed graph */
typedef struct DIGRAPH_t
{
	VERTEX_t *v;			// List of Nodes
	EDGE_t *e;				// List of Edges
	int num_v;				// Number of vertecies
	int num_e;				// Number of edges
	int mst_sum;			// MST sum
} DIGRAPH_t;

/* Type of a cyclic loop TODO DELETE */
typedef struct CYCLEDATA_t
{
	unsigned int curr;
	unsigned int mrkr;
	unsigned int cyc;
	unsigned int state;
	unsigned int self_fnd;
	unsigned int start;
	unsigned int max;
} CYCLEDATA_t;

/* ADD_COMMENT
 * - Describe the high-level functionality provided by the function below, along with their parameters.
 * - Specify which functions run on the host and which on the device.
 */

/* Wrapper HOST function to check if there is any error countered, if there is, then prints it out 
 * @ce The return of the call executed.
 * @returns void
 */
void cudaCheckError (cudaError_t ce);

/* HOST function to add an edge to the directed graph 
 * @d Pointer to directed graph
 * @addr Last node just added
 * @returns void
 */
void addEdge (DIGRAPH_t * d, int addr);

/* DEVICE function to trim a spanning tree
 * @e List of edges
 * @v List of vertecies/nodes
 * @returns void
 */
__global__ void trimSpanningTree (EDGE_t * e, VERTEX_t * v);

/* DEVICE function to find if there is a cycle in the tree
 * @e List of edges
 * @v List of vertecies/nodes
 * @num_v Number of vertecies
 * @returns void
 */
__global__ void findCycles (EDGE_t * e, VERTEX_t * v, int num_v);

/* HOST function to restore the spanning tree from the more abstracted one
 * @d Pointer to directed graph
 * @returns Number of cycles found
 */
int restoreSpanningTree (DIGRAPH_t * d);


inline void
cudaCheckError (cudaError_t ce)
{
	if (ce != cudaSuccess)
	{
		printf ("Cuda error: %s\n\n", cudaGetErrorString (ce));
		exit (1);
	}
}

__global__ void
trimSpanningTree (EDGE_t * e, VERTEX_t * v, int num_v)
{
	int max, max_addr, last, last_max, next;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check if vertex is in bounds and not the root
	if ((id < num_v) && (id != 0))
	{
		max = 0;
		max_addr = -1;
		last = -1;
		last_max = -1;
		
		// Get head of the linked list
		next = v[id].ei;
		
		// While the tail is not found
		while (next != -1)
		{
			// Check max and mark
			if (e[next].w > max)
			{
				max = e[next].w;
				if (max_addr != -1)
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
		if (last_max != -1)
		{
			next = e[max_addr].next_i;
			
			e[last_max].next_i = next;
			e[max_addr].next_i = v[id].ei;
			v[id].ei = max_addr;
		}
	}
}

__global__ void
findCycles (EDGE_t * e, VERTEX_t * v, int num_v)
{
	int i;
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	int curr = id;
	int start = 0;
	int self_fnd = 0;
	int cyc_found = 1;
	int max = 0;
	
	// Check if vertex is in bounds and not the root
	if ((id < num_v) && (id != 0)) {
		// The edges can be backtracked (# of vertices) times until
		// it is known whether the initial vertex is connected to the root
		for (i = 0; i < num_v; i++) {
			// Check if root found
			if (curr == 0) {
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
		if (cyc_found == 1) {
			
			max = 0;
			
			// If the initial vertex is found at the start or while scanning for the largest vertex in the cycle
			// than it belongs to the actual cycle and is not a branch off of it
			if (start == id)
			{
				self_fnd = 1;
			}
			
			
			// Scan for the max vertex number in the cycle. This is how we will know what cycle a vertex belongs to
			// later on
			while (curr != max) {
				if (curr > max) {
					max = curr;
				}
				// Mark self found
				if (curr == id) {
					self_fnd = 1;
				}
				// go to next vertex
				curr = e[v[curr].ei].vo;
				/*
				 *	         if(curr == start)
				 *	         {
				 *	         break;
				 } */	
			}
			
			// If self was found in the scanning, mark the vertex with a cycle number
			// equal to the max found
			if (self_fnd == 1) {
				v[id].cyc = max;
			}
			// Otherwise, mark it as not a cycle
			else {
				v[id].cyc = 0;
			}
		} else {
			v[id].cyc = 0;
		}
	}
}
	
void
addEdge (DIGRAPH_t * d, int addr)
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

int
restoreSpanningTree (DIGRAPH_t * d)
{
	
	int i = 0;
	
	int cyc_found = 0;
	
	int max = 0;
	
	int next, num_proc, cycle, cyc_max, cyc_max_addr, prev, after;
	
	// Find the max adjusted incoming for each vertex and store it in the vertex
	for (i = 0; i < d->num_v; i++)
	{
		if (d->v[i].cyc > 0)
		{
			d->v[i].max_adj = NO_EDGE_FND;
			max = 0;
			
			next = d->v[i].ei;
			while (next != LIST_END)
			{
				if ((d->e[next].rmvd == 1)
					&& (d->v[i].cyc != d->v[d->e[next].vo].cyc)
					&& (d->e[next].buried != 1))
				{
					d->e[next].adj_w = d->e[next].w - d->e[d->v[i].ei].w;
					
					if (d->e[next].w > max)
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
	
	for (i = 0; i < d->num_v; i++)
	{
		if (d->v[i].cyc == 0)
		{
			d->v[i].proc = 1;
			num_proc++;
		}
		else
		{
			cyc_found = 1;
		}
	}
	
	while (num_proc != d->num_v)
	{
		cycle = 0;
		cyc_max = -100000;
		cyc_max_addr = -1;
		
		for (i = 0; i < d->num_v; i++)
		{
			
			if (d->v[i].proc == 1)
			{
				continue;
			}
			
			if (cycle == 0)
			{
				cycle = d->v[i].cyc;
			}
			
			if (d->v[i].cyc == cycle)
			{
				
				if (d->v[i].max_adj != -1)
				{
					if ((d->e[d->v[i].max_adj].adj_w > cyc_max))
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
		while (next != -1)
		{
			if (d->e[next].next_i == cyc_max_addr)
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
	
	for (i = 0; i < d->num_e; i++)
	{
		d->e[i].dead = 0;
		d->e[i].adj_w = -100000000;
	}
	
	for (i = 0; i < d->num_v; i++)
	{
		d->v[i].max_adj = -1;
	}
	
	return (cyc_found);
	
}

int
main (int argc, char **argv)
{
	
	/* ADD_COMMENT
		* - Indicate the use of the variable below, and whether they reside on host or on device.
		*/
	
	// HOST Cursor for the file handles, ingraph and outgraph
	FILE 	*fin, *fout; 
	
	// HOST temp variables: i, fnd_c - cycles found
	int 	i, fnd_c; 
	
	// HOST directed graph
	DIGRAPH_t d; 
	
	// edges on DEVICE and HOST respectively
	EDGE_t *e_gpu, *e; 
	
	// nodes on DEVICE and HOST respectively
	VERTEX_t *v_gpu, *v; 
	
	if (argc < 2) {
		fprintf (stderr,
		 "Error: Must have ast least 1 argument\n"
			"%s {input graph} [-o output_file]\n", argv[0]
		);
		abort();
	}
	
	fin = fopen (argv[1], "r");
	
	if (fin == NULL) {
		fprintf (stderr, "Error: Could not open input file");
		abort();
	}
	
	// Try to determine if the last param is the output to file directive
	// if not, then we can try to output to stdout
	if (strncmp(argv[argc - 2], "-o", 16) == 0) {
		fout = fopen(argv[argc - 1], "w");
	} else {
		fprintf (stderr, "No [-o outputfile] parameter specified.\n"
		"Outputting to standard output\n");
		fout = stdout;
	}
	
	if (fout == NULL) {
		fprintf (stderr, "Error: Could not open output file");
		abort();
	}
	
	
	fscanf (fin, "%u", &d.num_v);
	fscanf (fin, "%u", &d.num_e);
	
	/* ADD_COMMENT
		* - instruction below
		*/
	
	/* ADD_CODE
		* - compute the size (in Bytes) of the data structures that must be transferred to the device, and save it in dev_mem_req variable
		* - query the device memory availability of the GPU, and save it to dev_mem_ava
		* - print the hardware configuration of the underlying device
		* - if dev_mem_req > 75% of dev_mem_ava, stop and return the message: "too much memory required on device"
		*/
	
	// Direct the CUDA device that we'll want to be using
	cudaCheckError (cudaSetDevice (CUDA_DEVICE)); 
	
	// Print the configuration 
	// Query the device to get an idea about the device
	cudaDeviceProp * prop = new cudaDeviceProp;
	cudaGetDeviceProperties(prop, CUDA_DEVICE);
	
	size_t mem_free = 0;
	size_t mem_total = 0;
	cudaMemGetInfo(&mem_free, &mem_total);
	fprintf(stderr, "-------------- GPU: %s --------------\n"
		"\tCompute Cap: \t%d.%d\n"
		"\tWarp size: \t%d\n"
		"\tMemory: \tFree: %luMB, Total: %luMB\n", 
		prop->name,
		prop->major, prop->minor,
		prop->warpSize,
		mem_free/1024/1024, mem_free/1024/1024);
	delete prop;
	
	// See if we could allocate that much memory safely
	unsigned int mem_required = 0;
	mem_required = 
		d.num_v * sizeof (VERTEX_t) + d.num_e * sizeof (EDGE_t);
		
	if (mem_free * FREEMEM_ULTILIZATION < mem_required) {
		fprintf(stderr, 
			"FATAL: too much memory required on device [needed %d bytes]\n",
		  mem_required);
		abort();
	}
	
	/* ADD_COMMENT
		* - instructions below (please add a comment at each blank line)
		*/
	
	// Allocate memory on the host for the array of edges and nodes needed for computation
	v = (VERTEX_t *) malloc (d.num_v * sizeof (VERTEX_t));
	e = (EDGE_t *) malloc (d.num_e * sizeof (EDGE_t));
	
	// Set the pointer of the edges and nodes to the one that we just allocated
	d.v = v;
	d.e = e;
	
	// Clear up memory regions
	memset (d.v, sizeof (VERTEX_t) * d.num_v, 0);
	memset (d.e, sizeof (EDGE_t) * d.num_e, 0);
	
	// Initialize all the vertecies (nodes)
	// Setting default value, not connected to anything in or out
	for (i = 0; i < d.num_v; i++) {
		d.v[i].num = i;
		d.v[i].eo = -1;
		d.v[i].ei = -1;
	}
	
	// Populate the edges with the information we have in file
	for (i = 0; i < d.num_e; i++)
	{
		fscanf (fin, "%i\t%i\t%i", &d.e[i].vo, &d.e[i].vi, &d.e[i].w);
		addEdge (&d, i);
	}
	
	// Calculate number of thread/block and block/grid
	dim3 threads_per_block (THREADS_PER_BLOCK);
	dim3 blocks_per_grid (d.num_v / THREADS_PER_BLOCK + 2);
	
	// Allocate memory on device
	cudaCheckError (cudaMalloc ((void **) &e_gpu, d.num_e * sizeof (EDGE_t)));
	cudaCheckError (cudaMalloc ((void **) &v_gpu, d.num_v * sizeof (VERTEX_t)));
	
	// Memcpy from HOST->DEV
	cudaCheckError (cudaMemcpy
	((void *) e_gpu, d.e, d.num_e * sizeof (EDGE_t),
		cudaMemcpyHostToDevice));
	cudaCheckError (cudaMemcpy
	((void *) v_gpu, d.v, d.num_v * sizeof (VERTEX_t),
		cudaMemcpyHostToDevice));
	
	// Do initial trimming
	trimSpanningTree <<< blocks_per_grid, threads_per_block >>> (e_gpu, v_gpu,
																	d.num_v);
	cudaCheckError (cudaGetLastError ());
	
	// Copy the result back
	cudaCheckError (cudaMemcpy
	((void *) d.e, (void *) e_gpu, d.num_e * sizeof (EDGE_t),
		cudaMemcpyDeviceToHost));
	cudaCheckError (cudaMemcpy
	((void *) d.v, (void *) v_gpu, d.num_v * sizeof (VERTEX_t),
		cudaMemcpyDeviceToHost));
	
	fnd_c = 1;
	
	// Loop while we're still have cyclic sub-graphs
	while (fnd_c > 0)
	{
		cudaCheckError (cudaMemcpy
		((void *) e_gpu, d.e, d.num_e * sizeof (EDGE_t),
			cudaMemcpyHostToDevice));
		cudaCheckError (cudaMemcpy
		((void *) v_gpu, d.v, d.num_v * sizeof (VERTEX_t),
			cudaMemcpyHostToDevice));
		
		// Contract nodes and recalculate weights
		findCycles <<< blocks_per_grid, threads_per_block >>> (e_gpu, v_gpu,
																d.num_v);
		cudaCheckError (cudaGetLastError ());
		
		cudaCheckError (cudaMemcpy
		((void *) d.e, (void *) e_gpu,
			d.num_e * sizeof (EDGE_t), cudaMemcpyDeviceToHost));
		cudaCheckError (cudaMemcpy
		((void *) d.v, (void *) v_gpu,
			d.num_v * sizeof (VERTEX_t), cudaMemcpyDeviceToHost));
		
		// Uncontract nodes and recalculate weights
		fnd_c = restoreSpanningTree (&d);
	}
	
	/* ADD_CODE
		* - Check whether the found MST is indeed a directed spanning tree. You can implement this in a separate function and invoke it here.
		* - Print the found MST into file (the file name should be the last argument to the program). You can implement this in a separate function and invoke it here.
		* - Print to stdout the weight of the MST, and the number of iterations needed to find it.
		*/
	
	
	
	free (e);
	free (v);
	cudaCheckError (cudaFree (e_gpu));
	cudaCheckError (cudaFree (v_gpu));
	return (0);
}
	