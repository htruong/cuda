#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define NO_EDGE_FND -1
#define LIST_END -1

/* The device number we will use */
/* It's better to be able to specify the device 
 * that we'll use at compile time */
#ifndef CUDA_DEVICE
#define CUDA_DEVICE 0
#endif


/* All Huan's defs to improve the code will go here */
#define FREEMEM_ULTILIZATION 0.75

#define IMPR_BETTER_THREADS_PER_BLOCK
#define BETTER_CYCLE_CHECK

#ifndef IMPR_BETTER_THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 16
#else
#define THREADS_PER_BLOCK 256
#endif
/* ADD_COMMENT
 * - Describe all the data structures listed below, along with their fields.
 */

/* The parts marked with TODO DELETE can be removed, 
 *	However I'm keeping it here as you did not instruct me to delete them */

/* Type of a vertex aka node */
typedef struct VERTEX_t
{
	int num;			// Node number
	int ei;				// Edge in
	int eo;				// Edge out
	int cyc;			// Is the node in a cyclic drirected subgraph?
	int max_adj;		// Adjusted incoming cost
	int next_v;			// If it's going to be visited the next time
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

/* Type of a cyclic loop? TODO DELETE */
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

/* HOST function to check if the spanning tree is a legitimate one
 * This function does two things:
 * 	- It verifies that all nodes can be visited from the root node
 *  - It verifies that the graph is not cyclic
 * It will also calculate the weight of the MST while checking.
 * NOTICE it will not verify if the solution is the optimal one
 * @d Pointer to directed graph
 * @returns 	true if pass
 */
bool verify_st(DIGRAPH_t * d);

inline void
cudaCheckError (cudaError_t ce)
{
	if (ce != cudaSuccess)
	{
		printf ("Cuda error: %s\n\n", cudaGetErrorString (ce));
		exit (1);
	}
}

/* HOST function to get the time 
 * @returns the current timestamp
 * source: cuda needle gpu code <becchim>
 * */
inline double gettime() {
	struct timeval t;
	gettimeofday(&t,NULL);
	return t.tv_sec+t.tv_usec*1e-6;
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
			if (curr == 0) {
				// Mark cycle as zero and break
				cyc_found = 0;
				#ifdef BETTER_CYCLE_CHECK
				v[id].cyc = 0;
				return;
				#endif
				break;
			}
			
			// Get next vertex
			curr = e[v[curr].ei].vo;
		}
		
		// Mark starting point in the cycle
		start = curr;
		
		// If the root was not found within (# of vertices) backtrackings,
		// then a cycle has been found
		#ifndef BETTER_CYCLE_CHECK
		if (cyc_found == 1) {
			max = 0;
		#endif	
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
		#ifndef BETTER_CYCLE_CHECK
		} else {
			v[id].cyc = 0;
		}
		#endif
	}
}
	
void
addEdge (DIGRAPH_t * d, int addr)
{
	// Next list item
	int next;
	
	// Insert edge at head of the outgoing list
	// What this does:
	// - It gets the edge id (addr), get the edge out of the node
	// - It assigns the edge out of the node to the edge id
	// - Then it assigns the next_o of the edge to be 
	//   the old edge out of the node (? - TOINVESTIGATE)
	next = d->v[d->e[addr].vo].eo;
	d->v[d->e[addr].vo].eo = addr;
	d->e[addr].next_o = next;
	
	// Insert edge at the head of the incoming list
	// Same here
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

bool 
verify_st(DIGRAPH_t * d) {
	// First try to visit all nodes to mark them unprocessed/unvisited
	for (int i = 0; i < d->num_v; i++) {
		d->v[i].proc = 0;
		d->v[i].next_v = 0;
	}
	
	// At the same time let's also try to update the mst sum
	d->mst_sum = 0;
	
	int next_v_count = 0;
	// Now try to start from the start node
	d->v[0].next_v = 1;
	next_v_count++;
	
	while (next_v_count > 0) {
		// Walk through all nodes, visit all nodes that are marked for visiting
		for (int i = 0; i < d->num_v; i++) {
			// if it's marked for visited...
			if (d->v[i].next_v == 1) {
				//fprintf(stderr, "Visiting node %d ->", i);
				// but not visited yet, then visit it
				if (d->v[i].proc == 0) {
					d->v[i].proc = 1;
					// find all the nodes that this node could visit
					for (int j = 0; j < d->num_e; j++) {
						if ((d->e[j].vo == i) && (!d->e[j].rmvd) && (!d->e[j].dead) && (!d->e[j].buried)) {
							//fprintf(stderr, "%d, ", d->e[j].vi);
							// Mark it for next visit
							d->v[d->e[j].vi].next_v = 1;
							d->mst_sum += d->e[j].w;
							next_v_count ++;
						}
					}
					//fprintf(stderr, "\n", 0);
				} else {
					// This node is already visited and is asked to visit again
					fprintf(stderr, "OOOPS: Node %d is already visited *Cyclic!!!*\n", i);
					return false;
				}
			d->v[i].next_v = 0;
			next_v_count --;
			break;
			}
		}
	}
	
	// At this point we're sure we have visited all the nodes that we could visit
	// And there are no cyclic nodes found
	// Make sure we have visited every node
	for (int i = 0; i < d->num_v; i++) {
		if (!d->v[i].proc) {
			fprintf(stderr, "OOOPS: Node %d is unreachable!\n", i);
			return false;
		}
	}
	
	// if we could reach here the graph should be fine
	return true;
}

void print_to_file(DIGRAPH_t * d, FILE * fout) {
	for (int n = 0; n < d->num_v; n++) {
		for (int i = d->num_e - 1; i > -1; i--) {
			if ((n == d->e[i].vo) && (!d->e[i].rmvd) && (!d->e[i].dead) && (!d->e[i].buried)) {
				fprintf(fout, "%d\t%d\t%d\t\n", n, d->e[i].w, d->e[i].vi);
				break;
			}
		}
	}
}

int
main (int argc, char **argv)
{
	
	/* ADD_COMMENT
		* - Indicate the use of the variable below, and whether they reside on host or on device.
		*/
	
	// HOST Cursor for the file handles, ingraph and outgraph
	FILE 	* fin, * fout; 
	
	// HOST temp variables: i, fnd_c - cycles found, interations_count
	int 	i, fnd_c; 
	unsigned long interations_count = 0;
	
	// HOST directed graph
	DIGRAPH_t d; 
	
	// edges on DEVICE and HOST respectively
	EDGE_t *e_gpu, *e; 
	
	// nodes on DEVICE and HOST respectively
	VERTEX_t *v_gpu, *v; 
	
	// Timing variables
	double time_start, time_init, time_memcpy_fwd, time_memcpy_bck, 
		time_trim, time_find_cyc, time_restore, time_total, time_total_start;
	
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
	// if not, then we can try to output to stderr
	if (strncmp(argv[argc - 2], "-o", 16) == 0) {
		fout = fopen(argv[argc - 1], "w");
	} else {
		fprintf (stderr, "No [-o outputfile] parameter specified.\n"
		"Outputting to standard error output\n");
		fout = stderr;
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
	} else {
	fprintf(stderr, 
			"Allocating %d bytes for nodes and %d bytes for edges\n",
		  d.num_v * sizeof (VERTEX_t), d.num_e * sizeof (EDGE_t));
	}
	
	/* ADD_COMMENT
		* - instructions below (please add a comment at each blank line)
		*/
	
	
	time_start = gettime();
	
	time_total_start = time_start;
	
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
	
	time_init = gettime() - time_start;
	
	time_start = gettime();
	// Memcpy from HOST->DEV
	cudaCheckError (cudaMemcpy
	((void *) e_gpu, d.e, d.num_e * sizeof (EDGE_t),
		cudaMemcpyHostToDevice));
	cudaCheckError (cudaMemcpy
	((void *) v_gpu, d.v, d.num_v * sizeof (VERTEX_t),
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	time_memcpy_fwd = gettime() - time_start;
	
	time_start = gettime();
	// Do initial trimming
	trimSpanningTree <<< blocks_per_grid, threads_per_block >>> (e_gpu, v_gpu,
																	d.num_v);
	cudaCheckError (cudaGetLastError ());
	cudaDeviceSynchronize();
	time_trim = gettime() - time_start;
	
	time_start = gettime();
	// Copy the result back
	cudaCheckError (cudaMemcpy
	((void *) d.e, (void *) e_gpu, d.num_e * sizeof (EDGE_t),
		cudaMemcpyDeviceToHost));
	cudaCheckError (cudaMemcpy
	((void *) d.v, (void *) v_gpu, d.num_v * sizeof (VERTEX_t),
		cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	time_memcpy_bck = gettime() - time_start;
	
	
	fnd_c = 1;
	interations_count = 0;
	time_find_cyc = 0;
	time_restore = 0;
	
	// Loop while we're still have cyclic sub-graphs
	while (fnd_c > 0) {
		
		time_start = gettime();
		cudaCheckError (cudaMemcpy
		((void *) e_gpu, d.e, d.num_e * sizeof (EDGE_t),
			cudaMemcpyHostToDevice));
		cudaCheckError (cudaMemcpy
		((void *) v_gpu, d.v, d.num_v * sizeof (VERTEX_t),
			cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		time_memcpy_fwd += gettime() - time_start;
		
		time_start = gettime();
		// Contract nodes and recalculate weights
		findCycles <<< blocks_per_grid, threads_per_block >>> (e_gpu, v_gpu,
																d.num_v);
		cudaCheckError (cudaGetLastError ());
		cudaDeviceSynchronize();
		time_find_cyc += gettime() - time_start;
	
		time_start = gettime();
		cudaCheckError (cudaMemcpy
		((void *) d.e, (void *) e_gpu,
			d.num_e * sizeof (EDGE_t), cudaMemcpyDeviceToHost));
		cudaCheckError (cudaMemcpy
		((void *) d.v, (void *) v_gpu,
			d.num_v * sizeof (VERTEX_t), cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		time_memcpy_bck += gettime() - time_start;
		
		// Uncontract nodes and recalculate weights
		time_start = gettime();
		fnd_c = restoreSpanningTree (&d);
		time_restore += gettime() - time_start;
		interations_count ++;
	}
	
	// Now print timing information
	time_total = gettime() - time_total_start;
	printf("TIMINGPROFILE, %d, %d, %f, %d, %f, %f, %f, %f, %f, %f\n", 
		d.num_v, d.num_e,
		time_total,
		interations_count,
		time_init, 
		time_memcpy_fwd, time_memcpy_bck,
		time_trim, time_find_cyc, time_restore
	);
	
	/* ADD_CODE
		* - Check whether the found MST is indeed a directed spanning tree. You can implement this in a separate function and invoke it here.
		* - Print the found MST into file (the file name should be the last argument to the program). You can implement this in a separate function and invoke it here.
		* - Print to stdout the weight of the MST, and the number of iterations needed to find it.
		*/
	
	if (verify_st(&d)) {
		printf("Weight of spanning tree: %d.\n"
		"Found after %d iterations.\n", d.mst_sum, interations_count);
		print_to_file(&d, fout);
	} else {
		fprintf(stderr, 
			"FATAL: The tree we found has got errors.\n",
		  d.mst_sum);
		abort();
	}
	
	free (e);
	free (v);
	cudaCheckError (cudaFree (e_gpu));
	cudaCheckError (cudaFree (v_gpu));
	return (0);
}
