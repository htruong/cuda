#define TIMES 10
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>



double
gettime ()
{
  struct timeval t;
  gettimeofday (&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

__global__ void
dummy_function (int *array, unsigned int howlarge)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tot = 0;
  int pos = 0;
  for (int i = 0; i < 100; i++) {
      for (int dummy = 0; dummy < 1000; dummy++) {
	    tot = tot + (i * dummy + threadIdx.x) & 0x000f;
	    array[pos++] = tot;
	  }
    }
}


void
runTest ()
{
  double start, end;
  unsigned int nints = 100 * 1024 * 1024;
  unsigned int sz = nints * sizeof (int);

  unsigned int nints_small = 1 * 1024 * 1024;
  unsigned int sz_small = nints_small * sizeof (int);

  cudaStream_t stream1, stream2 ;
	cudaStreamCreate ( &stream1) ;
	cudaStreamCreate ( &stream2) ;


#ifdef _LP64
  printf ("Running on a 64-bit platform!\n", 0);
#else
#endif

  int *dummy_cpu, *dummy_cpu2, *dummy_small_cpu, *dummy_small_cpu2;
  dummy_cpu = (int *) malloc (sz);
  dummy_cpu2 = (int *) malloc (sz);
  dummy_small_cpu = (int *) malloc (sz_small);
  dummy_small_cpu2 = (int *) malloc (sz_small);

  int *dummy_gpu, *dummy_gpu2, *dummy_small_gpu, *dummy_small_gpu2;
  cudaMalloc ((void **) &dummy_gpu, sz);
  cudaMalloc ((void **) &dummy_gpu2, sz);
  cudaMalloc ((void **) &dummy_small_gpu, sz_small);
  cudaMalloc ((void **) &dummy_small_gpu2, sz_small);

  double kernelt = 0, memcpyt = 0, st = 0, ast = 0;


  for (int i = 0; i < TIMES; i++) {
      start = gettime ();
      dummy_function <<< 100, 512 >>> (dummy_gpu, sz);
      cudaDeviceSynchronize ();
      end = gettime ();
      printf ("kernel time: %f\n", end - start);
      kernelt += end - start;

      start = gettime ();
      cudaMemcpy (dummy_cpu, dummy_gpu, sz, cudaMemcpyDeviceToHost);
      end = gettime ();
      printf ("memcpy time: %f\n", end - start);
      memcpyt += end - start;


      start = gettime ();


      // Do the sync routine
      cudaMemcpy (dummy_small_gpu, dummy_small_cpu, sz_small,
		  cudaMemcpyHostToDevice);
      dummy_function <<< 100, 512 >>> (dummy_gpu, sz);
      cudaMemcpy (dummy_cpu, dummy_gpu, sz, cudaMemcpyDeviceToHost);
      cudaMemcpy (dummy_small_gpu2, dummy_small_cpu2, sz_small,
		  cudaMemcpyHostToDevice);
      dummy_function <<< 100, 512 >>> (dummy_gpu2, sz);
      cudaMemcpy (dummy_cpu2, dummy_gpu2, sz, cudaMemcpyDeviceToHost);

      cudaDeviceSynchronize ();
      end = gettime ();
      printf ("sync time: %f\n", end - start);
      st += end - start;

      start = gettime ();

      // Do the async routine
      cudaMemcpyAsync (dummy_small_gpu, dummy_small_cpu, sz_small,
		  cudaMemcpyHostToDevice, stream1);
      dummy_function <<< 100, 512, 0, stream1 >>> (dummy_gpu, sz);
      cudaDeviceSynchronize ();
      cudaMemcpyAsync (dummy_cpu, dummy_gpu, sz, cudaMemcpyDeviceToHost, stream2);
      cudaMemcpyAsync (dummy_small_gpu2, dummy_small_cpu2, sz_small,
		  cudaMemcpyHostToDevice, stream1);
      dummy_function <<< 100, 512, 0, stream1 >>> (dummy_gpu2, sz);
      cudaDeviceSynchronize ();
      cudaMemcpyAsync (dummy_cpu2, dummy_gpu2, sz, cudaMemcpyDeviceToHost, stream2);

      cudaDeviceSynchronize ();
      end = gettime ();
      printf ("async time: %f\n", end - start);
      ast += end - start;

      printf ("-- Done round %d --\n", i);
    }
  printf ("Average:\nkerneltime=%f, memcpy=%f, sync=%f, async=%f\n",
	  kernelt / TIMES, memcpyt / TIMES, st / TIMES, ast / TIMES);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main (int argc, char **argv)
{
  runTest ();
  return EXIT_SUCCESS;
}
