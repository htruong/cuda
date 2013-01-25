#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>


// CUDA runtime
#include <cuda_runtime.h>

typedef unsigned int uint;

using namespace std;

// C[m,k] = A[m,n] * B[n,k]
// m rows, k columns

// C[m,k] is stored as such
// i[m=1, k=1], i[m=1, k=2],
// i[m=2, k=1], i[m=2, k=2], and so on


void print_matrix(float *matrix, uint size) {
  uint how_many_elements = 25;

  if (size < 25) {
    how_many_elements = size;
  }

  for (uint i = 0; i < how_many_elements; i++) {
    printf(" %f", matrix[i]);
  }

  printf("\n");
}

void host_matmul(float *a, float *b, float *c, uint m, uint n, uint k) {
  // Go row by row
  for (uint i = 0; i < m; i++) {
    // Go column by column
    for (uint j = 0; j < k; j++) {
      // Go through the cells of each dest matrix
      for (uint t = 0; t < n; t++) {
	// c (i, j) += a(i, t) * b(t, j)
	c[i * k + j] += a[i * n + t] * b[t * k + j];
      }
    }
  }
}

__global__ void kernel_mathmul(float * a, float * b, float * c, uint k, uint n)
{
	// This does not work yet when the columns count of the first matrix is > 256 or whatever...
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < k) {
      /*
      printf("[%d] ", i);

	  if (i == 0) {
		  printf("a now is: ");
	  	  for (uint j = 0; j< n; j++) {
		  	  printf("%f ", a[j]);
	  	  }
	  	  printf("\n");
	  }
	  */
      c[i] = 0; // reset it first, remember it doesn't get cleared everytime
      
      for (uint t=0; t<n; t++) {
    	  	  c[i] += a[t] * b[i + t*k];
      }
    }

}

void dev_matmul(float *a, float *b, float *c, uint m, uint n, uint k) {
    int threads_per_block = 256;
    int blocks =  ceil((float)(m) / threads_per_block);
    // printf("Number of blocks: [%d]\n", blocks);
    
    float * dev_a_onerow;
	cudaMalloc(&dev_a_onerow, n*sizeof(float));
    
    float * dev_c_onerow;
    cudaMalloc(&dev_c_onerow, k*sizeof(float));
    
    float * dev_b;
    cudaMalloc(&dev_b, m*n*sizeof(float));
    
    cudaMemcpy(dev_b, b, m*n*sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute the rows of resulting matrix one line by one line.
    for (uint i = 0; i < m; i++) {
      // copy one row of the a matrix to the device
      cudaMemcpy(dev_a_onerow, a + (i * n), n * sizeof(float), cudaMemcpyHostToDevice);

      //printf("Calling kernel mathmul\n");
      kernel_mathmul <<<blocks,threads_per_block>>> (dev_a_onerow, dev_b, dev_c_onerow, k, n);
      cudaThreadSynchronize();

      // copy the resulting row back
      cudaMemcpy(c + i * k, dev_c_onerow, k * sizeof(float), cudaMemcpyDeviceToHost);

      //print_matrix(c, m*k);
    }
    
    cudaFree(dev_a_onerow);
    cudaFree(dev_b);
    cudaFree(dev_c_onerow);
}

void init_matrix(float *matrix, uint size) {
  for (uint i = 0; i < size; i++) {
    matrix[i] = ((float) rand()) / RAND_MAX;
    //matrix[i] = 1.0;
  }
}

void clear_matrix(float *matrix, uint size) {
  for (uint i = 0; i < size; i++) {
    matrix[i] = 0.0;
  }
}


void verify_matrix(const float *matrix1, const float *matrix2, const uint size) {
  for (uint i = 0; i < size; i++) {
    assert(matrix1[i] == matrix2[i]);
  }
}



int main() {
  uint m = 200;
  uint n = 200;
  uint k = 200;
  
  srand( time (NULL) );
  
  float * a_matrix = new float[m*n];
  float * b_matrix = new float[n*k];
  float * c_matrix = new float[m*k];
  float * c_dev_matrix = new float[m*k];

  clock_t start;
  clock_t end;
  /*
  printf("Initializing matrices with sample numbers\n", 2, 2);
  a_matrix[0] = 1; a_matrix[2] = -2; a_matrix[4] = 3; a_matrix[5] = -1;
  b_matrix[1] = 3; b_matrix[2] = -2; b_matrix[3] = -1; b_matrix[5] = 4;
  
  printf("Here comes the first 25 elements of a, b, c before:\n");
  print_matrix(a_matrix, m*n);
  print_matrix(b_matrix, n*k);
  print_matrix(c_matrix, m*k);
  
  printf("Do host-calculation:\n");
  host_matmul(a_matrix, b_matrix, c_matrix, m, n, k);
  
  printf("Here comes the first 25 elements of a, b, c after:\n");
  print_matrix(a_matrix, m*n);
  print_matrix(b_matrix, n*k);
  print_matrix(c_matrix, m*k);
  
  printf("Asserting that host-calculation is correct: ");
  assert(c_matrix[0] == 0);
  assert(c_matrix[1] == -5);
  assert(c_matrix[2] == -6);
  assert(c_matrix[3] == -7);
  printf("It is.\n");
  ////////////////////////////////////////////////
  */
  printf("Intialize random matrices: ");
  init_matrix(a_matrix, m*n);
  init_matrix(b_matrix, n*k);
  printf("Done. \n");
  print_matrix(a_matrix, m*n);
  print_matrix(b_matrix, n*k);
  print_matrix(c_matrix, m*k);
  ///////////////////////////////////////////////

  printf("Do host-calculation: ");
  clear_matrix(c_matrix, m*k);
  start = clock();
  host_matmul(a_matrix, b_matrix, c_matrix, m, n, k);
  end = clock();
  printf("Took %d clock cycles\n", end - start);

  printf("Here comes the first 25 elements of c:\n");
  print_matrix(c_matrix, m*k);

  printf("Do device-calculation: ");
  start = clock();
  dev_matmul(a_matrix, b_matrix, c_dev_matrix, m, n, k);
  end = clock();
  printf("Took %d clock cycles\n", end - start);

  printf("Here comes the first 25 elements of c on device:\n");
  print_matrix(c_dev_matrix, m*k);
  printf("Checking if GPU result is correct: ");
  verify_matrix(c_matrix, c_dev_matrix, m*k);
  printf("It is.");

  delete a_matrix;
  delete b_matrix;
  delete c_matrix;
  delete c_dev_matrix;
}
