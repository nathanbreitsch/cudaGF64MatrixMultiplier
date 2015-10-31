#include <stdio.h>

//define matrix type
typedef struct{
  int row_count;
  int column_count;
  int* elements;
} Matrix;


__global__ void multiply_kernel_stupid(const Matrix left, const Matrix right, Matrix result);
Matrix ones(int row_count, int column_count);
Matrix multiply(Matrix left, Matrix right);
void print_matrix(Matrix mat);

int main(){
  //make the matrices you want to multiply
  Matrix A = ones(10, 5);
  Matrix B = ones(5, 10);
  Matrix result = multiply(A, B);
  print_matrix(A);
  print_matrix(B);
  print_matrix(result);
}

//stupid kernel, one thread per result cell, global memory, no use of spacial locality.
__global__ void multiply_kernel_stupid(const Matrix left, const Matrix right, Matrix result){
  int sum = 0;
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = 0; i < left.column_count; i++){
    int left_index = row_index * left.column_count  + i;
    int right_index = column_index + i * right.column_count;
    sum += left.elements[left_index] * right.elements[right_index];
  }
  result.elements[row_index * result.row_count + column_index] = sum;

}

Matrix multiply(Matrix left, Matrix right){
  cudaError_t error;
  //step 1: allocate memory on the kernel for left, right, result
  Matrix left_d, right_d;
  left_d.row_count = left.row_count;
  left_d.column_count = left.column_count;
  size_t left_size = left.row_count * left.column_count * sizeof(int);
  error = cudaMalloc((void**) &left_d.elements, left_size);
  if(error != cudaSuccess){
    printf("error allocating left matrix\n");
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }


  right_d.row_count = right.row_count;
  right_d.column_count = right.column_count;
  size_t right_size = right.row_count * right.column_count * sizeof(int);
  error = cudaMalloc((void**) &right_d.elements, right_size);
  if(error != cudaSuccess){ printf("error allocating right matrix\n"); }

  //step 2: allocate memory on the host and device for result
  Matrix result, result_d;
  result.row_count = result_d.row_count = left.row_count;
  result.column_count = result_d.column_count = right.column_count;
  size_t result_size = result.row_count * result.column_count * sizeof(int);
  result.elements = (int*) malloc(result_size);
  error = cudaMalloc((void**) &result_d.elements, result_size);
  if(error != cudaSuccess){ printf("error allocating matrix\n"); }

  //step 3: copy left and right to device
  error = cudaMemcpy(left_d.elements, left.elements, left_size, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){ printf("error copying left matrix\n"); }
  error = cudaMemcpy(right_d.elements, right.elements, right_size, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){ printf("error copying right matrix\n"); }

  //step 4: launch kernel



  dim3 block_dims(32, 32);
  dim3 grid_dims(result.column_count / block_dims.x + 1, result.row_count / block_dims.y + 1);
  multiply_kernel_stupid <<<grid_dims, block_dims>>> (left_d, right_d, result_d);

  //step 5: copy results back to host
  error = cudaMemcpy(result.elements, result_d.elements, result_size, cudaMemcpyDeviceToHost);
  if(error != cudaSuccess){
  	printf("error copying result matrix\n");
  	printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  return result;
}

Matrix ones (int row_count, int column_count){
  Matrix result;
  result.row_count = row_count;
  result.column_count = column_count;
  result.elements = (int*) malloc(row_count * column_count * sizeof(int));
  for(int i = 0; i < row_count * column_count; i++){
    result.elements[i] = 1;
  }
  return result;
}

void print_matrix(Matrix mat){
  int num_elements = mat.row_count * mat.column_count;
  for(int i = 0; i < num_elements; i++){
    printf(" %d", mat.elements[i]);
    if(!((i + 1) % mat.column_count)){ printf("\n"); }
  }
}
