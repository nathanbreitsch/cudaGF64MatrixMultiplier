//question:
//does contention for accessing global memory affect read performance?
//for instance, n threads access each of the n rows in a column at the same time.
//would it be faster for thread i to access row i % n first and proceed to i + j % n?

#define TILE_WIDTH 32
#define DIVIDE_ROUND_UP(a, b)((a + b - 1) / b)
#define GET_INDEX(row, column, numcols)(row * numcols + column)

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

//global memory
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

//better kernel, shared memory
__global__ void multiply_kernel_smart(const Matrix left, const Matrix right, Matrix result){
  // allocate shared memory
  __shared__ int left_shared[TILE_WIDTH*TILE_WIDTH];
  __shared__ int right_shared[TILE_WIDTH*TILE_WIDTH];

  //get row/col indices
  int result_row_index = threadIdx.y + blockDim.y * blockIdx.y;
  int result_col_index = threadIdx.x + blockDim.x * blockIdx.x;
  //int grid_row = blockIdx.y;
  //int grid_col = blockIdx.x;
  int block_row = threadIdx.y;
  int block_col = threadIdx.x;

  //how many blocks do we need to multiply?
  int num_block_mult = DIVIDE_ROUND_UP(left.column_count, blockDim.x);

  //loop through the tiles that we need to multiply
  for(int i = 0; i < num_block_mult; i++){
    //copy relevant blocks to shared memory, watch out for overflow
    int left_col_index = i * blockDim.x + block_col;
    int& left_row_index = result_row_index; //different name for readability
    int& right_col_index = result_col_index; //different name for readability
    int right_row_index = i * blockDim.x + block_row;
    int shared_array_index = block_col + block_row * blockDim.x;
    int left_index = GET_INDEX(left_row_index, left_col_index, left.column_count);
    int right_index = GET_INDEX(right_row_index, right_col_index, right.column_count);

    if(left_col_index < left.column_count && left_row_index < left.row_count){
      left_shared[shared_array_index] = left.elements[left_index];
    }
    else{
      left_shared[shared_array_index] = 0;
    }

    if(right_col_index < right.column_count && right_row_index < right.row_count){
      right_shared[shared_array_index] = right.elements[right_index];
    }
    else{
      right_shared[shared_array_index] = 0;
    }

    //make sure threads all finish copying to shared memory before doing multiplications
    __syncthreads();

    //do multiplications
    int dot_product = 0;
    for(int k = 0; k < blockDim.x; k++){
      dot_product += left_shared[GET_INDEX(blockIdx.y, k, blockDim.x)] * right_shared[GET_INDEX(k, blockIdx.x, blockDim.x)];
    }

    //make sure threads all finish multiplying before writing resulting block to memory
    __syncthreads();

    //write to global memory
    result.elements[GET_INDEX(result_row_index, result_col_index, result.column_count)] = dot_product;
  }

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



  dim3 block_dims(TILE_WIDTH, TILE_WIDTH);
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
