#include "../../../icicle/icicle/curves/curve_config.cuh"
#include <cuda.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int MAX_THREAD_NUM = 256;

template <typename P>
void point_sum(P *h_outputs, P *h_inputs, unsigned nof_rows, unsigned nof_cols, unsigned l);

template<typename P>
__global__ void sum_kernel(P *d0, P *d1, unsigned nof_rows, unsigned nof_cols, unsigned l) {
  //Compute each thread's global row and column index
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned step = nof_cols / l;
  //Iterate over row, and down column
  d1[row * step + col] = P::zero();
  for (int k = 0; k < l; k++) {
    //Accumulate results for a single element
    d1[row * step + col] = d1[row * step + col] + d0[row * nof_cols + col + k * step];
  }
}

template <typename P>
void point_sum(P* h_outputs, P* h_inputs, unsigned nof_rows, unsigned nof_cols, unsigned l){
  unsigned step = nof_cols/l;
  P* d_outputs;
  P* d_inputs;
  cudaMalloc(&d_outputs, sizeof(P) * nof_cols * step);
  cudaMalloc(&d_inputs, sizeof(P) * nof_cols * nof_rows);
  //copy inputs to device
  cudaMemcpy(d_inputs, h_inputs, sizeof(P) * nof_cols * nof_rows, cudaMemcpyHostToDevice);

  unsigned NUM_THREADS = 1;
  unsigned THREADS_PER_ROW = (nof_rows + NUM_THREADS - 1) / NUM_THREADS;
  dim3 NUM_BLOCKS(THREADS_PER_ROW, step);

  sum_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_inputs, d_outputs, nof_rows, nof_cols, l);

  cudaDeviceSynchronize();

  //check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  cudaMemcpy(h_outputs, d_outputs, sizeof(P) * nof_rows * step, cudaMemcpyDeviceToHost);

  //free memory
  cudaFree(d_inputs);
  cudaFree(d_outputs);
}

extern "C" int sum_of_points(projective_t *out, projective_t in[], size_t nof_rows, size_t nof_cols, size_t l)
{
    try
    {
        point_sum<projective_t>(out, in, nof_rows, nof_cols, l);

        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what()); // TODO: error code and message
        // out->z = 0; //TODO: .set_infinity()
        return -1;
    }
}

template <typename T>
__global__ void shift_kernel(T *arr, unsigned nof_rows, unsigned nof_cols_div_2)
{
    // printf("block id: %d; thread id: %d \n", blockIdx.x, threadIdx.x);
    unsigned id = blockIdx.x * MAX_THREAD_NUM + threadIdx.x;

    if (id < nof_rows * nof_cols_div_2) {
        unsigned col_id = id % nof_cols_div_2;
        unsigned row_id = id / nof_cols_div_2;
        arr[row_id * 2 * nof_cols_div_2 + col_id] = arr[(2 * row_id + 1) * nof_cols_div_2 + col_id];
        arr[(2 * row_id + 1) * nof_cols_div_2 + col_id] = T::zero();
    }
}

extern "C" int shift_batch(projective_t *arr, size_t nof_rows, size_t nof_cols_div_2)
{
    try
    {
        int thread_num = MAX_THREAD_NUM;
        int block_num = (nof_rows * nof_cols_div_2) / thread_num;
        shift_kernel <projective_t> <<<block_num, thread_num>>> (arr, nof_rows, nof_cols_div_2);

        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what()); // TODO: error code and message
        return -1;
    }
}

// the shared-memory version of matrix transpose taken from here: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template <typename T>
__global__ void transpose_kernel(T *odata, const T *idata)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];
      
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*height + x] = tile[threadIdx.x][threadIdx.y+j];
}

extern "C" int transpose_matrix(scalar_field_t *out, scalar_field_t *in, size_t nof_rows, size_t nof_cols)
{
    try
    {
        dim3 dimGrid(nof_rows / TILE_DIM, nof_cols / TILE_DIM, 1);
        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
        transpose_kernel <scalar_t> <<<dimGrid, dimBlock>>> (out, in);

        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what()); // TODO: error code and message
        return -1;
    }
}
