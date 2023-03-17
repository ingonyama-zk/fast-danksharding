#include "../../../icicle/icicle/curves/curve_config.cuh"
#include <cuda.h>

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

extern "C" int sum_of_points(projective_t *out, projective_t in[], size_t nof_rows, size_t nof_cols, size_t l, size_t device_id = 0)
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
    }
}
