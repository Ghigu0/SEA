#include "./headers/kernel_hist_hamming.cuh"
#include "./headers/kernel_query_utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void kernel_hist_hamming(
    const uint64_t* __restrict__ d_hashes,
    int n,
    uint64_t q,
    uint32_t* __restrict__ d_hist65)
{
  __shared__ uint32_t sh[65];

  for (int i = threadIdx.x; i < 65; i += blockDim.x) sh[i] = 0;
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    uint32_t d = popc64(d_hashes[i] ^ q); // 0..64
    atomicAdd(&sh[d], 1u);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < 65; i += blockDim.x) {
    if (sh[i]) atomicAdd(&d_hist65[i], sh[i]);
  }
}
