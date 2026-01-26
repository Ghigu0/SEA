#include <cuda_runtime.h>
#include <cstdint>

#include "./headers/kernel_template_sad_batch.cuh"

__device__ __forceinline__ uint32_t uabsdiff_u8(uint8_t a, uint8_t b) {
  return (a > b) ? (uint32_t)(a - b) : (uint32_t)(b - a);
}

// 1 block per frame (grid.x = count)
// each block reduces SAD over bpf bytes
__global__ void kernel_template_sad_batch(
    const uint8_t* __restrict__ d_query,
    const uint8_t* __restrict__ d_batch,
    int bpf,
    int count,
    uint32_t* __restrict__ d_scores) {

  const int frame_i = (int)blockIdx.x;
  if (frame_i >= count) return;

  const uint8_t* cand = d_batch + (size_t)frame_i * (size_t)bpf;

  // partial sum per thread
  uint32_t local = 0;

  // stride over bytes
  for (int j = (int)threadIdx.x; j < bpf; j += (int)blockDim.x) {
    local += uabsdiff_u8(d_query[j], cand[j]);
  }

  // block reduction (uint32)
  __shared__ uint32_t sh[256]; // <-- assume blockDim.x <= 256 (come nel tuo main)
  sh[threadIdx.x] = local;
  __syncthreads();

  // reduce to sh[0]
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) sh[threadIdx.x] += sh[threadIdx.x + offset];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    d_scores[frame_i] = sh[0];
  }
}
