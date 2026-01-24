#include "kernel_query_ahash.cuh"
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint8_t rgb_to_luma_u8(uint8_t r, uint8_t g, uint8_t b) {
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

// grid.x = 64 (una cella), block.x = 256
__global__ void k_query_cellmean_u16_8x8(
    const uint8_t* d_frame,
    int w, int h, int c,
    int bytes_per_frame,
    uint16_t* d_cell_mean_u16_64)
{
  int cell_id = (int)blockIdx.x; // 0..63
  if (cell_id >= 64) return;

  int cx = cell_id & 7;
  int cy = cell_id >> 3;

  int cell_w = w / 8;
  int cell_h = h / 8;

  int x0 = cx * cell_w;
  int y0 = cy * cell_h;
  int x1 = (cx == 7) ? w : (x0 + cell_w);
  int y1 = (cy == 7) ? h : (y0 + cell_h);

  int rw = x1 - x0;
  int rh = y1 - y0;
  int total = rw * rh;

  uint32_t local_sum = 0;
  for (int t = (int)threadIdx.x; t < total; t += (int)blockDim.x) {
    int dy = t / rw;
    int dx = t - dy * rw;
    int x = x0 + dx;
    int y = y0 + dy;
    size_t p = ((size_t)y * (size_t)w + (size_t)x) * (size_t)c;

    uint8_t lum;
    if (c == 1) lum = d_frame[p];
    else lum = rgb_to_luma_u8(d_frame[p+0], d_frame[p+1], d_frame[p+2]);

    local_sum += (uint32_t)lum;
  }

  __shared__ uint32_t sh[256];
  sh[threadIdx.x] = local_sum;
  __syncthreads();

  for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    d_cell_mean_u16_64[cell_id] = (total > 0) ? (uint16_t)(sh[0] / (uint32_t)total) : 0;
  }
}

// 1 blocco, 256 thread (o anche 64 thread), ma qui semplice: 1 thread
__global__ void k_query_ahash64_from_cellmean(
    const uint16_t* d_cell_mean_u16_64,
    uint64_t* d_out_hash)
{
  // 1 thread basta
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  uint32_t sum = 0;
  #pragma unroll
  for (int k = 0; k < 64; ++k) sum += (uint32_t)d_cell_mean_u16_64[k];
  uint16_t mean = (uint16_t)(sum / 64u);

  uint64_t h = 0ull;
  #pragma unroll
  for (int k = 0; k < 64; ++k) {
    h |= ((uint64_t)(d_cell_mean_u16_64[k] > mean) << (uint64_t)k);
  }
  *d_out_hash = h;
}
