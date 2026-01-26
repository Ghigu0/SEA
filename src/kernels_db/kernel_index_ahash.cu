#include <cuda_runtime.h>
#include <cstdint>

#include "../../include/cuda_utils.h"
#include "./headers/kernel_index_ahash.cuh"

// =======================
// Utils device
// =======================
__device__ __forceinline__ uint8_t rgb_to_luma_u8(uint8_t r, uint8_t g, uint8_t b) {
  // Rec.601 approx: (77R + 150G + 29B) >> 8
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

// Grid: dim3(kept, 64, 1), blockDim.x = 256 (fisso per shared reduction)
__global__ void k_downsample8x8_cellmean_u16_kept(
    const uint8_t* __restrict__ d_frames,
    const int32_t* __restrict__ d_kept_ids,
    int kept,
    int w, int h, int c,
    int bytes_per_frame,
    uint16_t* __restrict__ d_cell_mean_u16
) {
  int j = (int)blockIdx.x;   // 0..kept-1 (indice compatto)
  int cell_id = (int)blockIdx.y; // 0..63
  if (j >= kept || cell_id >= 64) return;

  int i = (int)d_kept_ids[j]; // indice nel chunk originale
  const uint8_t* frame = d_frames + (size_t)i * (size_t)bytes_per_frame;

  int cx = cell_id & 7;
  int cy = cell_id >> 3;

  // Celle 8x8: gestiamo anche w/h non multipli di 8 usando l'ultima cella "piu larga"
  int cell_w = w / 8;
  int cell_h = h / 8;

  int x0 = cx * cell_w;
  int y0 = cy * cell_h;
  int x1 = (cx == 7) ? w : (x0 + cell_w);
  int y1 = (cy == 7) ? h : (y0 + cell_h);

  int rw = x1 - x0;
  int rh = y1 - y0;
  int total = rw * rh;
  if (total <= 0) {
    if (threadIdx.x == 0) d_cell_mean_u16[(size_t)j * 64u + (size_t)cell_id] = 0;
    return;
  }

  uint32_t local_sum = 0;

  for (int t = (int)threadIdx.x; t < total; t += (int)blockDim.x) {
    int dy = t / rw;
    int dx = t - dy * rw;

    int x = x0 + dx;
    int y = y0 + dy;

    size_t p = ((size_t)y * (size_t)w + (size_t)x) * (size_t)c;

    uint8_t lum;
    if (c == 1) {
      lum = frame[p];
    } else {
      // assume RGB interleaved
      uint8_t r = frame[p + 0];
      uint8_t g = frame[p + 1];
      uint8_t b = frame[p + 2];
      lum = rgb_to_luma_u8(r, g, b);
    }
    local_sum += (uint32_t)lum;
  }

  __shared__ uint32_t sh[256];
  sh[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    uint32_t sum = sh[0];
    uint16_t mean = (uint16_t)(sum / (uint32_t)total);
    d_cell_mean_u16[(size_t)j * 64u + (size_t)cell_id] = mean;
  }
}

__global__ void k_ahash64_from_cellmean_kept(
    const uint16_t* __restrict__ d_cell_mean_u16,
    int kept,
    uint64_t* __restrict__ d_out_hashes
) {
  int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (j >= kept) return;

  const uint16_t* cells = d_cell_mean_u16 + (size_t)j * 64u;

  uint32_t sum = 0;
  #pragma unroll
  for (int k = 0; k < 64; k++) sum += (uint32_t)cells[k];
  uint16_t mean = (uint16_t)(sum / 64u);

  uint64_t h = 0ull;
  #pragma unroll
  for (int k = 0; k < 64; k++) {
    uint64_t bit = (cells[k] > mean) ? 1ull : 0ull;
    h |= (bit << (uint64_t)k);
  }

  d_out_hashes[j] = h;
}
