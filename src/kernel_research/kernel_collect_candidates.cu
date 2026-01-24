#include "kernel_collect_candidates.cuh"
#include "kernel_query_utils.cuh"   // per popc64

/* inclusa dagli headers
struct Cand {
  int32_t idx;   // indice nel newDB (0..N-1)
  uint8_t dist;  // Hamming distance
};*/

__global__ void kernel_collect_candidates(const uint64_t* __restrict__ d_hashes,
                                         int n,
                                         uint64_t q,
                                         uint8_t thresh,
                                         Cand* __restrict__ d_out,
                                         int topk,
                                         int* __restrict__ d_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    uint32_t d = popc64(d_hashes[i] ^ q);
    if (d <= (uint32_t)thresh) {
      int pos = atomicAdd(d_count, 1);
      if (pos < topk) {
        d_out[pos].idx  = (int32_t)i;
        d_out[pos].dist = (uint8_t)d;
      }
    }
  }
}