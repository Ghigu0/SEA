// frame_hash_ahash.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

static inline void cuda_check(cudaError_t e, const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
    std::exit(1);
  }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

static constexpr int W = 1280;
static constexpr int H = 720;
static constexpr int CELLS_X = 8;
static constexpr int CELLS_Y = 8;
static constexpr int CELL_W = W / CELLS_X; // 160
static constexpr int CELL_H = H / CELLS_Y; // 90

__device__ __forceinline__ uint8_t rgb_to_luma_u8(uint8_t r, uint8_t g, uint8_t b) {
  // Rec.601 approx: (77R + 150G + 29B) >> 8
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

// Grid: dim3(frames, 64, 1). blockDim.x should be power-of-two (e.g. 256)
__global__ void k_downsample8x8_cellmean_u16(
    const uint8_t* __restrict__ frames_rgb,
    int frames,
    uint16_t* __restrict__ cell_mean_u16
) {
  int frame_id = (int)blockIdx.x;
  int cell_id  = (int)blockIdx.y; // 0..63
  if (frame_id >= frames || cell_id >= 64) return;

  int cx = cell_id & 7;
  int cy = cell_id >> 3;

  int x0 = cx * CELL_W;
  int y0 = cy * CELL_H;

  const uint8_t* frame = frames_rgb + (size_t)frame_id * (size_t)W * (size_t)H * 3u;

  uint32_t local_sum = 0;
  int total = CELL_W * CELL_H; // 14400

  for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x) {
    int dy = i / CELL_W;
    int dx = i - dy * CELL_W;

    int x = x0 + dx;
    int y = y0 + dy;

    size_t p = ((size_t)y * (size_t)W + (size_t)x) * 3u;
    uint8_t r = frame[p + 0];
    uint8_t g = frame[p + 1];
    uint8_t b = frame[p + 2];

    local_sum += (uint32_t)rgb_to_luma_u8(r, g, b);
  }

  // reduction
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
    cell_mean_u16[(size_t)frame_id * 64u + (size_t)cell_id] = mean;
  }
}

__global__ void k_ahash64_from_cellmean(
    const uint16_t* __restrict__ cell_mean_u16,
    int frames,
    uint64_t* __restrict__ out_hash64
) {
  int frame_id = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (frame_id >= frames) return;

  const uint16_t* cells = cell_mean_u16 + (size_t)frame_id * 64u;

  uint32_t sum = 0;
  #pragma unroll
  for (int i = 0; i < 64; i++) sum += (uint32_t)cells[i];
  uint16_t mean = (uint16_t)(sum / 64u);

  uint64_t h = 0ull;
  #pragma unroll
  for (int i = 0; i < 64; i++) {
    uint64_t bit = (cells[i] > mean) ? 1ull : 0ull;
    h |= (bit << (uint64_t)i);
  }

  out_hash64[frame_id] = h;
}

void compute_ahash64_batch(
    const uint8_t* frames_rgb_dev,
    int frames,
    uint64_t* out_hash64_dev
) {
  uint16_t* cell_mean_dev = nullptr;
  CUDA_CHECK(cudaMalloc(&cell_mean_dev, (size_t)frames * 64u * sizeof(uint16_t)));

  dim3 block1(256, 1, 1);
  dim3 grid1((unsigned)frames, 64u, 1u);
  k_downsample8x8_cellmean_u16<<<grid1, block1>>>(frames_rgb_dev, frames, cell_mean_dev);
  CUDA_CHECK(cudaGetLastError());

  dim3 block2(256, 1, 1);
  dim3 grid2((unsigned)((frames + (int)block2.x - 1) / (int)block2.x), 1u, 1u);
  k_ahash64_from_cellmean<<<grid2, block2>>>(cell_mean_dev, frames, out_hash64_dev);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(cell_mean_dev));
}
