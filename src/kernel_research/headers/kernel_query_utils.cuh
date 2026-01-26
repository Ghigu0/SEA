#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Utility condivise tra kernel query
__device__ __forceinline__ uint32_t popc64(uint64_t x) {
  return (uint32_t)__popcll((unsigned long long)x);
}
