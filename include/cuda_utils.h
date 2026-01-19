#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t err__ = (x);                                                 \
    if (err__ != cudaSuccess) {                                              \
      throw std::runtime_error(                                              \
        std::string("CUDA error: ") +                                        \
        cudaGetErrorString(err__) +                                          \
        " (" + std::to_string(static_cast<int>(err__)) + ")" +               \
        " @ " + __FILE__ + ":" + std::to_string(__LINE__)                    \
      );                                                                     \
    }                                                                        \
  } while (0)
