#pragma once
#include <cstdint>

__global__ void kernel_hist_hamming(
    const uint64_t* __restrict__ d_hashes,
    int n,
    uint64_t q,
    uint32_t* __restrict__ d_hist65);
