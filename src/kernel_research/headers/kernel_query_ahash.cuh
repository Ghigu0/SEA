#pragma once
#include <cstdint>

__global__ void k_query_cellmean_u16_8x8(
    const uint8_t* d_frame,
    int w, int h, int c,
    int bytes_per_frame,
    uint16_t* d_cell_mean_u16_64);

__global__ void k_query_ahash64_from_cellmean(
    const uint16_t* d_cell_mean_u16_64,
    uint64_t* d_out_hash);
