#pragma once
#include <cstdint>

__global__ void k_downsample8x8_cellmean_u16_kept(
    const uint8_t* d_frames,
    const int32_t* d_kept_ids,
    int kept,
    int w, int h, int c,
    int bytes_per_frame,
    uint16_t* d_cell_mean_u16
);

__global__ void k_ahash64_from_cellmean_kept(
    const uint16_t* d_cell_mean_u16,
    int kept,
    uint64_t* d_out_hashes
);
