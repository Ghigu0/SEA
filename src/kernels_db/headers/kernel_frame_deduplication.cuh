#pragma once
#include <cstdint>

__global__ void dedup_kernel_downsample_sad_blockperframe(
    const uint8_t* d_frames,
    uint8_t* d_keep,
    int n,
    int w, int h, int c,
    int bytes_per_frame,
    int threshold
);
