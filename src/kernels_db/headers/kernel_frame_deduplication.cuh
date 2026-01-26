#pragma once
#include <cstdint>

// Kernel di deduplicazione temporale:
// - confronta frame i con i-1
// - downsample su griglia fissa (es. 32x18)
// - metrica SAD su luminanza
// - scrive d_keep[i] = 1 se il frame va tenuto, 0 se duplicato
//
// d_frames: [n * bytes_per_frame]  (RAW RGB o grayscale)
// d_keep:   [n] flag 0/1
// n:        numero di frame nel chunk
// w,h,c:    dimensioni frame
// bytes_per_frame: w*h*c
// threshold: soglia SAD

__global__ void dedup_kernel_downsample_sad(
    const uint8_t* d_frames,
    uint8_t* d_keep,
    int n,
    int w, int h, int c,
    int bytes_per_frame,
    int threshold
);
