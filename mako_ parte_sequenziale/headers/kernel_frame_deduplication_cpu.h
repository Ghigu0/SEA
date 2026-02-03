#pragma once
#include <cstdint>

uint32_t frame_deduplication_cpu(
    const uint8_t* frames,
    uint32_t N,
    int w, int h, int c,
    uint16_t threshold_mean_sad,
    int stride,
    uint8_t* keep);
