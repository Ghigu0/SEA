#pragma once
#include <cstdint>

__global__ void kernel_template_sad_batch(
    const uint8_t* __restrict__ d_query,  // bpf bytes
    const uint8_t* __restrict__ d_batch,  // count*bpf bytes (frame i at d_batch + i*bpf)
    int bpf,
    int count,
    uint32_t* __restrict__ d_scores       // count scores (SAD)
);
