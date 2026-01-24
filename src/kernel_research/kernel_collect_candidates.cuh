#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "../../include/research_types.h"

// ============================================================================
// Kernel: collect candidates
//
// Scopo:
//   Scansiona tutti gli hash del DB e raccoglie (fino a topk) gli indici
//   dei frame la cui distanza di Hamming dall'hash query è <= thresh.
//
// Input:
//   - d_hashes : array di hash64 del DB (size = n)
//   - n        : numero di frame nel DB
//   - q        : hash64 della query
//   - thresh   : soglia di distanza di Hamming
//
// Output:
//   - d_out    : array di Cand (size = topk)
//   - d_count  : contatore atomico dei candidati trovati
//
// Note:
//   - d_count deve essere inizializzato a 0 dal chiamante
//   - se i candidati > topk, quelli in eccesso vengono ignorati
// ============================================================================

__global__ void kernel_collect_candidates(
    const uint64_t* __restrict__ d_hashes,
    int n,
    uint64_t q,
    uint8_t thresh,
    Cand* __restrict__ d_out,
    int topk,
    int* __restrict__ d_count
);
