#pragma once
#include <cstdint>

// Struttura identica (o compatibile) a research_types.h
struct Cand {
  int32_t idx;   // indice nel newDB (0..N-1)
  uint8_t dist;  // Hamming distance
};

// popcount 64-bit (equivalente a popc64 del device)
uint32_t popc64_cpu(uint64_t x);

// Versione sequenziale equivalente al kernel CUDA kernel_collect_candidates.
// Input:
//  - hashes: array di hash 64-bit del DB
//  - n: numero di hash
//  - q: hash query
//  - thresh: soglia (0..64)
//  - topk: massimo numero di candidati da scrivere in out[0..topk-1]
// Output:
//  - out: array Cand di capacità almeno topk
//  - out_count_total: conteggio totale di elementi che passano la soglia (può essere > topk)
// Return:
//  - numero di candidati effettivamente scritti in out (min(out_count_total, topk))
int collect_candidates_cpu(
    const uint64_t* hashes,
    int n,
    uint64_t q,
    uint8_t thresh,
    Cand* out,
    int topk,
    int* out_count_total);
