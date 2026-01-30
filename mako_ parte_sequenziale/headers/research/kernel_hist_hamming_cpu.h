#pragma once
#include <cstdint>

// popcount 64-bit (equivalente a popc64 del device)
uint32_t popc64_cpu(uint64_t x);

// Istogramma delle distanze di Hamming (0..64) tra q e hashes[0..n-1].
// Output: hist65[0..64] (uint32_t), viene azzerato e riempito dalla funzione.
void hist_hamming65_cpu(
    const uint64_t* hashes,
    int n,
    uint64_t q,
    uint32_t* hist65);
