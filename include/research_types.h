#pragma once
#include <cstdint>

struct Cand {
  int32_t idx;   // indice nel newDB (0..N-1)
  uint8_t dist;  // Hamming distance
};