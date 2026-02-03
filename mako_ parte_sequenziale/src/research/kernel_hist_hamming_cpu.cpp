#include "kernel_hist_hamming_cpu.h"
#include <stdexcept>

uint32_t popc64_cpu(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return (uint32_t)__builtin_popcountll((unsigned long long)x);
#elif defined(_MSC_VER)
  #include <intrin.h>
  return (uint32_t)__popcnt64(x);
#else
  uint32_t c = 0;
  while (x) { x &= (x - 1); ++c; }
  return c;
#endif
}

void hist_hamming65_cpu(
    const uint64_t* hashes,
    int n,
    uint64_t q,
    uint32_t* hist65)
{
  if (n < 0) throw std::runtime_error("hist_hamming65_cpu: n < 0");
  if (!hist65) throw std::runtime_error("hist_hamming65_cpu: hist65 is null");

  // azzera hist[0..64] (come lo shared init del kernel)
  for (int i = 0; i < 65; ++i) hist65[i] = 0;

  if (n == 0) return;
  if (!hashes) throw std::runtime_error("hist_hamming65_cpu: hashes is null");

  for (int i = 0; i < n; ++i) {
    const uint32_t d = popc64_cpu(hashes[i] ^ q); // 0..64
    // d è garantito 0..64, ma per sicurezza:
    if (d <= 64u) {
      hist65[d] += 1u;
    } else {
      // impossibile, ma se succede è un bug (popcount > 64)
      throw std::runtime_error("hist_hamming65_cpu: popcount out of range");
    }
  }
}
