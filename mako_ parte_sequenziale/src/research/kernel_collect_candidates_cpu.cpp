#include "kernel_collect_candidates_cpu.h"
#include <stdexcept>

uint32_t popc64_cpu(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return (uint32_t)__builtin_popcountll((unsigned long long)x);
#elif defined(_MSC_VER)
  #include <intrin.h>
  return (uint32_t)__popcnt64(x);
#else
  // fallback portable
  uint32_t c = 0;
  while (x) { x &= (x - 1); ++c; }
  return c;
#endif
}

int collect_candidates_cpu(
    const uint64_t* hashes,
    int n,
    uint64_t q,
    uint8_t thresh,
    Cand* out,
    int topk,
    int* out_count_total)
{
  if (n < 0) throw std::runtime_error("collect_candidates_cpu: n < 0");
  if (topk < 0) throw std::runtime_error("collect_candidates_cpu: topk < 0");
  if (n == 0 || topk == 0) {
    if (out_count_total) *out_count_total = 0;
    return 0;
  }

  if (!hashes) throw std::runtime_error("collect_candidates_cpu: hashes is null");
  if (!out)    throw std::runtime_error("collect_candidates_cpu: out is null");
  if (!out_count_total) throw std::runtime_error("collect_candidates_cpu: out_count_total is null");

  int count_total = 0;
  int written = 0;

  for (int i = 0; i < n; ++i) {
    const uint32_t d = popc64_cpu(hashes[i] ^ q);
    if (d <= (uint32_t)thresh) {
      const int pos = count_total; // equivalente a atomicAdd(d_count, 1) che ritorna il vecchio valore
      count_total++;

      if (pos < topk) {
        out[written].idx  = (int32_t)i;
        out[written].dist = (uint8_t)d;
        written++;
      }
    }
  }

  *out_count_total = count_total;
  return written;
}
