#include "kernel_template_sad_batch_cpu.h"
#include <stdexcept>

uint32_t uabsdiff_u8_cpu(uint8_t a, uint8_t b) {
  return (a > b) ? (uint32_t)(a - b) : (uint32_t)(b - a);
}

uint32_t template_sad_single_cpu(
    const uint8_t* query,
    const uint8_t* cand,
    int bpf)
{
  if (bpf < 0) throw std::runtime_error("template_sad_single_cpu: bpf < 0");
  if (bpf == 0) return 0;
  if (!query) throw std::runtime_error("template_sad_single_cpu: query is null");
  if (!cand)  throw std::runtime_error("template_sad_single_cpu: cand is null");

  uint32_t sum = 0;
  for (int j = 0; j < bpf; ++j) {
    sum += uabsdiff_u8_cpu(query[j], cand[j]);
  }
  return sum;
}

void template_sad_batch_cpu(
    const uint8_t* query,
    const uint8_t* batch,
    int bpf,
    int count,
    uint32_t* scores)
{
  if (bpf < 0) throw std::runtime_error("template_sad_batch_cpu: bpf < 0");
  if (count < 0) throw std::runtime_error("template_sad_batch_cpu: count < 0");

  if (count == 0) return;
  if (bpf == 0) {
    // SAD su zero byte => 0 per tutti
    if (!scores) throw std::runtime_error("template_sad_batch_cpu: scores is null");
    for (int i = 0; i < count; ++i) scores[i] = 0;
    return;
  }

  if (!query)  throw std::runtime_error("template_sad_batch_cpu: query is null");
  if (!batch)  throw std::runtime_error("template_sad_batch_cpu: batch is null");
  if (!scores) throw std::runtime_error("template_sad_batch_cpu: scores is null");

  for (int i = 0; i < count; ++i) {
    const uint8_t* cand = batch + (size_t)i * (size_t)bpf;
    uint32_t sum = 0;

    for (int j = 0; j < bpf; ++j) {
      sum += uabsdiff_u8_cpu(query[j], cand[j]);
    }

    scores[i] = sum;
  }
}
