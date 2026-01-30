// kernel_frame_deduplication_cpu.h/.cpp (CPU sequenziale)
// Versione equivalente "logica" del kernel CUDA di frame dedup temporale:
// - Confronta frame i con frame i-1
// - Calcola SAD su luminanza (Rec.601) con downsample (stride)
// - Se la differenza media supera una soglia => keep[i]=1, altrimenti keep[i]=0
//
// Nota: questa è pensata per rimpiazzare la semantica tipica del kernel:
// - keep[0] = 1
// - per i>0: keep[i] dipende da distanza tra frame consecutivi
//
// Input frame: RGB interleaved (rgb24), quindi c=3.
// frames layout: frames[i * (w*h*c) + ...]
//
// Puoi mettere questo file in src/kernels_db/cpu/ o dove preferisci.

#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

// -----------------------------
// Helpers
// -----------------------------
static inline uint8_t rgb_to_luma_u8(uint8_t r, uint8_t g, uint8_t b) {
  // Rec.601 approx: (77R + 150G + 29B) >> 8
  return (uint8_t)((77u * (uint32_t)r + 150u * (uint32_t)g + 29u * (uint32_t)b) >> 8);
}

static inline uint32_t u32_abs_diff(uint32_t a, uint32_t b) {
  return (a > b) ? (a - b) : (b - a);
}

// Calcola la "mean SAD" su luminanza campionando 1 pixel ogni `stride` in x e y.
// Restituisce un valore in [0..255] circa (media delle differenze assolute).
static inline uint16_t mean_sad_luma_downsampled(
    const uint8_t* __restrict__ frameA,
    const uint8_t* __restrict__ frameB,
    int w, int h, int c,
    int stride)
{
  if (!frameA || !frameB) throw std::runtime_error("mean_sad_luma_downsampled: null frame ptr");
  if (w <= 0 || h <= 0) throw std::runtime_error("mean_sad_luma_downsampled: invalid w/h");
  if (c != 3) throw std::runtime_error("mean_sad_luma_downsampled: expected c=3 (RGB24)");
  stride = std::max(1, stride);

  const int step_x = stride;
  const int step_y = stride;

  uint64_t sum = 0;
  uint32_t count = 0;

  // Campionamento regolare (downsample)
  for (int y = 0; y < h; y += step_y) {
    const size_t row_off = (size_t)y * (size_t)w * (size_t)c;
    for (int x = 0; x < w; x += step_x) {
      const size_t idx = row_off + (size_t)x * (size_t)c;

      const uint8_t ar = frameA[idx + 0];
      const uint8_t ag = frameA[idx + 1];
      const uint8_t ab = frameA[idx + 2];

      const uint8_t br = frameB[idx + 0];
      const uint8_t bg = frameB[idx + 1];
      const uint8_t bb = frameB[idx + 2];

      const uint32_t la = (uint32_t)rgb_to_luma_u8(ar, ag, ab);
      const uint32_t lb = (uint32_t)rgb_to_luma_u8(br, bg, bb);

      sum += (uint64_t)u32_abs_diff(la, lb);
      count++;
    }
  }

  if (count == 0) return 0;
  const uint32_t mean = (uint32_t)(sum / (uint64_t)count);
  return (uint16_t)std::min<uint32_t>(mean, 65535u);
}

// -----------------------------
// API principale (CPU)
// -----------------------------
//
// frames: puntatore al buffer contenente N frame consecutivi (RGB interleaved)
// N: numero frame nel chunk
// w,h,c: dimensioni frame (c deve essere 3)
// threshold_mean_sad: soglia sulla DIFFERENZA MEDIA in luminanza (0..255 tipicamente)
// stride: downsample (es. 4, 8, 16). Più alto => più veloce ma meno preciso.
// keep: output array di N elementi (0/1). keep[0]=1.
//
// Ritorna: numero di frame tenuti (somma keep).
static inline uint32_t frame_deduplication_cpu(
    const uint8_t* __restrict__ frames,
    uint32_t N,
    int w, int h, int c,
    uint16_t threshold_mean_sad,
    int stride,
    uint8_t* __restrict__ keep)
{
  if (!frames) throw std::runtime_error("frame_deduplication_cpu: frames is null");
  if (!keep)   throw std::runtime_error("frame_deduplication_cpu: keep is null");
  if (N == 0)  return 0;
  if (w <= 0 || h <= 0) throw std::runtime_error("frame_deduplication_cpu: invalid w/h");
  if (c != 3) throw std::runtime_error("frame_deduplication_cpu: expected c=3 (RGB24)");

  const size_t bytes_per_frame = (size_t)w * (size_t)h * (size_t)c;

  keep[0] = 1;
  uint32_t kept = 1;

  for (uint32_t i = 1; i < N; ++i) {
    const uint8_t* prev = frames + (size_t)(i - 1) * bytes_per_frame;
    const uint8_t* curr = frames + (size_t)i * bytes_per_frame;

    const uint16_t mean_sad = mean_sad_luma_downsampled(prev, curr, w, h, c, stride);

    // Regola: se la differenza media supera la soglia => è "nuovo" => keep
    const uint8_t k = (mean_sad > threshold_mean_sad) ? 1u : 0u;
    keep[i] = k;
    kept += (uint32_t)k;
  }

  return kept;
}
