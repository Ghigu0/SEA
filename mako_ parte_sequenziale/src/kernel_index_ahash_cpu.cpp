#include "kernel_index_ahash_cpu.h"
#include <stdexcept>
#include <vector>
#include <algorithm>

uint8_t rgb_to_luma_u8_cpu(uint8_t r, uint8_t g, uint8_t b) {
  // identico al kernel: (77R + 150G + 29B) >> 8
  return (uint8_t)((77u * (uint32_t)r + 150u * (uint32_t)g + 29u * (uint32_t)b) >> 8);
}

static inline void validate_rgb24_dims(int w, int h, int c, int bytes_per_frame) {
  if (w <= 0 || h <= 0) throw std::runtime_error("aHash CPU: invalid w/h");
  if (c != 3) throw std::runtime_error("aHash CPU: expected c=3 (RGB24)");
  const int expected = w * h * c;
  if (bytes_per_frame != expected) {
    // nel tuo progetto bytes_per_frame è spesso proprio w*h*c
    throw std::runtime_error("aHash CPU: bytes_per_frame mismatch (expected w*h*c)");
  }
}

void downsample8x8_cellmean_u16_kept_cpu(
    const uint8_t* frames,
    const int32_t* kept_ids,
    int kept,
    int w, int h, int c,
    int bytes_per_frame,
    uint16_t* out_cell_mean_u16)
{
  if (kept < 0) throw std::runtime_error("downsample8x8_cellmean: kept < 0");
  if (kept == 0) return;

  if (!frames) throw std::runtime_error("downsample8x8_cellmean: frames is null");
  if (!kept_ids) throw std::runtime_error("downsample8x8_cellmean: kept_ids is null");
  if (!out_cell_mean_u16) throw std::runtime_error("downsample8x8_cellmean: out_cell_mean_u16 is null");

  validate_rgb24_dims(w, h, c, bytes_per_frame);

  // come nel kernel: cell_w = w/8, cell_h = h/8 (divisione intera)
  const int cell_w = w / 8;
  const int cell_h = h / 8;
  if (cell_w <= 0 || cell_h <= 0) {
    throw std::runtime_error("downsample8x8_cellmean: frame too small for 8x8 grid");
  }

  for (int j = 0; j < kept; ++j) {
    const int32_t i = kept_ids[j];
    if (i < 0) throw std::runtime_error("downsample8x8_cellmean: negative kept_id");

    const uint8_t* frame = frames + (size_t)i * (size_t)bytes_per_frame;

    for (int cell_id = 0; cell_id < 64; ++cell_id) {
      const int cx = cell_id % 8;
      const int cy = cell_id / 8;

      const int x0 = cx * cell_w;
      const int y0 = cy * cell_h;
      const int x1 = x0 + cell_w;
      const int y1 = y0 + cell_h;

      const int rw = x1 - x0;
      const int rh = y1 - y0;
      const int total = rw * rh;

      // Nel kernel: somma su luminanza di tutti i pixel della cella
      uint64_t sum = 0;

      for (int yy = y0; yy < y1; ++yy) {
        for (int xx = x0; xx < x1; ++xx) {
          const size_t p = ((size_t)yy * (size_t)w + (size_t)xx) * (size_t)c;

          const uint8_t r = frame[p + 0];
          const uint8_t g = frame[p + 1];
          const uint8_t b = frame[p + 2];

          const uint8_t lum = rgb_to_luma_u8_cpu(r, g, b);
          sum += (uint64_t)lum;
        }
      }

      const uint16_t mean = (total > 0) ? (uint16_t)(sum / (uint64_t)total) : 0;
      out_cell_mean_u16[(size_t)j * 64u + (size_t)cell_id] = mean;
    }
  }
}

void ahash64_from_cellmean_kept_cpu(
    const uint16_t* cell_mean_u16,
    int kept,
    uint64_t* out_hashes)
{
  if (kept < 0) throw std::runtime_error("ahash64_from_cellmean: kept < 0");
  if (kept == 0) return;

  if (!cell_mean_u16) throw std::runtime_error("ahash64_from_cellmean: cell_mean_u16 is null");
  if (!out_hashes) throw std::runtime_error("ahash64_from_cellmean: out_hashes is null");

  for (int j = 0; j < kept; ++j) {
    const uint16_t* cells = cell_mean_u16 + (size_t)j * 64u;

    uint32_t sum = 0;
    for (int k = 0; k < 64; ++k) sum += (uint32_t)cells[k];

    const uint16_t mean = (uint16_t)(sum / 64u);

    uint64_t h = 0ull;
    for (int k = 0; k < 64; ++k) {
      const uint64_t bit = (cells[k] > mean) ? 1ull : 0ull;
      h |= (bit << (uint64_t)k);
    }

    out_hashes[j] = h;
  }
}

void compute_ahash64_kept_cpu(
    const uint8_t* frames,
    const int32_t* kept_ids,
    int kept,
    int w, int h, int c,
    int bytes_per_frame,
    uint64_t* out_hashes,
    uint16_t* optional_cell_mean_u16)
{
  if (kept < 0) throw std::runtime_error("compute_ahash64_kept_cpu: kept < 0");
  if (kept == 0) return;

  if (!out_hashes) throw std::runtime_error("compute_ahash64_kept_cpu: out_hashes is null");

  if (optional_cell_mean_u16) {
    downsample8x8_cellmean_u16_kept_cpu(
        frames, kept_ids, kept, w, h, c, bytes_per_frame, optional_cell_mean_u16);
    ahash64_from_cellmean_kept_cpu(optional_cell_mean_u16, kept, out_hashes);
    return;
  }

  // buffer temporaneo interno (64 * kept)
  std::vector<uint16_t> tmp((size_t)kept * 64u);
  downsample8x8_cellmean_u16_kept_cpu(
      frames, kept_ids, kept, w, h, c, bytes_per_frame, tmp.data());
  ahash64_from_cellmean_kept_cpu(tmp.data(), kept, out_hashes);
}
