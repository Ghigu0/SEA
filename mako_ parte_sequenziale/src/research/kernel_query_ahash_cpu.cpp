#include "kernel_query_ahash_cpu.h"
#include <stdexcept>

uint8_t rgb_to_luma_u8_cpu(uint8_t r, uint8_t g, uint8_t b) {
  return (uint8_t)((77u * (uint32_t)r + 150u * (uint32_t)g + 29u * (uint32_t)b) >> 8);
}

static inline void validate_dims(int w, int h, int c, int bytes_per_frame) {
  if (w <= 0 || h <= 0) throw std::runtime_error("query_ahash CPU: invalid w/h");
  if (c != 1 && c != 3) throw std::runtime_error("query_ahash CPU: expected c=1 or c=3");
  const int expected = w * h * c;
  if (bytes_per_frame != expected) {
    throw std::runtime_error("query_ahash CPU: bytes_per_frame mismatch (expected w*h*c)");
  }
}

void query_cellmean_u16_8x8_cpu(
    const uint8_t* frame,
    int w, int h, int c,
    int bytes_per_frame,
    uint16_t* out_cell_mean_u16_64)
{
  if (!frame) throw std::runtime_error("query_cellmean_u16_8x8_cpu: frame is null");
  if (!out_cell_mean_u16_64) throw std::runtime_error("query_cellmean_u16_8x8_cpu: out is null");
  validate_dims(w, h, c, bytes_per_frame);

  const int cell_w = w / 8;
  const int cell_h = h / 8;
  if (cell_w <= 0 || cell_h <= 0) {
    throw std::runtime_error("query_cellmean_u16_8x8_cpu: frame too small for 8x8 grid");
  }

  for (int cell_id = 0; cell_id < 64; ++cell_id) {
    const int cx = (cell_id & 7);
    const int cy = (cell_id >> 3);

    const int x0 = cx * cell_w;
    const int y0 = cy * cell_h;

    // identico al kernel query: ultima colonna/ultima riga prendono il resto (fino a w/h)
    const int x1 = (cx == 7) ? w : (x0 + cell_w);
    const int y1 = (cy == 7) ? h : (y0 + cell_h);

    const int rw = x1 - x0;
    const int rh = y1 - y0;
    const int total = rw * rh;

    uint64_t sum = 0;

    for (int yy = y0; yy < y1; ++yy) {
      for (int xx = x0; xx < x1; ++xx) {
        const size_t p = ((size_t)yy * (size_t)w + (size_t)xx) * (size_t)c;

        uint8_t lum;
        if (c == 1) {
          lum = frame[p];
        } else {
          const uint8_t r = frame[p + 0];
          const uint8_t g = frame[p + 1];
          const uint8_t b = frame[p + 2];
          lum = rgb_to_luma_u8_cpu(r, g, b);
        }

        sum += (uint64_t)lum;
      }
    }

    out_cell_mean_u16_64[cell_id] = (total > 0) ? (uint16_t)(sum / (uint64_t)total) : 0;
  }
}

uint64_t query_ahash64_from_cellmean_cpu(const uint16_t* cell_mean_u16_64) {
  if (!cell_mean_u16_64) throw std::runtime_error("query_ahash64_from_cellmean_cpu: null input");

  uint32_t sum = 0;
  for (int k = 0; k < 64; ++k) sum += (uint32_t)cell_mean_u16_64[k];
  const uint16_t mean = (uint16_t)(sum / 64u);

  uint64_t h = 0ull;
  for (int k = 0; k < 64; ++k) {
    h |= ((uint64_t)(cell_mean_u16_64[k] > mean) << (uint64_t)k);
  }
  return h;
}

uint64_t compute_query_ahash64_cpu(
    const uint8_t* frame,
    int w, int h, int c,
    int bytes_per_frame)
{
  uint16_t cells[64];
  query_cellmean_u16_8x8_cpu(frame, w, h, c, bytes_per_frame, cells);
  return query_ahash64_from_cellmean_cpu(cells);
}
