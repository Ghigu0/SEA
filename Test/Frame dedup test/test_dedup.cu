// test_dedup_dump_ppm.cu
// Legge chunk via RawDbReader, lancia dedup kernel, salva i frame TENUTI come PPM ASCII (P3).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>


#include "../../include/config.h"
#include "../../include/cuda_utils.h"
#include "../../include/raw_db_reader.h"

// ---------------------------
// Kernel: downsample + SAD
// ---------------------------
__device__ __forceinline__ uint8_t rgb_to_luma(uint8_t r, uint8_t g, uint8_t b) {
  // approx BT.601 luma
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

__global__ void dedup_kernel_downsample_sad(
    const uint8_t* d_frames,
    uint8_t* d_keep,
    int n,
    int w, int h, int c,
    int bytes_per_frame,
    int threshold)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  if (i == 0) { d_keep[i] = 1; return; }

  const uint8_t* cur  = d_frames + (size_t)i * (size_t)bytes_per_frame;
  const uint8_t* prev = d_frames + (size_t)(i - 1) * (size_t)bytes_per_frame;

  // scegli la griglia qui
  const int GX = 32;
  const int GY = 18;

  unsigned int sad = 0;

  for (int yy = 0; yy < GY; ++yy) {
    int y = (yy * h) / GY;
    for (int xx = 0; xx < GX; ++xx) {
      int x = (xx * w) / GX;
      int idx = (y * w + x) * c;

      uint8_t a, b;
      if (c >= 3) {
        a = rgb_to_luma(cur[idx + 0],  cur[idx + 1],  cur[idx + 2]);
        b = rgb_to_luma(prev[idx + 0], prev[idx + 1], prev[idx + 2]);
      } else {
        a = cur[idx];
        b = prev[idx];
      }
      sad += (a > b) ? (a - b) : (b - a);
    }
  }

  d_keep[i] = (sad > (unsigned int)threshold) ? 1 : 0;
}

// ---------------------------
// Utility: salva frame come PPM ASCII (P3) NON binario
// ---------------------------
static void save_ppm_p3_ascii(const std::string& path,
                             const uint8_t* rgb, int w, int h, int c)
{
  // Salviamo sempre PPM RGB. Se c==1 duplico il canale.
  std::ofstream f(path, std::ios::out);
  if (!f) throw std::runtime_error("Impossibile aprire file: " + path);

  f << "P3\n" << w << " " << h << "\n255\n";

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int idx = (y * w + x) * c;
      int r, g, b;
      if (c >= 3) {
        r = rgb[idx + 0];
        g = rgb[idx + 1];
        b = rgb[idx + 2];
      } else {
        r = g = b = rgb[idx];
      }
      f << r << " " << g << " " << b << "  ";
    }
    f << "\n";
  }
}

int main(int argc, char** argv) {
  try {
    Config cfg;

    // ---- Config minimale: puoi anche leggere da file se già lo fate altrove ----
    // Qui uso valori di default dal tuo struct, MA setto i path necessari.
    if (argc < 2) {
      std::cerr << "Uso:\n  " << argv[0] << " <full_db_path> [out_dir]\n\n"
                << "Esempio:\n  " << argv[0] << " /path/db_raw ./dedup_out\n";
      return 1;
    }
    cfg.full_db_path = argv[1];
    std::string out_dir = (argc >= 3) ? argv[2] : std::string("./dedup_out");

    // Parametri utili per test:
    cfg.gpu_id = 0;
    cfg.enable_dedup = true;
    cfg.chunk_frames = 256;        // per debug: piccolo
    cfg.dedup_threshold = 8000;    // start per 32x18 (tuning dopo)
    // cfg.frame_w / frame_h / channels devono essere coerenti col RawDbReader
    // Se il RawDbReader ricava dimensioni da sorgente, assicurati che cfg combaci.

    CUDA_CHECK(cudaSetDevice(cfg.gpu_id));

    std::filesystem::create_directories(out_dir);

    // ---- Alloca buffer GPU/CPU ----
    const int w = cfg.frame_w;
    const int h = cfg.frame_h;
    const int c = cfg.channels;
    const int max_n = cfg.chunk_frames;

    const size_t bytes_per_frame = (size_t)w * (size_t)h * (size_t)c;

    uint8_t* d_frames = nullptr;
    uint8_t* d_keep = nullptr;

    std::vector<uint8_t> h_keep(max_n);

    CUDA_CHECK(cudaMalloc(&d_frames, max_n * bytes_per_frame));
    CUDA_CHECK(cudaMalloc(&d_keep,   max_n * sizeof(uint8_t)));

    // ---- Reader ----
    RawDbReader reader(cfg);

    int chunk_idx = 0;
    uint64_t total = 0, kept_total = 0;

    while (reader.has_next()) {
      HostChunk ch = reader.next_chunk(cfg.chunk_frames, bytes_per_frame);
      if (ch.n <= 0) break;

      total += (uint64_t)ch.n;

      // Upload frame bytes
      CUDA_CHECK(cudaMemcpy(d_frames, ch.frames.data(),
                            (size_t)ch.n * bytes_per_frame,
                            cudaMemcpyHostToDevice));

      // Launch dedup
      CUDA_CHECK(cudaMemset(d_keep, 0, (size_t)ch.n * sizeof(uint8_t)));

      dim3 block(256);
      dim3 grid((ch.n + block.x - 1) / block.x);

      dedup_kernel_downsample_sad<<<grid, block>>>(
          d_frames, d_keep, ch.n,
          w, h, c,
          (int)bytes_per_frame,
          cfg.dedup_threshold
      );
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // Download keep flags
      CUDA_CHECK(cudaMemcpy(h_keep.data(), d_keep,
                            (size_t)ch.n * sizeof(uint8_t),
                            cudaMemcpyDeviceToHost));

      // Dump solo i frame kept come PPM ASCII
      int kept_in_chunk = 0;
      for (int i = 0; i < ch.n; ++i) {
        if (!h_keep[i]) continue;
        kept_in_chunk++;

        const uint8_t* frame_i = ch.frames.data() + (size_t)i * bytes_per_frame;

        std::ostringstream name;
        // Nome file utile per capire da dove viene:
        // chunk0001_vid000123_f000045.ppm
        name << out_dir << "/chunk"
             << std::setw(4) << std::setfill('0') << chunk_idx
             << "_vid" << ch.video_id[i]
             << "_f"   << ch.frame_id[i]
             << ".ppm";

        save_ppm_p3_ascii(name.str(), frame_i, w, h, c);
      }

      kept_total += (uint64_t)kept_in_chunk;

      std::cerr << "[chunk " << chunk_idx << "] n=" << ch.n
                << " kept=" << kept_in_chunk
                << " threshold=" << cfg.dedup_threshold
                << " out_dir=" << out_dir << "\n";

      chunk_idx++;

      // Per debug puoi fermarti dopo pochi chunk:
      // if (chunk_idx >= 3) break;
    }

    std::cerr << "[DONE] total_frames=" << total
              << " kept_frames=" << kept_total
              << " ratio=" << (total ? (double)kept_total / (double)total : 0.0)
              << "\n";

    cudaFree(d_keep);
    cudaFree(d_frames);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERRORE: " << e.what() << "\n";
    return 1;
  }
}
