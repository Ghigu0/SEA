// database_loader.cpp (CPU-only)
// Build DB: load -> dedup -> compaction -> index -> pack -> write (SoA)
//
// CPU-only per confronti seri CPU vs GPU.
//
// DEDUP: IDENTICA al kernel CUDA dedup_kernel_downsample_sad:
//   - griglia 32x18
//   - luma BT.601 approx: (77R + 150G + 29B) >> 8
//   - SAD su 576 campioni tra frame i e frame i-1
//   - keep = 1 se sad > threshold, else 0
//   - frame 0 del chunk sempre keep
//   - + reset al cambio video_id (per dataset multi-video quando un chunk attraversa video)
//
// INDEX (aHash64): allineata ai kernel query aHash che hai incollato:
//   - luma BT.601
//   - celle 8x8 con cell_w=w/8, cell_h=h/8, ultima cella prende il resto
//   - bit = (cell_mean > global_mean)  (STRICT >)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/config.h"
#include "../include/db_types.h"
#include "../include/database_loader.h"
#include "../include/I_O/raw_db_reader.h"
#include "../include/I_O/new_db_writer.h"

// --------------------------------------------------------------------------------------
// Workspace CPU locale (solo host)
// --------------------------------------------------------------------------------------
struct CpuWorkspace {
  int max_frames = 0;
  int w = 0, h = 0, c = 0;
  size_t bytes_per_frame = 0;

  std::vector<uint8_t>  keep;      // size n: 1=kept, 0=dropped
  std::vector<int32_t>  kept_ids;  // indici nel chunk originale (compattati)
  std::vector<uint64_t> hashes;    // aHash per kept (stesso ordine)
};

static void workspace_init(CpuWorkspace& ws, const Config& cfg) {
  ws.max_frames = cfg.chunk_frames;
  ws.w = cfg.frame_w;
  ws.h = cfg.frame_h;
  ws.c = cfg.channels;
  ws.bytes_per_frame = static_cast<size_t>(ws.w) * (size_t)ws.h * (size_t)ws.c;

  ws.keep.assign((size_t)ws.max_frames, 0);
  ws.kept_ids.resize((size_t)ws.max_frames);
  ws.hashes.resize((size_t)ws.max_frames);
}

static void workspace_reset_for_n(CpuWorkspace& ws, int n) {
  if (n < 0) n = 0;
  if ((size_t)n > ws.keep.size()) ws.keep.resize((size_t)n);
  std::fill(ws.keep.begin(), ws.keep.begin() + n, 0);
}

// --------------------------------------------------------------------------------------
// Luma BT.601 approx (identica ai kernel GPU: rgb_to_luma_u8 / rgb_to_luma)
// --------------------------------------------------------------------------------------
static inline uint8_t rgb_to_luma_cpu(uint8_t r, uint8_t g, uint8_t b) {
  return (uint8_t)((77u * (unsigned)r + 150u * (unsigned)g + 29u * (unsigned)b) >> 8);
}

// --------------------------------------------------------------------------------------
// Dedup CPU = dedup_kernel_downsample_sad (GPU)
// --------------------------------------------------------------------------------------
static int run_dedup_cpu_like_gpu(CpuWorkspace& ws, const Config& cfg, const HostChunk& ch) {
  const int n = ch.n;
  workspace_reset_for_n(ws, n);
  if (n <= 0) return 0;

  const int w = ws.w;
  const int h = ws.h;
  const int c = ws.c;
  const int bpf = (int)ws.bytes_per_frame;

  const int GX = 32;
  const int GY = 18;

  ws.keep[0] = 1; // primo frame chunk

  for (int i = 1; i < n; ++i) {
    // Reset al cambio video: evita confronto tra video diversi quando un chunk attraversa cartelle
    if (ch.video_id[i] != ch.video_id[i - 1]) {
      ws.keep[i] = 1;
      continue;
    }

    const uint8_t* cur  = ch.frames.data() + (size_t)i * (size_t)bpf;
    const uint8_t* prev = ch.frames.data() + (size_t)(i - 1) * (size_t)bpf;

    unsigned int sad = 0;

    for (int yy = 0; yy < GY; ++yy) {
      const int y = (yy * h) / GY;
      for (int xx = 0; xx < GX; ++xx) {
        const int x = (xx * w) / GX;
        const int idx = (y * w + x) * c;

        uint8_t a, b;
        if (c >= 3) {
          a = rgb_to_luma_cpu(cur[idx + 0],  cur[idx + 1],  cur[idx + 2]);
          b = rgb_to_luma_cpu(prev[idx + 0], prev[idx + 1], prev[idx + 2]);
        } else {
          a = cur[idx];
          b = prev[idx];
        }

        sad += (a > b) ? (unsigned)(a - b) : (unsigned)(b - a);
      }
    }

    ws.keep[i] = (sad > (unsigned)cfg.dedup_threshold) ? 1 : 0;
  }

  int kept = 0;
  for (int i = 0; i < n; ++i) kept += (ws.keep[i] ? 1 : 0);
  return kept;
}

// --------------------------------------------------------------------------------------
// Compaction CPU
// --------------------------------------------------------------------------------------
static int run_compaction_cpu(CpuWorkspace& ws, int n) {
  int kept = 0;
  if ((size_t)n > ws.kept_ids.size()) ws.kept_ids.resize((size_t)n);

  for (int i = 0; i < n; ++i) {
    if (ws.keep[i]) {
      ws.kept_ids[kept] = (int32_t)i;
      ++kept;
    }
  }
  return kept;
}

// --------------------------------------------------------------------------------------
// aHash64 CPU (ALLINEATO ai kernel query aHash incollati):
// - celle 8x8 con cell_w=w/8, cell_h=h/8, ultima cella prende il resto
// - luma BT.601
// - bit = (cell_mean > mean) (STRICT >)
// --------------------------------------------------------------------------------------
static uint64_t ahash64_one_like_gpu_query(const uint8_t* frame, int w, int h, int c) {
  uint16_t cell_mean[64];

  const int cell_w = w / 8;
  const int cell_h = h / 8;

  for (int cell_id = 0; cell_id < 64; ++cell_id) {
    const int cx = cell_id & 7;
    const int cy = cell_id >> 3;

    const int x0 = cx * cell_w;
    const int y0 = cy * cell_h;
    const int x1 = (cx == 7) ? w : (x0 + cell_w);
    const int y1 = (cy == 7) ? h : (y0 + cell_h);

    const int rw = x1 - x0;
    const int rh = y1 - y0;
    const int total = rw * rh;

    uint32_t sum = 0;
    if (total > 0) {
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const size_t p = ((size_t)y * (size_t)w + (size_t)x) * (size_t)c;
          uint8_t lum;
          if (c == 1) lum = frame[p];
          else        lum = rgb_to_luma_cpu(frame[p + 0], frame[p + 1], frame[p + 2]);
          sum += (uint32_t)lum;
        }
      }
      cell_mean[cell_id] = (uint16_t)(sum / (uint32_t)total);
    } else {
      cell_mean[cell_id] = 0;
    }
  }

  uint32_t sum_all = 0;
  for (int k = 0; k < 64; ++k) sum_all += (uint32_t)cell_mean[k];
  const uint16_t mean = (uint16_t)(sum_all / 64u);

  uint64_t h64 = 0ull;
  for (int k = 0; k < 64; ++k) {
    // STRICT > come nel kernel GPU
    h64 |= ((uint64_t)(cell_mean[k] > mean) << (uint64_t)k);
  }
  return h64;
}

static void run_index_cpu(CpuWorkspace& ws, const HostChunk& ch, int kept) {
  const size_t bpf = ws.bytes_per_frame;
  if ((size_t)kept > ws.hashes.size()) ws.hashes.resize((size_t)kept);

  for (int j = 0; j < kept; ++j) {
    const int32_t i = ws.kept_ids[j];
    const uint8_t* frame = ch.frames.data() + (size_t)i * bpf;
    ws.hashes[j] = ahash64_one_like_gpu_query(frame, ws.w, ws.h, ws.c);
  }
}

// --------------------------------------------------------------------------------------
// Packing SoA
// --------------------------------------------------------------------------------------
static DbSoAChunk build_soa_chunk_from_results(const HostChunk& in, const CpuWorkspace& ws, int kept) {
  DbSoAChunk out;
  out.hashes.resize(kept);
  out.video_id.resize(kept);
  out.frame_id.resize(kept);

  out.bytes_per_frame = (int)ws.bytes_per_frame;
  out.frames_raw.resize((size_t)kept * ws.bytes_per_frame);

  for (int j = 0; j < kept; ++j) {
    const int32_t i = ws.kept_ids[j];

    out.hashes[j]   = ws.hashes[j];
    out.video_id[j] = in.video_id[i];
    out.frame_id[j] = in.frame_id[i];

    const uint8_t* src = in.frames.data() + (size_t)i * ws.bytes_per_frame;
    uint8_t*       dst = out.frames_raw.data() + (size_t)j * ws.bytes_per_frame;
    std::memcpy(dst, src, ws.bytes_per_frame);
  }

  return out;
}

// --------------------------------------------------------------------------------------
// Verbose
// --------------------------------------------------------------------------------------
static void verbose_print_chunk_info(const Config& cfg, int chunk_idx, const HostChunk& ch, int kept) {
  if (!cfg.verbose) return;

  std::vector<int> vids;
  vids.reserve((size_t)ch.n);
  for (int i = 0; i < ch.n; ++i) vids.push_back(ch.video_id[i]);

  std::sort(vids.begin(), vids.end());
  vids.erase(std::unique(vids.begin(), vids.end()), vids.end());

  std::cout << "[CHUNK " << chunk_idx << "] "
            << "n=" << ch.n
            << " kept=" << kept
            << " videos={";
  for (size_t i = 0; i < vids.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << vids[i];
  }
  std::cout << "}\n";
}

// ============================================================
// Entry point
// ============================================================
BuildStats carica_db(const Config& cfg) {
  CpuWorkspace ws;
  workspace_init(ws, cfg);

  RawDbReader reader(cfg);
  NewDbWriter writer(cfg);

  BuildStats stats{};
  int chunk_idx = 0;

  if (cfg.verbose) {
    std::cout << "\nFASE 1: GENERAZIONE DEL NUOVO DATABASE =========================================\n\n";
  }

  while (reader.has_next()) {
    ++chunk_idx;

    HostChunk ch = reader.next_chunk(cfg.chunk_frames, ws.bytes_per_frame);
    if (ch.n <= 0) break;

    stats.frames_total += (uint64_t)ch.n;

    // 1) dedup (1:1 GPU)
    (void)run_dedup_cpu_like_gpu(ws, cfg, ch);

    // 2) compaction
    const int kept = run_compaction_cpu(ws, ch.n);
    stats.frames_after_dedup += (uint64_t)kept;

    // 3) index (aHash 1:1 con i kernel query aHash che hai incollato)
    run_index_cpu(ws, ch, kept);
    stats.signatures_written += (uint64_t)kept;

    if (cfg.verbose) {
      verbose_print_chunk_info(cfg, chunk_idx, ch, kept);
    }

    // 4) pack + write
    DbSoAChunk out = build_soa_chunk_from_results(ch, ws, kept);
    writer.write_chunk(out);
  }

  return stats;
}
