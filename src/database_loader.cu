// database_loader.cu
// Build DB: load -> upload -> dedup -> compaction -> index -> download -> pack -> write (SoA)

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <algorithm> // sort, unique, min/max

#include "../include/config.h"
#include "../include/cuda_utils.h"
#include "../include/db_types.h"
#include "../include/workspace.h"
#include "../include/database_loader.h"
#include "../include/I_O/raw_db_reader.h"
#include "../include/I_O/new_db_writer.h"
#include "./kernels_db/headers/compaction.cuh"
#include "./kernels_db/headers/kernel_frame_deduplication.cuh"
#include "./kernels_db/headers/kernel_index_ahash.cuh"

static void workspace_init(Workspace& ws, const Config& cfg) {
  ws.max_frames = cfg.chunk_frames;
  ws.w = cfg.frame_w;
  ws.h = cfg.frame_h;
  ws.c = cfg.channels;
  ws.bytes_per_frame = static_cast<size_t>(ws.w) * ws.h * ws.c;

  // Device buffers
  CUDA_CHECK(cudaMalloc(&ws.d_frames,   (size_t)ws.max_frames * ws.bytes_per_frame));
  CUDA_CHECK(cudaMalloc(&ws.d_keep,     (size_t)ws.max_frames * sizeof(uint8_t)));

  CUDA_CHECK(cudaMalloc(&ws.d_kept_ids, (size_t)ws.max_frames * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_hashes,   (size_t)ws.max_frames * sizeof(uint64_t)));

  // CUB compaction helpers
  CUDA_CHECK(cudaMalloc(&ws.d_all_ids,    (size_t)ws.max_frames * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_kept_count, sizeof(int32_t)));

  // temp per aHash (kept * 64)
  CUDA_CHECK(cudaMalloc(&ws.d_cell_mean_u16, (size_t)ws.max_frames * 64u * sizeof(uint16_t)));

  // Host pinned buffers (download veloce)
  CUDA_CHECK(cudaMallocHost(&ws.h_hashes,   (size_t)ws.max_frames * sizeof(uint64_t)));
  CUDA_CHECK(cudaMallocHost(&ws.h_kept_ids, (size_t)ws.max_frames * sizeof(int32_t)));

  // temp CUB (allocato "lazy" dentro run_compaction_cub di solito)
  ws.d_temp = nullptr;
  ws.d_temp_bytes = 0;
}

static void workspace_destroy(Workspace& ws) {
  if (ws.d_temp)         cudaFree(ws.d_temp);
  if (ws.d_cell_mean_u16)cudaFree(ws.d_cell_mean_u16);
  if (ws.d_kept_count)   cudaFree(ws.d_kept_count);
  if (ws.d_all_ids)      cudaFree(ws.d_all_ids);

  if (ws.d_hashes)       cudaFree(ws.d_hashes);
  if (ws.d_kept_ids)     cudaFree(ws.d_kept_ids);
  if (ws.d_keep)         cudaFree(ws.d_keep);
  if (ws.d_frames)       cudaFree(ws.d_frames);

  if (ws.h_kept_ids)     cudaFreeHost(ws.h_kept_ids);
  if (ws.h_hashes)       cudaFreeHost(ws.h_hashes);

  ws = Workspace{};
}

static void upload_frames(Workspace& ws, const HostChunk& ch) {
  const size_t bytes = (size_t)ch.n * ws.bytes_per_frame;
  CUDA_CHECK(cudaMemcpyAsync(ws.d_frames, ch.frames.data(), bytes, cudaMemcpyHostToDevice, 0));
}

static void run_dedup(Workspace& ws, const Config& cfg, int n) {
  CUDA_CHECK(cudaMemsetAsync(ws.d_keep, 0, (size_t)n * sizeof(uint8_t), 0));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);

  dedup_kernel_downsample_sad<<<grid, block>>>(
      ws.d_frames,
      ws.d_keep,
      n,
      ws.w, ws.h, ws.c,
      (int)ws.bytes_per_frame,
      cfg.dedup_threshold
  );
  CUDA_CHECK(cudaGetLastError());
}

static void run_index(Workspace& ws, int kept) {
  if (kept <= 0) return;

  // 1) mean per ciascuna delle 64 celle (8x8) per ogni frame kept
  dim3 block1(256, 1, 1);
  dim3 grid1((unsigned)kept, 64u, 1u);

  k_downsample8x8_cellmean_u16_kept<<<grid1, block1>>>(
      ws.d_frames,
      ws.d_kept_ids,
      kept,
      ws.w, ws.h, ws.c,
      (int)ws.bytes_per_frame,
      ws.d_cell_mean_u16
  );
  CUDA_CHECK(cudaGetLastError());

  // 2) hash64 confrontando ciascuna cella con la media globale
  dim3 block2(256, 1, 1);
  dim3 grid2((unsigned)((kept + (int)block2.x - 1) / (int)block2.x), 1u, 1u);

  k_ahash64_from_cellmean_kept<<<grid2, block2>>>(
      ws.d_cell_mean_u16,
      kept,
      ws.d_hashes
  );
  CUDA_CHECK(cudaGetLastError());
}

static void download_results(Workspace& ws, int kept) {
  CUDA_CHECK(cudaMemcpyAsync(ws.h_hashes, ws.d_hashes, (size_t)kept * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, 0));
  CUDA_CHECK(cudaMemcpyAsync(ws.h_kept_ids, ws.d_kept_ids, (size_t)kept * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, 0));
}

static DbSoAChunk build_soa_chunk_from_results(const HostChunk& in, const Workspace& ws, int kept) {
  DbSoAChunk out;
  out.hashes.resize(kept);
  out.video_id.resize(kept);
  out.frame_id.resize(kept);

  out.bytes_per_frame = (int)ws.bytes_per_frame;
  out.frames_raw.resize((size_t)kept * ws.bytes_per_frame);

  for (int j = 0; j < kept; ++j) {
    const int32_t i = ws.h_kept_ids[j]; // indice nel chunk originale

    // Metadati + hash
    out.hashes[j]   = ws.h_hashes[j];
    out.video_id[j] = in.video_id[i];
    out.frame_id[j] = in.frame_id[i];

    // Copia frame RAW: src = frame i nel chunk, dst = frame j nel DB nuovo
    const uint8_t* src = in.frames.data() + (size_t)i * ws.bytes_per_frame;
    uint8_t*       dst = out.frames_raw.data() + (size_t)j * ws.bytes_per_frame;
    std::memcpy(dst, src, ws.bytes_per_frame);
  }

  return out;
}

//funzione che serve solo ed esclusivamente per stampare a video quanti frame sono stati ridotti per chunk
static void verbose_print_chunk_info(const Config& cfg, int chunk_idx, const HostChunk& ch, int kept ) {
  if (!cfg.verbose) return;

  std::vector<int> vids;
  vids.reserve((size_t)ch.n);
  for (int i = 0; i < ch.n; ++i)
    vids.push_back(ch.video_id[i]);

  std::sort(vids.begin(), vids.end());
  vids.erase(std::unique(vids.begin(), vids.end()), vids.end());

  std::cerr << "[CHUNK " << chunk_idx << "] "
            << "n=" << ch.n
            << " kept=" << kept
            << " videos={";

  for (size_t i = 0; i < vids.size(); ++i) {
    if (i) std::cerr << ",";
    std::cerr << vids[i];
  }

  std::cerr << "}\n";
}


// ============================================================
// Entry point
// ============================================================
BuildStats carica_db(const Config& cfg) {
  CUDA_CHECK(cudaSetDevice(cfg.gpu_id));

  Workspace ws;
  workspace_init(ws, cfg);

  RawDbReader reader(cfg);
  NewDbWriter writer(cfg);

  BuildStats stats{};

  int chunk_idx = 0;
  if (cfg.verbose){
    std::cout << "\nFASE 1: GENERAZIONE DEL NUOVO DATABASE =========================================" << "\n\n";
  }
  try {
    while (reader.has_next()) {
      ++chunk_idx;
      HostChunk ch = reader.next_chunk(cfg.chunk_frames, ws.bytes_per_frame);
      if (ch.n <= 0) break;

      stats.frames_total += (uint64_t)ch.n;

      upload_frames(ws, ch);

      // dedup sempre attivo (feature toggle rimosso)
      run_dedup(ws, cfg, ch.n);
      int kept = run_compaction_cub(ws, ch.n);

      stats.frames_after_dedup += (uint64_t)kept;

      run_index(ws, kept);
      stats.signatures_written += (uint64_t)kept;

      // output verbose per chunk (quali video + quanti frame tenuti)
      if (cfg.verbose){
        verbose_print_chunk_info(cfg, chunk_idx, ch, kept);
      }

      download_results(ws, kept);

      // qui serve una sync perché usiamo ws.h_* su CPU
      CUDA_CHECK(cudaDeviceSynchronize());

      DbSoAChunk out = build_soa_chunk_from_results(ch, ws, kept);

      writer.write_chunk(out);
    }
  } catch (...) {
    workspace_destroy(ws);
    throw;
  }

  workspace_destroy(ws);
  return stats;
}
