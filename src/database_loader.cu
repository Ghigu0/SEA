// database_loader.cu
// Build DB: load -> upload -> dedup -> compaction -> index -> download -> pack -> write (SoA)

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

#include "../include/config.h"
#include "../include/cuda_utils.h"
#include "../include/db_types.h"
#include "../include/raw_db_reader.h"
#include "../include/workspace.h"
#include "../include/compaction.cuh"

#include "./kernels_db/kernel_frame_deduplication.cuh"
#include "./kernels_db/kernel_index_ahash.cuh"

// ============================================================
// Writer SoA (placeholder) - poi lo sposti in un file dedicato
// ============================================================
struct SoaWriter {
  explicit SoaWriter(const Config&) {
    // TODO: apri file/crea cartella/inizializza header
  }
  void write_chunk(const DbSoAChunk& out) {
    // TODO: scrivi hashes + metadati su disco (SoA)
    (void)out;
  }
};

// ============================================================
// Helpers pipeline
// ============================================================

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
  // offset_bytes: lo riempi quando implementi davvero il writer binario (se ti serve)

  for (int j = 0; j < kept; ++j) {
    const int32_t i = ws.h_kept_ids[j]; // indice nel chunk originale
    out.hashes[j]   = ws.h_hashes[j];
    out.video_id[j] = in.video_id[i];
    out.frame_id[j] = in.frame_id[i];
  }
  return out;
}

// ============================================================
// Entry point
// ============================================================
BuildStats carica_db(const Config& cfg) {
  CUDA_CHECK(cudaSetDevice(cfg.gpu_id));

  Workspace ws;
  workspace_init(ws, cfg);

  RawDbReader reader(cfg);
  SoaWriter writer(cfg);

  BuildStats stats{};

  try {
    while (reader.has_next()) {
      HostChunk ch = reader.next_chunk(cfg.chunk_frames, ws.bytes_per_frame);
      if (ch.n <= 0) break;

      stats.frames_total += (uint64_t)ch.n;

      upload_frames(ws, ch);

      int kept = ch.n;
      if (cfg.enable_dedup) {
        run_dedup(ws, cfg, ch.n);
        kept = run_compaction_cub(ws, ch.n);
      } else {
        // se dedup disabilitato, compaction deve semplicemente tenere tutti
        kept = run_compaction_cub(ws, ch.n);
      }

      stats.frames_after_dedup += (uint64_t)kept;

      run_index(ws, kept);
      stats.signatures_written += (uint64_t)kept;

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
