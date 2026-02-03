// workspace.h  (CPU-only)
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

// Workspace CPU: solo memoria host.
// Nessun puntatore device, nessun CUDA, nessun CUB.
struct Workspace {

  // info generali
  int max_frames = 0;          // tipicamente cfg.chunk_frames
  int w = 0, h = 0, c = 0;     // frame width / height / channels
  size_t bytes_per_frame = 0;

  // ============================================================
  // Buffer CPU
  // ============================================================

  // RAW frames del chunk: max_frames * bytes_per_frame
  std::vector<uint8_t> frames;

  // flag di dedup: 1 = keep, 0 = drop
  std::vector<uint8_t> keep;

  // lista compatta degli indici tenuti
  // es: keep = [1,0,1] → kept_ids = [0,2]
  std::vector<int32_t> kept_ids;

  // hash aHash64 per i frame tenuti
  std::vector<uint64_t> hashes;

  // temp per aHash: 64 celle per frame (opzionale)
  std::vector<uint16_t> cell_mean_u16;

  // ============================================================
  // helper
  // ============================================================
  void reset(int n) {
    if (n < 0) n = 0;
    if ((int)keep.size() < n) keep.resize(n);
    std::fill(keep.begin(), keep.begin() + n, 0);
    kept_ids.clear();
    hashes.clear();
  }
};
