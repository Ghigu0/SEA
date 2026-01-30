#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../include/cuda_utils.h"
#include "../include/config.h"
#include "../include/frame_research.h"
#include "../include/research_types.h"
#include "../include/I_O/frame_reader.h"
#include "../include/I_O/new_db_reader.h"

// --- GPU kernels (come prima) ---
#include "../src/kernel_research/headers/kernel_query_ahash.cuh"
#include "../src/kernel_research/headers/kernel_template_sad_batch.cuh"
#include "../src/kernel_research/headers/kernel_hist_hamming.cuh"
#include "../src/kernel_research/headers/kernel_collect_candidates.cuh"

// --- CPU equivalents (nuovi) ---
// Adatta i path a dove li hai salvati tu.
#include "../src/kernel_research/cpu/kernel_query_ahash_cpu.h"
#include "../src/kernel_research/cpu/kernel_hist_hamming_cpu.h"
#include "../src/kernel_research/cpu/kernel_collect_candidates_cpu.h"
#include "../src/kernel_research/cpu/kernel_template_sad_batch_cpu.h"

// =============================================================================================
// GPU: calcolo qhash (2 kernel)
static uint64_t compute_query_hash64_gpu(const std::string& imgPath, const Config& cfg) {
  FrameReader fr(cfg.frame_w, cfg.frame_h, cfg.channels);
  std::vector<uint8_t> h_frame = fr.read_raw_frame(imgPath);
  const int bpf = fr.bytes_per_frame();

  uint8_t*  d_frame  = nullptr;
  uint16_t* d_cells  = nullptr;
  uint64_t* d_hash   = nullptr;

  CUDA_CHECK(cudaMalloc(&d_frame, (size_t)bpf));
  CUDA_CHECK(cudaMalloc(&d_cells, 64u * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_hash,  sizeof(uint64_t)));

  CUDA_CHECK(cudaMemcpy(d_frame, h_frame.data(), (size_t)bpf, cudaMemcpyHostToDevice));

  dim3 block1(256, 1, 1);
  dim3 grid1(64, 1, 1);
  dim3 block2(1, 1, 1);
  dim3 grid2(1, 1, 1);

  k_query_cellmean_u16_8x8<<<grid1, block1>>>(
      d_frame,
      cfg.frame_w, cfg.frame_h, cfg.channels,
      bpf,
      d_cells
  );
  CUDA_CHECK(cudaGetLastError());

  k_query_ahash64_from_cellmean<<<grid2, block2>>>(
      d_cells,
      d_hash
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  uint64_t h_hash = 0;
  CUDA_CHECK(cudaMemcpy(&h_hash, d_hash, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_hash));
  CUDA_CHECK(cudaFree(d_cells));
  CUDA_CHECK(cudaFree(d_frame));

  return h_hash;
}

// CPU: calcolo qhash (sequenziale)
static uint64_t compute_query_hash64_cpu(const std::string& imgPath, const Config& cfg) {
  FrameReader fr(cfg.frame_w, cfg.frame_h, cfg.channels);
  std::vector<uint8_t> h_frame = fr.read_raw_frame(imgPath);
  const int bpf = fr.bytes_per_frame();
  if ((int)h_frame.size() != bpf) {
    throw std::runtime_error("[QUERY][CPU] query frame size mismatch");
  }
  return compute_query_ahash64_cpu(h_frame.data(), cfg.frame_w, cfg.frame_h, cfg.channels, bpf);
}

//===============================================================================
// per l'individuazione della soglia minima
static uint8_t choose_threshold_for_topk(const uint32_t hist[65], int topk) {
  uint64_t cum = 0;
  for (int d = 0; d <= 64; ++d) {
    cum += hist[d];
    if (cum >= (uint64_t)topk) return (uint8_t)d;
  }
  return 64;
}

// helper: stampa istogramma Hamming
static void verbose_print_hamming_hist(const Config& cfg, const uint32_t hist[65], int topk, uint8_t thresh) {
  if (!cfg.verbose) return;

  std::cout << "[QUERY] Hamming histogram (Hamming dist 0–64):\n";
  uint64_t cum = 0;
  for (int d = 0; d <= 64; ++d) {
    if (hist[d] == 0) continue;
    cum += hist[d];
    std::cout << "  d=" << d << " count=" << hist[d]
              << "  cum=" << cum;
    if (d == (int)thresh) std::cout << "  <-- thresh";
    std::cout << "\n";
    if (cum >= (uint64_t)topk && d >= (int)thresh) break;
  }
}

// helper: stampa top candidati
static void verbose_print_top_candidates(const Config& cfg, const SoADbReader& db,
                                        const std::vector<Cand>& cands, int max_show) {
  if (!cfg.verbose) return;

  const int show = std::min((int)cands.size(), max_show);
  std::cout << "[QUERY] Top candidates (after sort by Hamming dist), show=" << show << "\n";

  const auto& vids = db.video_id();
  const auto& fids = db.frame_id();

  for (int i = 0; i < show; ++i) {
    const auto& c = cands[i];
    std::cout << "  #" << i
              << " idx=" << c.idx
              << " dist=" << (int)c.dist;

    if ((int)vids.size() > c.idx && (int)fids.size() > c.idx) {
      std::cout << " video_id=" << vids[c.idx]
                << " frame_id=" << fids[c.idx];
    }
    std::cout << "\n";
  }
}

// =============================================================================================
// CPU path: pipeline completa query
static QueryResult ricerca_frame_cpu(const Config& cfg) {
  QueryResult res{};

  SoADbReader db(cfg);
  db.validate(true, true);

  const uint64_t qhash = compute_query_hash64_cpu(cfg.query_frame_path, cfg);

  if (cfg.verbose) {
    std::cout << "\nFASE 2: RICERCA DEL FRAME ====================================================\n\n";
    std::cout << "[QUERY][CPU] query_frame_path=" << cfg.query_frame_path << "\n";
    std::cout << "[QUERY][CPU] qhash=0x" << std::hex << qhash << std::dec << "\n";
  }

  db.load_hashes();
  db.load_meta();

  const int N = static_cast<int>(db.hashes().size());
  if (N <= 0) {
    if (cfg.verbose) std::cout << "[QUERY][CPU] Empty DB hashes\n";
    return res;
  }
  if (cfg.verbose) {
    std::cout << "[QUERY][CPU] N_frames_new_db=" << N << " topk=" << cfg.topk << "\n";
  }

  // 1) Hist Hamming su CPU
  uint32_t h_hist[65];
  hist_hamming65_cpu(db.hashes().data(), N, qhash, h_hist);

  const uint8_t thresh = choose_threshold_for_topk(h_hist, cfg.topk);

  if (cfg.verbose) {
    std::cout << "[QUERY][CPU] Hamming cutoff (max dist for topk): " << (int)thresh << "\n";
  }
  verbose_print_hamming_hist(cfg, h_hist, cfg.topk, thresh);

  // 2) Collect candidates su CPU
  std::vector<Cand> cands((size_t)cfg.topk);
  int count_total = 0;
  int written = collect_candidates_cpu(
      db.hashes().data(), N, qhash, thresh,
      cands.data(), cfg.topk, &count_total);
  cands.resize((size_t)written);

  if (cfg.verbose) {
    std::cout << "[QUERY][CPU] Candidates collected: " << written
              << " (total passing thresh=" << count_total
              << ", cap topk=" << cfg.topk << ")\n";
  }

  if (cands.empty()) {
    res.found = false;
    return res;
  }

  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& b){ return a.dist < b.dist; });

  verbose_print_top_candidates(cfg, db, cands, 10);

  // 3) Template matching su CPU (SAD)
  FrameReader fr(cfg.frame_w, cfg.frame_h, cfg.channels);
  std::vector<uint8_t> query_raw = fr.read_raw_frame(cfg.query_frame_path);
  const int bpf = fr.bytes_per_frame();
  if ((int)query_raw.size() != bpf) {
    throw std::runtime_error("[QUERY][CPU] query_raw size mismatch: expected bpf bytes");
  }

  db.open_frames();

  const int total = (int)cands.size();
  const int BATCH = 64;

  uint32_t best_score = std::numeric_limits<uint32_t>::max();
  int best_idx = -1;
  uint8_t best_dist = 255;

  std::vector<uint8_t> batch_host((size_t)BATCH * (size_t)bpf);
  std::vector<uint32_t> scores((size_t)BATCH);

  if (cfg.verbose) {
    std::cout << "[TM][CPU] Processing candidate chunk with template matching\n";
  }

  for (int base = 0; base < total; base += BATCH) {
    const int count = std::min(BATCH, total - base);

    if (cfg.verbose) {
      std::cout << "[TM][CPU] batch base=" << base << " count=" << count << "\n";
    }

    for (int i = 0; i < count; ++i) {
      const int cand_idx = cands[base + i].idx;
      db.read_frame((size_t)cand_idx,
                    batch_host.data() + (size_t)i * (size_t)bpf,
                    (size_t)bpf);
    }

    // SAD batch CPU (equivalente al kernel)
    template_sad_batch_cpu(
        query_raw.data(),
        batch_host.data(),
        bpf,
        count,
        scores.data());

    for (int i = 0; i < count; ++i) {
      const uint32_t s = scores[i];
      const Cand& c = cands[base + i];
      if (s < best_score) {
        best_score = s;
        best_idx = c.idx;
        best_dist = c.dist;
      }
    }
  }

  db.close_frames();

  if (best_idx < 0) {
    res.found = false;
    return res;
  }

  res.found = true;
  res.best_db_index = best_idx;
  res.video_id = db.video_id()[best_idx];
  res.frame_id = db.frame_id()[best_idx];
  res.score = (float)best_score;

  if (cfg.verbose) {
    std::cout << "\n\n[RESULT][CPU] idx=" << best_idx
              << " video_id=" << res.video_id
              << " frame_id=" << res.frame_id
              << " dist=" << (int)best_dist
              << " best_SAD=" << best_score
              << "\n";
  }

  return res;
}

// =============================================================================================
// GPU path: il tuo codice (spostato in funzione per chiarezza)
static QueryResult ricerca_frame_gpu(const Config& cfg) {
  QueryResult res{};

  SoADbReader db(cfg);
  db.validate(true, true);

  const uint64_t qhash = compute_query_hash64_gpu(cfg.query_frame_path, cfg);

  if (cfg.verbose) {
    std::cout << "\nFASE 2: RICERCA DEL FRAME ====================================================\n\n";
    std::cout << "[QUERY] query_frame_path=" << cfg.query_frame_path << "\n";
    std::cout << "[QUERY] qhash=0x" << std::hex << qhash << std::dec << "\n";
  }

  db.load_hashes();
  db.load_meta();

  const int N = static_cast<int>(db.hashes().size());
  if (N <= 0) {
    if (cfg.verbose) std::cout << "[QUERY] Empty DB hashes\n";
    return res;
  }
  if (cfg.verbose) {
    std::cout << "[QUERY] N_frames_new_db=" << N << " topk=" << cfg.topk << "\n";
  }

  uint64_t* d_hashes = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hashes, (size_t)N * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_hashes, db.hashes().data(), (size_t)N * sizeof(uint64_t), cudaMemcpyHostToDevice));

  uint32_t* d_hist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hist, 65 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_hist, 0, 65 * sizeof(uint32_t)));

  const int block = 256;
  int grid = (N + block - 1) / block;
  if (grid > 120) grid = 120;
  if (grid < 1)   grid = 1;

  kernel_hist_hamming<<<grid, block>>>(d_hashes, N, qhash, d_hist);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_hist[65];
  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 65 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  const uint8_t thresh = choose_threshold_for_topk(h_hist, cfg.topk);

  if (cfg.verbose) {
    std::cout << "[QUERY] Hamming cutoff (max dist for topk): " << (int)thresh << "\n";
  }
  verbose_print_hamming_hist(cfg, h_hist, cfg.topk, thresh);

  Cand* d_cands = nullptr;
  int* d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cands, (size_t)cfg.topk * sizeof(Cand)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  kernel_collect_candidates<<<grid, block>>>(d_hashes, N, qhash, thresh, d_cands, cfg.topk, d_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_count = 0;
  CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  h_count = std::min(h_count, cfg.topk);

  if (cfg.verbose) {
    std::cout << "[QUERY] Candidates collected: " << h_count << " (cap topk=" << cfg.topk << ")\n";
  }

  std::vector<Cand> h_cands((size_t)h_count);
  if (h_count > 0) {
    CUDA_CHECK(cudaMemcpy(h_cands.data(), d_cands, (size_t)h_count * sizeof(Cand), cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_cands));
  CUDA_CHECK(cudaFree(d_hist));
  CUDA_CHECK(cudaFree(d_hashes));

  if (h_count == 0) {
    res.found = false;
    return res;
  }

  std::sort(h_cands.begin(), h_cands.end(), [](const Cand& a, const Cand& b){ return a.dist < b.dist; });
  verbose_print_top_candidates(cfg, db, h_cands, 10);

  FrameReader fr(cfg.frame_w, cfg.frame_h, cfg.channels);
  std::vector<uint8_t> query_raw = fr.read_raw_frame(cfg.query_frame_path);
  const int bpf = fr.bytes_per_frame();
  if ((int)query_raw.size() != bpf) {
    throw std::runtime_error("[QUERY] query_raw size mismatch: expected bpf bytes");
  }

  db.open_frames();

  const int BATCH = 64;
  const int total = h_count;
  uint8_t best_dist = 255;

  std::vector<uint8_t> batch_host((size_t)BATCH * (size_t)bpf);

  uint8_t* d_query = nullptr;
  uint8_t* d_batch = nullptr;
  uint32_t* d_scores = nullptr;

  CUDA_CHECK(cudaMalloc(&d_query, (size_t)bpf));
  CUDA_CHECK(cudaMemcpy(d_query, query_raw.data(), (size_t)bpf, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_batch, (size_t)BATCH * (size_t)bpf));
  CUDA_CHECK(cudaMalloc(&d_scores, (size_t)BATCH * sizeof(uint32_t)));

  uint32_t best_score = std::numeric_limits<uint32_t>::max();
  int best_idx = -1;

  if (cfg.verbose) {
    std::cout << "[TM] Processing candidate chunk with template matching \n";
  }

  for (int base = 0; base < total; base += BATCH) {
    const int count = std::min(BATCH, total - base);

    if (cfg.verbose) {
      std::cout << "[TM] batch base=" << base << " count=" << count << "\n";
    }

    for (int i = 0; i < count; ++i) {
      const int cand_idx = h_cands[base + i].idx;
      db.read_frame((size_t)cand_idx, batch_host.data() + (size_t)i * (size_t)bpf, (size_t)bpf);
    }

    CUDA_CHECK(cudaMemcpy(d_batch, batch_host.data(), (size_t)count * (size_t)bpf, cudaMemcpyHostToDevice));

    dim3 blk(256);
    dim3 grd(count);
    kernel_template_sad_batch<<<grd, blk>>>(d_query, d_batch, bpf, count, d_scores);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> scores((size_t)count);
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, (size_t)count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < count; ++i) {
      const uint32_t s = scores[i];
      const Cand& c = h_cands[base + i];
      if (s < best_score) {
        best_score = s;
        best_idx = c.idx;
        best_dist = c.dist;
      }
    }
  }

  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_batch));
  CUDA_CHECK(cudaFree(d_query));

  db.close_frames();

  if (best_idx < 0) {
    res.found = false;
    return res;
  }

  res.found = true;
  res.best_db_index = best_idx;
  res.video_id = db.video_id()[best_idx];
  res.frame_id = db.frame_id()[best_idx];
  res.score = (float)best_score;

  if (cfg.verbose) {
    std::cout << "\n\n[RESULT] idx=" << best_idx
              << " video_id=" << res.video_id
              << " frame_id=" << res.frame_id
              << " dist=" << (int)best_dist
              << " best_SAD=" << best_score
              << "\n";
  }

  return res;
}

// =============================================================================================
// Entry point unico
QueryResult ricerca_frame(const Config& cfg) {
  // Se vuoi tenere CUDA solo per GPU:
  // - CPU path non chiama cudaSetDevice né alloca GPU.
  if (cfg.use_cpu) {
    return ricerca_frame_cpu(cfg);
  }

  CUDA_CHECK(cudaSetDevice(cfg.gpu_id));
  return ricerca_frame_gpu(cfg);
}
