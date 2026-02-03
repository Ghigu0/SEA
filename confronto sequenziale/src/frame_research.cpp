// frame_research.cpp (CPU-only)
// Niente CUDA. Ricerca sequenziale su CPU, con componenti allineate ai kernel GPU incollati.
//
// Allineamenti 1:1 con i kernel GPU:
// - aHash query: celle 8x8 con cell_w=w/8, cell_h=h/8, ultima cella prende il resto
// - luma BT.601: (77R + 150G + 29B) >> 8
// - bit = (cell_mean > global_mean) (STRICT >)
// - Hamming: popcount64(hash ^ qhash)
// - Template SAD: somma absdiff byte-wise su RAW, accumulo uint32_t
//
// Nota: la selezione candidati qui è DETERMINISTICA (sort + tie-break su idx).
// La GPU con atomicAdd può avere ordine non deterministico; per comparazione scientifica è meglio così.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/config.h"
#include "../include/frame_research.h"
#include "../include/research_types.h"
#include "../include/I_O/frame_reader.h"
#include "../include/I_O/new_db_reader.h"

// --------------------------------------------------------------------------------------
// Luma BT.601 approx (come kernel_query_ahash.cu)
// --------------------------------------------------------------------------------------
static inline uint8_t rgb_to_luma_cpu(uint8_t r, uint8_t g, uint8_t b) {
  return (uint8_t)((77u * (unsigned)r + 150u * (unsigned)g + 29u * (unsigned)b) >> 8);
}

// --------------------------------------------------------------------------------------
// aHash64 identico a (k_query_cellmean_u16_8x8 + k_query_ahash64_from_cellmean)
// --------------------------------------------------------------------------------------
static uint64_t ahash64_like_gpu_query(const uint8_t* frame, int w, int h, int c) {
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

static uint64_t compute_query_hash64(const std::string& imgPath, const Config& cfg) {
  FrameReader fr(cfg.frame_w, cfg.frame_h, cfg.channels);
  std::vector<uint8_t> raw = fr.read_raw_frame(imgPath);

  const int bpf = fr.bytes_per_frame();
  if ((int)raw.size() != bpf) {
    throw std::runtime_error("[QUERY] FrameReader: wrong file size for RAW frame (query).");
  }

  return ahash64_like_gpu_query(raw.data(), cfg.frame_w, cfg.frame_h, cfg.channels);
}

// --------------------------------------------------------------------------------------
// popcount64 / Hamming (0..64)
// --------------------------------------------------------------------------------------
#if defined(_MSC_VER)
  #include <intrin.h>
#endif

static inline uint32_t popcount64(uint64_t x) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
  return (uint32_t)__popcnt64(x);
#elif defined(_MSC_VER)
  return (uint32_t)(__popcnt((uint32_t)x) + __popcnt((uint32_t)(x >> 32)));
#else
  uint32_t c = 0;
  while (x) { x &= (x - 1); ++c; }
  return c;
#endif
}

static inline uint8_t hamming64(uint64_t a, uint64_t b) {
  return (uint8_t)popcount64(a ^ b);
}

// --------------------------------------------------------------------------------------
// Soglia per topk
// --------------------------------------------------------------------------------------
static uint8_t choose_threshold_for_topk(const uint32_t hist[65], int topk) {
  uint64_t cum = 0;
  for (int d = 0; d <= 64; ++d) {
    cum += hist[d];
    if (cum >= (uint64_t)topk) return (uint8_t)d;
  }
  return 64;
}

// Verbose helpers
static void verbose_print_hamming_hist(const Config& cfg, const uint32_t hist[65], int topk, uint8_t thresh) {
  if (!cfg.verbose) return;

  std::cout << "[QUERY] Hamming histogram (Hamming dist 0–64):\n";
  uint64_t cum = 0;
  for (int d = 0; d <= 64; ++d) {
    if (hist[d] == 0) continue;
    cum += hist[d];
    std::cout << "  d=" << d << " count=" << hist[d] << "  cum=" << cum;
    if (d == (int)thresh) std::cout << "  <-- thresh";
    std::cout << "\n";
    if (cum >= (uint64_t)topk && d >= (int)thresh) break;
  }
}

static void verbose_print_top_candidates(const Config& cfg, const SoADbReader& db,
                                        const std::vector<Cand>& cands, int max_show) {
  if (!cfg.verbose) return;

  const int show = std::min((int)cands.size(), max_show);
  std::cout << "[QUERY] Top candidates (after sort by Hamming dist), show=" << show << "\n";

  const auto& vids = db.video_id();
  const auto& fids = db.frame_id();

  for (int i = 0; i < show; ++i) {
    const auto& c = cands[i];
    std::cout << "  #" << i << " idx=" << c.idx << " dist=" << (int)c.dist;

    if ((int)vids.size() > c.idx && (int)fids.size() > c.idx) {
      std::cout << " video_id=" << vids[c.idx]
                << " frame_id=" << fids[c.idx];
    }
    std::cout << "\n";
  }
}

// --------------------------------------------------------------------------------------
// Template matching SAD byte-wise (identico a kernel_template_sad_batch)
// --------------------------------------------------------------------------------------
static inline uint32_t uabsdiff_u8(uint8_t a, uint8_t b) {
  return (a > b) ? (uint32_t)(a - b) : (uint32_t)(b - a);
}

static uint32_t sad_bytes_u32(const uint8_t* query, const uint8_t* cand, int bpf) {
  uint32_t s = 0;
  for (int j = 0; j < bpf; ++j) s += uabsdiff_u8(query[j], cand[j]);
  return s;
}

// --------------------------------------------------------------------------------------
// Ricerca frame
// --------------------------------------------------------------------------------------
QueryResult ricerca_frame(const Config& cfg) {
  QueryResult res{};

  SoADbReader db(cfg);
  db.validate(true, true);

  const uint64_t qhash = compute_query_hash64(cfg.query_frame_path, cfg);

  if (cfg.verbose) {
    std::cout << "\nFASE 2: RICERCA DEL FRAME ====================================================\n\n";
    std::cout << "[QUERY] query_frame_path=" << cfg.query_frame_path << "\n";
    std::cout << "[QUERY] qhash=0x" << std::hex << qhash << std::dec << "\n";
  }

  db.load_hashes();
  db.load_meta();

  const int N = (int)db.hashes().size();
  if (N <= 0) {
    if (cfg.verbose) std::cout << "[QUERY] Empty DB hashes\n";
    return res;
  }
  if (cfg.verbose) {
    std::cout << "[QUERY] N_frames_new_db=" << N << " topk=" << cfg.topk << "\n";
  }

  // 1) Istogramma Hamming
  uint32_t hist[65];
  std::memset(hist, 0, sizeof(hist));

  // Salvo anche le distanze per evitare un secondo popcount (CPU)
  std::vector<uint8_t> dists((size_t)N);
  for (int i = 0; i < N; ++i) {
    const uint8_t d = hamming64(db.hashes()[i], qhash);
    dists[(size_t)i] = d;
    ++hist[d];
  }

  const uint8_t thresh = choose_threshold_for_topk(hist, cfg.topk);

  if (cfg.verbose) {
    std::cout << "[QUERY] Hamming cutoff (max dist for topk): " << (int)thresh << "\n";
  }
  verbose_print_hamming_hist(cfg, hist, cfg.topk, thresh);

  // 2) Collezione candidati deterministica
  // GPU kernel_collect_candidates prende i primi topk che passano (non deterministico).
  // Qui facciamo una selezione deterministica: tutti quelli <= thresh e poi topk migliori.
  std::vector<Cand> cands;
  cands.reserve((size_t)cfg.topk * 2);

  for (int i = 0; i < N; ++i) {
    const uint8_t d = dists[(size_t)i];
    if (d <= thresh) cands.push_back(Cand{ (int32_t)i, d });
  }

  if ((int)cands.size() > cfg.topk) {
    auto mid = cands.begin() + cfg.topk;
    std::nth_element(cands.begin(), mid, cands.end(),
                     [](const Cand& a, const Cand& b) {
                       if (a.dist != b.dist) return a.dist < b.dist;
                       return a.idx < b.idx;
                     });
    cands.resize((size_t)cfg.topk);
  }

  // ordino topk per dist, tie-break idx (deterministico)
  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& b) {
              if (a.dist != b.dist) return a.dist < b.dist;
              return a.idx < b.idx;
            });

  const int h_count = (int)cands.size();
  if (cfg.verbose) {
    std::cout << "[QUERY] Candidates collected: " << h_count << " (cap topk=" << cfg.topk << ")\n";
  }

  if (h_count == 0) {
    res.found = false;
    return res;
  }

  verbose_print_top_candidates(cfg, db, cands, 10);

  // 3) carico query raw
  FrameReader fr(cfg.frame_w, cfg.frame_h, cfg.channels);
  std::vector<uint8_t> query_raw = fr.read_raw_frame(cfg.query_frame_path);
  const int bpf = fr.bytes_per_frame();
  if ((int)query_raw.size() != bpf) {
    throw std::runtime_error("[QUERY] query_raw size mismatch: expected bpf bytes");
  }

  // 4) template matching SAD
  db.open_frames();

  uint32_t best_score = std::numeric_limits<uint32_t>::max();
  int best_idx = -1;
  uint8_t best_dist = 255;

  std::vector<uint8_t> cand_raw((size_t)bpf);

  if (cfg.verbose) {
    std::cout << "[TM] Processing candidates with template matching (CPU)\n";
  }

  for (int i = 0; i < h_count; ++i) {
    const int cand_idx = cands[i].idx;
    db.read_frame((size_t)cand_idx, cand_raw.data(), (size_t)bpf);

    const uint32_t s = sad_bytes_u32(query_raw.data(), cand_raw.data(), bpf);
    if (s < best_score) {
      best_score = s;
      best_idx = cand_idx;
      best_dist = cands[i].dist;
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
    std::cout << "\n\n[RESULT] idx=" << best_idx
              << " video_id=" << res.video_id
              << " frame_id=" << res.frame_id
              << " dist=" << (int)best_dist
              << " sad=" << best_score << "\n";
  }

  return res;
}
