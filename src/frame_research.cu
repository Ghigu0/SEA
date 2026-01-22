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
#include "../include/frame_research.cuh"

// ===============================
// Helpers file
// ===============================
static std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/' || a.back() == '\\') return a + b;
  return a + "/" + b;
}

template <typename T>
static std::vector<T> read_bin_vector(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open: " + path);
  f.seekg(0, std::ios::end);
  std::streamoff sz = f.tellg();
  if (sz < 0) throw std::runtime_error("tellg failed: " + path);
  if (sz % (std::streamoff)sizeof(T) != 0) {
    throw std::runtime_error("File size not multiple of element size: " + path);
  }
  const size_t n = (size_t)(sz / (std::streamoff)sizeof(T));
  std::vector<T> v(n);
  f.seekg(0, std::ios::beg);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)sz);
  if (!f) throw std::runtime_error("Read failed: " + path);
  return v;
}


// DOBBIAMO RIUTILIZZARE IL KERNEL NOSTRO
// =============================================================================================
// Per ora placeholder: devi sostituirla con la tua funzione di hash query (CPU o GPU).
static uint64_t compute_query_hash64_stub(const std::string& /*imgPath*/) {
  // TODO: carica immagine, riduci a 8x8/16x16 e calcola aHash/dHash come avete fatto in build
  // Qui ritorna un valore fisso solo per far compilare.
  return 0ULL;
}
// =============================================================================================



// ===============================
// GPU kernels: histogram + collect
// ===============================
__device__ __forceinline__ uint32_t popc64(uint64_t x) {
  // __popcll ritorna int, cast a uint32_t
  return (uint32_t)__popcll((unsigned long long)x);
}

__global__ void kernel_hist_hamming(const uint64_t* __restrict__ d_hashes,
                                   int n,
                                   uint64_t q,
                                   uint32_t* __restrict__ d_hist65) {
  // istogramma per blocco in shared
  __shared__ uint32_t sh[65];
  for (int i = threadIdx.x; i < 65; i += blockDim.x) sh[i] = 0;
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    uint32_t d = popc64(d_hashes[i] ^ q); // 0..64
    atomicAdd(&sh[d], 1u);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < 65; i += blockDim.x) {
    if (sh[i]) atomicAdd(&d_hist65[i], sh[i]);
  }
}

// raccoglie fino a topk indici con dist <= thresh
struct Cand {
  int32_t idx;
  uint8_t dist;
};

__global__ void kernel_collect_candidates(const uint64_t* __restrict__ d_hashes,
                                         int n,
                                         uint64_t q,
                                         uint8_t thresh,
                                         Cand* __restrict__ d_out,
                                         int topk,
                                         int* __restrict__ d_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    uint32_t d = popc64(d_hashes[i] ^ q);
    if (d <= (uint32_t)thresh) {
      int pos = atomicAdd(d_count, 1);
      if (pos < topk) {
        d_out[pos].idx = (int32_t)i;
        d_out[pos].dist = (uint8_t)d;
      }
    }
  }
}

// ===============================
// CPU: scegli threshold che copre topk
// ===============================
static uint8_t choose_threshold_for_topk(const uint32_t hist[65], int topk) {
  uint64_t cum = 0;
  for (int d = 0; d <= 64; ++d) {
    cum += hist[d];
    if (cum >= (uint64_t)topk) return (uint8_t)d;
  }
  return 64;
}


QueryResult ricerca_frame(const Config& cfg) {

  QueryResult res;

  // CAMBIA I PATH PERCHE' IL PATH LO SCEGLIAMO NOI IN FASE DI DOWNLOAD
  //==========================================================================================
  // ---- 0) paths dei file SoA
  // Convenzione: dentro cfg.deduplicated_db_path ci sono questi file.
  // Se i tuoi nomi sono diversi, cambia qui.
  const std::string p_hashes  = join_path(cfg.deduplicated_db_path, "hashes.bin");
  const std::string p_vid    = join_path(cfg.deduplicated_db_path, "video_id.bin");
  const std::string p_fid    = join_path(cfg.deduplicated_db_path, "frame_id.bin");
  const std::string p_offsets= join_path(cfg.deduplicated_db_path, "offset_bytes.bin");
  // frames.bin lo userai dopo per template matching:
  // const std::string p_frames = join_path(cfg.deduplicated_db_path, "frames.bin");
  //============================================================================================


  // COME GIA' DETTO NELL'IMPLEMENTAZIONE, DOBBIAMO RICHIAMARE IL NOSTRO KERNEL ( E SPERARE VADA BENE )
  // ===========================================================================================
  const uint64_t qhash = compute_query_hash64_stub(cfg.query_frame_path);
  // ===========================================================================================


  
  // ---- 2) load SoA su CPU
  // Nota: per index-match ti serve davvero solo hashes. Video/frame li userai per il best.
  std::vector<uint64_t> h_hashes = read_bin_vector<uint64_t>(p_hashes);
  const int64_t N64 = (int64_t)h_hashes.size();
  if (N64 <= 0) {
    if (cfg.verbose) std::cerr << "[QUERY] Empty DB hashes\n";
    return res;
  }
  if (N64 > std::numeric_limits<int>::max()) {
    throw std::runtime_error("DB too large for int indexing (N > 2^31-1).");
  }
  const int N = (int)N64;

  std::vector<int32_t> h_video_id = read_bin_vector<int32_t>(p_vid);
  std::vector<int32_t> h_frame_id = read_bin_vector<int32_t>(p_fid);

  if ((int)h_video_id.size() != N || (int)h_frame_id.size() != N) {
    throw std::runtime_error("SoA size mismatch (video_id/frame_id vs hashes).");
  }

  // offset_bytes ti serve solo se fai template matching.
  // Lo leggiamo solo se abilitato.
  std::vector<uint64_t> h_offsets;
  if (cfg.enable_template_match) {
    h_offsets = read_bin_vector<uint64_t>(p_offsets);
    if ((int)h_offsets.size() != N) {
      throw std::runtime_error("SoA size mismatch (offset_bytes vs hashes).");
    }
  }

  if (cfg.verbose) {
    std::cerr << "[QUERY] N=" << N
              << " topk=" << cfg.topk
              << " index=" << (cfg.enable_index_match ? "on" : "off")
              << " template=" << (cfg.enable_template_match ? "on" : "off")
              << "\n";
  }

  // Se index match è OFF: candidati = tutti (ma topk ha poco senso).
  // Qui per semplicità forziamo index-match ON: altrimenti scegli 0 come best.
  if (!cfg.enable_index_match) {
    res.found = true;
    res.best_db_index = 0;
    res.video_id = h_video_id[0];
    res.frame_id = h_frame_id[0];
    res.score = 0.0f;
    return res;
  }

  // ---- 3) Copia hashes in GPU (resident)
  uint64_t* d_hashes = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hashes, (size_t)N * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_hashes, h_hashes.data(), (size_t)N * sizeof(uint64_t), cudaMemcpyHostToDevice));

  // ---- 4) Pass 1: histogram distanze (0..64)
  uint32_t* d_hist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hist, 65 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_hist, 0, 65 * sizeof(uint32_t)));

  const int block = 256;
  // grid non troppo grande; basta saturare la GPU senza overhead enorme
  int grid = 0;
  {
    int sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, cfg.gpu_id));
    grid = sm * 8; // heuristic
    if (grid < 1) grid = 1;
  }

  kernel_hist_hamming<<<grid, block>>>(d_hashes, N, qhash, d_hist);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_hist[65];
  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 65 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  const uint8_t thresh = choose_threshold_for_topk(h_hist, cfg.topk);
  if (cfg.verbose) std::cerr << "[QUERY] Hamming threshold for topk: " << (int)thresh << "\n";

  // ---- 5) Pass 2: collect candidates <= thresh (fino a topk)
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

  std::vector<Cand> h_cands((size_t)h_count);
  if (h_count > 0) {
    CUDA_CHECK(cudaMemcpy(h_cands.data(), d_cands, (size_t)h_count * sizeof(Cand), cudaMemcpyDeviceToHost));
  }

  // ---- 6) pick best (min dist; tie-break: primo)
  if (h_count == 0) {
    // nessun candidato raccolto: DB vuoto o qualcosa di strano
    res.found = false;
  } else {
    auto best_it = std::min_element(h_cands.begin(), h_cands.end(),
      [](const Cand& a, const Cand& b) { return a.dist < b.dist; });

    const int best_idx = best_it->idx;
    const uint8_t best_dist = best_it->dist;

    res.found = true;
    res.best_db_index = best_idx;
    res.video_id = h_video_id[best_idx];
    res.frame_id = h_frame_id[best_idx];
    res.score = (float)best_dist;

    // ---- 7) Template matching (hook)
    // Qui hai già topK candidati in h_cands e (se enable_template_match) hai h_offsets.
    // Strategia:
    //  - leggi solo i frame dei candidati (offset_bytes + frames.bin)
    //  - fai SAD/NCC e scegli best finale
    //
    // Nota: se quei K frame non stanno in VRAM insieme:
    //  - fai batch (es. 10 per volta) copiando in GPU e matchando.
    //
    // if (cfg.enable_template_match) { ... }
  }

  // ---- cleanup
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_cands));
  CUDA_CHECK(cudaFree(d_hist));
  CUDA_CHECK(cudaFree(d_hashes));

  return res;
}
