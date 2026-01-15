// database_loader.cu
// Scheletro “fase build DB”: load -> dedup -> compaction -> index -> write (SoA)
// Nota: è uno scheletro: i kernel e l’I/O vero li metti nei rispettivi moduli.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

// ============================================================
//  Utils
// ============================================================
#define CUDA_CHECK(x)                                                      \
  do {                                                                     \
    cudaError_t err__ = (x);                                               \
    if (err__ != cudaSuccess) {                                            \
      throw std::runtime_error(std::string("CUDA error: ") +               \
                               cudaGetErrorString(err__) +                \
                               " @ " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                                      \
  } while (0)

// ============================================================
//  Config + Stats (adatta ai tuoi)
// ============================================================
struct Config {
  std::string full_db_path;          // cartella/video list ecc.
  std::string deduplicated_db_path;  // output SoA (hashes + metadati)

  int frame_w = 640;
  int frame_h = 360;
  int channels = 1;                 // 1=grayscale, 3=RGB
  int chunk_frames = 2048;          // batch/chunk size
  bool enable_dedup = true;

  // parametri dedup/index (placeholder)
  int dedup_threshold = 0;
};

struct BuildStats {
  uint64_t frames_total = 0;
  uint64_t frames_after_dedup = 0;
  uint64_t signatures_written = 0;
};

// ============================================================
//  Layout SoA output (quello che scriverai su disco)
// ============================================================
struct DbSoAChunk {
  // Per i "kept" del chunk
  std::vector<uint64_t> hashes;   // [kept]
  std::vector<int32_t>  video_id; // [kept]
  std::vector<int32_t>  frame_id; // [kept]
  // opzionale: offset, timestamp, ecc.
  // std::vector<uint64_t> offset;
};

// ============================================================
//  Workspace (buffer riusabili) - qui lo definiamo nello stesso file
// ============================================================
struct Workspace {
  // dimensionamento
  int max_frames = 0;
  int w = 0, h = 0, c = 0;
  size_t bytes_per_frame = 0;

  // --- GPU buffers ---
  uint8_t*  d_frames = nullptr;     // [max_frames * bytes_per_frame]
  uint8_t*  d_keep   = nullptr;     // [max_frames] 0/1
  int32_t*  d_pos    = nullptr;     // [max_frames] (scan output / posizioni)
  int32_t*  d_kept_ids = nullptr;   // [max_frames] lista compatta degli indici kept
  uint64_t* d_hashes = nullptr;     // [max_frames] hashes per kept (puoi anche allocare max_frames)

  // --- Host pinned (per scaricare veloce) ---
  uint64_t* h_hashes = nullptr;     // [max_frames]
  int32_t*  h_kept_ids = nullptr;   // [max_frames]

  // --- CUDA objects ---
  cudaStream_t stream = nullptr;

  // --- temp memory per scan/compaction (placeholder) ---
  void*  d_temp = nullptr;
  size_t d_temp_bytes = 0;

  Workspace() = default;
};

// Inizializza workspace (alloca una volta)
static void workspace_init(Workspace& ws, const Config& cfg) {
  ws.max_frames = cfg.chunk_frames;
  ws.w = cfg.frame_w;
  ws.h = cfg.frame_h;
  ws.c = cfg.channels;
  ws.bytes_per_frame = static_cast<size_t>(ws.w) * ws.h * ws.c;

  CUDA_CHECK(cudaStreamCreate(&ws.stream));

  CUDA_CHECK(cudaMalloc(&ws.d_frames,   ws.max_frames * ws.bytes_per_frame));
  CUDA_CHECK(cudaMalloc(&ws.d_keep,     ws.max_frames * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_pos,      ws.max_frames * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_kept_ids, ws.max_frames * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_hashes,   ws.max_frames * sizeof(uint64_t)));

  CUDA_CHECK(cudaMallocHost(&ws.h_hashes,   ws.max_frames * sizeof(uint64_t)));
  CUDA_CHECK(cudaMallocHost(&ws.h_kept_ids, ws.max_frames * sizeof(int32_t)));

  // d_temp/d_temp_bytes: lo dimensioni quando integri CUB (DeviceScan/DeviceSelect)
  ws.d_temp = nullptr;
  ws.d_temp_bytes = 0;

  std::cerr << "[Workspace] init max_frames=" << ws.max_frames
            << " bytes_per_frame=" << ws.bytes_per_frame << "\n";
}

// Libera workspace (RAII manuale)
static void workspace_destroy(Workspace& ws) {
  if (ws.d_temp)     cudaFree(ws.d_temp);
  if (ws.d_hashes)   cudaFree(ws.d_hashes);
  if (ws.d_kept_ids) cudaFree(ws.d_kept_ids);
  if (ws.d_pos)      cudaFree(ws.d_pos);
  if (ws.d_keep)     cudaFree(ws.d_keep);
  if (ws.d_frames)   cudaFree(ws.d_frames);

  if (ws.h_kept_ids) cudaFreeHost(ws.h_kept_ids);
  if (ws.h_hashes)   cudaFreeHost(ws.h_hashes);

  if (ws.stream)     cudaStreamDestroy(ws.stream);

  ws = Workspace{};
}

// ============================================================
//  I/O: placeholder (tu li implementi davvero altrove)
// ============================================================

// Un chunk CPU: frame bytes contigui + metadati associati
struct HostChunk {
  std::vector<uint8_t> frames;   // size = n * bytes_per_frame
  std::vector<int32_t> video_id; // size = n
  std::vector<int32_t> frame_id; // size = n
  int n = 0;
};

// Lettore DB grezzo (placeholder)
struct RawDbReader {
  explicit RawDbReader(const Config&) {}

  bool has_next() const {
    // TODO: implementa iterazione su video/frame
    return false;
  }

  HostChunk next_chunk(int max_frames, size_t bytes_per_frame) {
    HostChunk ch;
    // TODO: carica fino a max_frames dal DB grezzo
    // ch.n = ...
    // ch.frames.resize(ch.n * bytes_per_frame)
    // ch.video_id.resize(ch.n)
    // ch.frame_id.resize(ch.n)
    return ch;
  }
};

// Writer SoA (placeholder)
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
//  Kernel stubs (dichiarazioni) - definiscili in kernels/*.cu
// ============================================================

// d_frames: [n * bytes_per_frame], produce d_keep: [n] 0/1
__global__ void dedup_kernel_stub(const uint8_t* /*d_frames*/,
                                  uint8_t* /*d_keep*/,
                                  int /*n*/,
                                  int /*bytes_per_frame*/,
                                  int /*threshold*/) {
  // TODO: implementa dedup vero (temporale o altra logica)
}

// index solo sui kept_ids: produce d_hashes[k]
__global__ void index_kernel_stub(const uint8_t* /*d_frames*/,
                                  const int32_t* /*d_kept_ids*/,
                                  uint64_t* /*d_hashes*/,
                                  int /*kept*/,
                                  int /*bytes_per_frame*/) {
  // TODO: implementa hash/firma vera
}

// ============================================================
//  Passaggi pipeline (upload / dedup / compaction / index / download)
// ============================================================

static void upload_frames(Workspace& ws, const HostChunk& ch) {
  const size_t bytes = static_cast<size_t>(ch.n) * ws.bytes_per_frame;
  CUDA_CHECK(cudaMemcpyAsync(ws.d_frames, ch.frames.data(), bytes,
                             cudaMemcpyHostToDevice, ws.stream));
}

// Dedup: riempie d_keep
static void run_dedup(Workspace& ws, const Config& cfg, int n) {
  // Azzera keep (opzionale, dipende dal kernel)
  CUDA_CHECK(cudaMemsetAsync(ws.d_keep, 0, n * sizeof(uint8_t), ws.stream));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);

  dedup_kernel_stub<<<grid, block, 0, ws.stream>>>(
      ws.d_frames, ws.d_keep, n,
      static_cast<int>(ws.bytes_per_frame),
      cfg.dedup_threshold);

  CUDA_CHECK(cudaGetLastError());
}

// Compaction: da keep[] ottieni kept_ids[] e kept_count
// Qui metto UNO scheletro: in pratica userai CUB (DeviceSelect::Flagged oppure scan+scatter).
static int run_compaction_stub(Workspace& ws, int n) {
  // TODO reale: usare CUB.
  // Per ora: finto (tiene tutto)
  // Riempie kept_ids = [0..n-1]
  std::vector<int32_t> tmp(n);
  for (int i = 0; i < n; ++i) tmp[i] = i;
  CUDA_CHECK(cudaMemcpyAsync(ws.d_kept_ids, tmp.data(), n * sizeof(int32_t),
                             cudaMemcpyHostToDevice, ws.stream));
  return n;
}

static void run_index(Workspace& ws, int kept) {
  dim3 block(256);
  dim3 grid((kept + block.x - 1) / block.x);

  index_kernel_stub<<<grid, block, 0, ws.stream>>>(
      ws.d_frames, ws.d_kept_ids, ws.d_hashes,
      kept, static_cast<int>(ws.bytes_per_frame));

  CUDA_CHECK(cudaGetLastError());
}

static void download_hashes(Workspace& ws, int kept) {
  CUDA_CHECK(cudaMemcpyAsync(ws.h_hashes, ws.d_hashes,
                             kept * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, ws.stream));
  CUDA_CHECK(cudaMemcpyAsync(ws.h_kept_ids, ws.d_kept_ids,
                             kept * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, ws.stream));
}

// Pack output SoA per writer (CPU side)
static DbSoAChunk build_soa_chunk_from_results(const HostChunk& in,
                                               const Workspace& ws,
                                               int kept) {
  DbSoAChunk out;
  out.hashes.resize(kept);
  out.video_id.resize(kept);
  out.frame_id.resize(kept);

  for (int j = 0; j < kept; ++j) {
    const int32_t i = ws.h_kept_ids[j]; // indice dentro il chunk originale
    out.hashes[j] = ws.h_hashes[j];
    out.video_id[j] = in.video_id[i];
    out.frame_id[j] = in.frame_id[i];
  }
  return out;
}

// ============================================================
//  Entry point della fase: carica DB + dedup + index + write
// ============================================================
BuildStats carica_db(const Config& cfg) {
  Workspace ws;
  workspace_init(ws, cfg);

  RawDbReader reader(cfg);
  SoaWriter writer(cfg);

  BuildStats stats{};

  try {
    while (reader.has_next()) {
      HostChunk ch = reader.next_chunk(cfg.chunk_frames, ws.bytes_per_frame);
      if (ch.n <= 0) break;

      stats.frames_total += static_cast<uint64_t>(ch.n);

      // 1) upload
      upload_frames(ws, ch);

      // 2) dedup
      int kept = ch.n;
      if (cfg.enable_dedup) {
        run_dedup(ws, cfg, ch.n);

        // 3) compaction (CUB in produzione)
        kept = run_compaction_stub(ws, ch.n);
      } else {
        kept = run_compaction_stub(ws, ch.n); // o riempi kept_ids = [0..n-1]
      }

      stats.frames_after_dedup += static_cast<uint64_t>(kept);

      // 4) index sui kept
      run_index(ws, kept);
      stats.signatures_written += static_cast<uint64_t>(kept);

      // 5) download risultati
      download_hashes(ws, kept);

      // sync solo per usare ws.h_* su CPU
      CUDA_CHECK(cudaStreamSynchronize(ws.stream));

      // 6) build SoA e write
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

// ============================================================
//  Nota: se vuoi esporre la funzione al main, metti un header
//  database_loader.cuh con: BuildStats carica_db(const Config&);
// ============================================================
