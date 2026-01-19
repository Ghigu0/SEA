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
#include <filesystem>
#include <fstream>
#include <algorithm>

#include "../include/config.h"
#include "../include/cuda_utils.h"
#include "../include/db_types.h"
#include "../include/raw_db_reader.h"   // <-- necessario: RawDbReader + HostChunk
#include "./kernels_db/kernel_frame_deduplication.cuh"
#include "../include/workspace.h"
#include "../include/compaction.cuh"
#include "../src/kernels_db/kernel_index_ahash.cuh"
/* =========================================================================================================== 
  
strutture dati usate (dichiarate negli header)

struct Config {

    std::string full_db_path;                    // path del DB da prelevare e snellire
    std::string deduplicated_db_path;            // path dove salvare il DB snellito dopo la frame deduplication
    int gpu_id = 0;                              // serve per settare quale GPU usare (se ne hai più di una) nel codice è presente la funzione cudaSetDevice( gpu_id )
    bool enable_dedup = true;                    // per dire al programma se effettuare o meno la deduplication ( utile per effettuare poi dei confronti )
    int frame_w = 1280;                          // larghezza frame (es. HD 1280x720)
    int frame_h = 720;                           // altezza frame
    int channels = 3;                            // 1=grayscale, 3=RGB
    int chunk_frames = 2048;                     // batch/chunk size (vedremo poi con nsight)
    int dedup_threshold = 0;                     // per definire quando due frame devono essere considerati duplicati
    std::string query_frame_path;                // path del frame da cercare
    bool enable_index_match = true;              // in caso per fare un paragone si può disabilitare la ricerca preliminare per indici
    int topk = 50;                               // nel caso in cui sia abilitata, se non si specifica nulla per default il template matching sarà sui primi 50 risultati dagli indici
    bool enable_template_match = true;           // per abilitare il template matching
    bool verbose = false;
};

struct BuildStats {
  uint64_t frames_total = 0;        // quanti frame abbiamo letto
  uint64_t frames_after_dedup = 0;  // quanti frame ci sono rimasti dopo il dedup
  uint64_t signatures_written = 0;  // quante firme hai salvato ( direi che deve essere uguale a frames_after_dedup)
};

struct DbSoAChunk {
  std::vector<uint64_t> hashes;
  std::vector<int32_t>  video_id;
  std::vector<int32_t>  frame_id;
  std::vector<uint64_t> offset_bytes;
};

struct HostChunk {
  std::vector<uint8_t> frames;   // size = n * bytes_per_frame
  std::vector<int32_t> video_id; // size = n
  std::vector<int32_t> frame_id; // size = n
  int n = 0;
};


struct Workspace {

  int max_frames = 0;               // max frames che il worksapce elabora insieme: sarà cfg.chunk_frames per reference
  int w = 0, h = 0, c = 0;          // info che verranno prese sempre da Config
  size_t bytes_per_frame = 0;

  // buffer pe la GPU
  uint8_t*  d_frames = nullptr;     // [max_frames * bytes_per_frame] ovvero contiene tutti i frame del chunk considerato
  uint8_t*  d_keep   = nullptr;     // [max_frames] array di 0 / 1 per capire se il frame è scartato o salvato

  int32_t*  d_pos    = nullptr;     // [max_frames] (scan output / posizioni)

  int32_t*  d_kept_ids = nullptr;   // [max_frames] lista compatta degli indici da tenere,
                                   // ovvero se keep è [1, 0, 0, 1, 1] allora d_kept_ids sarà [0, 3, 4]

  uint64_t* d_hashes = nullptr;     // [max_frames] hashes per kept (puoi anche allocare max_frames)


  int32_t*  d_all_ids = nullptr;    // [max_frames] array con [0..n-1] (input items per CUB)
  int32_t*  d_kept_count = nullptr; // [1] numero di elementi selezionati (kept) scritto da CUB (su device)

  // --- Host pinned (per scaricare veloce) ---
  uint64_t* h_hashes = nullptr;     // per salvare le firme di quelli che tieni
  int32_t*  h_kept_ids = nullptr;   // per salvare gli id compatti di quelli che tieni (lato host )

  // buffer di appoggio temporanei (usato anche da CUB)
  void*  d_temp = nullptr;
  size_t d_temp_bytes = 0;

  uint16_t* d_cell_mean_u16 = nullptr; // [max_frames * 64] temp per aHash

};

============================================================================================================*/




/*==========================================================================================*/
/*Inizializiamo il workspace, allocando sulla GPU gli spazi di memoria che ci serviranno. 
L'allocazione avviene prima dell'esecuzione degli algortimi, i quali essendo iterativi
 comportrebbero una continua allocazione e deallocazione della memoria
*/
static void workspace_init(Workspace& ws, const Config& cfg) {
  ws.max_frames = cfg.chunk_frames;
  ws.w = cfg.frame_w;
  ws.h = cfg.frame_h;
  ws.c = cfg.channels;
  ws.bytes_per_frame = static_cast<size_t>(ws.w) * ws.h * ws.c;

  CUDA_CHECK(cudaMalloc(&ws.d_frames,   ws.max_frames * ws.bytes_per_frame));
  CUDA_CHECK(cudaMalloc(&ws.d_keep,     ws.max_frames * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_pos,      ws.max_frames * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_kept_ids, ws.max_frames * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_hashes,   ws.max_frames * sizeof(uint64_t)));

  // cudaMallocHost permette di allocare della memoria sull'host NON SWAPPABILE dal SO
  CUDA_CHECK(cudaMallocHost(&ws.h_hashes,   ws.max_frames * sizeof(uint64_t)));
  CUDA_CHECK(cudaMallocHost(&ws.h_kept_ids, ws.max_frames * sizeof(int32_t)));

  CUDA_CHECK(cudaMalloc(&ws.d_all_ids,    ws.max_frames * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_kept_count, sizeof(int32_t)));

  // per l'algoritmo di hashing
  CUDA_CHECK(cudaMalloc(&ws.d_cell_mean_u16, (size_t)ws.max_frames * 64u * sizeof(uint16_t)));


  // d_temp/d_temp_bytes: lo dimensioni quando integri CUB (DeviceScan/DeviceSelect)
  ws.d_temp = nullptr;
  ws.d_temp_bytes = 0;

  std::cerr << "[Workspace] init max_frames=" << ws.max_frames
            << " bytes_per_frame=" << ws.bytes_per_frame << "\n";
}
/*==========================================================================================*/


/*==========================================================================================*/
// deallocazione del Workspace
static void workspace_destroy(Workspace& ws) {
  if (ws.d_temp)     cudaFree(ws.d_temp);
  if (ws.d_hashes)   cudaFree(ws.d_hashes);
  if (ws.d_kept_ids) cudaFree(ws.d_kept_ids);
  if (ws.d_pos)      cudaFree(ws.d_pos);
  if (ws.d_keep)     cudaFree(ws.d_keep);
  if (ws.d_frames)   cudaFree(ws.d_frames);
  if (ws.h_kept_ids) cudaFreeHost(ws.h_kept_ids);
  if (ws.h_hashes)   cudaFreeHost(ws.h_hashes);
  if (ws.d_kept_count) cudaFree(ws.d_kept_count);
  if (ws.d_all_ids)    cudaFree(ws.d_all_ids);
  if (ws.d_cell_mean_u16) cudaFree(ws.d_cell_mean_u16);


  ws = Workspace{};
}
/*==========================================================================================*/



/*=========================================================================================*/
// va tolto, è un placeholder
struct SoaWriter {
  explicit SoaWriter(const Config&) {
    // TODO: apri file/crea cartella/inizializza header
  }
  void write_chunk(const DbSoAChunk& out) {
    // TODO: scrivi hashes + metadati su disco (SoA)
    (void)out;
  }
};




/*====================================================================================================*/
// carica i frame di un chunk in GPU 
static void upload_frames(Workspace& ws, const HostChunk& ch) {
  const size_t bytes = static_cast<size_t>(ch.n) * ws.bytes_per_frame;
  CUDA_CHECK(cudaMemcpyAsync(ws.d_frames, ch.frames.data(), bytes, cudaMemcpyHostToDevice, 0));
}
/*====================================================================================================*/


/*====================================================================================================*/
// chiama il kernel per la deduplicazione dei frame
static void run_dedup(Workspace& ws, const Config& cfg, int n) {
  // Azzera keep (opzionale, dipende dal kernel)
  CUDA_CHECK(cudaMemsetAsync(ws.d_keep, 0, n * sizeof(uint8_t), 0));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);

  // Kernel su default stream (nessun parametro stream)
 dedup_kernel_downsample_sad<<<grid, block>>>(
    ws.d_frames,
    ws.d_keep,
    n,
    ws.w, ws.h, ws.c,
    static_cast<int>(ws.bytes_per_frame),
    cfg.dedup_threshold
);
  CUDA_CHECK(cudaGetLastError());
}
/*=====================================================================================================*/


/*==================================================================================================== */
// kernel degli hash
static void run_index(Workspace& ws, int kept) {
  if (kept <= 0) return;

  // 1) Kernel 1: calcola mean per ciascuna delle 64 celle (8x8) per ogni frame kept
  // grid: (kept, 64, 1)  -> un blocco per (frame_kept, cella)
  // block: 256 thread -> riduzione nel blocco (shared[256])
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

  // 2) Kernel 2: costruisce hash64 confrontando ciascuna cella con la media globale
  dim3 block2(256, 1, 1);
  dim3 grid2((unsigned)((kept + (int)block2.x - 1) / (int)block2.x), 1u, 1u);

  k_ahash64_from_cellmean_kept<<<grid2, block2>>>(
      ws.d_cell_mean_u16,
      kept,
      ws.d_hashes
  );
  CUDA_CHECK(cudaGetLastError());
}
/*==================================================================================================== */


/*==================================================================================================== */
// trasferisco dalla GPU alla CPU le informazioni che costituiscono il nuovo database
static void download_hashes(Workspace& ws, int kept) {
  CUDA_CHECK(cudaMemcpyAsync(ws.h_hashes, ws.d_hashes,
                             kept * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, 0));
  CUDA_CHECK(cudaMemcpyAsync(ws.h_kept_ids, ws.d_kept_ids,
                             kept * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, 0));
}
/*===================================================================================================== */


/*===================================================================================================== */
// costruisco il chunk di informazioni da salvare in memoria
// ( considerando quindi solo i frame che ci servono)
static DbSoAChunk build_soa_chunk_from_results(const HostChunk& in, const Workspace& ws, int kept) {
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
/*==================================================================================================== */


// entry point
BuildStats carica_db(const Config& cfg) {
  // Se vuoi usare gpu_id, fallo subito qui
  CUDA_CHECK(cudaSetDevice(cfg.gpu_id));

  Workspace ws;
  workspace_init(ws, cfg);
  RawDbReader reader(cfg);
  // ancora da implementare, sarà un file esterno 
  SoaWriter writer(cfg);
  BuildStats stats{};

  try {
    while (reader.has_next()) {
      HostChunk ch = reader.next_chunk(cfg.chunk_frames, ws.bytes_per_frame);
      if (ch.n <= 0) break;

      // incrementiamo il contatore dei frame processati
      stats.frames_total += static_cast<uint64_t>(ch.n);
      // carico i frame in GPU
      upload_frames(ws, ch);
      // fase di deduplicazione e compattazione
      int kept = ch.n;
      if (cfg.enable_dedup) {
        run_dedup(ws, cfg, ch.n);
        kept = run_compaction_cub(ws, ch.n);
      } else {
        kept = run_compaction_cub(ws, ch.n); // o riempi kept_ids = [0..n-1]
      }
      stats.frames_after_dedup += static_cast<uint64_t>(kept);
      // calcolo degli hash sui frame filtrati
      run_index(ws, kept);
      stats.signatures_written += static_cast<uint64_t>(kept);
      //GPU -> RAM degli hash
      download_hashes(ws, kept);
      // garantisce che i trasferimenti di memoria siano avvenuti con successo
      CUDA_CHECK(cudaDeviceSynchronize());
      // 6) build SoA e write
      DbSoAChunk out = build_soa_chunk_from_results(ch, ws, kept);
      // scrive in file binari diversi le informazioni di un DbSoaChunk
      writer.write_chunk(out);

    }
  } catch (...) {
    workspace_destroy(ws);
    throw;
  }

  workspace_destroy(ws);
  return stats;
}

