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
#include "../include/new_db_reader.cuh"  
#include "../src/kernel_research/kernel_hist_hamming.cuh"
#include "../include/research_types.h"
#include "../src/kernel_research/kernel_collect_candidates.cuh"
#include "../include/frame_reader.h"
#include "../src/kernel_research/kernel_query_ahash.cuh"

// =============================================================================================
// chiamata kernel per calcolare l'hash dell'immagine da ricercare 
  static uint64_t compute_query_hash64(const std::string& imgPath, const Config& cfg) {
  //frameReader è un oggetto che permette di leggere un frame
  FrameReader fr(cfg.frame_w, cfg.frame_h, cfg.channels);
  std::vector<uint8_t> h_frame = fr.read_raw_frame(imgPath);
  //dimensione del frame
  const int bpf = fr.bytes_per_frame();

  // allocazioni su GPU ( frame, cella 8x8 e il risultato)
  uint8_t*  d_frame  = nullptr;   // RAW frame query
  uint16_t* d_cells  = nullptr;   // 64 mean (8x8)
  uint64_t* d_hash   = nullptr;   // 1 hash64

  CUDA_CHECK(cudaMalloc(&d_frame, (size_t)bpf));
  CUDA_CHECK(cudaMalloc(&d_cells, 64u * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_hash,  sizeof(uint64_t)));

  // caricamento frame in GPU
  CUDA_CHECK(cudaMemcpy(d_frame, h_frame.data(), (size_t)bpf, cudaMemcpyHostToDevice));

  dim3 block1(256, 1, 1);
  dim3 grid1(64, 1, 1);
  dim3 block2(1, 1, 1);
  dim3 grid2(1, 1, 1);

  // produce i 64 valori
  k_query_cellmean_u16_8x8<<<grid1, block1>>>(
      d_frame,
      cfg.frame_w, cfg.frame_h, cfg.channels,
      bpf,
      d_cells
  );
  CUDA_CHECK(cudaGetLastError());
  // calcola la media globale e produce l'hash
  k_query_ahash64_from_cellmean<<<grid2, block2>>>(
      d_cells,
      d_hash
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //passo il risultato alla CPU
  uint64_t h_hash = 0;
  CUDA_CHECK(cudaMemcpy(&h_hash, d_hash, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  // clenaup
  CUDA_CHECK(cudaFree(d_hash));
  CUDA_CHECK(cudaFree(d_cells));
  CUDA_CHECK(cudaFree(d_frame));

  return h_hash;
}


//===============================================================================
// per l'individuazione della soglia minima ( guardare commento nel main )
static uint8_t choose_threshold_for_topk(const uint32_t hist[65], int topk) {
  uint64_t cum = 0;
  for (int d = 0; d <= 64; ++d) {
    cum += hist[d];
    if (cum >= (uint64_t)topk) return (uint8_t)d;
  }
  return 64;
}

// =============================================================================================
// Helper: carica un file intero (per query frame RAW già in rgb24 sul disco).
// Se la query è PNG/JPG, userai un loader immagini (stb_image / opencv / ffmpeg).
static std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open file: " + path);
  f.seekg(0, std::ios::end);
  std::streamoff sz = f.tellg();
  if (sz < 0) throw std::runtime_error("tellg failed: " + path);
  std::vector<uint8_t> data((size_t)sz);
  f.seekg(0, std::ios::beg);
  if (sz > 0) {
    f.read(reinterpret_cast<char*>(data.data()), sz);
    if (!f) throw std::runtime_error("Read failed: " + path);
  }
  return data;
}





QueryResult ricerca_frame(const Config& cfg) {

  //variabile per il risultato
  QueryResult res{};
  //inizializzo l'oggetto SoADbReader (che mi serve per leggere i file binari che compongono il nuovo database)
  SoADbReader db(cfg);
  //controlla che i contenuti (principlamente numero di byte) dei file binari siano coerenti tra di loro
  db.validate(true, cfg.enable_template_match);

  // KERNEL 
  // calcolo l'hash del frame che carica l'utente ( questa funzione chiama poi 2 KERNEL )
  const uint64_t qhash = compute_query_hash64(cfg.query_frame_path, cfg);

  // ci permette di usare db.hashes che contiene gli hash dei frame "sopravvisuti" dopo la fram dedup
  db.load_hashes();
  // in modo analogo possiamo usare db.video_id() e db.frame_id()
  db.load_meta();

  // N rappresenta il numero di frame presenti nel databse (contanto il numeo di hash di 64 bit presenti nel vettore db.hashes)
  const int N = static_cast<int>(db.hashes().size());
  //controlli: se non ci sono frame il db è vuoto
  if (N<= 0) {
    if (cfg.verbose) std::cerr << "[QUERY] Empty DB hashes\n";
    return res;
  }
  if (cfg.verbose) {
    std::cerr << "[QUERY] N=" << N << " topk=" << cfg.topk << "\n";
  }

  /* copio gli hash in memoria GPU ( non ci si pongono problemi di spazio, un hash ha dimensione di un byte)
  nel caso avessimo 1 milione di hash occuperemmo 8 MB */
  uint64_t* d_hashes = nullptr; // puntatore
  //alloco memoria nella GPU
  CUDA_CHECK(cudaMalloc(&d_hashes, (size_t)N * sizeof(uint64_t)));
  //copio i dati nella GPU
  CUDA_CHECK(cudaMemcpy(d_hashes, db.hashes().data(), (size_t)N * sizeof(uint64_t), cudaMemcpyHostToDevice));

  // cominciamo con la ricerca del frame 

  // d_hist è un istogramma delle distanze di Hamming ( vogliamo appunto verificare quanto gli hash siano simili con il nostro hash in input )
  uint32_t* d_hist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hist, 65 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_hist, 0, 65 * sizeof(uint32_t)));

  //kernel dell'istogramma 
  const int block = 256;
  int grid = (N + block - 1) / block;
  if (grid > 120) grid = 120;
  if (grid < 1)   grid = 1;
  kernel_hist_hamming<<<grid, block>>>(d_hashes, N, qhash, d_hist);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // riporto l'istogramma dalla GPU alla CPU
  uint32_t h_hist[65];
  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, 65 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  /* A partire dall’istogramma delle distanze di Hamming (0..64),
  calcoliamo la soglia minima che garantisce almeno topk frame.
  I frame con distanza <= soglia saranno considerati candidati
  per la fase successiva (template matching). */
  const uint8_t thresh = choose_threshold_for_topk(h_hist, cfg.topk);

  if (cfg.verbose){
    std::cerr << "[QUERY] Hamming threshold for topk: " << (int)thresh << "\n";
  }

  // collezionamento dei candidati 
      /* inclusa dal file header
      struct Cand {
      int32_t idx;   // indice nel newDB (0..N-1)
      uint8_t dist;  // Hamming distance
      };*/

  Cand* d_cands = nullptr;
  int* d_count = nullptr;
  // allochiamo spazio per massimo i topk candidati
  CUDA_CHECK(cudaMalloc(&d_cands, (size_t)cfg.topk * sizeof(Cand)));

  // contatore condiviso tra i thread GPU 
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  // lancio kernel per ottenere i candidati ( ogni candidato ha indice del frame nel db e la distanza di hamming)
  kernel_collect_candidates<<<grid, block>>>(d_hashes, N, qhash, thresh, d_cands, cfg.topk, d_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_count = 0;
  //riporto il su CPU, dove d_count può essere anhe più grande di topk
  CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  // prendo il minimo tra h_count e cfg.topk
  h_count = std::min(h_count, cfg.topk);

  std::vector<Cand> h_cands((size_t)h_count);
  if (h_count > 0) {
    CUDA_CHECK(cudaMemcpy(h_cands.data(), d_cands, (size_t)h_count * sizeof(Cand), cudaMemcpyDeviceToHost));
  }

  // non mi servono più 
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_cands));
  CUDA_CHECK(cudaFree(d_hist));
  CUDA_CHECK(cudaFree(d_hashes));

  //se non ho trovato candidati
  if (h_count == 0) {
    res.found = false; // non troveremo un risultato ( non abbiamo con chi fare template matching)
  } else {

  // ordiniamo i candidati, in modo da partire dal più "promettente "
  std::sort(h_cands.begin(), h_cands.end(),[](const Cand& a, const Cand& b){ return a.dist < b.dist; });

  }
    
  // vedi ultima risposta chat gpt 


 
  return res;
}
