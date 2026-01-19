#pragma once
#include <cstdint>
#include <cstddef>

struct Workspace {

  int max_frames = 0;               // max frames che il worksapce elabora insieme: sarà cfg.chunk_frames per reference
  int w = 0, h = 0, c = 0;          // info che verranno prese sempre da Config
  size_t bytes_per_frame = 0;

  // buffer pe la GPU
  uint8_t*  d_frames = nullptr;     // [max_frames * bytes_per_frame] ovvero contiene tutti i frame del chunk considerato
  uint8_t*  d_keep   = nullptr;     // [max_frames] array di 0 / 1 per capire se il frame è scartato o salvato

  /*Su GPU ogni thread lavora su un indice i diverso.
    Ogni thread che ha d_keep[i] == 1 deve sapere:
    “In quale posizione dell’array compatto devo scrivere i?”
    d_pos risponde esattamente a questa domanda.*/
  int32_t*  d_pos    = nullptr;     // [max_frames] (scan output / posizioni)

  int32_t*  d_kept_ids = nullptr;   // [max_frames] lista compatta degli indici da tenere,
                                   // ovvero se keep è [1, 0, 0, 1, 1] allora d_kept_ids sarà [0, 3, 4]

  uint64_t* d_hashes = nullptr;     // [max_frames] hashes per kept (puoi anche allocare max_frames)

  // ============================================================
  //  AGGIUNTE per CUB compaction (DeviceSelect::Flagged)
  // ============================================================

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
