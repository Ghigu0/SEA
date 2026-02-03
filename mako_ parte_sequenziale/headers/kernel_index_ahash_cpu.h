#pragma once
#include <cstdint>

// Converte RGB->luminanza (stessa formula del kernel CUDA)
uint8_t rgb_to_luma_u8_cpu(uint8_t r, uint8_t g, uint8_t b);

// Step 1 (equivalente a k_downsample8x8_cellmean_u16_kept):
// Per ogni frame "kept", calcola 64 medie di luminanza (8x8) e le salva in out_cell_mean_u16.
// Layout output: out_cell_mean_u16[j*64 + cell_id], con j in [0..kept-1], cell_id in [0..63].
void downsample8x8_cellmean_u16_kept_cpu(
    const uint8_t* frames,        // chunk di frame RGB interleaved
    const int32_t* kept_ids,      // indici dei frame validi nel chunk
    int kept,                    // numero frame validi
    int w, int h, int c,          // dimensioni frame (c deve essere 3)
    int bytes_per_frame,          // bytes per frame (w*h*c)
    uint16_t* out_cell_mean_u16   // output: 64*kept
);

// Step 2 (equivalente a k_ahash64_from_cellmean_kept):
// Converte le 64 medie per frame in hash 64-bit.
// Layout input: cell_mean_u16[j*64 + k]
// Layout output: out_hashes[j]
void ahash64_from_cellmean_kept_cpu(
    const uint16_t* cell_mean_u16,
    int kept,
    uint64_t* out_hashes
);

// Funzione “comoda” che fa entrambi i passi:
// frames + kept_ids -> out_hashes (e opzionalmente out_cell_mean_u16 se vuoi salvarla per debug)
void compute_ahash64_kept_cpu(
    const uint8_t* frames,
    const int32_t* kept_ids,
    int kept,
    int w, int h, int c,
    int bytes_per_frame,
    uint64_t* out_hashes,
    uint16_t* optional_cell_mean_u16 // può essere nullptr: in quel caso usa un buffer temporaneo interno
);
