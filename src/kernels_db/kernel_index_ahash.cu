#include <cuda_runtime.h>
#include <cstdint>

#include "../../include/cuda_utils.h"
#include "./headers/kernel_index_ahash.cuh"

// forceline fa si che non sia una vera chiamata a funzione, serve solo per rendere il codice più leggibile
__device__ __forceinline__ uint8_t rgb_to_luma_u8(uint8_t r, uint8_t g, uint8_t b) {
  // Rec.601 approx: (77R + 150G + 29B) >> 8
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}


/* primo kernel per il calcolo dell'hash di un frame: concretamente, per ogni frame che abbiamo tenuto
dopo la frame deduplication, calcola una rappresentazione compatta 8x8 (divide l'immagine in 64 celle con 8 colonne x 8 righe)
per ogni cella calcola la media della luminanza dei pixel contenuti e salva il risultato in d_cell_mean_u16 */
__global__ void k_downsample8x8_cellmean_u16_kept(
    const uint8_t* __restrict__ d_frames,   // chunk di frame
    const int32_t* __restrict__ d_kept_ids, // indice dei frame da elaborare relativo al chun
    int kept,                               // numero dei frame da elaborare 
    int w, int h, int c,                    // dimensioni dei frame da elaborare
    int bytes_per_frame,                    // bytes per frame
    uint16_t* __restrict__ d_cell_mean_u16  // array di 64 valori * kept
) {
  int j = (int)blockIdx.x;   // 0..kept-1 (indice compatto)
  int cell_id = (int)blockIdx.y; // 0..63
  if (j >= kept || cell_id >= 64) return;

  int i = (int)d_kept_ids[j]; // indice nel chunk originale
  const uint8_t* frame = d_frames + (size_t)i * (size_t)bytes_per_frame;

  int cx = cell_id & 7;
  int cy = cell_id >> 3;

  // Celle 8x8: gestiamo anche w/h non multipli di 8 usando l'ultima cella "piu larga"
  int cell_w = w / 8;
  int cell_h = h / 8;

  int x0 = cx * cell_w;
  int y0 = cy * cell_h;
  int x1 = (cx == 7) ? w : (x0 + cell_w);
  int y1 = (cy == 7) ? h : (y0 + cell_h);

  int rw = x1 - x0;
  int rh = y1 - y0;
  int total = rw * rh;
  if (total <= 0) {
    if (threadIdx.x == 0) d_cell_mean_u16[(size_t)j * 64u + (size_t)cell_id] = 0;
    return;
  }

  uint32_t local_sum = 0;

  for (int t = (int)threadIdx.x; t < total; t += (int)blockDim.x) {
    int dy = t / rw;
    int dx = t - dy * rw;

    int x = x0 + dx;
    int y = y0 + dy;

    size_t p = ((size_t)y * (size_t)w + (size_t)x) * (size_t)c;

    uint8_t lum;
    if (c == 1) {
      lum = frame[p];
    } else {
      // assume RGB interleaved
      uint8_t r = frame[p + 0];
      uint8_t g = frame[p + 1];
      uint8_t b = frame[p + 2];
      lum = rgb_to_luma_u8(r, g, b);
    }
    local_sum += (uint32_t)lum;
  }

  __shared__ uint32_t sh[256];
  sh[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    uint32_t sum = sh[0];
    uint16_t mean = (uint16_t)(sum / (uint32_t)total);
    d_cell_mean_u16[(size_t)j * 64u + (size_t)cell_id] = mean;
  }
}


/* l'output del kernel "k_downsample8x8_cellmean_u16_kept" produce PER OGNI FRAME 64 valori uint16.
Il compito di questo kernel è di creare un unico valore a 64 bit ( l'has che vogliamo ottenere ) a partire dai 64 valori
iniziali*/

/*
const uint16_t* d_cell_mean_u16 tiene le media delle celle 8x8 di ogni frame 
int kep numero di frame validi (per sapere quanti hash calcolare)
uint64_t* d_out_hashes dove salvare il risultato

*/
__global__ void k_ahash64_from_cellmean_kept( const uint16_t* __restrict__ d_cell_mean_u16, int kept, uint64_t* __restrict__ d_out_hashes ) {
  
  // metodo di mappatura delle coordinate
  int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  /* per i thread che non hanno dati su cui lavorare ( l'ultimo blocco della
   grid sostanzialmente) */
  if (j >= kept) return;

  const uint16_t* cells = d_cell_mean_u16 + (size_t)j * 64u;

  uint32_t sum = 0;

  #pragma unroll
  for (int k = 0; k < 64; k++) sum += (uint32_t)cells[k];
  uint16_t mean = (uint16_t)(sum / 64u);

  uint64_t h = 0ull;

  /*confronta ciascuna delle 64 celle con la media globale e codifica il 
  risultato in un hash binario a 64 bit, un bit per cella.*/

  #pragma unroll
  for (int k = 0; k < 64; k++) {
    uint64_t bit = (cells[k] > mean) ? 1ull : 0ull;
    h |= (bit << (uint64_t)k);
  }

  d_out_hashes[j] = h;
}
