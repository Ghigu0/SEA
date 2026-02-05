// ============================================================================
// KERNEL 3) QUERY HASH (aHash 64 bit)
// Qui ci sono due kernel:
//  A) k_query_cellmean_u16_8x8  -> produce 64 valori: media luminanza per cella 8x8
//  B) k_query_ahash64_from_cellmean -> calcola la media globale e crea l'hash 64-bit
//
// Pipeline (classica aHash):
// 1) riduco immagine in 8x8 "blocchi" (non ridimensiono con interpolazione: faccio medie per blocco)
// 2) calcolo la luminanza media di ogni cella -> 64 valori
// 3) faccio la media globale dei 64 valori
// 4) bit k = 1 se cell[k] > media_globale, altrimenti 0
//
// Risultato: hash robusto a piccole variazioni e compressione, utile per ricerca "approx".
// ============================================================================

#include "./headers/kernel_query_ahash.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// helper device: RGB -> luminanza (approx BT.601) in interi veloci
__device__ __forceinline__ uint8_t rgb_to_luma_u8(uint8_t r, uint8_t g, uint8_t b) {
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

// ----------------------------------------------------------------------------
// A) Produce 64 medie: una per ciascuna cella dell'8x8
//
// Config di lancio prevista:
// - grid.x = 64  (1 blocco = 1 cella)
// - block.x = 256 (thread che collaborano per sommare i pixel della cella)
//
// Ogni blocco:
// - identifica la cella (cell_id)
// - scorre i pixel di quella cella in parallelo
// - somma luminanze
// - riduce in shared
// - thread0 scrive la media in d_cell_mean_u16_64[cell_id]
// ----------------------------------------------------------------------------
__global__ void k_query_cellmean_u16_8x8(
    const uint8_t* d_frame,
    int w, int h, int c,
    int bytes_per_frame,
    uint16_t* d_cell_mean_u16_64)
{
  // cella gestita da questo blocco (0..63)
  int cell_id = (int)blockIdx.x;
  if (cell_id >= 64) return;

  // coord cella (cx, cy) in griglia 8x8
  int cx = cell_id & 7;    // modulo 8
  int cy = cell_id >> 3;   // /8

  // dimensione base di una cella (approssimata): w/8 e h/8
  // gli ultimi blocchi (cx==7 o cy==7) "prendono il resto" per coprire tutta l'immagine
  int cell_w = w / 8;
  int cell_h = h / 8;

  // bounding box della cella
  int x0 = cx * cell_w;
  int y0 = cy * cell_h;
  int x1 = (cx == 7) ? w : (x0 + cell_w);
  int y1 = (cy == 7) ? h : (y0 + cell_h);

  int rw = x1 - x0;      // larghezza reale cella (può essere un po' più grande nell'ultimo blocco)
  int rh = y1 - y0;      // altezza reale cella
  int total = rw * rh;   // numero totale di pixel nella cella

  // somma parziale per thread
  uint32_t local_sum = 0;

  // ogni thread prende più pixel con stride blockDim.x
  for (int t = (int)threadIdx.x; t < total; t += (int)blockDim.x) {
    int dy = t / rw;
    int dx = t - dy * rw;
    int x = x0 + dx;
    int y = y0 + dy;

    // offset nel frame: (y*w + x) * c
    size_t p = ((size_t)y * (size_t)w + (size_t)x) * (size_t)c;

    // luminanza: se grayscale (c==1) prendo il byte, altrimenti converto RGB->luma
    uint8_t lum;
    if (c == 1) lum = d_frame[p];
    else        lum = rgb_to_luma_u8(d_frame[p + 0], d_frame[p + 1], d_frame[p + 2]);

    local_sum += (uint32_t)lum;
  }

  // riduzione in shared: sommo tutti i local_sum dei thread del blocco
  __shared__ uint32_t sh[256]; // assumiamo blockDim.x <= 256
  sh[threadIdx.x] = local_sum;
  __syncthreads();

  // riduzione a potenze di 2 (classica)
  for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) {
      sh[threadIdx.x] += sh[threadIdx.x + s];
    }
    __syncthreads();
  }

  // thread 0: calcola media e la scrive
  if (threadIdx.x == 0) {
    d_cell_mean_u16_64[cell_id] =
        (total > 0) ? (uint16_t)(sh[0] / (uint32_t)total) : 0;
  }
}

// ----------------------------------------------------------------------------
// B) Trasforma i 64 valori in hash 64-bit
//
// Config prevista: 1 blocco, 1 thread (qui scegli semplicemente 1 thread).
// - calcolo la media globale dei 64 valori
// - imposto bit k se cell[k] > media
// ----------------------------------------------------------------------------
__global__ void k_query_ahash64_from_cellmean(
    const uint16_t* d_cell_mean_u16_64,
    uint64_t* d_out_hash)
{
  // qui decidiamo volutamente: un solo thread fa tutto (costo trascurabile)
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  // media globale dei 64 valori
  uint32_t sum = 0;
  #pragma unroll
  for (int k = 0; k < 64; ++k) {
    sum += (uint32_t)d_cell_mean_u16_64[k];
  }
  uint16_t mean = (uint16_t)(sum / 64u);

  // costruzione hash: bit k = 1 se cell[k] > mean
  uint64_t h = 0ull;
  #pragma unroll
  for (int k = 0; k < 64; ++k) {
    h |= ((uint64_t)(d_cell_mean_u16_64[k] > mean) << (uint64_t)k);
  }

  *d_out_hash = h;
}
