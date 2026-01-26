#include "./headers/kernel_frame_deduplication.cuh"
#include <cstdint>


/*Come proseguiamo (scelta concreta per v1)

Implementiamo una dedup temporale così:

Per ogni frame i (dentro il chunk) creiamo una mini-immagine implicita campionando una griglia fissa (es. 32×18 o 16×16)

Confrontiamo mini-frame i con mini-frame i-1 usando:

SAD = somma delle differenze assolute tra campioni

Se SAD <= threshold → consideriamo il frame duplicato → keep=0
Altrimenti → keep=1

Perché questa è robusta

compressione/rumore cambiano pixel singoli → SAD resta basso se la scena è uguale

piccoli micro-movimenti → SAD cresce gradualmente (dipende dal threshold)





*/
/* SPIEGAZIONE PARAMETRI:
    - const uint8_t* d_frames       -> contiene i frame del chunk corrente che stiamo analizzando (const perchè i frame non vanno toccati/modificati in questa funzione)
    - uint8_t* d_keep               -> array di flag che contiene valori 0/1 in modo da determinare quali frame teniamo e quali scartiamo 
    - int n                         -> numero di frame che stiamo processando
    - int bytes_per_frame           -> numero di byte per singolo frame 
    - int threshold                 -> per definire dopo quanto due frame sono considerabili uguali */


__device__ __forceinline__ uint8_t rgb_to_luma(uint8_t r, uint8_t g, uint8_t b) {
  // luminanza veloce (approx BT.601)
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

__global__ void dedup_kernel_downsample_sad(
    const uint8_t* d_frames,
    uint8_t* d_keep,
    int n,
    int w, int h, int c,
    int bytes_per_frame,
    int threshold)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Primo frame del chunk: tienilo
  if (i == 0) { d_keep[i] = 1; return; }

  const uint8_t* cur  = d_frames + (size_t)i * (size_t)bytes_per_frame;
  const uint8_t* prev = d_frames + (size_t)(i - 1) * (size_t)bytes_per_frame;

  // Downsample grid (puoi cambiare: 16x16 è più leggero, 32x18 più robusto)
  const int GX = 32;
  const int GY = 18;

  unsigned int sad = 0;

  for (int yy = 0; yy < GY; ++yy) {
    int y = (yy * h) / GY;
    for (int xx = 0; xx < GX; ++xx) {
      int x = (xx * w) / GX;
      int idx = (y * w + x) * c;

      uint8_t a, b;
      if (c >= 3) {
        a = rgb_to_luma(cur[idx + 0],  cur[idx + 1],  cur[idx + 2]);
        b = rgb_to_luma(prev[idx + 0], prev[idx + 1], prev[idx + 2]);
      } else {
        a = cur[idx];
        b = prev[idx];
      }

      sad += (a > b) ? (a - b) : (b - a);
    }
  }

  // Decisione: se differenza totale è piccola → duplicato
  d_keep[i] = (sad > (unsigned int)threshold) ? 1 : 0;
}