#include <cuda_runtime.h>
#include <cstdint>



  // forceline indica al compilatore che questa non è una chiamata a funzione, ma verrà direttamente sostituita nel codice ( in modo da evitare appunto l'overhead di 
  // una chiamata a funzione )

__device__ __forceinline__ uint8_t rgb_to_luma_u8(uint8_t r, uint8_t g, uint8_t b) {
  // approx BT.601: Y ≈ (0.299R + 0.587G + 0.114B)
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

// calcola |a-b|
__device__ __forceinline__ unsigned int abs_diff(uint8_t a, uint8_t b) {
  return (a > b) ? (unsigned int)(a - b) : (unsigned int)(b - a);
}


/*Questa funzione implementa una riduzione di somma a livello di warp utilizzando le istruzioni di shuffle. 
I thread del warp cooperano per sommare i loro valori senza usare shared memory né sincronizzazioni esplicite.*/

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int v) {
 
  constexpr unsigned mask = 0xffffffffu;
  
  // questa itruzione permette di prendere il valore dal registro di un thread tramite lane+offset
  v += __shfl_down_sync(mask, v, 16);// i primi 16 thread prendono il valore v che si trova 16 lane più avanti nel warp, e lo somma al prorpio v
  v += __shfl_down_sync(mask, v, 8); // i primi 8 thread prendono il valore v che si trova 8 lane più avanti ( che rappresentavano la somma del thread x + il thread x + 16) e lo somma al proprio v
  v += __shfl_down_sync(mask, v, 4); // e così via
  v += __shfl_down_sync(mask, v, 2);
  v += __shfl_down_sync(mask, v, 1);
  return v;
}

/*
  Deduplicazione temporale basata su downsampling e SAD 

  Ogni blocco CUDA è associato a un singolo frame i del chunk.
  I thread del blocco cooperano per confrontare il frame i con il frame precedente (i-1)
  campionando un insieme fisso di punti distribuiti uniformemente sull’intero frame
  (griglia GX×GY).

  Ogni thread calcola una parte della Sum of Absolute Differences (SAD) sulla luminanza
  dei pixel campionati.
  Le somme parziali vengono poi ridotte prima a livello di warp
  tramite shuffle e successivamente a livello di blocco tramite shared memory.

  La SAD totale rappresenta una misura approssimata della variazione globale tra due
  frame consecutivi: se supera una soglia prefissata (la threshold, configurabile tramite l'opzione --threshold quando 
  si lancia il programma ) il frame viene mantenuto, altrimenti viene considerato duplicato e scartato.
*/

__global__ void dedup_kernel_downsample_sad_blockperframe(
    const uint8_t* __restrict__ d_frames,
    uint8_t* __restrict__ d_keep,
    int n,
    int w, int h, int c,
    int bytes_per_frame,
    int threshold)
{
  
  const int i = (int)blockIdx.x;
  if (i >= n) return;

  // First frame always kept
  if (i == 0) {
    if (threadIdx.x == 0) d_keep[i] = 1;
    return;
  }

  const uint8_t* cur  = d_frames + (size_t)i       * (size_t)bytes_per_frame;
  const uint8_t* prev = d_frames + (size_t)(i - 1) * (size_t)bytes_per_frame;

  // Sample grid (same as your original)
  constexpr int GX = 32;
  constexpr int GY = 18;
  constexpr int S  = GX * GY; // 576 samples

  unsigned int local_sum = 0;

  // Each thread processes samples: s = tid, tid+blockDim, ...
  for (int s = (int)threadIdx.x; s < S; s += (int)blockDim.x) {
    const int yy = s / GX;
    const int xx = s - yy * GX;

    const int y = (yy * h) / GY;
    const int x = (xx * w) / GX;

    const int idx = (y * w + x) * c;

    // Read and compute luma
    uint8_t a, b;
    if (c >= 3) {
      a = rgb_to_luma_u8(cur[idx + 0],  cur[idx + 1],  cur[idx + 2]);
      b = rgb_to_luma_u8(prev[idx + 0], prev[idx + 1], prev[idx + 2]);
    } else {
      a = cur[idx];
      b = prev[idx];
    }

    local_sum += abs_diff(a, b);
  }

  // Reduce within each warp
  unsigned int warp_sum = warp_reduce_sum(local_sum);

  // One value per warp -> reduce across warps
  __shared__ unsigned int warp_partials[32]; // enough for up to 1024 threads => 32 warps

  const int lane   = (int)(threadIdx.x & 31);
  const int warpId = (int)(threadIdx.x >> 5);

  if (lane == 0) warp_partials[warpId] = warp_sum;
  __syncthreads();

  // Final reduce by first warp
  unsigned int total = 0;
  if (warpId == 0) {
    // Number of warps in this block
    const int numWarps = ((int)blockDim.x + 31) / 32;
    total = (lane < numWarps) ? warp_partials[lane] : 0;
    total = warp_reduce_sum(total);

    if (lane == 0) {
      d_keep[i] = (total > (unsigned int)threshold) ? 1 : 0;
    }
  }
}



// per reference, viene lasciata la prima versione dell'algortimo, la quale prevedeva un thread per il calcolo di un frame ( due in realtà, in quanto per capire se un frame 
// andava tenuto o meno, veniva calcolato anche il sad del frame precedente)

/*
#include "./headers/kernel_frame_deduplication.cuh"
#include <cstdint>


Come proseguiamo (scelta concreta per v1)

Implementiamo una dedup temporale così:

Per ogni frame i all’interno del chunk, il kernel non costruisce una mini-immagine continua, ma campiona un
insieme fisso di punti distribuiti uniformemente sull’intero frame, secondo una griglia regolare di dimensione 
GX × GY (ad esempio 32×18).

I pixel campionati del frame i vengono confrontati con quelli del frame precedente (i−1) calcolando una SAD (Sum of Absolute Differences) 
sulla luminanza. La SAD risultante fornisce una misura approssimata della variazione globale tra i due frame.
Se SAD <= threshold → consideriamo il frame duplicato → keep=0
                                           Altrimenti → keep=1

Nota sull'algoritmo: si confronta il frame[i] con il frame[i-1] anche se il frame[i-1] è stato marcato come duplicato (keep=0).


  // forceline indica al compilatore che questa non è una chiamata a funzione, ma verrà direttamente sostituita nel codice ( in modo da evitare appunto l'overhead di 
  // una chiamata a funzione )
__device__ __forceinline__ uint8_t rgb_to_luma(uint8_t r, uint8_t g, uint8_t b) {
  // luminanza veloce (approx BT.601)
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

__global__ void dedup_kernel_downsample_sad(
    const uint8_t* d_frames,                   // puntatore a tutti i frame del chunk
    uint8_t* d_keep,                           // array di output che contiene un flag (un bit 0 o 1 ) per ogni frame analizzato (1 lo manteniamo, 0 lo scartiamo)
    int n,                                     // numero di frame del chunk corrente
    int w, int h, int c,                       // dimensione e canali dei frame (ipotesi di dominio: 1280:720 con 3 canali)
    int bytes_per_frame,                       // numero totale di bytes per frame
    int threshold)                             // !!! THRESHOLD: definibile anche da linea di comando con l'opzione --threshold è la soglia che vienbe usata sulla SAD. 
                                               //       se SAD <= threshold -> frame duplicato e quindi scartato, altrimenti lo si tiene 
{
  
  // mappatura basata sul metodo delle coordinate: ogni thread sarà associato a un frame
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // il primo frame del chunk va tenuto, non ha infatti un frame precedente con cui confrontarlo
  if (i == 0) { d_keep[i] = 1; return; }

  // calcolo dell'indirizzo i-esimo
  const uint8_t* cur  = d_frames + (size_t)i * (size_t)bytes_per_frame;
  // calcolo dell'indirizzo i-1 esimo
  const uint8_t* prev = d_frames + (size_t)(i - 1) * (size_t)bytes_per_frame;

  // definiamo la dimensione della "sotto-immagine"
  const int GX = 32;
  const int GY = 18;

  unsigned int sad = 0;

  // loop sulle righe ( asse verticale )
  for (int yy = 0; yy < GY; ++yy) {
    // dalle coordinate yy della sottogriglia, ottengo una coordinata reale nel frame (es: [0,GY] -> [0, h] e così via).
    // di conseguenza per yy=0,1,...17 otterremo y=0,40,80,..680 (nel caso di una immagine 720)
    // i pixel che consideriamo sono quindi di coordinate costanti per ogni frame, ma non sono pixel adiacenti, solo nel loro insieme formano una sottogriglia di 32x18
    int y = (yy * h) / GY;
    // stesso ragionamento per l'asse orizzontale 
    for (int xx = 0; xx < GX; ++xx) {
      int x = (xx * w) / GX;

      // calcoliamo quindi l'indice del pixel su cui dobbiamo calcolare la luminanza
      int idx = (y * w + x) * c;

      uint8_t a, b;
      // i frame sono nel formato RGB interleaved 
      a = rgb_to_luma(cur[idx + 0],  cur[idx + 1],  cur[idx + 2]);
      b = rgb_to_luma(prev[idx + 0], prev[idx + 1], prev[idx + 2]);
     
      // calcolo la somma delle differenze delle luminanze del singolo pixel tra il frame i e il frame i-1
      sad += (a > b) ? (a - b) : (b - a);
    }
  }

  // Decisione: se differenza totale è piccola → duplicato
  d_keep[i] = (sad > (unsigned int)threshold) ? 1 : 0;
}



, l’approccio 1 thread = 1 frame ricalcola due volte la luminanza campionata per i frame interni. 
Tuttavia il tentativo di riusare tali valori in shared memory non è praticabile perché richiederebbe 
memorizzare GX×GY campioni per molti frame del blocco, superando la shared memory disponibile, e 
inoltre non permette condivisione tra blocchi.*/












