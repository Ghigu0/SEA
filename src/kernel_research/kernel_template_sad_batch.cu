// ============================================================================
// KERNEL 4) TEMPLATE MATCHING (SAD) A BATCH
// Scopo: dopo che hai ristretto la ricerca usando gli hash (Hamming),
//        fai una verifica "più precisa" confrontando i pixel del frame query
//        con i pixel dei frame candidati.
//
// Qui usi SAD (Sum of Absolute Differences):
// SAD = somma_j | query[j] - cand[j] |  su tutti i byte del frame (bpf bytes)
//
// Interpretazione:
// - SAD piccolo => immagini più simili
// - SAD grande  => molto diverse
//
// Config di lancio prevista (come nel tuo main):
// - grid.x  = count  (1 blocco per frame candidato nel batch)
// - block.x = 256    (i thread si dividono i byte da confrontare)
// Ogni blocco produce 1 score in d_scores[frame_i].
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include "./headers/kernel_template_sad_batch.cuh"

// abs diff su byte, restituito in uint32 (per evitare overflow di somme)
__device__ __forceinline__ uint32_t uabsdiff_u8(uint8_t a, uint8_t b) {
  return (a > b) ? (uint32_t)(a - b) : (uint32_t)(b - a);
}

__global__ void kernel_template_sad_batch(
    const uint8_t* __restrict__ d_query,
    const uint8_t* __restrict__ d_batch,
    int bpf,
    int count,
    uint32_t* __restrict__ d_scores)
{
  // 1 blocco = 1 frame candidato (frame_i)
  const int frame_i = (int)blockIdx.x;
  if (frame_i >= count) return;

  // puntatore al frame candidato i-esimo dentro il batch
  // (frames contigui: cand0, cand1, ..., ognuno lungo bpf bytes)
  const uint8_t* cand = d_batch + (size_t)frame_i * (size_t)bpf;

  // somma parziale del thread (ognuno accumula la sua parte)
  uint32_t local = 0;

  // ogni thread confronta byte j = threadIdx.x, threadIdx.x+blockDim.x, ...
  for (int j = (int)threadIdx.x; j < bpf; j += (int)blockDim.x) {
    local += uabsdiff_u8(d_query[j], cand[j]);
  }

  // -------------- riduzione nel blocco --------------
  // sommo tutti i "local" dei thread per ottenere SAD totale del frame
  __shared__ uint32_t sh[256]; // valido perché nel main usi block=256
  sh[threadIdx.x] = local;
  __syncthreads();

  // riduzione a potenze di 2 fino a sh[0]
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sh[threadIdx.x] += sh[threadIdx.x + offset];
    }
    __syncthreads();
  }

  // thread 0 del blocco scrive lo score finale del frame candidato
  if (threadIdx.x == 0) {
    d_scores[frame_i] = sh[0];
  }
}
