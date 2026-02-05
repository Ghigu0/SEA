// ============================================================================
// KERNEL 2) HIST HAMMING
// Scopo: costruire l'istogramma delle distanze di Hamming tra query hash q
//        e tutti gli hash nel DB.
//        L'istogramma ha 65 bin: distanze 0..64.
//
// Perché serve:
// - tu poi scegli la soglia "thresh" tale che i frame con dist<=thresh
//   siano almeno topk (così non rischi di restare senza candidati).
//
// Ottimizzazione:
// - invece di fare atomicAdd direttamente su d_hist65 (memoria globale) per ogni hash,
//   ogni blocco accumula prima in shared memory sh[65] (molto più veloce),
//   poi alla fine fa il merge su globale.
// ============================================================================

#include "./headers/kernel_hist_hamming.cuh"
#include "./headers/kernel_query_utils.cuh"   // popc64
#include <cuda_runtime.h>
#include <cstdint>

__global__ void kernel_hist_hamming(
    const uint64_t* __restrict__ d_hashes,
    int n,
    uint64_t q,
    uint32_t* __restrict__ d_hist65)
{
  // shared memory: istogramma locale per blocco (65 bin)
  __shared__ uint32_t sh[65];

  // 1) inizializzo sh[] a zero (tutti i thread collaborano)
  for (int i = threadIdx.x; i < 65; i += blockDim.x) {
    sh[i] = 0;
  }
  __syncthreads(); // mi assicuro che sh sia completamente azzerato

  // tid/stride per grid-stride loop
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // 2) scorro tutti gli hash e aggiorno l'istogramma in shared
  for (int i = tid; i < n; i += stride) {
    uint32_t d = popc64(d_hashes[i] ^ q); // distanza 0..64
    // atomicAdd in shared: più veloce di global (e riduce contesa globale)
    atomicAdd(&sh[d], 1u);
  }

  __syncthreads(); // prima di copiare su globale, sh deve essere "completo"

  // 3) merge: sommo l'istogramma locale del blocco dentro l'istogramma globale
  for (int i = threadIdx.x; i < 65; i += blockDim.x) {
    if (sh[i]) {
      atomicAdd(&d_hist65[i], sh[i]);
    }
  }
}
