// ============================================================================
// KERNEL 1) COLLECT CANDIDATES
// Scopo: dato l'hash della query (q) e tutti gli hash del DB (d_hashes),
//        selezionare i frame "abbastanza simili" (distanza di Hamming <= thresh)
//        e salvarli in un array di candidati (d_out), fino a un massimo di topk.
//
// Output:
// - d_out[pos] = { idx = i, dist = distanza } per ogni candidato accettato
// - d_count    = contatore globale di quanti candidati avremmo trovato in totale
//
// Nota importante:
// - d_count può crescere oltre topk: noi però scriviamo in d_out solo se pos < topk
//   (così non sforiamo l'array d_out).
// - l'ordine dei candidati in d_out NON è garantito (perché usiamo atomicAdd e thread paralleli).
//   Per questo poi su CPU fai std::sort per ordinarli per distanza.
// ============================================================================

#include "./headers/kernel_collect_candidates.cuh"
#include "./headers/kernel_query_utils.cuh"   // popc64: popcount su 64 bit (bitcount)

__global__ void kernel_collect_candidates(const uint64_t* __restrict__ d_hashes,
                                         int n,
                                         uint64_t q,
                                         uint8_t thresh,
                                         Cand* __restrict__ d_out,
                                         int topk,
                                         int* __restrict__ d_count)
{
  // tid = indice globale del thread (unico su tutta la griglia)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // stride = quanto "salto" tra un lavoro e l'altro quando faccio il loop
  // (pattern classico: grid-stride loop)
  int stride = blockDim.x * gridDim.x;

  // Ogni thread scansiona più elementi dell'array (i = tid, tid+stride, tid+2*stride, ...)
  for (int i = tid; i < n; i += stride) {

    // distanza di Hamming tra hash DB e hash query:
    // 1) XOR: i bit diversi diventano 1
    // 2) popcount: conto quanti bit a 1 -> numero differenze -> distanza 0..64
    uint32_t d = popc64(d_hashes[i] ^ q);

    // filtro: se dist <= soglia, questo frame è un candidato
    if (d <= (uint32_t)thresh) {

      // prendo una "posizione libera" nell'array candidati
      // atomicAdd garantisce che ogni thread ottenga una pos diversa
      int pos = atomicAdd(d_count, 1);

      // ma scrivo solo se rientro nel buffer d_out (dimensione topk)
      if (pos < topk) {
        d_out[pos].idx  = (int32_t)i;   // indice del frame nel newDB (0..N-1)
        d_out[pos].dist = (uint8_t)d;   // distanza di Hamming (0..64)
      }
    }
  }
}
