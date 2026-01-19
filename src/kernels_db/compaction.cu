#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
#include "../../include/cuda_utils.h"
#include "../../include/compaction.cuh"
#include "../../include/workspace.h"

__global__ void iota_kernel(int32_t* out, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = i;
}

// tiene tutti i frame (ignora la deduplication, per eventuali test e paragoni)
int init_kept_ids_identity(Workspace& ws, int n) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  iota_kernel<<<grid, block>>>(ws.d_kept_ids, n);
  CUDA_CHECK(cudaGetLastError());
  return n;
}

int run_compaction_cub(Workspace& ws, int n) {

  //  Prepara ws.d_all_ids = [0..n-1]
  {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    iota_kernel<<<grid, block>>>(ws.d_all_ids, n);
    CUDA_CHECK(cudaGetLastError());
  }

  //  Dry-run: calcola quanti bytes servono per il temp storage CUB
  size_t required_temp_bytes = 0;
  cub::DeviceSelect::Flagged(
      nullptr, required_temp_bytes,
      ws.d_all_ids,     // input items (0..n-1)
      ws.d_keep,        // flags 0/1
      ws.d_kept_ids,    // output items
      ws.d_kept_count,  // output count (device)
      n);

  // Alloca/Riusa temp storage nel workspace
  if (ws.d_temp == nullptr || ws.d_temp_bytes < required_temp_bytes) {
    if (ws.d_temp) CUDA_CHECK(cudaFree(ws.d_temp));
    CUDA_CHECK(cudaMalloc(&ws.d_temp, required_temp_bytes));
    ws.d_temp_bytes = required_temp_bytes;
  }

  // Compaction con CUB ( operazione Flagged)
  cub::DeviceSelect::Flagged(
      ws.d_temp, ws.d_temp_bytes,
      ws.d_all_ids,
      ws.d_keep,
      ws.d_kept_ids,
      ws.d_kept_count,
      n);

  // Copia kept_count su host (sincrona: niente stream)
  int kept = 0;
  CUDA_CHECK(cudaMemcpy(&kept, ws.d_kept_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  return kept;
}
