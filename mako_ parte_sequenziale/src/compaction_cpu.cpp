#include "compaction_cpu.h"
#include <stdexcept>

static inline void iota_cpu(int32_t* out, int n) {
  if (!out && n > 0) throw std::runtime_error("iota_cpu: out is null");
  for (int i = 0; i < n; ++i) out[i] = i;
}

int init_kept_ids_identity_cpu(WorkspaceCPU& ws, int n) {
  if (n < 0) throw std::runtime_error("init_kept_ids_identity_cpu: n < 0");
  if (n == 0) return 0;
  if (!ws.kept_ids) throw std::runtime_error("init_kept_ids_identity_cpu: kept_ids is null");

  // equivalente a iota_kernel<<<...>>>(ws.d_kept_ids, n)
  iota_cpu(ws.kept_ids, n);
  return n;
}

int run_compaction_cpu(WorkspaceCPU& ws, int n) {
  if (n < 0) throw std::runtime_error("run_compaction_cpu: n < 0");
  if (n == 0) return 0;

  if (!ws.keep)     throw std::runtime_error("run_compaction_cpu: keep is null");
  if (!ws.kept_ids) throw std::runtime_error("run_compaction_cpu: kept_ids is null");
  if (!ws.all_ids)  throw std::runtime_error("run_compaction_cpu: all_ids is null");

  // 1) Prepara all_ids = [0..n-1]  (equivalente al tuo blocco con iota_kernel su ws.d_all_ids)
  iota_cpu(ws.all_ids, n);

  // 2) Equivalent di cub::DeviceSelect::Flagged
  int out = 0;
  for (int i = 0; i < n; ++i) {
    if (ws.keep[i]) {
      ws.kept_ids[out++] = ws.all_ids[i];
    }
  }

  // 3) In CUDA copiavi kept_count device->host; qui out è già host
  return out;
}
