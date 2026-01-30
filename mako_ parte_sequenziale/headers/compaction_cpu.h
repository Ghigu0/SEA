#pragma once
#include <cstdint>

struct WorkspaceCPU {
  // Flags 0/1 per ogni frame (input dedup)
  const uint8_t* keep = nullptr;

  // Output: indici kept (0..n-1 filtrati)
  int32_t* kept_ids = nullptr;

  // Buffer temporaneo: all_ids (0..n-1) se vuoi replicare la pipeline identica
  // In CPU potresti anche non usarlo, ma lo teniamo per essere fedeli.
  int32_t* all_ids = nullptr;
};

// Tiene tutti i frame (ignora deduplication): kept_ids = [0..n-1], return n
int init_kept_ids_identity_cpu(WorkspaceCPU& ws, int n);

// Compaction "flagged": prepara all_ids = [0..n-1], poi filtra con keep -> kept_ids
// Return: kept_count
int run_compaction_cpu(WorkspaceCPU& ws, int n);
