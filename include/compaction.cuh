#pragma once
#include <cstdint>


struct Workspace;

int init_kept_ids_identity(Workspace& ws, int n);

int run_compaction_cub(Workspace& ws, int n);
