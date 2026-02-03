// cuda_utils.h  (CPU-only stub)
#pragma once

// In build CPU-only CUDA non esiste.
// Manteniamo CUDA_CHECK solo per compatibilità sintattica.

#include <stdexcept>
#include <string>

#define CUDA_CHECK(x) \
  do {               \
    (void)(x);       \
  } while (0)
