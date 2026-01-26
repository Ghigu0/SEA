#pragma once
#include <cstdint>
#include "../include/config.h"

struct QueryResult {
  bool found = false;
  int32_t video_id = -1;
  int32_t frame_id = -1;
  float score = 0.0f;   // qui: hamming distance (più basso = migliore)
  int32_t best_db_index = -1; // indice nel DB deduplicato (utile per debug/template)
};

QueryResult ricerca_frame(const Config& cfg);
