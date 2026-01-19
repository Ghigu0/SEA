#pragma once

#include <cstdint>
#include <vector>

// Chunk in formato SoA (Structure of Arrays) del DB "snellito":
// hashes + metadati per ogni frame tenuto dopo dedup/compaction.
struct DbSoAChunk {
  std::vector<uint64_t> hashes;        // hash64 del frame deduplicato
  std::vector<int32_t>  video_id;      // id del video nel DB originale
  std::vector<int32_t>  frame_id;      // id del frame nel DB originale
  std::vector<uint64_t> offset_bytes;  // offset nel file binario del nuovo DB
};

// Chunk host con i frame grezzi + metadati (input della pipeline):
// tipicamente letto dal RawDbReader, poi copiato in GPU.
struct HostChunk {
  std::vector<uint8_t>  frames;    // size = n * bytes_per_frame
  std::vector<int32_t>  video_id;  // size = n
  std::vector<int32_t>  frame_id;  // size = n
  int n = 0;                       // numero di frame nel chunk
};
