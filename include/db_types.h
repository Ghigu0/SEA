#pragma once
#include <vector>
#include <cstdint>

struct DbSoAChunk {
  
  std::vector<uint64_t> hashes;             // del frame deduplicato
  std::vector<int32_t>  video_id;           // id del video nel db originale
  std::vector<int32_t>  frame_id;           // id del frame nel db originale
  std::vector<uint64_t> offset_bytes;       // offset nel file binario del nuovo DB 
 
};


struct HostChunk {
  std::vector<uint8_t> frames;   // size = n * bytes_per_frame
  std::vector<int32_t> video_id; // size = n
  std::vector<int32_t> frame_id; // size = n
  int n = 0;
};