#pragma once
#include <cstdint>
#include <string>
#include <filesystem>

#include "config.h"
#include "db_types.h"
#include "workspace.h"

// Writer SoA: scrive arrays separati (hashes/video_id/frame_id) + header
class SoaWriter {
public:
  SoaWriter(const Config& cfg, const Workspace& ws);
  ~SoaWriter();

  SoaWriter(const SoaWriter&) = delete;
  SoaWriter& operator=(const SoaWriter&) = delete;

  void write_chunk(const DbSoAChunk& out);
  void finalize();                 // idempotente
  uint64_t total_written() const;  // utile per debug/log

private:
  struct Impl;
  Impl* p_;
};
