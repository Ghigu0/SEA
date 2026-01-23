#pragma once
#include <cstdint>
#include <fstream>

#include "../include/config.h"
#include "../include/db_types.h"

// Writer DB snellito in formato SoA su 3 file binari.
class NewDbWriter {
public:
  explicit NewDbWriter(const Config& cfg);
  ~NewDbWriter();

  NewDbWriter(const NewDbWriter&) = delete;
  NewDbWriter& operator=(const NewDbWriter&) = delete;
 
  // Scrive un chunk SoA in coda ai file.
  void write_chunk(const DbSoAChunk& out);

  // Flush + close espliciti (opzionali, il distruttore li chiama comunque)
  void finalize();

  uint64_t total_written() const { return total_written_; }

private:
  std::ofstream f_hash_;
  std::ofstream f_vid_;
  std::ofstream f_fid_;
  std::ofstream f_frames_;

  uint64_t total_written_ = 0;
  bool finalized_ = false;

private:
  static void must(bool ok, const char* msg);
};
