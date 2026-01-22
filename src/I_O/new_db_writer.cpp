#include "../../include/new_db_writer.h"

#include <stdexcept>
#include <string>

// ======================================================================
// METTI QUI I TUOI PATH COMPLETI (hardcoded) - UN SOLO PUNTO DA CAMBIARE
// ======================================================================
static constexpr const char* kHashesPath  = "./../../newDB/hashes.bin";
static constexpr const char* kVideoIdPath = "./../../newDB/video_id.bin";
static constexpr const char* kFrameIdPath = "./../../newDB/frame_id.bin";
// ======================================================================

void NewDbWriter::must(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(std::string("NewDbWriter: ") + msg);
}

NewDbWriter::NewDbWriter(const Config& /*cfg*/) {
  // Build “pulita”: trunc (riscrive da zero)
  f_hash_.open(kHashesPath,  std::ios::binary | std::ios::out | std::ios::trunc);
  f_vid_.open (kVideoIdPath, std::ios::binary | std::ios::out | std::ios::trunc);
  f_fid_.open (kFrameIdPath, std::ios::binary | std::ios::out | std::ios::trunc);

  must((bool)f_hash_, "cannot open hashes.bin");
  must((bool)f_vid_,  "cannot open video_id.bin");
  must((bool)f_fid_,  "cannot open frame_id.bin");
}

NewDbWriter::~NewDbWriter() {
  try { finalize(); } catch (...) {}
}

void NewDbWriter::write_chunk(const DbSoAChunk& out) {
  if (finalized_) throw std::runtime_error("NewDbWriter: write_chunk after finalize");

  const size_t n = out.hashes.size();
  if (out.video_id.size() != n || out.frame_id.size() != n) {
    throw std::runtime_error("NewDbWriter: chunk vectors size mismatch (hashes/video_id/frame_id)");
  }
  if (n == 0) return;

  // Scrittura SoA contigua
  f_hash_.write(reinterpret_cast<const char*>(out.hashes.data()),
                (std::streamsize)(n * sizeof(uint64_t)));
  must((bool)f_hash_, "write failed on hashes.bin");

  f_vid_.write(reinterpret_cast<const char*>(out.video_id.data()),
               (std::streamsize)(n * sizeof(int32_t)));
  must((bool)f_vid_, "write failed on video_id.bin");

  f_fid_.write(reinterpret_cast<const char*>(out.frame_id.data()),
               (std::streamsize)(n * sizeof(int32_t)));
  must((bool)f_fid_, "write failed on frame_id.bin");

  total_written_ += (uint64_t)n;
}

void NewDbWriter::finalize() {
  if (finalized_) return;

  if (f_hash_) f_hash_.flush();
  if (f_vid_)  f_vid_.flush();
  if (f_fid_)  f_fid_.flush();

  if (f_hash_.is_open()) f_hash_.close();
  if (f_vid_.is_open())  f_vid_.close();
  if (f_fid_.is_open())  f_fid_.close();

  finalized_ = true;
}
