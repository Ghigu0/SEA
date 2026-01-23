#include "../../include/new_db_writer.h"

#include <stdexcept>
#include <string>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

void NewDbWriter::must(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(std::string("NewDbWriter: ") + msg);
}

NewDbWriter::NewDbWriter(const Config& cfg) {
  // Cartella output decisa nel main (es: "<working_dir>/newDB" o path assoluto)
  fs::path out_dir = cfg.deduplicated_db_path.empty()
      ? fs::path("newDB")
      : fs::path(cfg.deduplicated_db_path);

  std::error_code ec;
  fs::create_directories(out_dir, ec);
  must(!ec, "cannot create output directory");

  const fs::path hashes = out_dir / "hashes.bin";
  const fs::path vid    = out_dir / "video_id.bin";
  const fs::path fid    = out_dir / "frame_id.bin";
  const fs::path frames = out_dir / "raw_frame.bin";  

  // TRUNC qui = svuota SOLO all'inizio run (costruttore chiamato una volta)
  f_hash_.open(hashes, std::ios::binary | std::ios::out | std::ios::trunc);
  f_vid_.open (vid,    std::ios::binary | std::ios::out | std::ios::trunc);
  f_fid_.open (fid,    std::ios::binary | std::ios::out | std::ios::trunc);
  f_frames_.open(frames, std::ios::binary | std::ios::out | std::ios::trunc); 

  must((bool)f_hash_,   "cannot open hashes.bin");
  must((bool)f_vid_,    "cannot open video_id.bin");
  must((bool)f_fid_,    "cannot open frame_id.bin");
  must((bool)f_frames_, "cannot open raw_frame.bin");
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

  // Se stai scrivendo anche i frame RAW, devono esserci e la size deve tornare
  if (!out.frames_raw.empty()) {
    const size_t expected = n * (size_t)out.bytes_per_frame;
    if ((size_t)out.bytes_per_frame == 0) {
      throw std::runtime_error("NewDbWriter: bytes_per_frame == 0 but frames_raw provided");
    }
    if (out.frames_raw.size() != expected) {
      throw std::runtime_error("NewDbWriter: frames_raw size mismatch (expected n*bytes_per_frame)");
    }
  }

  // 1) Scrittura SoA contigua
  f_hash_.write(reinterpret_cast<const char*>(out.hashes.data()),
                (std::streamsize)(n * sizeof(uint64_t)));
  must((bool)f_hash_, "write failed on hashes.bin");

  f_vid_.write(reinterpret_cast<const char*>(out.video_id.data()),
               (std::streamsize)(n * sizeof(int32_t)));
  must((bool)f_vid_, "write failed on video_id.bin");

  f_fid_.write(reinterpret_cast<const char*>(out.frame_id.data()),
               (std::streamsize)(n * sizeof(int32_t)));
  must((bool)f_fid_, "write failed on frame_id.bin");

  // 2) Scrittura dei frame RAW (se presenti)
  if (!out.frames_raw.empty()) {
    f_frames_.write(reinterpret_cast<const char*>(out.frames_raw.data()),
                    (std::streamsize)out.frames_raw.size());
    must((bool)f_frames_, "write failed on raw_frame.bin");
  }

  total_written_ += (uint64_t)n;
}

void NewDbWriter::finalize() {
  if (finalized_) return;

  if (f_hash_)   f_hash_.flush();
  if (f_vid_)    f_vid_.flush();
  if (f_fid_)    f_fid_.flush();
  if (f_frames_) f_frames_.flush();

  if (f_hash_.is_open())   f_hash_.close();
  if (f_vid_.is_open())    f_vid_.close();
  if (f_fid_.is_open())    f_fid_.close();
  if (f_frames_.is_open()) f_frames_.close();

  finalized_ = true;
}
