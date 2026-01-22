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

  // Crea la directory se non esiste (non errore se già esiste)
  std::error_code ec;
  fs::create_directories(out_dir, ec);
  must(!ec, "cannot create output directory");

  const fs::path hashes = out_dir / "hashes.bin";
  const fs::path vid    = out_dir / "video_id.bin";
  const fs::path fid    = out_dir / "frame_id.bin";

  // TRUNC qui = svuota SOLO all'inizio run (costruttore chiamato una volta)
  f_hash_.open(hashes, std::ios::binary | std::ios::out | std::ios::trunc);
  f_vid_.open (vid,    std::ios::binary | std::ios::out | std::ios::trunc);
  f_fid_.open (fid,    std::ios::binary | std::ios::out | std::ios::trunc);

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
