#include "../include/soa_writer.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <system_error>

#pragma pack(push, 1)
struct DbSoAHeader {
  char     magic[8];      // "SEADB\0\0\0"
  uint32_t version;       // 1
  uint32_t header_bytes;  // sizeof(DbSoAHeader)
  uint64_t total_entries; // totale firme scritte
  uint32_t w, h, c;       // formato frame
  uint32_t reserved0;
};
#pragma pack(pop)

static_assert(sizeof(DbSoAHeader) == (8 + 4 + 4 + 8 + 4 + 4 + 4 + 4), "DbSoAHeader size mismatch");

struct SoaWriter::Impl {
  std::filesystem::path out_dir;
  std::ofstream f_hdr;
  std::ofstream f_hash;
  std::ofstream f_vid;
  std::ofstream f_fid;

  DbSoAHeader hdr{};
  uint64_t total_written = 0;
  bool finalized = false;

  static void must(bool ok, const std::string& msg) {
    if (!ok) throw std::runtime_error(msg);
  }
};

SoaWriter::SoaWriter(const Config& cfg, const Workspace& ws) : p_(new Impl()) {
  p_->out_dir = std::filesystem::path(cfg.deduplicated_db_path);

  std::error_code ec;
  std::filesystem::create_directories(p_->out_dir, ec);
  if (ec) {
    throw std::runtime_error("SoaWriter: create_directories failed: " + ec.message());
  }

  const auto p_hdr  = p_->out_dir / "header.bin";
  const auto p_hash = p_->out_dir / "hashes.u64";
  const auto p_vid  = p_->out_dir / "video_id.i32";
  const auto p_fid  = p_->out_dir / "frame_id.i32";

  // Build pulita: trunc
  p_->f_hdr.open(p_hdr,  std::ios::binary | std::ios::out | std::ios::trunc);
  p_->f_hash.open(p_hash, std::ios::binary | std::ios::out | std::ios::trunc);
  p_->f_vid.open(p_vid,  std::ios::binary | std::ios::out | std::ios::trunc);
  p_->f_fid.open(p_fid,  std::ios::binary | std::ios::out | std::ios::trunc);

  Impl::must((bool)p_->f_hdr && (bool)p_->f_hash && (bool)p_->f_vid && (bool)p_->f_fid,
             "SoaWriter: cannot open output files in " + p_->out_dir.string());

  std::memset(&p_->hdr, 0, sizeof(p_->hdr));
  std::memcpy(p_->hdr.magic, "SEADB", 5);
  p_->hdr.version      = 1;
  p_->hdr.header_bytes = (uint32_t)sizeof(DbSoAHeader);
  p_->hdr.total_entries = 0;
  p_->hdr.w = (uint32_t)ws.w;
  p_->hdr.h = (uint32_t)ws.h;
  p_->hdr.c = (uint32_t)ws.c;

  // Header placeholder
  p_->f_hdr.write(reinterpret_cast<const char*>(&p_->hdr), sizeof(p_->hdr));
  Impl::must((bool)p_->f_hdr, "SoaWriter: header write failed");
}

SoaWriter::~SoaWriter() {
  if (!p_) return;
  try { finalize(); } catch (...) {}
  delete p_;
  p_ = nullptr;
}

void SoaWriter::write_chunk(const DbSoAChunk& out) {
  if (!p_ || p_->finalized) {
    throw std::runtime_error("SoaWriter: write_chunk called after finalize");
  }

  const size_t n = out.hashes.size();
  if (out.video_id.size() != n || out.frame_id.size() != n) {
    throw std::runtime_error("SoaWriter: chunk vectors size mismatch");
  }
  if (n == 0) return;

  p_->f_hash.write(reinterpret_cast<const char*>(out.hashes.data()), n * sizeof(uint64_t));
  p_->f_vid.write(reinterpret_cast<const char*>(out.video_id.data()), n * sizeof(int32_t));
  p_->f_fid.write(reinterpret_cast<const char*>(out.frame_id.data()), n * sizeof(int32_t));

  Impl::must((bool)p_->f_hash && (bool)p_->f_vid && (bool)p_->f_fid,
             "SoaWriter: data write failed");

  p_->total_written += (uint64_t)n;
}

void SoaWriter::finalize() {
  if (!p_ || p_->finalized) return;

  p_->hdr.total_entries = p_->total_written;

  // riscrivi header all'inizio
  p_->f_hdr.seekp(0, std::ios::beg);
  p_->f_hdr.write(reinterpret_cast<const char*>(&p_->hdr), sizeof(p_->hdr));
  p_->f_hdr.flush();

  p_->f_hash.flush();
  p_->f_vid.flush();
  p_->f_fid.flush();

  p_->finalized = true;
}

uint64_t SoaWriter::total_written() const {
  return p_ ? p_->total_written : 0ULL;
}
