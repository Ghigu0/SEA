#include "../../include/new_db_reader.cuh"

#include <fstream>
#include <stdexcept>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// ======================================================================
// I path NON sono più hardcoded: vengono costruiti dal Config
// ======================================================================


//======================================================================================
// ritorna la dimensione in byte del file indicato
std::size_t SoADbReader::file_size_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error(std::string("Cannot open file for size: ") + path);
  std::streamoff sz = f.tellg();
  if (sz < 0) throw std::runtime_error(std::string("tellg failed for: ") + path);
  return static_cast<std::size_t>(sz);
}
//======================================================================================


//======================================================================================
// verificare che la dimensione di un file binario sia compatibile con il tipo T
// che pensi di leggerci dentro.
template <typename T>
void SoADbReader::require_multiple_of(const std::string& path, std::size_t bytes) {
  if (bytes % sizeof(T) != 0) {
    throw std::runtime_error(std::string("File size not multiple of element size (") +
                             std::to_string(sizeof(T)) + "): " + path);
  }
}
//======================================================================================


// ======================================================================================
/* leggere un intero file binario e interpretarlo come array contiguo di elementi di
   tipo T, restituendolo come std::vector<T>.
   NOTA: non verrà mai usata per leggere il file dei frame, in quanto potrebbe
         riempire la RAM */
template <typename T>
std::vector<T> SoADbReader::read_bin_vector(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("Cannot open: ") + path);

  f.seekg(0, std::ios::end);
  std::streamoff sz = f.tellg();
  if (sz < 0) throw std::runtime_error(std::string("tellg failed: ") + path);

  const std::size_t bytes = static_cast<std::size_t>(sz);
  require_multiple_of<T>(path, bytes);

  const std::size_t n = bytes / sizeof(T);
  std::vector<T> v(n);

  f.seekg(0, std::ios::beg);
  if (bytes > 0) {
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error(std::string("Read failed: ") + path);
  }
  return v;
}
// ======================================================================================


//========================================================================================
/* Determina quanti record contiene il DB (N) e verifica che tutti i file richiesti
   abbiano esattamente N elementi, senza caricare nulla in RAM.

   - hashes.bin       -> N uint64
   - video_id.bin     -> N int32   (opzionale)
   - frame_id.bin     -> N int32   (opzionale)
   - raw_frame.bin    -> N * bytes_per_frame byte
*/
void SoADbReader::validate(bool require_meta, bool require_frames) {
  // hashes definisce N
  const std::size_t bytes_h = file_size_bytes(hashes_path_);
  require_multiple_of<uint64_t>(hashes_path_, bytes_h);
  const std::size_t n = bytes_h / sizeof(uint64_t);

  if (require_meta) {
    const std::size_t bytes_v = file_size_bytes(video_id_path_);
    require_multiple_of<int32_t>(video_id_path_, bytes_v);
    const std::size_t nv = bytes_v / sizeof(int32_t);

    const std::size_t bytes_f = file_size_bytes(frame_id_path_);
    require_multiple_of<int32_t>(frame_id_path_, bytes_f);
    const std::size_t nf = bytes_f / sizeof(int32_t);

    if (nv != n || nf != n) {
      throw std::runtime_error("SoA size mismatch: hashes=" + std::to_string(n) +
                               " video_id=" + std::to_string(nv) +
                               " frame_id=" + std::to_string(nf));
    }
  }

  if (require_frames) {
    const std::size_t bytes_fr = file_size_bytes(frames_path_);
    const std::size_t expected = n * (std::size_t)bytes_per_frame_;
    if (bytes_fr != expected) {
      throw std::runtime_error("SoA size mismatch: raw_frame.bin bytes=" +
                               std::to_string(bytes_fr) +
                               " expected=" + std::to_string(expected));
    }
  }

  n_ = n;
  validated_ = true;
}

void SoADbReader::ensure_validated() const {
  if (!validated_) {
    throw std::runtime_error("SoADbReader: call validate(...) before loading.");
  }
}

void SoADbReader::ensure_size_matches(std::size_t got, const char* what) const {
  if (got != n_) {
    throw std::runtime_error(std::string("SoADbReader: size mismatch loading ") +
                             what + ": got=" + std::to_string(got) +
                             " expected=" + std::to_string(n_));
  }
}
//========================================================================================


//=========================================================================================
// FUNZIONI PRINCIPALI CHE CHIAMERA' IL FILE FRAME_RESEARCH
//=========================================================================================


// carica il file degli hash
void SoADbReader::load_hashes() {
  ensure_validated();
  hashes_ = read_bin_vector<uint64_t>(hashes_path_);
  ensure_size_matches(hashes_.size(), "hashes");
}

// carica i file che contengono il video di appartenenza e il frame_id
void SoADbReader::load_meta() {
  ensure_validated();
  video_id_ = read_bin_vector<int32_t>(video_id_path_);
  frame_id_ = read_bin_vector<int32_t>(frame_id_path_);
  ensure_size_matches(video_id_.size(), "video_id");
  ensure_size_matches(frame_id_.size(), "frame_id");
}

// libera memoria (hash)
void SoADbReader::clear_hashes() {
  hashes_.clear();
  hashes_.shrink_to_fit();
}

// libera memoria (metadati)
void SoADbReader::clear_meta() {
  video_id_.clear(); video_id_.shrink_to_fit();
  frame_id_.clear(); frame_id_.shrink_to_fit();
}


//=========================================================================================
// GESTIONE FILE DEI FRAME (streaming, NO RAM)
//=========================================================================================


// apre raw_frame.bin (una sola volta)
void SoADbReader::open_frames() {
  ensure_validated();
  if (frames_file_.is_open()) return;

  frames_file_.open(frames_path_, std::ios::binary);
  if (!frames_file_) {
    throw std::runtime_error("SoADbReader: cannot open raw_frame.bin");
  }
}

// chiude raw_frame.bin
void SoADbReader::close_frames() {
  if (frames_file_.is_open()) {
    frames_file_.close();
  }
}

// legge il frame k-esimo (indice nel DB snellito)
// NOTA: offset = k * bytes_per_frame (frame di dimensione fissa)
void SoADbReader::read_frame(std::size_t k, uint8_t* dst, std::size_t dst_bytes) {
  ensure_validated();

  if (!frames_file_.is_open()) {
    throw std::runtime_error("SoADbReader: call open_frames() before read_frame()");
  }

  if (k >= n_) {
    throw std::runtime_error("SoADbReader: frame index out of range");
  }

  if (dst_bytes < (std::size_t)bytes_per_frame_) {
    throw std::runtime_error("SoADbReader: destination buffer too small");
  }

  const std::size_t offset = k * (std::size_t)bytes_per_frame_;
  frames_file_.seekg((std::streamoff)offset, std::ios::beg);
  if (!frames_file_) {
    throw std::runtime_error("SoADbReader: seekg failed");
  }

  frames_file_.read(reinterpret_cast<char*>(dst),
                    (std::streamsize)bytes_per_frame_);
  if (!frames_file_) {
    throw std::runtime_error("SoADbReader: read failed");
  }
}
