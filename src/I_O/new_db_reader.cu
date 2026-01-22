#include "../../include/new_db_reader.cuh"

#include <fstream>
#include <stdexcept>
#include <string>

// ======================================================================
// METTI QUI I TUOI PATH COMPLETI (hardcoded)
// ======================================================================
static constexpr const char* kHashesPath  = "/ABS/PATH/TO/DB/hashes.bin";
static constexpr const char* kVideoIdPath = "/ABS/PATH/TO/DB/video_id.bin";
static constexpr const char* kFrameIdPath = "/ABS/PATH/TO/DB/frame_id.bin";
static constexpr const char* kOffsetsPath = "/ABS/PATH/TO/DB/offset_bytes.bin";
// ======================================================================


//======================================================================================
// ritorna la dimensione in byte del file indicato
std::size_t SoADbReader::file_size_bytes(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error(std::string("Cannot open file for size: ") + path);
  std::streamoff sz = f.tellg();
  if (sz < 0) throw std::runtime_error(std::string("tellg failed for: ") + path);
  return static_cast<std::size_t>(sz);
}//======================================================================================


//======================================================================================
//verificare che la dimensione di un file binario sia compatibile con il tipo T
//che pensi di leggerci dentro.
template <typename T>
void SoADbReader::require_multiple_of(const char* path, std::size_t bytes) {
  if (bytes % sizeof(T) != 0) {
    throw std::runtime_error(std::string("File size not multiple of element size (") +
                             std::to_string(sizeof(T)) + "): " + path);
  }
}//======================================================================================


// ======================================================================================
/* leggere un intero file binario e interpretarlo come array contiguo di elementi di 
 tipo T, restituendolo come std::vector<T>.
 NOTA: non verrà mai usata per leggere il file dei frame, in quanto potrebbe riempire la ram */
template <typename T>
std::vector<T> SoADbReader::read_bin_vector(const char* path) {
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
}// ======================================================================================


//========================================================================================
/*Determina quanti record contiene il DB (N) e verifica che tutti i file richiesti 
    abbiano esattamente N elementi, senza caricare nulla in RAM.*/

void SoADbReader::validate(bool require_meta, bool require_offsets) {
  // hashes definisce N
  const std::size_t bytes_h = file_size_bytes(kHashesPath);
  require_multiple_of<uint64_t>(kHashesPath, bytes_h);
  const std::size_t n = bytes_h / sizeof(uint64_t);

  if (require_meta) {
    const std::size_t bytes_v = file_size_bytes(kVideoIdPath);
    require_multiple_of<int32_t>(kVideoIdPath, bytes_v);
    const std::size_t nv = bytes_v / sizeof(int32_t);

    const std::size_t bytes_f = file_size_bytes(kFrameIdPath);
    require_multiple_of<int32_t>(kFrameIdPath, bytes_f);
    const std::size_t nf = bytes_f / sizeof(int32_t);

    if (nv != n || nf != n) {
      throw std::runtime_error("SoA size mismatch: hashes=" + std::to_string(n) +
                               " video_id=" + std::to_string(nv) +
                               " frame_id=" + std::to_string(nf));
    }
  }

  if (require_offsets) {
    const std::size_t bytes_o = file_size_bytes(kOffsetsPath);
    require_multiple_of<uint64_t>(kOffsetsPath, bytes_o);
    const std::size_t no = bytes_o / sizeof(uint64_t);

    if (no != n) {
      throw std::runtime_error("SoA size mismatch: hashes=" + std::to_string(n) +
                               " offset_bytes=" + std::to_string(no));
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
}//========================================================================================

//=========================================================================================
// FUNZIONI PRINCIPALI CHE CHIAMERA' IL FILE FRAME_RESEARCH
//=========================================================================================

// carica il file degli hash
void SoADbReader::load_hashes() {
  ensure_validated();
  hashes_ = read_bin_vector<uint64_t>(kHashesPath);
  ensure_size_matches(hashes_.size(), "hashes");
}

// carica i file che contiene il video di appartenenza e il frame_id 
void SoADbReader::load_meta() {
  ensure_validated();
  video_id_ = read_bin_vector<int32_t>(kVideoIdPath);
  frame_id_ = read_bin_vector<int32_t>(kFrameIdPath);
  ensure_size_matches(video_id_.size(), "video_id");
  ensure_size_matches(frame_id_.size(), "frame_id");
}

// carica in ram il file degli offset 
void SoADbReader::load_offsets() {
  ensure_validated();
  offset_bytes_ = read_bin_vector<uint64_t>(kOffsetsPath);
  ensure_size_matches(offset_bytes_.size(), "offset_bytes");
}

// libera memoria 
void SoADbReader::clear_hashes() {
  hashes_.clear();
  hashes_.shrink_to_fit();
}

void SoADbReader::clear_meta() {
  video_id_.clear(); video_id_.shrink_to_fit();
  frame_id_.clear(); frame_id_.shrink_to_fit();
}

void SoADbReader::clear_offsets() {
  offset_bytes_.clear();
  offset_bytes_.shrink_to_fit();
}
