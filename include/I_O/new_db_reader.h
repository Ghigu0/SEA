#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>

// Reader per DB snellito in formato SoA (4 file binari).
//
// Layout atteso:
//  - hashes.bin      : uint64_t[N]
//  - video_id.bin    : int32_t[N]
//  - frame_id.bin    : int32_t[N]
//  - raw_frame.bin   : uint8_t[N * bytes_per_frame]  (NON viene caricato tutto in RAM)
//
// NOTA: i path non sono più hardcoded: vengono costruiti dal Config (cartella newDB).
class SoADbReader {
public:
  // Costruisce i path dai parametri del Config (deduplicated_db_path)
  explicit SoADbReader(const struct Config& cfg);

  // Validazione dimensioni (senza caricare i file in RAM).
  // Imposta N prendendolo da hashes e controlla coerenza con gli altri file richiesti.
  void validate(bool require_meta, bool require_frames);

  std::size_t size() const { return n_; }
  bool validated() const { return validated_; }

  // Load selettivo (carica SOLO quel file in RAM)
  void load_hashes();   // carica hashes
  void load_meta();     // carica video_id + frame_id

  // Accesso ai buffer caricati
  const std::vector<uint64_t>& hashes() const { return hashes_; }
  const std::vector<int32_t>&  video_id() const { return video_id_; }
  const std::vector<int32_t>&  frame_id() const { return frame_id_; }

  bool has_hashes() const { return !hashes_.empty(); }
  bool has_meta() const { return !video_id_.empty() && !frame_id_.empty(); }

  // Utility per liberare RAM
  void clear_hashes();
  void clear_meta();

  // ============================================================
  // Accesso ai frame RAW (streaming, NO RAM full)
  // ============================================================

  // bytes per frame (frame_w * frame_h * channels)
  int bytes_per_frame() const { return bytes_per_frame_; }

  // apre raw_frame.bin (da chiamare una volta prima delle letture)
  void open_frames();

  // chiude raw_frame.bin
  void close_frames();

  bool frames_open() const { return frames_file_.is_open(); }

  // legge il frame k-esimo (indice nel DB snellito) dentro un buffer esterno
  // NOTA: offset = k * bytes_per_frame (frame di dimensione fissa)
  void read_frame(std::size_t k, uint8_t* dst, std::size_t dst_bytes);

private:
  std::size_t n_ = 0;
  bool validated_ = false;

  std::vector<uint64_t> hashes_;
  std::vector<int32_t>  video_id_;
  std::vector<int32_t>  frame_id_;

  // Path ai file SoA (costruiti dal Config)
  std::string hashes_path_;
  std::string video_id_path_;
  std::string frame_id_path_;
  std::string frames_path_;

  int bytes_per_frame_ = 0;

  // stream per raw_frame.bin (lettura a seek)
  std::ifstream frames_file_;

private:
  // helpers file
  static std::size_t file_size_bytes(const std::string& path);

  template <typename T>
  static void require_multiple_of(const std::string& path, std::size_t bytes);

  template <typename T>
  static std::vector<T> read_bin_vector(const std::string& path);

  void ensure_validated() const;
  void ensure_size_matches(std::size_t got, const char* what) const;
};
