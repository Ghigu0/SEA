#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Reader per DB snellito in formato SoA (4 file binari).
// I PATH COMPLETI sono hardcoded nel .cpp, non in Config e non passati al costruttore.
//
// Layout atteso:
//  - hashes.bin        : uint64_t[N]
//  - video_id.bin      : int32_t[N]
//  - frame_id.bin      : int32_t[N]
//  - offset_bytes.bin  : uint64_t[N] (opzionale)
class SoADbReader {
public:
  SoADbReader() = default;

  // Validazione dimensioni (senza caricare i file in RAM).
  // Imposta N prendendolo da hashes e controlla coerenza con gli altri file richiesti.
  void validate(bool require_meta, bool require_offsets);

  std::size_t size() const { return n_; }
  bool validated() const { return validated_; }

  // Load selettivo (carica SOLO quel file in RAM)
  void load_hashes();   // carica hashes
  void load_meta();     // carica video_id + frame_id
  void load_offsets();  // carica offset_bytes

  // Accesso ai buffer caricati
  const std::vector<uint64_t>& hashes() const { return hashes_; }
  const std::vector<int32_t>&  video_id() const { return video_id_; }
  const std::vector<int32_t>&  frame_id() const { return frame_id_; }
  const std::vector<uint64_t>& offset_bytes() const { return offset_bytes_; }

  bool has_hashes() const { return !hashes_.empty(); }
  bool has_meta() const { return !video_id_.empty() && !frame_id_.empty(); }
  bool has_offsets() const { return !offset_bytes_.empty(); }

  // Utility per liberare RAM
  void clear_hashes();
  void clear_meta();
  void clear_offsets();

private:
  std::size_t n_ = 0;
  bool validated_ = false;

  std::vector<uint64_t> hashes_;
  std::vector<int32_t>  video_id_;
  std::vector<int32_t>  frame_id_;
  std::vector<uint64_t> offset_bytes_;

private:
  static std::size_t file_size_bytes(const char* path);

  template <typename T>
  static void require_multiple_of(const char* path, std::size_t bytes);

  template <typename T>
  static std::vector<T> read_bin_vector(const char* path);

  void ensure_validated() const;
  void ensure_size_matches(std::size_t got, const char* what) const;
};
