#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Lettore di un singolo frame RAW da disco.
//
// Formato atteso:
// - RGB interleaved (c=3) oppure grayscale (c=1)
// - dimensione fissa: bytes_per_frame = w*h*c
// - il file deve contenere ESATTAMENTE bytes_per_frame byte
class FrameReader {
public:
  FrameReader(int w, int h, int c);

  int w() const { return w_; }
  int h() const { return h_; }
  int c() const { return c_; }
  int bytes_per_frame() const { return bpf_; }

  // Legge un frame RAW da file e ritorna un buffer contiguo.
  // Lancia std::runtime_error se:
  // - non riesce ad aprire
  // - la size non è esattamente bytes_per_frame
  std::vector<uint8_t> read_raw_frame(const std::string& path) const;

private:
  int w_ = 0;
  int h_ = 0;
  int c_ = 0;
  int bpf_ = 0;
};
