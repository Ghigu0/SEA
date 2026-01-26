#include "../../include/I_O/frame_reader.h"

#include <fstream>
#include <stdexcept>
#include <string>

FrameReader::FrameReader(int w, int h, int c) : w_(w), h_(h), c_(c) {
  if (w_ <= 0 || h_ <= 0) {
    throw std::runtime_error("FrameReader: invalid w/h");
  }
  if (c_ != 1 && c_ != 3) {
    throw std::runtime_error("FrameReader: channels must be 1 or 3");
  }
  // attenzione overflow int (non serve paranoie qui, ma facciamo pulito)
  const long long bpf = 1LL * w_ * h_ * c_;
  if (bpf <= 0 || bpf > (1LL << 31)) {
    throw std::runtime_error("FrameReader: bytes_per_frame overflow/invalid");
  }
  bpf_ = (int)bpf;
}

std::vector<uint8_t> FrameReader::read_raw_frame(const std::string& path) const {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("FrameReader: cannot open: " + path);
  }

  std::streamoff sz = f.tellg();
  if (sz < 0) {
    throw std::runtime_error("FrameReader: tellg failed: " + path);
  }

  if (sz != (std::streamoff)bpf_) {
    throw std::runtime_error(
        "FrameReader: wrong file size for RAW frame. got=" + std::to_string((long long)sz) +
        " expected=" + std::to_string(bpf_) + " path=" + path);
  }

  std::vector<uint8_t> buf((size_t)bpf_);

  f.seekg(0, std::ios::beg);
  f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size());
  if (!f) {
    throw std::runtime_error("FrameReader: read failed: " + path);
  }

  return buf;
}
