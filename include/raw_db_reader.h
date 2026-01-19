#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <filesystem>

#include "config.h"
#include "db_types.h"

struct RawDbReader {
  explicit RawDbReader(const Config& cfg);

  bool has_next() const;
  HostChunk next_chunk(int max_frames, size_t bytes_per_frame);

private:
  std::string root_;
  size_t bytes_per_frame_expected_ = 0;

  std::vector<std::filesystem::path> video_dirs_;

  // stato iterazione
  size_t cur_video_idx_ = 0;
  size_t cur_frame_idx_ = 0;

  // lista frame del video corrente
  std::vector<std::filesystem::path> cur_video_frames_;

  void load_video_frames_list_(size_t video_idx);
  static void read_exact_file_(const std::filesystem::path& p, uint8_t* out, size_t bytes);
};
