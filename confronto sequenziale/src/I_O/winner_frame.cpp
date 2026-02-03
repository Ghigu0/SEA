#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "../../include/I_O/new_db_reader.h"  
#include "../../include/I_O/winner_frame.h"

static void write_ppm_rgb(const std::string& path,
                          const uint8_t* rgb,
                          int w, int h,
                          int channels) {
  if (channels != 3) {
    throw std::runtime_error("[DUMP] PPM writer supports only RGB (3 channels)");
  }

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("[DUMP] Cannot open output file: " + path);

  // PPM P6 header
  f << "P6\n" << w << " " << h << "\n255\n";
  f.write(reinterpret_cast<const char*>(rgb), (size_t)w * (size_t)h * 3);

  if (!f) throw std::runtime_error("[DUMP] Write failed: " + path);
}

bool dump_winner_frame_ppm(const Config& cfg,
                           const QueryResult& r,
                           const std::string& out_path) {
  if (!r.found) return false;

  // Reader newDB
  SoADbReader db(cfg);

  // Se vuoi essere super-sicuro (consigliato)
  db.validate(true, /*enable_template_match=*/true);

  // Per leggere frame.bin
  db.open_frames();

  const int w = cfg.frame_w;
  const int h = cfg.frame_h;
  const int c = cfg.channels;
  const size_t bpf = (size_t)w * (size_t)h * (size_t)c;

  std::vector<uint8_t> frame(bpf);

  // r.best_db_index è indice nel newDB (compatto)
  db.read_frame((size_t)r.best_db_index, frame.data(), frame.size());

  db.close_frames();

  write_ppm_rgb(out_path, frame.data(), w, h, c);
  return true;
}
