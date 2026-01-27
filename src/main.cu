// main.cu
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "../include/config.h"
#include "../include/cuda_utils.h"
#include "../include/database_loader.h"
#include "../include/research_types.h"
#include "../include/frame_research.h"
#include "../include/I_O/winner_frame.h"

// ------------------------------------------------------------
// Timer RAII (debug)
// ------------------------------------------------------------
struct ScopedTimer {
  std::string name;
  std::chrono::high_resolution_clock::time_point t0;

  explicit ScopedTimer(std::string n)
      : name(std::move(n)), t0(std::chrono::high_resolution_clock::now()) {}

  ~ScopedTimer() {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "[TIMER] " << name << ": " << ms << " ms\n";
  }
};

// ------------------------------------------------------------
// CLI parsing
// ------------------------------------------------------------
static void print_usage(const char* prog) {
  std::cout
      << "Usage:\n"
      << "  " << prog << " <full-db_PATH> <frame_PATH> [options]\n"
      << "Notes:\n"
      << "  The deduplicated DB will be written to: ./newDB (relative to current working directory)\n"
      << "Options:\n"
      << "  --gpu <id>          GPU id (default 0)\n"
      << "  --verbose           Logging verboso\n"
      << "  --topk <k>          Numero candidati per template matching (fase query)\n"
      << "  --no-template       Disabilita template matching (fase query)\n";
}

static bool is_flag(const std::string& s, const char* f) { return s == f; }

static int to_int_or_die(const std::string& s, const char* what) {
  char* end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (!end || *end != '\0') {
    std::cerr << "Invalid integer for " << what << ": " << s << "\n";
    std::exit(2);
  }
  return static_cast<int>(v);
}

static Config parse_args(int argc, char** argv) {
  Config cfg;

  // two positional args: full db path + frame path
  if (argc < 3) {
    print_usage(argv[0]);
    std::exit(0);
  }

  cfg.full_db_path     = argv[1];
  cfg.query_frame_path = argv[2];

  // default output for deduplicated db: ./newDB (cwd)
  std::filesystem::path exe = std::filesystem::current_path();
  std::filesystem::path out = exe / "newDB";
  cfg.deduplicated_db_path = out.string();
  std::filesystem::create_directories(out);

  // options start from argv[3]
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];

    if (is_flag(a, "--help") || is_flag(a, "-h")) {
      print_usage(argv[0]);
      std::exit(0);
    } else if (is_flag(a, "--gpu")) {
      if (++i >= argc) { std::cerr << "--gpu needs a value\n"; std::exit(2); }
      cfg.gpu_id = to_int_or_die(argv[i], "--gpu");
    } else if (is_flag(a, "--verbose")) {
      cfg.verbose = true;
    } else if (is_flag(a, "--topk")) {
      if (++i >= argc) { std::cerr << "--topk needs a value\n"; std::exit(2); }
      cfg.topk = to_int_or_die(argv[i], "--topk");
    } else if (is_flag(a, "--no-template")) {
      cfg.enable_template_match = false;
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      std::exit(2);
    }
  }

  if (cfg.topk <= 0) {
    std::cerr << "--topk must be > 0\n";
    std::exit(2);
  }

  return cfg;
}

// ------------------------------------------------------------
// GPU info
// ------------------------------------------------------------
static void print_gpu_info(int gpu_id) {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count == 0) {
    std::cerr << "No CUDA devices found\n";
    std::exit(1);
  }
  if (gpu_id < 0 || gpu_id >= count) {
    std::cerr << "Invalid --gpu " << gpu_id << " (device count=" << count << ")\n";
    std::exit(2);
  }

  CUDA_CHECK(cudaSetDevice(gpu_id));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
  std::cerr << "[GPU] Using device " << gpu_id << ": " << prop.name
            << " (cc " << prop.major << "." << prop.minor << ")\n";
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);
  print_gpu_info(cfg.gpu_id);

  // opzionale ma utile per chiarezza a runtime
  if (cfg.verbose) {
    std::cerr << "[CFG] full_db_path=" << cfg.full_db_path << "\n";
    std::cerr << "[CFG] query_frame_path=" << cfg.query_frame_path << "\n";
    std::cerr << "[CFG] deduplicated_db_path=" << cfg.deduplicated_db_path << "\n";
    std::cerr << "[CFG] topk=" << cfg.topk
              << " template_match=" << (cfg.enable_template_match ? "on" : "off") << "\n";
  }

  try {
    {
      ScopedTimer t("carica_db");
      BuildStats st = carica_db(cfg);

      // Se vuoi che il timer includa davvero il lavoro GPU, sincronizza qui:
      CUDA_CHECK(cudaDeviceSynchronize());

      std::cerr << "[BUILD DONE] frames_total=" << st.frames_total
                << " frames_after_dedup=" << st.frames_after_dedup
                << " signatures_written=" << st.signatures_written << "\n";
    }

    // TODO: quando implementi la fase query, chiamala qui.
    QueryResult r = ricerca_frame(cfg);

    // se ricerca_frame lancia kernel, questa sync aiuta a beccare errori e garantisce completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // salvo il frame vincitore
    dump_winner_frame_ppm(cfg, r, "winner.ppm");
    std::cout << "Saved winner.ppm\n";

  } catch (const std::exception& e) {
    std::cerr << "[FATAL] Exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
