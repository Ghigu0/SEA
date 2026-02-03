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

//====================================================================================================================
// per cronometrare il tempo di esecuzione dell'applicazione
struct ScopedTimer {
  std::string name;
  std::chrono::high_resolution_clock::time_point t0;
  explicit ScopedTimer(std::string n): name(std::move(n)), t0(std::chrono::high_resolution_clock::now()) {}
  ~ScopedTimer() {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[TIMER] " << name << ": " << ms << " ms\n";
  }
};


//====================================================================================================================
// per cronometrare il tempo di esecuzione dell'applicazione
static void print_usage(const char* prog) {
  std::cout
      << "Usage:\n"
      << "  " << prog << " <full-db_PATH> <frame_PATH> [options]\n"
      << "Notes:\n"
      << "  The deduplicated database will be written to: ./newDB\n"
      << "Options:\n"
      << "  --gpu <id>              GPU id (default 0)\n"
      << "  --verbose               Verbose logging\n"
      << "  --topk <k>              Number of candidates for template matching (query phase)\n"
      << "  --dedup-threshold <t>   SAD threshold for frame deduplication (build phase)\n"
      << "  --chunk-frames <n>      Number of frames per chunk/batch (default 128)\n";
}


//====================================================================================================================
// funzioni per il parsing delle opzioni
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
    } else if (is_flag(a, "--dedup-threshold")) {
      if (++i >= argc) { std::cerr << "--dedup-threshold needs a value\n"; std::exit(2); }
      cfg.dedup_threshold = to_int_or_die(argv[i], "--dedup-threshold");
    } else if (is_flag(a, "--chunk-frames")) {
      if (++i >= argc) { std::cerr << "--chunk-frames needs a value\n"; std::exit(2); }
      cfg.chunk_frames = to_int_or_die(argv[i], "--chunk-frames");
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      std::exit(2);
    }
  }

  if (cfg.topk <= 0) {
    std::cerr << "--topk must be > 0\n";
    std::exit(2);
  }
  if (cfg.dedup_threshold <= 0) {
    std::cerr << "--dedup-threshold must be > 0\n";
    std::exit(2);
  }
  if (cfg.chunk_frames <= 0) {
    std::cerr << "--chunk-frames must be > 0\n";
    std::exit(2);
  }

  return cfg;
}


//====================================================================================================================
// per stampare le informazioni relative alla scheda GPU che si sta utilizzando
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
  std::cout << "\n[GPU] Using device " << gpu_id << ": " << prop.name
            << " (cc " << prop.major << "." << prop.minor << ")\n";
}



//====================================================================================================================
//main
int main(int argc, char** argv) {

  ScopedTimer total("total_time_program: ");
  //parsing degli argomenti
  Config cfg = parse_args(argc, argv);

  // le stampe avvengono solo se l'opzione verbose è attiva
  if (cfg.verbose) {

    std::cout << "\nINFORMAZIONI GENERALI ==================================================" << "\n";
    print_gpu_info(cfg.gpu_id);
    std::cout << "\n[CFG] full_db_path=" << cfg.full_db_path << "\n";
    std::cout << "[CFG] query_frame_path=" << cfg.query_frame_path << "\n";
    std::cout << "[CFG] deduplicated_db_path=" << cfg.deduplicated_db_path << "\n";
    std::cout << "[CFG] frame_w=" << cfg.frame_w << " frame_h=" << cfg.frame_h
              << " channels=" << cfg.channels << "\n";
    std::cout << "[CFG] chunk_frames=" << cfg.chunk_frames << "\n";
    std::cout << "[CFG] topk=" << cfg.topk << "\n";
    std::cout << "[CFG] dedup_threshold=" << cfg.dedup_threshold << "\n";
  }

  try {
    
    // prima fase del programma: caricamento e deduplicazione del database
    BuildStats st = carica_db(cfg);

    // non si può iniziare la seconda fase di ricerca se non è garantito che la prima fase sia finita
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cfg.verbose){
      std::cout << "\n[BUILD DONE] frames_total=" << st.frames_total
                << " frames_after_dedup=" << st.frames_after_dedup
                << " signatures_written=" << st.signatures_written << "\n";
    }
    
    //seconda fase del programma: ricerca del frame
    QueryResult r = ricerca_frame(cfg);

    // necessaria configurazione per salvare poi il frame vincitore della ricerca
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
