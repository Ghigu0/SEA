// main.cu
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _err = (call);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "[CUDA ERROR] " << cudaGetErrorString(_err)                  \
                << " (" << static_cast<int>(_err) << ") at " << __FILE__        \
                << ":" << __LINE__ << "\n";                                     \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)


// funzione da chiamare prima di una funzione che cronometra il tempo. Termina in automatico prima della return e stampa il tempo
// verosibilmente non ci servirà perchè nsight di nvidia già ci cronometra i kernel e per la CPU possiamo direttamente contare i clock del processore
struct ScopedTimer {
  std::string name;
  std::chrono::high_resolution_clock::time_point t0;
  explicit ScopedTimer(std::string n)
      : name(std::move(n)), t0(std::chrono::high_resolution_clock::now()) {}
  ~ScopedTimer() {
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "[TIMER] " << name << ": " << ms << " ms\n";
  }
};

// questa Struct servirà per definire una variabile dove salveremo tutti gli argomenti in ingresso del main
struct Config {

    /* di solito in c++ si può mettere il namespace std ed evitare std:: ogni volta,
    chat consiglia di non usarlo in progetti grossi. ignora std::, in questo caso per std::string
    è come se leggessi solo string                                                               */

    // parametri per la prima fase del caricamento del database
    std::string full_db_path;                    // path del DB da prelevare e snellire
    std::string deduplicated_db_path;            // path dove salvare il DB snellito dopo la frame deduplication
    int gpu_id = 0;                              // server per settare quale GPU usare (se ne hai più di una) nel codice è presente la funzione cudaSetDevice( gpu_id )
    bool enable_dedup = true;                    // per dire al programma se effettuare o meno la deduplication ( utile per effettuare poi dei confronti )

    // parametri per la seconda fase della ricerca del frame
    std::string query_frame_path;               // path del frame da cercare
    bool enable_index_match = true;             // in caso per fare un paragone si può disabilitare la ricerca preliminare per indici
    int topk = 50;                              // nel caso in cui sia abilitata, se non si specifica nulla per default il template matching sarà sui primi 50 risultati dagli indici
    bool enable_template_match = true;          // per abilitare il template matching

    /* questo parametro serve per abilitare o disabilitare le printf. abilitarle significa ridurre di molto le performance.
    Nei kernel CUDA preferirei non mettere proprio le printf neanche dentro ai branch sinceramente                      */
    bool verbose = false;
};

// -------------------- Arg parsing / usage --------------------

struct ParsedArgs {
  Config cfg;
  bool is_query = false;   // true => fase 2, false => fase 1
  bool show_help = false;
};

static void print_usage(const char* prog) {
  std::cout
      << "Usage:\n"
      << "  " << prog << " [build options] --full-db <PATH> --dedup-db <PATH> [common]\n"
      << "  " << prog << " [query options] --dedup-db <PATH> --q <FRAME_PATH> [common]\n\n"
      << "Common options:\n"
      << "  --gpu <id>              GPU id (default 0)\n"
      << "  --verbose               Logging verboso\n"
      << "  --help, -h              Stampa questo help\n\n"
      << "Build (fase 1) options:\n"
      << "  --full-db <PATH>        Percorso DB grezzo da prelevare/sn ellire\n"
      << "  --dedup-db <PATH>       Percorso dove salvare il DB snellito\n"
      << "  --no-dedup              Disabilita deduplication\n\n"
      << "Query (fase 2) options:\n"
      << "  --q <FRAME_PATH>        Percorso del frame da cercare\n"
      << "  --no-index              Disabilita pre-match via indici (candidati = tutti)\n"
      << "  --topk <K>              Numero candidati post-index (default 50)\n"
      << "  --no-template           Disabilita template matching\n";
}

static bool is_flag(const std::string& s, const char* f) {
  return s == f;
}

static int to_int_or_die(const std::string& s, const char* what) {
  char* end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (!end || *end != '\0') {
    std::cerr << "Invalid integer for " << what << ": " << s << "\n";
    std::exit(2);
  }
  return static_cast<int>(v);
}

static ParsedArgs parse_args(int argc, char** argv) {
  ParsedArgs out;
  Config& cfg = out.cfg;

  if (argc < 2) {
    out.show_help = true;
    return out;
  }

  // Nota: non usiamo più "Mode". Decidiamo la fase in base alla presenza di --q
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];

    if (is_flag(a, "--help") || is_flag(a, "-h")) {
      out.show_help = true;
      return out;
    }

    if (is_flag(a, "--gpu")) {
      if (++i >= argc) { std::cerr << "--gpu needs a value\n"; std::exit(2); }
      cfg.gpu_id = to_int_or_die(argv[i], "--gpu");
      continue;
    }

    if (is_flag(a, "--verbose")) {
      cfg.verbose = true;
      continue;
    }

    // Build paths
    if (is_flag(a, "--full-db")) {
      if (++i >= argc) { std::cerr << "--full-db needs a value\n"; std::exit(2); }
      cfg.full_db_path = argv[i];
      continue;
    }

    if (is_flag(a, "--dedup-db")) {
      if (++i >= argc) { std::cerr << "--dedup-db needs a value\n"; std::exit(2); }
      cfg.deduplicated_db_path = argv[i];
      continue;
    }

    if (is_flag(a, "--no-dedup")) {
      cfg.enable_dedup = false;
      continue;
    }

    // Query
    if (is_flag(a, "--q")) {
      if (++i >= argc) { std::cerr << "--q needs a value\n"; std::exit(2); }
      cfg.query_frame_path = argv[i];
      out.is_query = true;
      continue;
    }

    if (is_flag(a, "--no-index")) {
      cfg.enable_index_match = false;
      continue;
    }

    if (is_flag(a, "--topk")) {
      if (++i >= argc) { std::cerr << "--topk needs a value\n"; std::exit(2); }
      cfg.topk = to_int_or_die(argv[i], "--topk");
      continue;
    }

    if (is_flag(a, "--no-template")) {
      cfg.enable_template_match = false;
      continue;
    }

    std::cerr << "Unknown arg: " << a << "\n";
    std::exit(2);
  }

  // Se non c'è --q allora è build (fase 1) per default.
  return out;
}

static void validate_or_die(const ParsedArgs& args) {
  const Config& cfg = args.cfg;

  if (args.show_help) return;

  // dedup-db serve sempre: in build è output, in query è input (DB snellito)
  if (cfg.deduplicated_db_path.empty()) {
    std::cerr << "Missing --dedup-db <PATH>\n";
    std::exit(2);
  }

  if (!args.is_query) {
    // BUILD (fase 1)
    if (cfg.full_db_path.empty()) {
      std::cerr << "build: missing --full-db <PATH>\n";
      std::exit(2);
    }
  } else {
    // QUERY (fase 2)
    if (cfg.query_frame_path.empty()) {
      std::cerr << "query: missing --q <FRAME_PATH>\n";
      std::exit(2);
    }
    if (cfg.topk <= 0) {
      std::cerr << "query: --topk must be > 0\n";
      std::exit(2);
    }
  }
}

// -------------------- Pipeline stubs --------------------
// Qui dentro per ora metti solo "glue". La logica vera la sposterai in moduli.

struct BuildStats {
  uint64_t frames_total = 0;
  uint64_t frames_after_dedup = 0;
  uint64_t signatures_written = 0;
};

static BuildStats build_db_pipeline(const Config& cfg) {
  ScopedTimer t("build_db_pipeline");

  if (cfg.verbose) {
    std::cerr << "[BUILD] full_db_path=" << cfg.full_db_path
              << " deduplicated_db_path=" << cfg.deduplicated_db_path
              << " dedup=" << (cfg.enable_dedup ? "on" : "off")
              << "\n";
  }

  // TODO:
  // 1) load DB grezzo dal path cfg.full_db_path (es. cartella con i video)
  // 2) frame dedup (se cfg.enable_dedup)
  // 3) compute signatures + index
  // 4) write DB snellito (SoA) to cfg.deduplicated_db_path

  BuildStats st;
  return st;
}

struct QueryResult {
  bool found = false;
  int video_id = -1;
  int frame_id = -1;
  float score = 0.0f;
};

static QueryResult query_pipeline(const Config& cfg) {
  ScopedTimer t("query_pipeline");

  if (cfg.verbose) {
    std::cerr << "[QUERY] deduplicated_db_path=" << cfg.deduplicated_db_path
              << " q=" << cfg.query_frame_path
              << " index=" << (cfg.enable_index_match ? "on" : "off")
              << " topk=" << cfg.topk
              << " template=" << (cfg.enable_template_match ? "on" : "off")
              << "\n";
  }

  // TODO:
  // 1) load DB snellito da cfg.deduplicated_db_path (mmap / read)
  // 2) se cfg.enable_index_match:
  //      - compute index for query frame
  //      - candidate_search (Hamming / buckets / etc.) => topk
  //    altrimenti:
  //      - candidates = tutti i frame
  // 3) se cfg.enable_template_match:
  //      - template matching sui candidati
  //    altrimenti:
  //      - usa lo score dell'indice (o un criterio semplice) per scegliere il best
  // 4) pick best result

  QueryResult r;
  return r;
}

// -------------------- Main --------------------

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

int main(int argc, char** argv) {
  ParsedArgs args = parse_args(argc, argv);
  if (args.show_help) {
    print_usage(argv[0]);
    return 0;
  }

  validate_or_die(args);
  print_gpu_info(args.cfg.gpu_id);

  try {
    if (!args.is_query) {
      ScopedTimer t("TOTAL build");
      BuildStats st = build_db_pipeline(args.cfg);
      std::cerr << "[BUILD DONE] frames_total=" << st.frames_total
                << " frames_after_dedup=" << st.frames_after_dedup
                << " signatures_written=" << st.signatures_written << "\n";
    } else {
      ScopedTimer t("TOTAL query");
      QueryResult r = query_pipeline(args.cfg);
      if (!r.found) {
        std::cerr << "[QUERY DONE] not found\n";
        return 1;
      }
      std::cerr << "[QUERY DONE] found video_id=" << r.video_id
                << " frame_id=" << r.frame_id
                << " score=" << r.score << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "[FATAL] Exception: " << e.what() << "\n";
    return 1;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
