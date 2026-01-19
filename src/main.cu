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
#include "../include/config.h"
#include "../include/cuda_utils.h"
#include "../include/raw_db_reader.h"
#include "../include/db_types.h"
#include "../include/database_loader.cuh"


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


/* inclusa dall'header
struct Config {
  
    std::string full_db_path;                    // path del DB da prelevare e snellire
    std::string deduplicated_db_path;            // path dove salvare il DB snellito dopo la frame deduplication
    int gpu_id = 0;                              // serve per settare quale GPU usare (se ne hai più di una) nel codice è presente la funzione cudaSetDevice( gpu_id )
    bool enable_dedup = true;                    // per dire al programma se effettuare o meno la deduplication ( utile per effettuare poi dei confronti )

   
    int frame_w = 1280;                          // larghezza frame (es. HD 1280x720)
    int frame_h = 720;                           // altezza frame
    int channels = 3;                            // 1=grayscale, 3=RGB
    int chunk_frames = 2048;                     // batch/chunk size (vedremo poi con nsight)
    int dedup_threshold = 10000;                     // per definire quando due frame devono essere considerati duplicati

   
    std::string query_frame_path;                // path del frame da cercare
    bool enable_index_match = true;              // in caso per fare un paragone si può disabilitare la ricerca preliminare per indici
    int topk = 50;                               // nel caso in cui sia abilitata, se non si specifica nulla per default il template matching sarà sui primi 50 risultati dagli indici
    bool enable_template_match = true;           // per abilitare il template matching

    bool verbose = false;
}; */



static void print_usage(const char* prog) {
  std::cout
      << "Usage:\n"
      << " <full-db_PATH> <dedup-db_PATH> <frame_PATH>\n"
      << "  --gpu <id>              GPU id (default 0)\n"
      << "  --verbose               Logging verboso\n"
      << "  --no-dedup              per disabilitare la deduplication "
      << "  --no-index              per disabilitare l'utilizzo degli indici "
      << "  --topk                  per specificare quanti indici salvare per proseguire con il template matching"
      << "  --no-template           per disabilitare il template matching ";
}

// guarda se uno degli argomenti è un flag o meno 
static bool is_flag(const std::string& s, const char* f) {
  return s == f;
}

// per convertire una stringa in un numero intero ( per il parsing degli argomenti del main )
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
  // default: se non passi niente, mostro help e termino
  if (argc < 4) {
    print_usage(argv[0]);
    std::exit(0);
  }
  cfg.full_db_path         = argv[1];
  cfg.deduplicated_db_path = argv[2];
  cfg.query_frame_path     = argv[3];
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (is_flag(a, "--help") || is_flag(a, "-h")) {
      print_usage(argv[0]);
      std::exit(0);
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
    if (is_flag(a, "--no-dedup")) {
      cfg.enable_dedup = false;
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

  // Validazione minima: siccome fai sempre entrambe le fasi,
  // mi aspetto che questi siano presenti.
 
  if (cfg.topk <= 0) {
    std::cerr << "--topk must be > 0\n";
    std::exit(2);
  }
  return cfg;
}





/* ########################################################################################################### 
  ricerca del frame, andrà in un file separato 

struct QueryResult {
  bool found = false;
  int video_id = -1;
  int frame_id = -1;
  float score = 0.0f;
};

static QueryResult ricerca_frame(const Config& cfg) {
  ScopedTimer t("ricerca_frame");

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
*/

//                                                   MAIN
/* ########################################################################################################### */

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

  Config cfg = parse_args(argc, argv); // verificare che gli argomenti da linea di comando siano corretti 
  print_gpu_info(cfg.gpu_id);          // stampa le caratteristiche della GPU utilizzata
  
  try {

    /* chiama la funzione build database e stampa informazioni sulla quantità di dati processati
    struct BuildStats {
      uint64_t frames_total = 0;        // quanti frame abbiamo letto
      uint64_t frames_after_dedup = 0;  // quanti frame ci sono rimasti dopo il dedup
      uint64_t signatures_written = 0;  // quante firme hai salvato ( direi che deve essere uguale a frames_after_dedup)
    }; */
      BuildStats st = carica_db(cfg);
      std::cerr << "[BUILD DONE] frames_total=" << st.frames_total
                << " frames_after_dedup=" << st.frames_after_dedup
                << " signatures_written=" << st.signatures_written << "\n";
    
    
    /* la sincornizzazione in dei kernel in realtà è a carico delle funzioni, tuttavia è estremamente importante che prima che inizi 
     la parte di ricerca frame il kernel riferito alla creazione del database sia finito  */   
    CUDA_CHECK(cudaDeviceSynchronize());


    // chiama la funzione di ricerca del frame (ancora non esiste ricerca_frame)
      QueryResult r = ricerca_frame(cfg);
      if (!r.found) {
        std::cerr << "[QUERY DONE] not found\n";
        return 1;
      }
      std::cerr << "[QUERY DONE] found video_id=" << r.video_id
                << " frame_id=" << r.frame_id
                << " score=" << r.score << "\n";
    
  } catch (const std::exception& e) {
    std::cerr << "[FATAL] Exception: " << e.what() << "\n";
    return 1;
  }

  // per sicurezza 
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;

}
