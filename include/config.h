#pragma once
#include <string>
#include <cstdint>

struct Config {
    // Parametri per la prima fase del caricamento del database
    std::string full_db_path;         // path del DB da prelevare e snellire
    std::string deduplicated_db_path; // path dove salvare il DB snellito dopo la frame dedup (file binario)
    int gpu_id = 0;                   // quale GPU usare (cudaSetDevice(gpu_id))
    bool enable_dedup = true;         // abilita/disabilita deduplication

    // Parametri comuni (formato frame + chunking)
    int frame_w = 1280;               // larghezza frame (es. 1280x720)
    int frame_h = 720;                // altezza frame
    int channels = 3;                 // 1=grayscale, 3=RGB
    int chunk_frames = 128;          // batch/chunk size

    // Soglia dedup: dipende dal kernel (es. SAD). Va TARATA con test reali.
    // Default iniziale "ragionevole" per non essere né troppo permissivo né troppo rigido.
    int dedup_threshold = 10000;

    // Parametri per la seconda fase (ricerca del frame)
    std::string query_frame_path;     // path del frame da cercare
    bool enable_index_match = true;   // abilita ricerca preliminare via indici/hash
    int topk = 50;                    // numero di candidati da passare al template matching
    bool enable_template_match = true;// abilita template matching

    // Debug/log: in CUDA le printf possono distruggere le performance
    bool verbose = false;
};

struct BuildStats {
    uint64_t frames_total = 0;        // quanti frame abbiamo letto
    uint64_t frames_after_dedup = 0;  // quanti frame sono rimasti dopo il dedup
    uint64_t signatures_written = 0;  // quante firme/hash salvate (tipicamente = frames_after_dedup)
};
