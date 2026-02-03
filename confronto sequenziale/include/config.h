#pragma once
#include <string>
#include <cstdint>

struct Config {
    // Parametri per la prima fase del caricamento del database
    std::string full_db_path;         // path del DB da prelevare e snellire
    std::string deduplicated_db_path; // path dove salvare il i file binari relativi al nuovo DB
    int gpu_id = 0;                   // quale GPU usare (cudaSetDevice(gpu_id))


    // Parametri comuni (formato frame + chunking)
    int frame_w = 1280;               // larghezza frame (es. 1280x720)
    int frame_h = 720;                // altezza frame
    int channels = 3;                 // 1=grayscale, 3=RGB
    int chunk_frames = 128;           // batch/chunk size

   
    //giochiamo intorno a questo valore qua, per ora nei 2 video uguale, salva circa la metà  
    int dedup_threshold = 8000;

    // path del frame da cercare
    std::string query_frame_path;     

    // numero di candidati da passare al template matching
    int topk = 50;                    

    // Debug/log: in CUDA le printf possono distruggere le performance
    bool verbose = false;
};

struct BuildStats {
    uint64_t frames_total = 0;        // quanti frame abbiamo letto
    uint64_t frames_after_dedup = 0;  // quanti frame sono rimasti dopo il dedup
    uint64_t signatures_written = 0;  // quante firme/hash salvate (tipicamente = frames_after_dedup)
};
