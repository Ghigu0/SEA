#pragma once
#include <string>

struct Config {

    /* di solito in c++ si può mettere il namespace std ed evitare std:: ogni volta,
    chat consiglia di non usarlo in progetti grossi. ignora std::, in questo caso per std::string
    è come se leggessi solo string                                                               */

    // parametri per la prima fase del caricamento del database
    std::string full_db_path;                    // path del DB da prelevare e snellire
    std::string deduplicated_db_path;            // path dove salvare il DB snellito dopo la frame deduplication ( per ragioni di efficienza sarà un file binario )
    int gpu_id = 0;                              // serve per settare quale GPU usare (se ne hai più di una) nel codice è presente la funzione cudaSetDevice( gpu_id )
    bool enable_dedup = true;                    // per dire al programma se effettuare o meno la deduplication ( utile per effettuare poi dei confronti )

    // parametri comuni (formato frame + chunking)
    int frame_w = 1280;                          // larghezza frame (es. HD 1280x720)
    int frame_h = 720;                           // altezza frame
    int channels = 3;                            // 1=grayscale, 3=RGB
    int chunk_frames = 2048;                     // batch/chunk size (vedremo poi con nsight)
    int dedup_threshold = 10000;                     // per definire quando due frame devono essere considerati duplicati

    // parametri per la seconda fase della ricerca del frame
    std::string query_frame_path;                // path del frame da cercare
    bool enable_index_match = true;              // in caso per fare un paragone si può disabilitare la ricerca preliminare per indici
    int topk = 50;                               // nel caso in cui sia abilitata, se non si specifica nulla per default il template matching sarà sui primi 50 risultati dagli indici
    bool enable_template_match = true;           // per abilitare il template matching

    /* questo parametro serve per abilitare o disabilitare le printf. abilitarle significa ridurre di molto le performance.
    Nei kernel CUDA preferirei non mettere proprio le printf neanche dentro ai branch sinceramente                      */
    bool verbose = false;
};


struct BuildStats {
  uint64_t frames_total = 0;        // quanti frame abbiamo letto
  uint64_t frames_after_dedup = 0;  // quanti frame ci sono rimasti dopo il dedup
  uint64_t signatures_written = 0;  // quante firme hai salvato ( direi che deve essere uguale a frames_after_dedup)
};