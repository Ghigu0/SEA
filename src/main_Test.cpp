#include <iostream>
#include <cstdint>

// include del tuo progetto
#include "../include/config.h"
#include "../include/raw_db_reader.h"
#include "../include/db_types.h"

int main() {
    try {
        Config cfg;

        //  METTI QUI IL TUO PATH REALE
        cfg.full_db_path = "C:\\Users\\Ghigu\\Desktop\\Progetto\\database";


        //  DEVONO CORRISPONDERE AI FRAME REALI
        cfg.frame_w  = 1280;
        cfg.frame_h  = 720;
        cfg.channels = 3;   // RGB

        RawDbReader reader(cfg);

        const int chunk_frames = 10; // piccolo per test visivo
        const size_t bytes_per_frame =
            static_cast<size_t>(cfg.frame_w) * cfg.frame_h * cfg.channels;

        int chunk_idx = 0;

        while (reader.has_next()) {
            HostChunk ch = reader.next_chunk(chunk_frames, bytes_per_frame);

            std::cout << "Chunk " << chunk_idx++
                      << " | n=" << ch.n << "\n";

            for (int i = 0; i < ch.n; ++i) {
                std::cout << "  frame[" << i << "] "
                          << "video_id=" << ch.video_id[i]
                          << " frame_id=" << ch.frame_id[i]
                          << "\n";
            }
        }

        std::cout << "\n[OK] Test RawDbReader completato\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\n[ERRORE] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
