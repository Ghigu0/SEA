#pragma once
#include <string>

#include "../config.h"
#include "../research_types.h"
#include "../frame_research.h"

// Legge il frame vincitore da frame.bin (newDB) e lo salva su disco come PPM (P6).
// Ritorna true se tutto ok, false se r.found == false.
// Lancia std::runtime_error su errori I/O o incoerenze.
bool dump_winner_frame_ppm(const Config& cfg,
                           const QueryResult& r,
                           const std::string& out_path);
