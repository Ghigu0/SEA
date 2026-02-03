#pragma once
#include <cstdint>

// abs diff su uint8 (equivalente a uabsdiff_u8 device)
uint32_t uabsdiff_u8_cpu(uint8_t a, uint8_t b);

// Calcola SAD per un batch di frame candidati (sequenziale).
// Input:
//  - query: buffer del frame query (bpf bytes)
//  - batch: buffer con "count" frame contigui, ciascuno lungo bpf bytes
//  - bpf: bytes per frame
//  - count: numero di frame nel batch
// Output:
//  - scores: array di count elementi (uint32), scores[i] = SAD(query, batch[i])
void template_sad_batch_cpu(
    const uint8_t* query,
    const uint8_t* batch,
    int bpf,
    int count,
    uint32_t* scores);

// Variante comoda: SAD tra query e un singolo candidato
uint32_t template_sad_single_cpu(
    const uint8_t* query,
    const uint8_t* cand,
    int bpf);
