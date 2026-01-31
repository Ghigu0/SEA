#include "./headers/kernel_frame_deduplication.cuh"
#include <cstdint>


/*Come proseguiamo (scelta concreta per v1)

Implementiamo una dedup temporale così:

Per ogni frame i all’interno del chunk, il kernel non costruisce una mini-immagine continua, ma campiona un
insieme fisso di punti distribuiti uniformemente sull’intero frame, secondo una griglia regolare di dimensione 
GX × GY (ad esempio 32×18).

I pixel campionati del frame i vengono confrontati con quelli del frame precedente (i−1) calcolando una SAD (Sum of Absolute Differences) 
sulla luminanza. La SAD risultante fornisce una misura approssimata della variazione globale tra i due frame.
Se SAD <= threshold → consideriamo il frame duplicato → keep=0
                                           Altrimenti → keep=1

Nota sull'algoritmo: si confronta il frame[i] con il frame[i-1] anche se il frame[i-1] è stato marcato come duplicato (keep=0).*/


  // forceline indica al compilatore che questa non è una chiamata a funzione, ma verrà direttamente sostituita nel codice ( in modo da evitare appunto l'overhead di 
  // una chiamata a funzione )
__device__ __forceinline__ uint8_t rgb_to_luma(uint8_t r, uint8_t g, uint8_t b) {
  // luminanza veloce (approx BT.601)
  return (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
}

__global__ void dedup_kernel_downsample_sad(
    const uint8_t* d_frames,                   // puntatore a tutti i frame del chunk
    uint8_t* d_keep,                           // array di output che contiene un flag (un bit 0 o 1 ) per ogni frame analizzato (1 lo manteniamo, 0 lo scartiamo)
    int n,                                     // numero di frame del chunk corrente
    int w, int h, int c,                       // dimensione e canali dei frame (ipotesi di dominio: 1280:720 con 3 canali)
    int bytes_per_frame,                       // numero totale di bytes per frame
    int threshold)                             // !!! THRESHOLD: definibile anche da linea di comando con l'opzione --threshold è la soglia che vienbe usata sulla SAD. 
                                               //       se SAD <= threshold -> frame duplicato e quindi scartato, altrimenti lo si tiene 
{
  
  // mappatura basata sul metodo delle coordinate: ogni thread sarà associato a un frame
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // il primo frame del chunk va tenuto, non ha infatti un frame precedente con cui confrontarlo
  if (i == 0) { d_keep[i] = 1; return; }

  // calcolo dell'indirizzo i-esimo
  const uint8_t* cur  = d_frames + (size_t)i * (size_t)bytes_per_frame;
  // calcolo dell'indirizzo i-1 esimo
  const uint8_t* prev = d_frames + (size_t)(i - 1) * (size_t)bytes_per_frame;

  // definiamo la dimensione della "sotto-immagine"
  const int GX = 32;
  const int GY = 18;

  unsigned int sad = 0;

  // loop sulle righe ( asse verticale )
  for (int yy = 0; yy < GY; ++yy) {
    // dalle coordinate yy della sottogriglia, ottengo una coordinata reale nel frame (es: [0,GY] -> [0, h] e così via).
    // di conseguenza per yy=0,1,...17 otterremo y=0,40,80,..680 (nel caso di una immagine 720)
    // i pixel che consideriamo sono quindi di coordinate costanti per ogni frame, ma non sono pixel adiacenti, solo nel loro insieme formano una sottogriglia di 32x18
    int y = (yy * h) / GY;
    // stesso ragionamento per l'asse orizzontale 
    for (int xx = 0; xx < GX; ++xx) {
      int x = (xx * w) / GX;

      // calcoliamo quindi l'indice del pixel su cui dobbiamo calcolare la luminanza
      int idx = (y * w + x) * c;

      uint8_t a, b;
      // i frame sono nel formato RGB interleaved 
      a = rgb_to_luma(cur[idx + 0],  cur[idx + 1],  cur[idx + 2]);
      b = rgb_to_luma(prev[idx + 0], prev[idx + 1], prev[idx + 2]);
     
      // calcolo la somma delle differenze delle luminanze del singolo pixel tra il frame i e il frame i-1
      sad += (a > b) ? (a - b) : (b - a);
    }
  }

  // Decisione: se differenza totale è piccola → duplicato
  d_keep[i] = (sad > (unsigned int)threshold) ? 1 : 0;
}



/*, l’approccio 1 thread = 1 frame ricalcola due volte la luminanza campionata per i frame interni. 
Tuttavia il tentativo di riusare tali valori in shared memory non è praticabile perché richiederebbe 
memorizzare GX×GY campioni per molti frame del blocco, superando la shared memory disponibile, e 
inoltre non permette condivisione tra blocchi.*/