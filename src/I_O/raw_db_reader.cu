
#include "../../include/db_types.h"
#include "../../include/raw_db_reader.h"
#include <stdexcept>
#include <fstream>
#include <algorithm>

// la struttura da leggere è la seguente: abbiamo una cartella che rappresenta il database, e, all'interno di tale
// cartella sono presenti delle sottocartelle che rappresentano i vari video


/* è il costruttore: riceve sempre cfg e inizializziamo le variabili che ci servono ( i path etc )*/
RawDbReader::RawDbReader(const Config& cfg) 
    //sintassi C++: serve per inizializzare i membri della classe mentre l'oggetto viene costruito 
    /* string */: root_(cfg.full_db_path), /* il path */
    /* size_t */bytes_per_frame_expected_(static_cast<size_t>(cfg.frame_w) * cfg.frame_h * cfg.channels) { /* come dice il nome, bytes per frame*/


  namespace fs = std::filesystem;

  // controllo che la cartella esista e che sia una cartella 
  if (!fs::exists(root_) || !fs::is_directory(root_)) {
    throw std::runtime_error("RawDbReader: full_db_path non esiste o non e' una cartella: " + root_);
  }

  /* il seguente codice riempe il vettore video_dirs_ con tutti i percorsi completi delle sottocartelle ( ovvero i video ) del Db */
  for (const auto& entry : fs::directory_iterator(root_)) {
    if (entry.is_directory()) {
      video_dirs_.push_back(entry.path());
    }
  }

  /* ordine il vettore: std::sort(inizio, fine, criterio_di_confronto) */

  /* BISOGNA ORDINARE SOLO A SCOPO DIDATTICO: il ciclo presente sopra, che riempe il vettore video_dirs_ non è un algoritmo deterministico,
  significa che a ogni esecuzione potrebbe riempire il vettore con un ordine casuale dei video presenti nel database. signifca che nella parte successiva, quella del 
  template matching, a parità di input potremo avere risultati di tempo diversi perchè magari alla prima esecuzione il video originale è il primo e nella seconda esecuzione
  è al secondo posto */

  std::sort(video_dirs_.begin(), video_dirs_.end(),
            [](const fs::path& a, const fs::path& b) { return a.filename().string() < b.filename().string(); });

  // carichiamo il primo video 
  if (!video_dirs_.empty()) {
    load_video_frames_list_(0);
  }

}

/* FUNZIONE HAS_NEXT
 dice se ci sono altri frame nel video corrente o nel video dopo */
bool RawDbReader::has_next() const {

  if (video_dirs_.empty()) return false; 
  if (cur_video_idx_ /* indice del video corrente */ >= video_dirs_.size()) return false;
  if (cur_frame_idx_ /* indice del frame corrente */ < cur_video_frames_.size() /* dimensione della lista dei frame del video corrente*/) return true;

  // verifichiamo nel video dopo 
  size_t v = cur_video_idx_ + 1;
  while (v < video_dirs_.size()) {
    for (const auto& entry : std::filesystem::directory_iterator(video_dirs_[v])) {
      if (entry.is_regular_file()) {
        return true;
      }
    }
    ++v;
  }
  return false;
}

/* FUNZIONE NEXT_CHUNK

*/
HostChunk RawDbReader::next_chunk(int max_frames /* numero massimo di frame che il chunk può contenere*/ , size_t bytes_per_frame /* byte attesi per ogni frame*/) { 

  // Inizio controlli
  if (bytes_per_frame != bytes_per_frame_expected_) {
    throw std::runtime_error("RawDbReader: bytes_per_frame non coerente con cfg (atteso " +
                             std::to_string(bytes_per_frame_expected_) + ", ricevuto " +
                             std::to_string(bytes_per_frame) + ")");
  }
  if (max_frames <= 0) {
    throw std::runtime_error("RawDbReader: max_frames deve essere > 0");
  }
  // Fine controlli 

  HostChunk ch;
  ch.n = 0;

  // i tre campi dei metadati dell'oggetto HostChunk
  ch.frames.resize(static_cast<size_t>(max_frames) * bytes_per_frame);
  ch.video_id.resize(max_frames);
  ch.frame_id.resize(max_frames);

  while (ch.n < max_frames) {
    if (cur_video_idx_ >= video_dirs_.size()) break;

    if (cur_frame_idx_ >= cur_video_frames_.size()) {
      ++cur_video_idx_;
      cur_frame_idx_ = 0;
      cur_video_frames_.clear();

      if (cur_video_idx_ < video_dirs_.size()) {
        load_video_frames_list_(cur_video_idx_);
        continue;
      } else {
        break;
      }
    }

    const std::filesystem::path& frame_path = cur_video_frames_[cur_frame_idx_];

    uint8_t* dst = ch.frames.data() + static_cast<size_t>(ch.n) * bytes_per_frame;
    read_exact_file_(frame_path, dst, bytes_per_frame);

    ch.video_id[ch.n] = static_cast<int32_t>(cur_video_idx_);
    ch.frame_id[ch.n] = static_cast<int32_t>(cur_frame_idx_);
    ++ch.n;
    ++cur_frame_idx_;
  }

  ch.frames.resize(static_cast<size_t>(ch.n) * bytes_per_frame);
  ch.video_id.resize(ch.n);
  ch.frame_id.resize(ch.n);

  return ch;
}


/* FUNZIONE LOAD_VIDEO_FRAMES_LIST_
    carica e ordina la lista dei frame del video video_idx*/
void RawDbReader::load_video_frames_list_(size_t video_idx) {
  namespace fs = std::filesystem;

  // svuotiamo la lista dei frame correnti 
  cur_video_frames_.clear();
  //controlliamo di non essere fuori range 
  if (video_idx >= video_dirs_.size()) return;

  // prendiamo la directrory del video 
  const fs::path& dir = video_dirs_[video_idx];

  //scorriamo tutti gli elementi dentro alla cartella e 
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file()) {
      // e mettiamo il path del frame nel vettore cur_video_frames_
      cur_video_frames_.push_back(entry.path());
    }
  }

  // ordini sia per esecuzione deterministica ma anche per assegnare nella funzione next_chunk gli id ai frame in modo corente
  // si assume quindi che nella cartella i frame siano del tipo frame_0001.xxx , frame_0002.xxx ( come effettivamente fanno i tools di 
  // estrazione dei frame come ffmpeg nel nostro caso )

  std::sort(cur_video_frames_.begin(), cur_video_frames_.end(),
            [](const fs::path& a, const fs::path& b) { return a.filename().string() < b.filename().string(); });
}


/* FUNZIONE READ_EXACT_FILE
legge il singolo frame preso in considerazione */
void RawDbReader::read_exact_file_(const std::filesystem::path& p, uint8_t* out, size_t bytes) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    throw std::runtime_error("RawDbReader: impossibile aprire file: " + p.string());
  }
  f.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(bytes));
  if (f.gcount() != static_cast<std::streamsize>(bytes)) {
    throw std::runtime_error("RawDbReader: file size non attesa (bytes letti=" +
                             std::to_string(f.gcount()) + ", attesi=" + std::to_string(bytes) +
                             ") file=" + p.string());
  }
}
