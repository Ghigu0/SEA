README – GPU Frame Search & Deduplication (Scroll down for the english version )

Questo programma implementa un sistema accelerato su GPU per la costruzione di un database di frame video deduplicato e per la ricerca di un frame query all’interno del database stesso. L’idea alla base è ridurre il numero di confronti necessari tramite una fase di deduplicazione e una pipeline di ricerca a più livelli.

UTILIZZO DEL PROGRAMMA

Il programma deve essere eseguito da linea di comando secondo la seguente sintassi:

./programma <full-db_PATH> <frame_PATH> [opzioni]

I parametri obbligatori sono:

    full-db_PATH: percorso alla directory che contiene il database completo dei frame (non deduplicato)

    frame_PATH: percorso al frame query da ricercare nel database

Le opzioni disponibili sono:

    --gpu <id> : seleziona l’ID della GPU da utilizzare (default 0)

    --verbose : abilita la stampa di informazioni dettagliate sull’esecuzione

    --topk <k> : numero di candidati considerati nella fase di ricerca                      ( default 50 )

    --dedup-threshold <t> : soglia SAD utilizzata nella fase di deduplicazione dei frame     (default 8000 )

    --chunk-frames <n> : numero di frame processati per batch                               (default 1200)

   

Esempio di esecuzione:

./programma ./database ./query/frame.raw --topk 50 --dedup-threshold 1200 --verbose

DOMINIO DEI DATI

Il database di input deve essere organizzato come una directory principale contenente una sottocartella per ogni video. Ogni sottocartella contiene i frame estratti dal video in formato raw.

Esempio di struttura:

database/
video_01/
frame_000001.raw
frame_000002.raw
...
video_02/
frame_000001.raw
...
video_03/
...

I frame sono immagini RGB in formato raw con dimensioni fissate, definite nel file di configurazione del progetto.

OUTPUT DEL PROGRAMMA

Durante la prima fase di esecuzione (build phase), il programma carica il database completo, processa i frame a blocchi e rimuove i frame ridondanti 
Il database deduplicato viene scritto automaticamente nella directory:

./newDB

Questa directory contiene i frame deduplicati e i file binari associati alle firme.

Al termine della seconda fase (query phase), il frame del nuovo database che risulta più simile al frame di input viene salvato nel file:

winner.ppm

Inoltre, sul terminale, verrà stampato a schermo il video di appartenenza del frame "vincitore", che corrisponde al video di appartenenza del frame caricato dall'utente






ENGLISH VERSION




README – GPU Frame Search & Deduplication

This program implements a GPU-accelerated system for building a deduplicated video frame database and for searching a query frame within it. The main idea is to reduce the number of required comparisons through a deduplication phase and a multi-level search pipeline.

PROGRAM USAGE

The program must be executed from the command line using the following syntax:

./program <full-db_PATH> <frame_PATH> [options]

The required parameters are:

full-db_PATH: path to the directory containing the complete (non-deduplicated) frame database

frame_PATH: path to the query frame to be searched


The available options are:

--gpu <id> : selects the GPU device ID to use (default 0)

--verbose : enables detailed execution logging

--topk <k> : number of candidates considered during the search phase              (default 50)

--dedup-threshold <t> : SAD threshold used during the frame deduplication phase   (default 8000)

--chunk-frames <n> : number of frames processed per batch                         (default 1200)


Example execution:

./program ./database ./query/frame.raw --topk 50 --dedup-threshold 1200 --verbose

DATA DOMAIN

The input database must be organized as a main directory containing one subdirectory for each video. Each subdirectory contains the frames extracted from the video in raw format.

Example structure:

database/
video_01/
frame_000001.raw
frame_000002.raw
...
video_02/
frame_000001.raw
...
video_03/
...

Frames are raw RGB images with fixed dimensions, defined in the project configuration file.

PROGRAM OUTPUT

During the first execution phase (build phase), the program loads the complete database, processes the frames in batches, and removes redundant frames.

The deduplicated database is automatically written to the following directory:

./newDB

This directory contains the deduplicated frames and the binary files associated with their signatures.

At the end of the second phase (query phase), the frame from the new database that is most similar to the input frame is saved to the file:

winner.ppm

Additionally, the program prints to the terminal the video to which the winning frame belongs, corresponding to the video associated with the frame provided by the user.