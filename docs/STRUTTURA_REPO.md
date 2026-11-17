# Struttura della repo

```
NLP/
├── README.md              panoramica, architettura, uso, limiti noti, divisione del lavoro
├── pyproject.toml         metadati del pacchetto e configurazione di pytest
├── requirements.txt       dipendenze (LangGraph, Chroma, sentence-transformers, ecc.)
├── .env.example           modello di configurazione: copiare in .env (che NON va su GitHub)
├── .gitignore             esclude materiali del corso, indice, cache OCR, risultati, .env
├── data/
├── src/nlp_agent/
├── eval/
├── tests/
├── scripts/
└── docs/
```

## data/
Dati di input e di lavoro.
- `course_map.yaml` — mappa **file → parte (I/II/III/Extra), macroargomento, tipo** (slide/articolo/tutorial/libro).
  Serve per citare le fonti e per valutare se il macroargomento indicato è giusto. Contiene anche l'elenco
  dei file esclusi (libro SLP) e il flag `stance: professor_opinion` per le slide con tesi del docente.
  Le voci con `verified: false` vanno controllate sulla scaletta del sito.
- `raw/` — qui si mettono i PDF/HTML del corso. **Non vanno su GitHub** (`.gitignore`): restano solo in locale.
- `index/` — indice Chroma generato da `cli index`. Non versionato, si rigenera.
- `ocr_cache/` — testo OCR dei PDF scansionati (Rabiner-Juang, LinguistGuide2Stats). Non versionato; evita di rifare l'OCR.

## src/nlp_agent/  (il codice dell'agente)
- `config.py` — legge le impostazioni da `.env` (LLM, modello di embedding, percorsi, top-k, max revisioni).
- `course_map.py` — carica `course_map.yaml` e normalizza i nomi file (`(1)`, `#U00e8`, maiuscole).
- `state.py` — lo **State** del grafo: il contratto tra i nodi (messaggi, passaggi, cicli di revisione).
- `prompts.py` — tutti i prompt (router, risposta, ripasso, judge, quiz).
- `llm.py` — interfaccia verso l'LLM (Ollama / Anthropic / OpenAI) e `extract_json`.
- `embeddings.py` — modelli di embedding: sentence-transformers (reale) e `HashEmbedder` (solo test/offline).
- `store.py` — vector store su Chroma: aggiunta chunk e ricerca per similarità.
- `graph.py` — **il grafo LangGraph**: router → retrieve → answer/quiz → judge → rewrite. Il file centrale.
- `cli.py` — comandi `index`, `chat`, `ask`.
- `evaluation.py` — metriche di valutazione (correttezza, accuratezza fonte, interventi del judge, baseline).
- `ingest/` — dai file ai chunk:
  - `loaders.py` lettura PDF/HTML, pulizia, rilevamento testo illeggibile, OCR con cache;
  - `chunking.py` divisione in chunk con metadati e posizione (slide N / pp. N / sezione);
  - `build_index.py` costruisce l'intero indice da `data/raw`.
- `nodes/helpers.py` — formattazione dei passaggi e costruzione dell'elenco "Fonti" dai metadati.

## eval/
- `questions.yaml` — domande con risposta attesa, file e macroargomento corretti (26 finora; da portare a 30-40).
- `run_eval.py` — esegue la valutazione agente vs stesso LLM senza retrieval; salva i dettagli in `eval/results/` (non versionato).

## tests/
Test automatici (`python -m pytest`) con LLM ed embedder finti: non servono modelli né rete.
`conftest.py` contiene i finti; gli altri file testano mappa, ingestion, grafo e valutazione.

## scripts/
- `inspect_ingest.py` — mostra quanti chunk esce da ogni file e gli avvisi, senza calcolare embeddings. Utile dopo aver cambiato il chunking.
- `ocr_scans.py` — esegue l'OCR dei due PDF scansionati e salva la cache.

## docs/
Documentazione di progetto (questo file).
