# NLP Study Agent

Assistente di studio per il corso di **Natural Language Processing** (prof. F. Tamburini, Unibo, a.a. 2026/27).
Progetto d'esame n. 1: agente LLM con **LangGraph**. Risponde a domande sui materiali del corso (slide, tutorial,
articoli), indica da quale file e macroargomento ricava la risposta, fa ripassi e genera quiz a risposta multipla.

Il prof ha approvato l'idea chiedendo di usare **tutti** i materiali linkati sul sito, non solo le slide.

## Architettura

```
START → router ─┬─ offtopic → END
                └─ retrieve ─┬─ quiz → END
                     ▲       └─ answer → judge ─┬─ ok → END
                     │                          └─ non sostenuta (max MAX_REVISIONS) → rewrite ─┘
```

| Nodo | Cosa fa | File |
|---|---|---|
| router | classifica: `theory` / `quiz` / `review` / `offtopic`; produce una query di ricerca **in inglese** (materiali in inglese, domande in italiano) | `graph.py`, `prompts.py` |
| retrieve | ricerca per similarità nel vector store (Chroma) | `store.py` |
| answer | risposta in italiano con marcatori `[n]`; le **fonti sono costruite dal codice** dai metadati, non dal modello | `graph.py`, `nodes/helpers.py` |
| judge | verifica che la risposta sia sostenuta dai passaggi; se no, riformula la query e torna a `retrieve` (ciclo limitato) | `graph.py` |
| quiz | domande a scelta multipla in JSON validato, con spiegazione e fonte | `graph.py` |

Lo **State** (`state.py`) è il contratto tra i nodi: messaggi, passaggi recuperati, numero di cicli di revisione.
Le tesi personali del docente (`stance: professor_opinion` in `data/course_map.yaml`) sono marcate nel prompt e
presentate come sue tesi.

## Come si usa

```bash
pip install -r requirements.txt
cp .env.example .env                  # scegliere LLM (Ollama locale o API) e modello di embedding
# mettere i materiali in data/raw/    (NON vanno su GitHub: sono coperti da .gitignore)
python -m nlp_agent.cli index         # ingestion + embeddings (usa OCR dove serve)
python -m nlp_agent.cli chat          # chat interattiva
python -m nlp_agent.cli ask "Che cos'è la self-attention?"
python eval/run_eval.py               # valutazione agente vs. stesso LLM senza retrieval
python -m pytest                      # test (senza modelli né rete: LLM ed embedder finti)
python scripts/inspect_ingest.py      # chunk per file e avvisi, senza embeddings
```

Per l'OCR serve `tesseract` installato nel sistema.

## Materiali e ingestion

`data/course_map.yaml` associa ogni file a parte (I/II/III/Extra), macroargomento e tipo (slide / articolo /
tutorial / libro). Il locator di citazione è *slide N* per le slide, *pp. N* per articoli, *sezione «…»* per gli HTML.

- **Libro SLP (Jurafsky & Martin): escluso** per scelta del team (`exclude:` nella mappa). Va dichiarato nella relazione.
- Voci con `verified: false` hanno la classificazione dedotta dal titolo: **controllatela sulla scaletta del sito**.
- Nomi file normalizzati: `GrammarsAndParsing (1).pdf` e `Milli#U00e8re2024.pdf` (artefatti dello zip) sono riconosciuti.
- `Cap2_RabinerJuang.pdf` e `LinguistGuide2Stats.pdf` sono **scansioni**: passano da OCR (Tesseract, con cache in
  `data/ocr_cache/`). Formule e tabelle OCR possono essere imprecise.

### Limiti noti
- **Slide quasi solo immagini** (es. `Agents.pdf`: 12 pagine su 13 con immagini, le slide di codice sono screenshot;
  anche `SpeechRecognition`, `SpeechSampling`, `NetEmbedding`). Il testo indicizzato è scarso: per queste servono i
  tutorial/articoli. Estensione possibile: descrivere le pagine con un LLM con visione in fase di ingestion.
- **Squilibrio dell'indice:** `FSTTutorial.pdf` (409 pagine) pesa circa il 29% dei chunk e può affollare i risultati.
  Da misurare con l'embedder vero; rimedi: limitare i risultati per file, o riclassificare.
- **Lingua:** documenti in inglese, domande in italiano → serve un embedder multilingua (default
  `intfloat/multilingual-e5-small`) e la query in inglese prodotta dal router.
- Il **judge** è lo stesso LLM che risponde: un judge debole lascia passare errori. Con modelli piccoli verificare a mano un campione.

## Valutazione

`eval/questions.yaml`: domande con risposta attesa, file e macroargomento corretti. Misura correttezza (agente vs.
baseline senza retrieval), accuratezza della fonte (file e macroargomento), interventi del judge, rifiuto delle
domande fuori tema, attribuzione delle tesi del docente. **Stato: 26 domande** (23 di teoria, 3 fuori tema),
scritte su Embeddings, DNNs, PromptingLLMs, SOTA4LLMs2026, Agents, NLPApplications, LangGraph, tokenizzazione, RAG.
**Da aggiungere** per arrivare a 30-40 e coprire Part I (n-grammi, probabilità, valutazione, corpora) e Part II
(ASR, parsing, WordNet/WSD, FrameNet).

## Divisione del lavoro

| | Jacopo | Giacomo |
|---|---|---|
| Ingestion e retrieval | verifica `course_map.yaml` sulla scaletta, OCR, chunking, misura recall con l'embedder vero | |
| Agente | | prompt di router/answer/judge/quiz, scelta di LLM e provider, interfaccia |
| Valutazione | domande Part I | domande Part II, esecuzione e analisi dei risultati |
| Relazione | metodo e ingestion | architettura e risultati |

(Proposta iniziale: scambiatevi i ruoli come preferite.)
