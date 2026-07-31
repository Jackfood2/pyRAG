# Offline RAG: Local Document Intelligence

This repository provides a complete, workstation-based Document Intelligence system that searches, analyzes, and answers questions using your private documents. It operates entirely offline without sending any data to the cloud.

The README is divided into two parts:
1. A simple, non-technical overview explaining what the application does and how to run it.
2. A detailed guide covering system architecture, algorithms, configuration, and API endpoints.

---
## What is Offline RAG?
Offline RAG is a private search and question-answering assistant for your personal computer. Think of it as a custom search engine combined with an assistant that reads your PDFs, Word documents, Excel spreadsheets, and Outlook emails, and then answers your questions based strictly on that information.

Every answer includes exact inline citations `[1]`, `[2]` so you can verify where the information came from. Hovering over or clicking a citation highlights the exact paragraph and source document used.

## Why Use It?
* **Complete Privacy**: Your files never leave your computer. Nothing is uploaded to the internet or external servers.
* **No Subscription Costs**: Uses free, open-source AI models running directly on your hardware.
* **Works Without Internet**: Runs completely offline during travel, in secure air-gapped environments, or during network outages.
* **Broad File Support**: Reads PDFs, Word files (`.docx`, `.doc`), Excel sheets (`.xlsx`), PowerPoint decks (`.pptx`), text files, and Outlook email archives (`.pst` and `.msg`).

## How to Get Started in 4 Steps (Windows)

### Step 1: Download the AI Models
Double-click `Install_Models.bat`. This downloads two small helper models into the `models/` folder:
* An **Embedding Model** that turns document text into mathematical search vectors.
* A **Reranker Model** that double-checks search results for accuracy.

### Step 2: (Optional) Enable Graphics Card Acceleration
If your PC has a graphics card (NVIDIA or Vulkan-compatible), double-click `Install_GPU.bat`. It automatically detects your graphics card, downloads the necessary files, and accelerates search speed. If you do not have a dedicated graphics card, skip this step.

### Step 3: Connect LM Studio
LM Studio is a separate desktop application that runs large language models on your PC.
1. Download and open **LM Studio**.
2. Download a text generation model (for example, Llama-3 or Mistral).
3. Load the model and click **Start Server** (it runs locally on port `1234`).

### Step 4: Launch the Application
Double-click `Start-Offline-RAG.bat`. A web browser window will open automatically at:
```
http://127.0.0.1:8765
```

## Basic How-To Guide

### How to Index Your Documents
1. Click the **Settings** gear icon in the top right corner.
2. Under **Indexing**, type or select the folder where your documents are stored.
3. Click **Add / update**. The system will read your files and build a searchable index.

### How to Ask Questions
1. Type any question into the search bar at the top and press **Enter**.
2. Watch the **Pipeline** panel show the search progress in real time.
3. Read the generated answer in the **Answer** box and hover over citation numbers `[1]` to see the exact source text highlighted in the **Evidence** box below.

### Running Without LLM / LM Studio (Retrieval-Only & Evidence Mode)

If you do not have LM Studio open, Offline RAG still works. It uses its built-in local engines to search your documents and display the exact matching text passages, page numbers, and file excerpts directly. You do not get a written AI summary, but search remains fast, intelligent, and accurate.

---

# PART 2: TECHNICAL MANUAL

## System Architecture

Offline RAG uses a multi-tier local architecture. The backend is written in Python (`app.py`), serving a single-page web interface (`index.html`). Local GGUF models are managed via background `llama-server` processes, while LLM generation is routed through an OpenAI-compatible endpoint (LM Studio).

```
                      +----------------------------------+
                      |       Web Frontend (HTML/JS)     |
                      |        http://127.0.0.1:8765     |
                      +----------------+-----------------+
                                       |
                                       | HTTP / Server-Sent Events (SSE)
                                       v
                      +----------------+-----------------+
                      |     Python Backend (app.py)      |
                      |   http.server ThreadingHTTPServer|
                      +---+------------+-------------+---+
                          |            |             |
       HTTP (Port 8787)   |            |             |   HTTP (Port 8788)
  +-----------------------+            |             +-----------------------+
  |                                    |                                     |
  v                                    v                                     v
+-+---------------------+   +----------+----------+   +----------------------+--+
| Local Embedding Server|   |   LM Studio Server   |   |  Local Reranker Server  |
|  (llama-server.exe)   |   |  http://127.0.0.1:1234|   |   (llama-server.exe)    |
+-----------------------+   +---------------------+   +-------------------------+

```

### Server Network Map
* **Port 8765**: Application Web Interface and REST/SSE API (`app.py`).
* **Port 8787**: Embedded `llama-server` instance handling embeddings.
* **Port 8788**: Embedded `llama-server` instance handling cross-encoder reranking.
* **Port 1234**: External LM Studio endpoint (`/v1/chat/completions`).

#### Technical Behavior and Pipeline Adjustments

When `chat_model` is blank, LM Studio is unreachable, or "Evidence only" is enabled in the UI, `app.py` automatically routes execution through a zero-LLM fast path:

1. **01 Understand Stage**: Skipped. Query rewriting and HyDE generation are bypassed. Search uses the exact user prompt.
2. **02 Retrieve Stage**: Active. Runs full hybrid retrieval (BM25 keyword search + local GGUF embedding vector similarity via `llama-server` on port 8787) followed by cross-encoder reranking (`llama-server` on port 8788).
3. **03 Analyze Stage**: Skipped. The pipeline returns full, unedited source chunks directly to the UI tagged as `RAW`.
4. **04 Answer Stage**: Skipped. The UI displays the retrieved passages in the **Evidence** panel while keeping the Answer box idle.

* **Latency**: Response times drop from seconds to milliseconds (typically under 100ms total search time).
* **API Dependencies**: 0 calls to external or local OpenAI/LLM endpoints.

---

#### Feature Availability Matrix

| Feature | With LM Studio / LLM | Without LM Studio / LLM |
| :--- | :--- | :--- |
| **BM25 Lexical Keyword Search** | Available | Available |
| **Local Semantic Vector Search** | Available (llama-server:8787) | Available (llama-server:8787) |
| **Local BGE Reranking** | Available (llama-server:8788) | Available (llama-server:8788) |
| **Full Chunk & Document Viewer** | Available | Available |
| **PST & Document Ingestion** | Available | Available |
| **LLM Query Rewriting & HyDE** | Available | Skipped |
| **LLM Passage Summarization** | Available | Skipped (Displays Raw Text) |
| **Iterative Multi-Wave Search** | Available | Single-Pass Search |
| **AI Answer Synthesis & Citations**| Available | Skipped (Evidence Only) |

---

---

## Directory Structure

```
OfflineRAG/
├── __pycache__/
├── data/                            # Application logs, settings, and database
│   ├── document_manifest.json       # Content fingerprint manifest for fallback tracking
│   ├── document_snapshots/          # Gzip-compressed extracted text snapshots
│   ├── lancedb/                     # LanceDB vector database directory
│   ├── logs/                        # Processing logs for servers and PST jobs
│   ├── settings.json                # Runtime configuration settings
│   └── index.json                   # Backup in-memory JSON index
├── models/                          # Local GGUF model storage
│   ├── catalog.json                 # Model metadata catalog
│   ├── nomic-embed-text-v1.5.Q4_K_M.gguf
│   └── bge-reranker-v2-m3-Q4_K_M.gguf
├── python/                          # Embedded portable Python 3.11 environment
│   ├── python.exe
│   ├── python311._pth
│   └── Lib/site-packages/           # Bundled packages (pywin32, openpyxl, rapidfuzz, etc.)
├── runtime/                         # Execution binaries and acceleration DLLs
│   ├── llama-server.exe
│   ├── ggml.dll
│   └── [ggml-cuda.dll / ggml-vulkan.dll]
├── vendor/                          # Embedded fallback libraries
│   └── pypdf/
├── Install_GPU.bat                  # CUDA/Vulkan runtime installer
├── Install_LanceDB.bat              # Enables site-packages and installs LanceDB
├── Install_Models.bat               # Downloads default GGUF models
├── Start-Offline-RAG.bat            # Application launcher batch script
├── README.md                        # Documentation
├── app.py                           # Application logic and HTTP server
└── index.html                       # Frontend user interface

```
---

## Search Engine and Retrieval Mechanics

Offline RAG employs a multi-stage hybrid retrieval strategy:

### 1. Hybrid Search
* **BM25 Lexical Scoring**: Calculates exact term frequency and inverse document frequency across the corpus, accounting for document lengths.
* **Vector Semantic Scoring**: Computes normalized dense vector cosine similarity using the local embedding model (`llama-server` on port 8787).
* **Reciprocal Rank Fusion (RRF)**: Merges lexical and semantic rank lists into a single fused candidate list using the formula:
  $$\text{RRF Score} = \sum_{m \in M} \frac{w_m}{60 + r_m(d)}$$
  where $w_m$ is the weight (semantic vs keyword) and $r_m(d)$ is the document rank in strategy $m$.

### 2. Fast Storage Layers
* **LanceDB**: Vector search database configured with float32 vector arrays.
* **In-Memory RAM Caching**: When running without LanceDB, vector arrays are pre-allocated into Python `array.array('f')` binary structures for rapid array dot-product computations.

### 3. Local Cross-Encoder Reranking
Top candidate passages are passed to a local BGE Reranker model running on port 8788. Cross-encoder models process the query and passage simultaneously, yielding higher precision relevance scores than dual-encoder cosine similarity alone.

---

## Corrective Adaptive RAG Pipeline

When a user query is received via Server-Sent Events (`/api/answer-stream`), `app.py` executes a four-stage process:

```
[01 Understand] ---> [02 Retrieve] ---> [03 Analyze & Assess] ---> [04 Answer]
```

### Stage 1: Understand
* The analysis LLM evaluates the user prompt.
* Generates an optimized query rewrite and up to 3 alternative search queries.
* Produces a Hypothetical Document Embedding (HyDE) response—a hypothetical answer passage used as an additional vector search query.

### Stage 2: Retrieve
* Runs hybrid retrieval across all query variants and HyDE passages.
* Applies Reciprocal Rank Fusion and passes candidate results through the cross-encoder reranker.

### Stage 3: Analyze & Assess (Corrective Loop)
* Candidate passages are analyzed individually by the evidence analysis LLM to produce concise factual summaries and assign a relevance classification:
  * `ANSWERS`: Directly contains requested details.
  * `PARTIAL`: On-topic, but missing specific requested details.
  * `RELATED`: Same domain, but different entity or direction.
  * `OFFTOPIC`: Irrelevant to the prompt.
* **Checkpoints**: Every $N$ passages (configurable via `rerank_count`), an assessment prompt asks the LLM if the collected evidence is sufficient to answer the query.
* **Multi-Wave Recovery**: If evidence is insufficient, the LLM articulates the specific information gap. The gap is converted into a new search query, triggering a new retrieval wave (up to 5 waves).

### Stage 4: Answer
* Collected evidence notes are compressed to fit within the LLM context window.
* The final answer LLM composes a response, enforcing strict inline citation tags `[n]`.
* If no citations are generated or single-source bias is detected despite multiple relevant notes, the backend automatically triggers a corrective regeneration pass.

---

## Outlook PST Ingestion Details

PST extraction runs via Windows COM automation using Python's `win32com.client`:

* **Threading & Deduplication**: Message items are grouped into conversation threads via `ConversationID` or subject normalized strings.
* **Fuzzy Attachment Matching**: Attachment filenames are normalized and evaluated using `rapidfuzz`. Attachments matching previous filenames above the configured similarity threshold (default 90%) are deduplicated, keeping only the latest version based on receipt timestamp.
* **Modes**:
  * `emails_only`: Parses message headers and body text; skips attachments.
  * `attachments_only`: Extracts file attachments while preserving email thread metadata.
  * `emails_and_attachments`: Full extraction of message bodies and attachments.

---

## Document Snapshots and Fallback Recovery

1. **Ingestion**: When files are indexed, extracted plain text is compressed using Gzip and written to `data/document_snapshots/<hash>.txt.gz`.
2. **Fingerprinting**: Files are assigned a SHA-256 fingerprint derived from file size and head/tail data blocks.
3. **Relocation Matching**: If an indexed document is moved or deleted, requesting its text triggers a scan of replacement folders (`/api/document-relocate`). The system matches candidates against expected relative paths, filenames, and SHA-256 fingerprints, restoring document access without requiring a full re-index.

---

## Batch Scripts Mechanics

* `Start-Offline-RAG.bat`: Sets the working directory, verifies `python\python.exe` and `runtime\llama-server.exe` exist, launches `app.py` in the background, waits 3 seconds, and opens `http://127.0.0.1:8765`.
* `Install_Models.bat`: Uses `curl` to fetch `nomic-embed-text-v1.5.Q4_K_M.gguf` and `bge-reranker-v2-m3-Q4_K_M.gguf` directly from Hugging Face into `models/`.
* `Install_LanceDB.bat`: Modifies `python311._pth` to uncomment `import site`, installs `pip` if missing, and runs `pip install lancedb numpy`.
* `Install_GPU.bat`: An embedded PowerShell script that checks for NVIDIA CUDA (`nvidia-smi`) or universal Vulkan GPUs (`Win32_VideoController`), fetches the latest matching `llama.cpp` release zip via the GitHub API, backs up `runtime/` to `runtime_cpu_backup/`, and copies GPU-enabled binaries into `runtime/`.

---

## Configuration Settings Reference

Stored in `data/settings.json`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source_folder` | String | Parent Folder | Directory containing documents to index. |
| `lmstudio_url` | String | `http://127.0.0.1:1234/v1` | LM Studio OpenAI API endpoint. |
| `analysis_model` | String | `""` | Model key for query rewriting and passage analysis. |
| `chat_model` | String | `""` | Model key for writing the final cited answer. |
| `embedding_model` | String | `nomic-embed-text-v1.5.Q4_K_M.gguf` | Embedding GGUF file inside `models/`. |
| `rerank_model` | String | `bge-reranker-v2-m3-Q4_K_M.gguf` | Reranker GGUF file inside `models/`. |
| `chunk_size` | Integer | `900` | Target passage size in characters. |
| `chunk_overlap` | Integer | `140` | Overlap between adjacent chunks in characters. |
| `candidate_count` | Integer | `32` | Candidate passages retrieved per search query. |
| `rerank_count` | Integer | `4` | Passages analyzed per sufficiency checkpoint. |
| `max_candidate_checks` | Integer | `24` | Hard cap on passages analyzed per user query. |
| `semantic_weight` | Float | `0.72` | Semantic vector search weight in RRF. |
| `keyword_weight` | Float | `0.28` | Lexical BM25 search weight in RRF. |
| `use_llm_rerank` | Boolean | `true` | Enables BGE cross-encoder reranking. |
| `adaptive_rag` | Boolean | `true` | Enables query rewriting, HyDE, and corrective search waves. |
| `use_lancedb` | Boolean | `true` | Uses LanceDB storage if installed. |
| `gpu_offload` | Boolean | `true` | Offloads model layers to GPU if available. |

---

## API Reference

### REST Endpoints

#### GET /api/status
Returns index status, total chunks, cache state, and GPU backend information.

#### GET /api/settings | POST /api/settings
Reads or updates configuration keys in `data/settings.json`.

#### GET /api/models
Queries LM Studio for a list of available local LLMs.

#### GET /api/local-models
Returns active status and context size limits for embedded embedding and reranker servers.

#### POST /api/select-local-model
Switches active embedding or reranker models and restarts the corresponding server.
* **Payload**: `{"kind": "embedding" | "reranker", "id": "<model_id>"}`

#### POST /api/pst/import
Launches an asynchronous PST archive extraction job.
* **Payload**: `{"pst_path": "<path_to_pst>"}`

#### GET /api/pst/status?job=<job_id>
Returns status, elapsed time, and logs for an active PST job.

#### GET /api/document-text
Retrieves full extracted text or snapshot for a document.
* **Query Parameters**: `path`, `document_id`, `offset`, `limit`

#### POST /api/document-relocate
Scans a target directory to match and re-link a missing file using SHA-256 fingerprints.
* **Payload**: `{"document_id": "<doc_id>", "folder": "<replacement_path>"}`

---

### Streaming Endpoints (Server-Sent Events)

#### POST /api/answer-stream
Streams processing events, pipeline stage transitions, evidence updates, and generated tokens.
* **Payload**:
  ```json
  {
    "query": "What were the Q3 sales figures?",
    "answer": true,
    "adaptive": true
  }
  ```

#### POST /api/ingest-stream
Streams progress, parsing logs, and vectorization status during document ingestion.
* **Payload**: `{"mode": "incremental" | "full"}`


'''
# Offline RAG — 310726 Fixes

This update improves retrieval accuracy and evidence usability, especially for short keyword searches across indexed emails and documents.

## What Was Fixed

- **Immediate settings changes**  
  Semantic and Keyword weighting is automatically saved and applied to the next enquiry. No application restart or index rebuild is required.

- **Collapsible evidence cards**  
  Completed evidence cards automatically collapse to show the evidence title. Click the title—or press `Enter` or `Space` while it is focused—to expand or collapse the chunk content.

- **Better exact-keyword retrieval**  
  Passages without a literal keyword match are excluded from keyword ranking, preventing unrelated results from receiving lexical rank positions.

- **Title and email-subject weighting**  
  Matches in `THREAD SUBJECT`, `SUBJECT`, or the document filename receive more weight than matches found only in the body content.

- **Short-query optimisation**  
  One- and two-keyword searches, such as `APAC`, receive a stronger title and subject boost.

- **Exact phrase boost**  
  An exact phrase found in an email subject or document title receives additional relevance weight.

- **Clearer evidence titles**  
  Evidence cards display the extracted email subject or document title. The original source filename remains available in the tooltip.

- **Improved citation navigation**  
  Clicking a citation in the generated answer automatically expands and scrolls to the corresponding evidence card.

---

## License

This software is released under the MIT License.
