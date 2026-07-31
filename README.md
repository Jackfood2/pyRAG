# Offline RAG: Local Document Intelligence

Offline RAG is a privacy-focused, workstation-based Document Intelligence and Retrieval-Augmented Generation (RAG) system. It runs entirely offline without relying on external cloud APIs or third-party data processing services.

The system combines a lightweight Python server (`app.py`) with a single-page web interface (`index.html`). It utilizes bundled `llama-server` instances for local embedding generation and reranking, while connecting to a local LM Studio instance for open-weights LLM inference.

---

## Key Features

### 1. Offline Execution & Privacy
* Zero cloud dependencies for core operations. Documents, embeddings, vectors, and chat interactions remain entirely local.
* Local embedding generation and reranking handled by bundled `llama-server` execution binaries.
* LLM generation handled via LM Studio or any local OpenAI-compatible API endpoint.

### 2. Advanced Retrieval Engine
* **Hybrid Search**: Combines BM25 lexical keyword scoring with vector-based semantic cosine similarity using Reciprocal Rank Fusion (RRF).
* **RAM In-Memory Acceleration**: Pre-allocated array buffers cache vector embeddings directly in RAM for rapid search iterations.
* **LanceDB Integration**: Automatic high-performance vector storage and retrieval via LanceDB when installed.
* **Local Reranking**: Re-scores search candidates using local cross-encoder reranker models (such as BGE Reranker v2-m3).

### 3. Adaptive Corrective Loop
* **Query Understanding**: Optionally rewrites user queries and generates Hypothetical Document Embeddings (HyDE) variants.
* **Multi-Wave Search**: Formulates new search queries based on identified information gaps if initial candidate passages are insufficient.
* **Passage Analysis**: Summarizes individual passages and assigns concrete relevance classifications (`ANSWERS`, `PARTIAL`, `RELATED`, `OFFTOPIC`).
* **Sufficiency Checkpoints**: Periodically assesses whether accumulated evidence notes are sufficient to answer the query, reducing unnecessary processing.

### 4. Comprehensive File Format Support
* Text formats: `.txt`, `.md`, `.csv`, `.json`, `.jsonl`
* Office documents: `.pdf`, `.docx`, `.doc`, `.pptx`, `.xlsx`, `.xlsm`, `.xls`
* Outlook messages: `.msg`
* Outlook PST archives: `.pst` (via Windows Outlook COM automation)

### 5. Resilient Document Management
* **Document Snapshots**: Retains compressed Gzip extracted-text snapshots along with SHA-256 fingerprinting.
* **Document Viewer & Relocation**: View passage chunks or full document texts directly within the web interface. If original files move or disappear, the system automatically suggests matches based on content fingerprints and relative paths.

### 6. Interactive Web Interface
* Streaming answer delivery with inline citations `[n]`. Hovering over citations highlights the corresponding source card.
* Detailed visual pipeline progress (Understand -> Retrieve -> Analyze -> Answer) and real-time telemetry logs.
* Customization options: Toggle Adaptive RAG, Evidence-only mode, dark/light theme, retrieval weights, model parameters, and search limits.

---

## Architecture and Services

```
                       +-----------------------------------+
                       |        Web Browser Interface     |
                       |           (index.html)            |
                       +-----------------+-----------------+
                                         |
                                         | HTTP / SSE (Port 8765)
                                         v
                       +-----------------+-----------------+
                       |       Offline RAG Backend        |
                       |             (app.py)              |
                       +----+------------+------------+----+
                            |            |            |
         HTTP (Port 8787)   |            |            |   HTTP (Port 8788)
   +------------------------+            |            +------------------------+
   |                                     |                                     |
   v                                     v                                     v
+--+---------------------+    +----------+----------+    +---------------------+--+
| Local Embedding Server |    |  LM Studio Server   |    | Local Reranker Server  |
|  (llama-server.exe)    |    | (OpenAI API /v1)    |    |  (llama-server.exe)    |
+------------------------+    +---------------------+    +------------------------+
```

### Server Port Allocation
* **Port 8765**: Main Application Server (Python `http.server.ThreadingHTTPServer`). Serves the web application and handles REST API and Server-Sent Events (SSE) endpoints.
* **Port 8787**: Local Embedding Server (`llama-server.exe`). Handles text vectorization requests.
* **Port 8788**: Local Reranker Server (`llama-server.exe`). Cross-encoder reranking endpoint.
* **Port 1234**: LM Studio Endpoint (Default: `http://127.0.0.1:1234/v1`). Provides chat completion models for query expansion, passage analysis, and final answer generation.

---

## Directory Structure

```
OfflineRAG/
├── app.py                      # Core Python HTTP server, pipeline, and PST backend
├── index.html                  # Single-page frontend interface
├── data/                       # System data, logs, index, and settings
│   ├── settings.json           # Active application configuration
│   ├── index.json              # Main document vector and lexical index
│   ├── document_manifest.json  # Fingerprint manifest for snapshots
│   ├── document_snapshots/     # Gzip-compressed extracted text snapshots
│   ├── lancedb/                 # LanceDB database directory (optional)
│   └── logs/                   # Server execution logs
├── models/                     # GGUF models for embedding and reranking
│   ├── nomic-embed-text-v1.5.Q4_K_M.gguf
│   ├── bge-reranker-v2-m3-Q4_K_M.gguf
│   └── catalog.json            # Model catalog configurations
├── runtime/                    # llama-server executable binaries & GPU DLLs
│   ├── llama-server.exe
│   ├── ggml-cuda.dll           # CUDA GPU acceleration library (optional)
│   └── ggml-vulkan.dll         # Vulkan GPU acceleration library (optional)
└── vendor/                     # Bundled Python library dependencies
```

---

## System Requirements

### Platform Support
* **Operating System**: Windows 10/11 (required for Outlook PST extraction via COM). Linux and macOS are supported for standard document processing and RAG operations.
* **Python**: Version 3.8 or higher.
* **RAM**: 8 GB minimum; 16 GB or more recommended when working with large document sets.
* **GPU (Optional)**: NVIDIA GPU with CUDA support or Vulkan-compatible GPU for accelerated local embeddings and reranking via `llama-server`.

### Optional Python Dependencies
The server uses standard library modules by default, but enhanced format support relies on the following packages:

```bash
# Basic optional capabilities
pip install numpy lancedb pypdf openpyxl python-docx xlrd

# Outlook MSG and PST support
pip install pywin32 extract-msg rapidfuzz
```

---

## Installation and Setup

### 1. Clone or Copy the Repository
Place `app.py` and `index.html` in your root project folder. Ensure the required directories (`data/`, `models/`, `runtime/`) exist or allow the application to create them at startup.

### 2. Add GGUF Local Models
Download the supported GGUF model files into the `models/` directory:

* **Embedding Model**: `nomic-embed-text-v1.5.Q4_K_M.gguf` or `bge-m3-Q4_K_M.gguf`
* **Reranker Model**: `bge-reranker-v2-m3-Q4_K_M.gguf`

*Note: You can also download these models directly through the application's Settings interface.*

### 3. Add llama-server Executables
Place `llama-server` (or `llama-server.exe` on Windows) inside the `runtime/` directory. Include any required acceleration DLLs (e.g., `ggml-cuda.dll` or `ggml-vulkan.dll`) in the same folder.

### 4. Configure LM Studio
1. Open **LM Studio**.
2. Load a fast small LLM (e.g., 0.5B to 3B parameters) for evidence analysis and a larger model for final answer generation.
3. Start the Local Server in LM Studio (default port `1234`).

---

## Running the Application

1. Launch the application server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8765/
   ```

---

## Ingesting Documents

### Standard Document Ingestion
1. Open the **Settings** panel (gear icon in top right).
2. Under the **Indexing** section, set the **Source folder** path containing your documents.
3. Specify the supported file extensions in **File types** (e.g., `.txt,.md,.pdf,.docx,.xlsx`).
4. Select an ingestion mode:
   * **Add / update**: Scans the folder, processes new or updated files, reuses existing vector embeddings, and retains previously indexed documents excluded by temporary extension filters.
   * **Rebuild from scratch**: Clears the index database and re-indexes all files from scratch.

### Outlook PST Archive Ingestion
1. In the **Settings** panel under **Outlook PST ingestion**, enter the full path to your `.pst` file.
2. Choose a **Processing mode**:
   * **Emails first**: Extracts email bodies and metadata (fastest).
   * **Attachments only**: Processes allowed file attachments with fuzzy filename deduplication.
   * **Emails and attachments together**: Fully extracts emails and matching attachments.
3. Click **Start selected mode**.
4. Once completed, run **Add / update** under Indexing to incorporate the extracted content into the main search database.

---

## Processing Pipeline Overview

When a user submits a query, the application executes a four-stage process:

1. **Understand Stage**:
   * Analyzes the question.
   * Generates search terms, optional query rewrites, and HyDE variants via the designated analysis LLM.

2. **Retrieve Stage**:
   * Performs hybrid lexical (BM25) and semantic vector search across the index.
   * Merges multi-query search results using Reciprocal Rank Fusion (RRF).
   * Passes the top candidates to the local cross-encoder reranker server.

3. **Analyze Stage**:
   * Evaluates candidate passages using the fast evidence analysis model.
   * Generates factual summaries and classifies each item (`ANSWERS`, `PARTIAL`, `RELATED`, `OFFTOPIC`).
   * Evaluates sufficiency at defined checkpoints. If information gaps exist, the engine formulates a revised query and initiates an additional retrieval wave.

4. **Answer Stage**:
   * Collects accepted evidence notes.
   * Streams the structured response from the final answer LLM, enforcing inline citations `[1]`, `[2]`, etc.

---

## API Reference

### GET /api/status
Returns index status, document counts, hardware details, and cache state.

### GET /api/settings
Returns active application configuration parameters.

### POST /api/settings
Updates application settings.
* **Payload**: JSON object containing configuration keys to update.

### GET /api/models
Queries LM Studio to list available chat completion models.

### GET /api/local-models
Returns catalog information and status for local embedding and reranker servers.

### POST /api/select-local-model
Switches the active embedding or reranking model.
* **Payload**: `{"kind": "embedding" | "reranker", "id": "<model_filename>"}`

### POST /api/answer-stream
Streams processing events, progress, and generated answers using Server-Sent Events (SSE).
* **Payload**:
  ```json
  {
    "query": "What are the flight departure times?",
    "answer": true,
    "adaptive": true
  }
  ```

### POST /api/ingest-stream
Triggers document scanning and vector ingestion via Server-Sent Events (SSE).
* **Payload**: `{"mode": "incremental" | "full"}`

### POST /api/pst/import
Initiates an asynchronous PST extraction job.
* **Payload**: `{"pst_path": "C:\\Path\\To\\Archive.pst"}`

### GET /api/pst/status?job=<job_id>
Polls status and progress of an active PST extraction task.

### GET /api/document-text
Extracts full text from a document or retrieves its saved snapshot.
* **Query Parameters**: `path`, `document_id`, `offset`, `limit`

### POST /api/document-relocate
Attempts to match a missing document against a replacement folder using SHA-256 fingerprint matching.
* **Payload**: `{"document_id": "<hash>", "folder": "D:\\NewLocation"}`

---

## Configuration Reference

Key application parameters stored in `data/settings.json`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `source_folder` | Project parent | Path to document directory for indexing. |
| `lmstudio_url` | `http://127.0.0.1:1234/v1` | URL endpoint for LM Studio server. |
| `chunk_size` | `900` | Target passage chunk size in characters. |
| `chunk_overlap` | `140` | Overlap between adjacent chunks in characters. |
| `candidate_count` | `32` | Initial candidate passages retrieved per query variant. |
| `rerank_count` | `4` | Number of analyzed passages per sufficiency checkpoint. |
| `max_candidate_checks` | `24` | Maximum candidate passages analyzed per query. |
| `semantic_weight` | `0.72` | Weighting factor assigned to semantic vector search. |
| `keyword_weight` | `0.28` | Weighting factor assigned to BM25 keyword search. |
| `use_llm_rerank` | `true` | Enables BGE local cross-encoder reranking. |
| `adaptive_rag` | `true` | Enables query rewrites, HyDE, and multi-wave search. |
| `use_lancedb` | `true` | Uses LanceDB storage if package is installed. |
| `gpu_offload` | `true` | Offloads llama-server layers to available GPU. |

---

## Troubleshooting

### Connection Error to Local Server
* Verify `llama-server.exe` exists in the `runtime/` directory.
* Check logs in `data/logs/embedding-server.log` or `data/logs/reranker-server.log`.

### LM Studio Models Not Visible
* Ensure the LM Studio local server is running on port `1234`.
* Confirm at least one model is loaded in LM Studio.

### Outlook PST Extraction Fails
* Verify Microsoft Outlook desktop application is installed and configured on Windows.
* Check required COM bindings by running `python -c "import win32com.client"`.

---

## License

This project is open-source and available under the MIT License.
