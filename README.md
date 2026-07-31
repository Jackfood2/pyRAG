OfflineRAG/
├── __pycache__/
├── data/
│   ├── document_manifest.json
│   ├── document_snapshots/
│   ├── logs/
│   │   ├── embedding-server.log
│   │   ├── pst_20260729_104312_ab4cb0ba.log
│   │   └── reranker-server.log
│   └── settings.json
├── models/
│   ├── bge-m3-Q4_K_M.gguf
│   ├── bge-reranker-v2-gemma.Q4_K_M.gguf
│   ├── bge-reranker-v2-m3-Q4_K_M.gguf
│   ├── catalog.json
│   ├── nomic-embed-text-v1.5.Q4_K_M.gguf
│   ├── nomic-embed-text-v1.5.Q8_0.gguf
│   └── snowflake-arctic-embed-l-v2.0-q4_k_m.gguf
├── python/                          # Embedded portable Python 3.11 environment
│   ├── python.exe
│   ├── python311.dll
│   ├── python311._pth
│   ├── Lib/site-packages/           # Bundled packages (pywin32, openpyxl, rapidfuzz, etc.)
│   └── Scripts/
├── runtime/                         # llama-server execution binaries and CPU/GPU DLLs
│   ├── llama-server.exe
│   ├── ggml.dll
│   └── [CPU/GPU backend DLLs]
├── vendor/                          # Bundled third-party libraries
│   └── pypdf/
├── Install_GPU.bat                  # One-click GPU runtime downloader (CUDA / Vulkan)
├── Install_LanceDB.bat              # Script to enable pip and install LanceDB
├── Install_Models.bat               # Automated downloader for default GGUF models
├── Start-Offline-RAG.bat            # One-click application launcher
├── README.md                        # Documentation file
├── app.py                           # Python backend application
└── index.html                       # Frontend user interface
```

---

### Complete Rewritten `README.md`

```markdown
# Offline RAG: Local Document Intelligence

Offline RAG is an enterprise-grade, workstation-based Retrieval-Augmented Generation (RAG) and Document Intelligence system. It operates 100% offline, keeping all documents, vector indices, search telemetry, and chat interactions strictly on your local machine.

The application includes an embedded portable Python environment, bundled `llama-server` model endpoints, and a lightweight web interface. It uses local GGUF models for vector embedding and cross-encoder reranking, and connects to local LLM engines like LM Studio for chat generation and document analysis.

---

## Repository Directory Structure

```
OfflineRAG/
├── __pycache__/
├── data/                            # Application logs, settings, index, and snapshots
│   ├── document_manifest.json       # SHA-256 fingerprint manifest for snapshot tracking
│   ├── document_snapshots/          # Gzip-compressed extracted text snapshots
│   ├── logs/                        # Server and PST extraction logs
│   ├── lancedb/                     # LanceDB vector database (created upon setup)
│   ├── settings.json                # System runtime configuration
│   └── index.json                   # In-memory JSON fallback vector index
├── models/                          # GGUF models for embedding and reranking
│   ├── catalog.json                 # Model catalog metadata
│   ├── nomic-embed-text-v1.5.Q4_K_M.gguf
│   └── bge-reranker-v2-m3-Q4_K_M.gguf
├── python/                          # Embedded portable Python 3.11 distribution
│   ├── python.exe
│   ├── python311._pth
│   └── Lib/site-packages/           # Bundled libraries (pywin32, openpyxl, rapidfuzz, etc.)
├── runtime/                         # Local llama-server execution engine
│   ├── llama-server.exe
│   ├── ggml.dll
│   └── [ggml-cuda.dll / ggml-vulkan.dll]
├── vendor/                          # Pure-Python fallback packages
│   └── pypdf/
├── Install_GPU.bat                  # Automatically installs CUDA or Vulkan llama.cpp runtime
├── Install_LanceDB.bat              # Installs LanceDB and NumPy into portable Python
├── Install_Models.bat               # Downloads default GGUF embedding and reranker models
├── Start-Offline-RAG.bat            # One-click launcher for Windows
├── README.md                        # Project documentation
├── app.py                           # Core Python backend server and RAG pipeline
└── index.html                       # Single-page web application interface
```

---

## Architectural Features

### 1. Zero-Cloud Privacy and Security
* All document processing, vectorization, reranking, and query handling are executed locally on your workstation.
* Network traffic is confined to `127.0.0.1` loopback interfaces. No external APIs or telemetry servers are contacted.

### 2. Hybrid Search Engine
* **BM25 Lexical Keyword Search**: Indexing engine with TF-IDF/BM25 scoring to match exact strings, serial numbers, codes, and names.
* **Vector Semantic Search**: High-dimensional cosine similarity matching powered by local GGUF embedding models.
* **Reciprocal Rank Fusion (RRF)**: Merges keyword and semantic result ranks into a unified candidate pool.
* **Cross-Encoder Reranking**: Re-evaluates retrieved candidates using local cross-encoder models (e.g., BGE Reranker v2-m3) via a separate llama-server instance.
* **LanceDB Storage Support**: Sub-millisecond vector querying via native LanceDB integration when installed.
* **RAM In-Memory Acceleration**: Uses native C-array memory buffers for fast vector searches when running without a vector database.

### 3. Adaptive Corrective RAG Pipeline
* **Query Expansion and HyDE**: Generates optimized search queries and Hypothetical Document Embeddings (HyDE) variants via an evidence analysis LLM.
* **Multi-Wave Corrective Retrieval**: If candidate passages are insufficient, the engine analyzes the missing information gap, generates revised search terms, and performs secondary retrieval waves.
* **Passage Analysis**: Summarizes individual passages and assigns explicit relevance classifications (`ANSWERS`, `PARTIAL`, `RELATED`, `OFFTOPIC`).
* **Sufficiency Checkpoints**: Periodically checks whether accumulated notes contain enough facts to answer the question, avoiding redundant LLM passes.

### 4. Comprehensive File Ingestion
* **Plaintext & Structured**: `.txt`, `.md`, `.csv`, `.json`, `.jsonl`
* **Documents & Presentations**: `.pdf`, `.docx`, `.doc` (via MS Word/Antiword), `.pptx`
* **Spreadsheets**: `.xlsx`, `.xlsm`, `.xls` (groups Excel rows into cohesive passage chunks)
* **Outlook Emails & Archives**: `.msg` files and full `.pst` archives using Outlook COM automation with conversation threading and fuzzy attachment deduplication.

### 5. Document Snapshots & Resilience
* **Gzip Text Snapshots**: Stores compressed full-text snapshots and SHA-256 content fingerprints during ingestion.
* **File Relocation Recovery**: If source files are moved or deleted, the system scans replacement directories and verifies content fingerprints to re-link original documents.
* **Document Viewer**: Interactive UI modal allows viewing passage chunks, extracted document text, or opening original files in Explorer.

---

## Port Allocation

* **Port 8765**: Application Web Interface and Backend Server (`app.py`).
* **Port 8787**: Embedded Local Embedding Server (`llama-server.exe`).
* **Port 8788**: Embedded Local Reranker Server (`llama-server.exe`).
* **Port 1234**: LM Studio Endpoint (Default: `http://127.0.0.1:1234/v1`).

---

## Quick Start Guide (Windows)

The repository includes a portable Python environment, pre-configured batch scripts, and embedded binaries for one-click setup on Windows.

### Step 1: Download Default Models
Run `Install_Models.bat` to download the default embedding model (`nomic-embed-text-v1.5.Q4_K_M.gguf`) and reranker model (`bge-reranker-v2-m3-Q4_K_M.gguf`) into the `models/` folder.

```cmd
Install_Models.bat
```

### Step 2: (Optional) Enable GPU Acceleration
By default, `llama-server` runs on CPU. To enable GPU offloading for NVIDIA (CUDA) or universal GPUs (Vulkan):

```cmd
Install_GPU.bat
```

This script detects your display hardware via `Win32_VideoController` and `nvidia-smi`, queries GitHub for the matching `llama.cpp` release binaries, backs up `runtime/` to `runtime_cpu_backup/`, and extracts the GPU-enabled server and DLLs.

### Step 3: (Optional) Enable LanceDB Vector Storage
To enable native LanceDB vector indexing for large document sets:

```cmd
Install_LanceDB.bat
```

This script enables site-packages in the portable Python environment (`python311._pth`) and installs `lancedb` and `numpy`.

### Step 4: Configure LM Studio
1. Launch **LM Studio**.
2. Load an Evidence Analysis model (a small, fast model like Qwen2.5-1.5B or Llama-3.2-3B is recommended) and a Final Answer model (such as Llama-3-8B or Mistral-7B).
3. Start the Local Server inside LM Studio on port `1234` (`http://127.0.0.1:1234/v1`).

### Step 5: Launch Offline RAG
Double-click `Start-Offline-RAG.bat` or run:

```cmd
Start-Offline-RAG.bat
```

The script starts `app.py` using the portable Python executable and opens `http://127.0.0.1:8765` in your default browser.

---

## Batch Scripts Summary

* `Start-Offline-RAG.bat`: Launches `app.py` in the background via `python\python.exe`, waits 3 seconds, and opens `http://127.0.0.1:8765` in your default web browser.
* `Install_Models.bat`: Downloads `nomic-embed-text-v1.5.Q4_K_M.gguf` and `bge-reranker-v2-m3-Q4_K_M.gguf` using `curl` from verified public Hugging Face mirrors.
* `Install_GPU.bat`: Embedded PowerShell installer wrapped in a batch file. Automatically selects between CUDA 12/13 or Vulkan, downloads the latest release from `ggml-org/llama.cpp`, and installs GPU runtime DLLs into `runtime/`.
* `Install_LanceDB.bat`: Configures site-packages access for the embedded portable Python engine and installs `lancedb` and `numpy`.

---

## Cross-Platform Setup (Linux / macOS / Non-Portable Python)

To run Offline RAG on Linux, macOS, or standard system Python installations:

1. **Install Dependencies**:
   ```bash
   pip install numpy lancedb pypdf openpyxl python-docx xlrd extract-msg rapidfuzz pywin32
   ```
2. **Setup Runtime Binaries**:
   Download the compiled `llama-server` binary for your operating system and place it inside the `runtime/` directory as `runtime/llama-server`.
3. **Download Models**:
   Place GGUF models in the `models/` directory.
4. **Run the Application**:
   ```bash
   python app.py
   ```

---

## Ingesting Documents

### Standard Document Ingestion
1. Open the **Settings** panel (gear icon in top right).
2. Set the **Source folder** path containing your documents.
3. Configure **File types** (e.g., `.txt,.md,.pdf,.docx,.xlsx,.msg`).
4. Select an action:
   * **Add / update**: Incrementally scans the target folder, processes new or modified files, reuses existing vector embeddings, and preserves previously indexed documents even if temporary file extension filters are changed.
   * **Rebuild from scratch**: Clears existing indices and rebuilds all vector representations from scratch.

### Outlook PST Ingestion
1. In the **Settings** panel under **Outlook PST Ingestion**, enter the full path to your `.pst` file (e.g., `C:\Users\Name\Documents\Archive.pst`).
2. Select a **Processing mode**:
   * **1. Emails first**: Extracts email message bodies and headers (fastest).
   * **2. Attachments only**: Extracts allowed file attachments with fuzzy filename deduplication.
   * **Emails and attachments together**: Processes both message bodies and attachments.
3. Click **Start selected mode**.
4. After extraction finishes, run **Add / update** under Indexing to index the output folder.

---

## Configuration Reference

Key application parameters (stored in `data/settings.json`):

| Key | Default | Description |
| :--- | :--- | :--- |
| `source_folder` | Root Parent | Path to the directory containing documents for indexing. |
| `lmstudio_url` | `http://127.0.0.1:1234/v1` | Endpoint for the local LM Studio server. |
| `analysis_model` | `""` | LM Studio model key for query planning and passage analysis. |
| `chat_model` | `""` | LM Studio model key for writing the final cited response. |
| `embedding_model` | `nomic-embed-text-v1.5.Q4_K_M.gguf` | GGUF model file used for text vectorization. |
| `rerank_model` | `bge-reranker-v2-m3-Q4_K_M.gguf` | GGUF model file used for cross-encoder reranking. |
| `chunk_size` | `900` | Target passage size in characters. |
| `chunk_overlap` | `140` | Character overlap between adjacent chunks. |
| `candidate_count` | `32` | Candidate passages retrieved per search query. |
| `rerank_count` | `4` | Passages evaluated per sufficiency assessment checkpoint. |
| `max_candidate_checks` | `24` | Hard cap on total candidate passages evaluated per query. |
| `semantic_weight` | `0.72` | Vector similarity weight in hybrid search. |
| `keyword_weight` | `0.28` | BM25 keyword weight in hybrid search. |
| `use_llm_rerank` | `true` | Enables BGE local cross-encoder reranking. |
| `adaptive_rag` | `true` | Enables query rewrites, HyDE, and multi-wave corrective search. |
| `use_lancedb` | `true` | Enables LanceDB vector storage when available. |
| `gpu_offload` | `true` | Offloads llama-server model layers to available GPU hardware. |

---

## API Endpoint Summary

* `GET /api/status`: Returns current document counts, memory index state, and hardware info.
* `GET /api/settings` / `POST /api/settings`: Reads or updates configuration options.
* `GET /api/models`: Queries LM Studio for available chat models.
* `GET /api/local-models`: Reports operational status for embedded embedding and reranker servers.
* `POST /api/select-local-model`: Switches active embedding or reranker models.
* `POST /api/answer-stream`: Streams multi-stage execution updates and final cited answers via Server-Sent Events (SSE).
* `POST /api/ingest-stream`: Streams progress events during document scanning and indexing.
* `POST /api/pst/import`: Starts an asynchronous Outlook PST extraction task.
* `GET /api/pst/status?job=<job_id>`: Returns status and progress for an active PST job.
* `GET /api/document-text`: Retrieves full extracted text or snapshot for a document.
* `POST /api/document-relocate`: Matches a missing document against a new folder using SHA-256 content fingerprints.

---

## License

This project is open-source and licensed under the MIT License.
```
