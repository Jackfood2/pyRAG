# Offline RAG — Local Document Intelligence

> **100% Private, Portable, and Local Retrieval-Augmented Generation Workstation**  
> Powered by an **Accumulate → Assess → Reformulate (CRAG-style)** pipeline, bundled GGUF embeddings, local reranking, and LM Studio integration.

---

## 📸 Overview

**Offline RAG** is a self-contained, fully offline document intelligence platform designed to ingest, index, analyze, and query your local documents without sending data to external APIs or cloud services.

It features a modern web workstation interface with real-time pipeline telemetry, interactive evidence cards with model output inspection, inline citation spotlighting, and streaming answers.

---

## ✨ Key Features

- 🔒 **100% Offline & Private:** Embeddings, reranking, and document processing run completely local on your CPU/GPU.
- 🔄 **Corrective / Adaptive RAG Pipeline:** Uses query expansion, HyDE (Hypothetical Document Embeddings), and iterative retrieval waves.
- 🎯 **Hybrid Retrieval Engine:** Combines dense semantic vector embeddings (`Nomic Embed v1.5`) with exact keyword search (`BM25`) using Reciprocal Rank Fusion (RRF).
- ⚡ **Bundled Reranking:** Re-scores retrieved passages using `BGE Reranker v2-m3` for ultra-precise relevance ranking.
- 🔍 **No-Drop Analysis Loop:** Every retrieved passage is analyzed, summarized, and tagged (`[ANSWERS]`, `[PARTIAL]`, `[RELATED]`, `[OFFTOPIC]`) by the local model—ensuring zero lost evidence.
- 🏷️ **Inline Citations & Evidence Spotlight:** Streamed answers feature clickable `[1]`, `[2]` citation tags that highlight the exact supporting source passage in real time.
- 📁 **Multi-Format Document Support:** Processes `.pdf`, `.docx`, `.xlsx`, `.xlsm`, `.csv`, `.md`, and `.txt` files.
- 🎛️ **Zero External Dependencies:** Portable Python environment bundled with local `llama-server.exe` binaries for background embedding and reranker inferencing.

---

## 📁 Repository Structure

```text
OfflineRAG/
├── data/                    # Local index storage & settings JSON
├── models/                  # GGUF models directory (populated via script)
│   ├── nomic-embed-text-v1.5.Q4_K_M.gguf
│   └── bge-reranker-v2-m3-Q4_K_M.gguf
├── python/                  # Portable Python runtime environment
├── runtime/                 # Portable llama-server binaries
├── vendor/                  # Bundled Python helper dependencies
├── app.py                   # RAG HTTP server & pipeline execution engine
├── index.html               # Web UI interface
├── Install_Models.bat       # Model downloader batch script
└── Start-Offline-RAG.bat    # Application launcher batch script
```

---

## ⚙️ Prerequisites & Requirements

- **OS:** Windows 10 or 11 (64-bit)
- **CPU:** Multi-core x86_64 processor (4+ CPU cores recommended)
- **RAM:** 8 GB minimum (16 GB+ recommended for large document collections)
- **LM Studio (Optional):** Required only if you want generated LLM text answers. The RAG retrieval, embedding, reranking, and evidence extraction work 100% locally even without LM Studio (Evidence-Only Mode).

---

## 🚀 Quick Start Guide

### Step 1: Download the RAG Models
Double-click **`Install_Models.bat`**.

This script automatically downloads the necessary GGUF embedding and reranker models into the `models/` directory using `curl`:
1. `nomic-embed-text-v1.5.Q4_K_M.gguf` (Embeddings)
2. `bge-reranker-v2-m3-Q4_K_M.gguf` (Reranker)

> **Note:** If a download fails due to network constraints, the script will automatically fallback to verified secondary mirrors.

---

### Step 2: (Optional) Launch LM Studio for Text Generation
If you want generated answers (in addition to retrieved evidence):
1. Open **LM Studio**.
2. Load any chat model of your choice (e.g., *Llama 3*, *Qwen 2.5*, *Mistral*, *Phi-3*).
3. Start the Local Inference Server on port **`127.0.0.1:1234`** (Default OpenAI-compatible endpoint).

*If LM Studio is not running, Offline RAG will automatically default to **Evidence-Only Mode**.*

---

### Step 3: Launch Offline RAG
Double-click **`Start-Offline-RAG.bat`**.

The launcher will automatically start three local processes in the background:
| Service | Endpoint | Role |
| :--- | :--- | :--- |
| **Embedding Server** | `http://127.0.0.1:8787` | Background `llama-server.exe` running Nomic Embed |
| **Reranker Server** | `http://127.0.0.1:8788` | Background `llama-server.exe` running BGE Reranker |
| **RAG Application** | `http://127.0.0.1:8765` | Python web application server |

Once initialized, your web browser will automatically open:
```text
http://127.0.0.1:8765
```

---

## 🛠️ How It Works (Pipeline Architecture)

Offline RAG employs an advanced 5-stage **Accumulate → Assess → Reformulate** architecture:

```
[01 UNDERSTAND] ──► [02 RETRIEVE] ──► [03 ANALYZE] ──► [04 ASSESS] ──► [05 ANSWER]
 Query Rewrite &      Hybrid Semantic     Passage Tagging     Check Sufficiency   Cited Streaming
 HyDE Expansion       + BM25 + Rerank      & Summarization     & Gap Search Wave    Answer Output
```

1. **01 UNDERSTAND (Query Optimization):**
   - Cleans and rewrites user prompts into targeted retrieval terms.
   - Generates HyDE (Hypothetical Document Embeddings) variations to capture conceptual matches.

2. **02 RETRIEVE (Fused Hybrid Search):**
   - Performs concurrent dense vector similarity search (`Nomic Embed`) and BM25 keyword matching.
   - Merges results using Reciprocal Rank Fusion (RRF) and re-scores candidates via `BGE Reranker v2-m3`.

3. **03 ANALYZE (No-Drop Passage Assessment):**
   - The model analyzes each passage and assigns a relevance tag (`[ANSWERS]`, `[PARTIAL]`, `[RELATED]`, `[OFFTOPIC]`).
   - Every passage is retained in memory—no raw text is discarded.

4. **04 ASSESS (Corrective Gap Loop):**
   - Evaluates accumulated notes against the question at designated checkpoints.
   - If evidence is incomplete, extracts the remaining query gap and executes a new corrective retrieval wave.

5. **05 ANSWER (Cited Generation):**
   - Generates a grounded, cited response using numbered inline references `[1]`, `[2]`.
   - Utilizes contrast notes (`[RELATED]` / `[OFFTOPIC]`) to explain document gaps if exact facts are missing, avoiding bare refusals.

---

## 🎛️ Configuration Options

Click the gear icon (**⚙**) in the top-right corner of the interface to adjust settings:

- **LM Studio Integration:**
  - Base URL (Default: `http://127.0.0.1:1234/v1`)
  - Model selection & hot-loading controls.
- **Indexing Options:**
  - Custom document folder path selection.
  - Supported file extension filters.
  - Passage size (chars) & passage overlap settings.
  - **Add / Update (Incremental):** Ingests new or modified documents while preserving existing cached vectors.
  - **Rebuild from Scratch:** Performs a complete clean re-indexing.
- **Retrieval Balance:**
  - Weighting slider for **Semantic Vectors** vs. **Exact Keywords**.
- **Assessment Parameters:**
  - Search candidate count, assess checkpoint intervals, and maximum hard limits.

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><b>Q: Do I need an internet connection to run this?</b></summary>
<b>A:</b> No. Once you run <code>Install_Models.bat</code> to download the GGUF models, the entire platform runs 100% offline without sending any data over the internet.
</details>

<details>
<summary><b>Q: What document types can I ingest?</b></summary>
<b>A:</b> Offline RAG supports <code>.txt</code>, <code>.md</code>, <code>.csv</code>, <code>.pdf</code>, <code>.docx</code>, <code>.xlsx</code>, and <code>.xlsm</code> files.
</details>

<details>
<summary><b>Q: Can I use Offline RAG without LM Studio?</b></summary>
<b>A:</b> Yes! If LM Studio is not connected, the application will operate in <b>Evidence Only</b> mode. It will retrieve, rank, analyze, and cite relevant passages directly from your documents.
</details>

<details>
<summary><b>Q: How do I change my document directory?</b></summary>
<b>A:</b> Open Settings (⚙ icon in top right), select or paste your desired directory under <b>Source folder</b>, and click <b>Add / update</b> or <b>Rebuild from scratch</b>.
</details>

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
```
