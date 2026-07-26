# README.md

```markdown
# Offline RAG — Local Document Intelligence

> **100% Private, Portable, and Local Retrieval-Augmented Generation Workstation**  
> Powered by an **Accumulate → Assess → Reformulate (CRAG-style)** pipeline, dual-LLM task delegation, bundled GGUF embeddings, local reranking, and LM Studio integration.

---

## 🆕 What's New & Key Updates

### ⚡ Dual-Model Local Inference Engine
Offline RAG now supports split-model execution in LM Studio to maximize both processing speed and answer quality:
- **Evidence Analysis Model (0.5B – 3B):** A small, ultra-fast local LLM handles low-latency tasks: query rewriting, passage summarization, tag classification (`[ANSWERS]`, `[PARTIAL]`, `[RELATED]`, `[OFFTOPIC]`), and assessment checkpoint checks.
- **Final Answer Model (7B+):** A larger, higher-capacity LLM reads the collated factual notes and generates the final, cited response.

### 🧩 Multi-Source Fact Synthesis
- **Complementary Passage Merging:** The prompt engine now instructs the answer LLM to combine distinct details from multiple matching passages (e.g., pulling names from Source A, fare details from Source B, and confirmation numbers from Source C).
- **Auto-Retry Validation:** If the pipeline detects that the answer model anchored onto a single source despite multiple contributing passages being available, it automatically triggers a synthesis retry.

### 🔌 Verified LM Studio Loader & Telemetry
- **One-Click Hot-Loading:** Select and load models directly from the settings panel with real-time status toasts and verification checks.
- **Model Telemetry Logging:** Every run logs the active analysis and answer models directly into the pipeline console.
- **UI Enhancements:** Added an evidence overview banner and raw model output toggles on evidence cards.

---

## 📸 Overview

**Offline RAG** is a complete, self-contained document intelligence workstation designed to index, search, and analyze your local files without sending any data over the internet or relying on external cloud APIs.

It features a modern web interface with live pipeline telemetry, interactive evidence cards, model raw output inspection, citation spotlighting, and streaming answers.

---

## ✨ Core Features

- 🔒 **100% Offline & Private:** Vector embeddings, reranking, and text analysis run locally on your hardware.
- 🔄 **Corrective / Adaptive Pipeline:** Uses query planning, HyDE (Hypothetical Document Embeddings), and iterative retrieval waves.
- 🎯 **Hybrid Search Engine:** Combines dense semantic vector similarity (`Nomic Embed v1.5`) with lexical keyword search (`BM25`) using Reciprocal Rank Fusion (RRF).
- ⚡ **Bundled Local Reranker:** Re-scores retrieved candidates using `BGE Reranker v2-m3`.
- 🔍 **No-Drop Analysis Loop:** Every retrieved passage is analyzed, summarized, and categorized—ensuring zero lost context.
- 🏷️ **Inline Citations & Evidence Spotlight:** Clickable `[1]`, `[2]` citation tags highlight supporting source cards in real time.
- 📁 **Multi-Format Ingestion:** Supports `.pdf`, `.docx`, `.xlsx`, `.xlsm`, `.csv`, `.md`, and `.txt` files.
- 📦 **Zero Python Installation Required:** Bundled portable Python runtime and `llama-server.exe` binaries.

---

## 📁 Repository Structure

```text
OfflineRAG/
├── data/                    # Local vector index storage & user settings JSON
├── models/                  # GGUF model binaries directory
│   ├── nomic-embed-text-v1.5.Q4_K_M.gguf
│   └── bge-reranker-v2-m3-Q4_K_M.gguf
├── python/                  # Portable Python runtime environment
├── runtime/                 # Portable llama-server binaries
├── vendor/                  # Python helper dependencies
├── app.py                   # RAG backend server & execution engine
├── index.html               # Web UI workstation interface
├── Install_Models.bat       # Automatic model installer script
└── Start-Offline-RAG.bat    # Workstation launcher script
```

---

## ⚙️ System Requirements

- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Processor:** 4+ CPU cores (x86_64)
- **RAM:** 8 GB minimum (16 GB+ recommended)
- **LM Studio (Optional):** Required only for LLM answer generation and passage summarization. RAG retrieval, embeddings, and reranking operate 100% locally even without LM Studio (Evidence-Only Mode).

---

## 🚀 Installation & Quick Start

### Step 1: Download RAG Models (`Install_Models.bat`)
Double-click **`Install_Models.bat`**.

This batch script automatically creates the `models/` directory and downloads the required GGUF files using `curl`:
1. **Embedding Model:** `nomic-embed-text-v1.5.Q4_K_M.gguf`
2. **Reranker Model:** `bge-reranker-v2-m3-Q4_K_M.gguf`

> **Note:** The installer includes automatic fallback mirrors in case primary downloads are blocked or slow.

---

### Step 2: Set Up LM Studio (Optional)
To enable generated answers and automated passage analysis:
1. Launch **LM Studio**.
2. Load your desired model(s). For optimal performance, load:
   - A fast model (e.g., `Qwen2.5-0.5B` or `Llama-3.2-1B`) for **Evidence Analysis**.
   - A larger model (e.g., `Llama-3.8B`, `Qwen2.5-7B`, or `Mistral-7B`) for **Final Answer Generation**.
3. Start the Local Server on port **`127.0.0.1:1234`** (Default API endpoint).

---

### Step 3: Launch Offline RAG (`Start-Offline-RAG.bat`)
Double-click **`Start-Offline-RAG.bat`**.

The launcher automatically spins up three background processes:
| Process | Port / Endpoint | Description |
| :--- | :--- | :--- |
| **Embedding Server** | `http://127.0.0.1:8787` | Local `llama-server.exe` running Nomic Embed |
| **Reranker Server** | `http://127.0.0.1:8788` | Local `llama-server.exe` running BGE Reranker |
| **RAG Application** | `http://127.0.0.1:8765` | Portable Python server running `app.py` |

Your default web browser will automatically open:
```text
http://127.0.0.1:8765
```

---

## 🛠️ Pipeline Architecture

Offline RAG processes queries through a 5-stage pipeline:

```
[01 UNDERSTAND] ──► [02 RETRIEVE] ──► [03 ANALYZE] ──► [04 ASSESS] ──► [05 ANSWER]
 Query Rewrite &      Hybrid Semantic     Passage Tagging     Check Sufficiency   Cited Streaming
 HyDE Expansion       + BM25 + Rerank      & Summarization     & Gap Search Wave    Answer Output
```

1. **01 UNDERSTAND:** Query optimizer generates alternative search queries and hypothetical answers (HyDE).
2. **02 RETRIEVE:** Performs dense similarity search and lexical BM25 matching, combined via Reciprocal Rank Fusion (RRF) and re-scored with `BGE Reranker v2-m3`.
3. **03 ANALYZE:** Every passage is summarized and tagged (`[ANSWERS]`, `[PARTIAL]`, `[RELATED]`, `[OFFTOPIC]`) by the local analysis model. No passages are discarded.
4. **04 ASSESS:** Evaluates accumulated notes against the query. If details are missing, extracts the factual gap and launches a new search wave.
5. **05 ANSWER:** Synthesizes notes into a final response with clickable `[1]` citations. Combines information across multiple sources and uses contrast notes to explain document gaps.

---

## 🎛️ Settings & Customization

Click the **⚙ Settings** icon in the top-right corner to configure:

- **LM Studio Integration:**
  - Base API URL (`http://127.0.0.1:1234/v1`).
  - **Evidence Analysis Model:** Dedicated drop-down & load button for small/fast LLMs.
  - **Final Answer Model:** Dedicated drop-down & load button for answer generation LLMs.
- **Document Indexing:**
  - Source folder selector & extension whitelist (`.pdf,.docx,.xlsx,.txt,.md,.csv`).
  - Chunk size & overlap parameters.
  - **Add / Update:** Incremental index update reusing existing embeddings.
  - **Rebuild from Scratch:** Clean index rebuild.
- **Retrieval Balance:**
  - Slider to balance Semantic Vector weighting vs. Exact Keyword (BM25) search.
- **Loop Parameters:**
  - Search candidate count, assess checkpoint frequency, and candidate check limits.

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><b>Q: Do I need internet access after setup?</b></summary>
<b>A:</b> No. Once <code>Install_Models.bat</code> finishes downloading the model files, Offline RAG runs completely disconnected from the internet.
</details>

<details>
<summary><b>Q: Can I run this without LM Studio?</b></summary>
<b>A:</b> Yes. Without LM Studio, Offline RAG runs in <b>Evidence Only</b> mode, retrieving, reranking, and displaying annotated document passages without generating a synthesized answer.
</details>

<details>
<summary><b>Q: How do I change the target folder for document ingestion?</b></summary>
<b>A:</b> Open Settings (⚙), pick or enter a directory path under <b>Source folder</b>, and click <b>Add / update</b> or <b>Rebuild from scratch</b>.
</details>

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
```
