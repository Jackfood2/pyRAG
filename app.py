#!/usr/bin/env python3

import array
import errno
import atexit
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse, parse_qs
import uuid
import tempfile
import hashlib
import gzip
from datetime import datetime
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Try to load LanceDB if the user ran Install_LanceDB.bat
try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
SETTINGS_FILE, INDEX_FILE = DATA / "settings.json", DATA / "index.json"

DEFAULT_EMBED_FILE = "nomic-embed-text-v1.5.Q4_K_M.gguf"
DEFAULT_RERANK_FILE = "bge-reranker-v2-m3-Q4_K_M.gguf"

EMBEDDING_URL = "http://127.0.0.1:8787"
RERANK_URL = "http://127.0.0.1:8788"

MAX_WAVES = 5                 
WAVE_EXTRA_CANDIDATES = 8     
ASSESS_CONFIDENCE_STOP = 0.8  

DEFAULTS = {
    "source_folder": str(ROOT.parent),
    "lmstudio_url": "http://127.0.0.1:1234/v1",
    "embedding_model": DEFAULT_EMBED_FILE,
    "analysis_model": "",          
    "chat_model": "",              
    "rerank_model": DEFAULT_RERANK_FILE,
    "chunk_size": 900,
    "chunk_overlap": 140,
    "candidate_count": 32,
    "rerank_count": 4,                 
    "max_candidate_checks": 24,        
    "semantic_weight": 0.72,
    "keyword_weight": 0.28,
    "use_llm_rerank": True,
    "answer_temperature": 0.1,
    "answer_tokens": 900,
    "include_extensions": ".txt,.md,.csv,.json,.jsonl,.pdf,.doc,.docx,.xls,.xlsx,.xlsm,.pptx,.msg",
    "max_row_chars": 5000,
    "adaptive_rag": True,
    "use_query_rewrite": True,
    "use_hyde": True,
    "use_lancedb": True, 
    "gpu_offload": True,  
    "gpu_layers": 99,   
    "local_server_parallel": 1,
    "embedding_workers": 1,
    "local_server_batch": 2048,
    "fast_path_score": 0.82,           
    "memory_fact_limit": 18,
    "context_window": 8192,
    "pst_path": "",
    "pst_similarity_threshold": 90,
    "pst_attachment_extensions": ".msg,.pdf,.xlsx,.xlsm,.docx",
    "pst_extract_attachments": True,
    "pst_processing_mode": "emails_only",
    "pst_stall_warning_seconds": 30,
}

lock = threading.Lock()
_JSON_LOCKS = {}
_JSON_LOCKS_GUARD = threading.Lock()
_EMBED_CACHE = {}
_DOCUMENT_TEXT_CACHE = {}
_DOCUMENT_TEXT_CACHE_LOCK = threading.Lock()
DOCUMENT_TEXT_CACHE_LIMIT = 12
DOCUMENT_VIEW_PAGE_CHARS = 200000
SNAPSHOT_DIR = DATA / "document_snapshots"
SNAPSHOT_MANIFEST_FILE = DATA / "document_manifest.json"
SNAPSHOT_DIR.mkdir(exist_ok=True)
SNAPSHOT_MANIFEST_LOCK = threading.RLock()

# --------------------------------------------------------------------------- #
#  RAM IN-MEMORY INDEX CACHE (Lightning fast retrieval)
# --------------------------------------------------------------------------- #
_INDEX_CACHE = None
_CACHE_MTIME = 0

def get_cached_index():
    global _INDEX_CACHE, _CACHE_MTIME
    if not INDEX_FILE.exists():
        return {"chunks": [], "lexical": {"lengths": [], "average_length": 1, "document_frequency": {}, "postings": {}}}
    
    mtime = INDEX_FILE.stat().st_mtime_ns
    if _INDEX_CACHE is not None and _CACHE_MTIME == mtime:
        return _INDEX_CACHE

    # Load 100MB+ JSON from disk ONLY ONCE into RAM
    data = load(INDEX_FILE, {"chunks": []})
    
    # Pre-allocate binary array buffers for ultra-fast cosine similarity loop
    for chunk in data.get("chunks", []):
        if "embedding" in chunk and isinstance(chunk["embedding"], list):
            chunk["_vec_buf"] = array.array('f', chunk["embedding"])

    _INDEX_CACHE = data
    _CACHE_MTIME = mtime
    return _INDEX_CACHE


def invalidate_index_cache():
    global _INDEX_CACHE, _CACHE_MTIME
    _INDEX_CACHE = None
    _CACHE_MTIME = 0

# --------------------------------------------------------------------------- #
#  local model catalog + llama-server process management
# --------------------------------------------------------------------------- #
RUNTIME_DIR = ROOT / "runtime"
LLAMA_SERVER = RUNTIME_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")
MODELS_DIR = ROOT / "models"
LOG_DIR = DATA / "logs"

def detect_gpu_backend():
    """Return 'cuda', 'vulkan' or 'cpu' from the backend DLLs sitting in runtime/."""
    if not RUNTIME_DIR.exists():
        return "cpu"
    names = {p.name.lower() for p in RUNTIME_DIR.iterdir()}
    if any(n == "ggml-cuda.dll" or n.startswith("ggml-cuda") for n in names):
        return "cuda"
    if any(n == "ggml-vulkan.dll" or n.startswith("ggml-vulkan") for n in names):
        return "vulkan"
    return "cpu"

EMBEDDING_PORT = 8787
RERANK_PORT = 8788

EST_CHARS_PER_TOKEN = 2.5     
CTX_SAFETY = 0.9              
RERANK_QUERY_RESERVE = 384    
MAX_SERVER_CTX = 8192

BUILTIN_CATALOG = [
    {"id": "nomic-embed-text-v1.5.Q4_K_M.gguf", "kind": "embedding",
     "name": "Nomic Embed v1.5 (Q4)", "ctx": 2048, "verified": True,
     "urls": ["https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"]},
    {"id": "bge-m3-Q4_K_M.gguf", "kind": "embedding",
     "name": "BGE-M3 multilingual (Q4)", "ctx": 8192, "verified": True,
     "urls": ["https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q4_K_M.gguf"]},
    {"id": "mxbai-embed-large-v1.Q4_K_M.gguf", "kind": "embedding",
     "name": "mxbai Embed Large v1 (Q4)", "ctx": 512, "verified": True,
     "urls": ["https://huggingface.co/ChristianAzinn/mxbai-embed-large-v1-gguf/resolve/main/mxbai-embed-large-v1.Q4_K_M.gguf"]},
    {"id": "snowflake-arctic-embed-l-v2.0-q4_k_m.gguf", "kind": "embedding",
     "name": "Snowflake Arctic Embed L v2.0 (Q4)", "ctx": 8192, "verified": False,
     "urls": ["https://huggingface.co/Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf/resolve/main/snowflake-arctic-embed-l-v2.0-q4_k_m.gguf"]},
    {"id": "bge-reranker-v2-m3-Q4_K_M.gguf", "kind": "reranker",
     "name": "BGE Reranker v2-m3 (Q4)", "ctx": 8192, "verified": True,
     "urls": ["https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf"]},
    {"id": "bge-reranker-v2-gemma.Q4_K_M.gguf", "kind": "reranker",
     "name": "BGE Reranker v2 Gemma (Q4)", "ctx": 8192, "verified": False,
     "urls": ["https://huggingface.co/mradermacher/bge-reranker-v2-gemma-GGUF/resolve/main/bge-reranker-v2-gemma.Q4_K_M.gguf"]},
]

LEGACY_EMBED_LABELS = {"Bundled Nomic Embed v1.5": DEFAULT_EMBED_FILE}

def normalize_embed_id(value):
    if not value:
        return DEFAULT_EMBED_FILE
    return LEGACY_EMBED_LABELS.get(value, value)

def load_catalog():
    cat = [dict(m) for m in BUILTIN_CATALOG]
    override = MODELS_DIR / "catalog.json"
    if override.exists():
        try:
            extra = json.loads(override.read_text(encoding="utf-8"))
            if isinstance(extra, list):
                ids = {m["id"] for m in cat}
                for m in extra:
                    if isinstance(m, dict) and m.get("id") and m["id"] not in ids:
                        cat.append(m)
        except Exception:
            pass
    return cat

def catalog_entry(file_id):
    for m in load_catalog():
        if m["id"] == file_id:
            return m
    return None

def model_max_chars(ctx, kind):
    usable = max(64.0, float(ctx) * CTX_SAFETY)
    if kind == "reranker":
        usable = max(64.0, usable - RERANK_QUERY_RESERVE)
    return int(usable * EST_CHARS_PER_TOKEN)

class LocalServer:
    def __init__(self, kind, port):
        self.kind = kind
        self.port = port
        self.proc = None
        self.file = None
        self.nominal_ctx = None
        self.eff_ctx = None
        self.state = "off"
        self.error = ""
        self._lock = threading.Lock()
        self.gpu_layers = 0          # layers actually offloaded this start (0 = CPU)
        self.gpu_fell_back = False   # True if we tried GPU and had to drop to CPU
        self.parallel = 1             # concurrent llama-server slots
        self.physical_batch = 2048    # tokens processed in one physical batch

    def build_args(self, threads, layers=0, parallel=1, physical_batch=2048):
        # llama-server divides the total context across parallel slots. Allocate
        # nominal_ctx per slot so increasing concurrency does not silently turn an
        # 8192-token model into 2048 tokens per request.
        per_slot_ctx = min(int(self.nominal_ctx or 2048), MAX_SERVER_CTX)
        parallel = positive(parallel, 1, 1, 16)
        total_ctx = per_slot_ctx * parallel
        # Context (-c) and physical batch (-b/-ub) are independent limits.
        # llama-server can have an 8K context yet reject a 529-token request when
        # the physical batch is only 512. Keep batch and micro-batch equal for
        # embedding/reranking to avoid llama.cpp reducing n_batch back to n_ubatch.
        physical_batch = positive(physical_batch, 2048, 512, MAX_SERVER_CTX)
        physical_batch = min(physical_batch, per_slot_ctx)
        args = [str(LLAMA_SERVER), "-m", str(MODELS_DIR / self.file),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-t", str(threads), "-c", str(total_ctx), "-np", str(parallel),
            "-b", str(physical_batch), "-ub", str(physical_batch)]
        if layers and layers > 0:
            args += ["-ngl", str(int(layers))]      # GPU layer offload
        args.append("--embedding" if self.kind == "embedding" else "--reranking")
        return args

    def stop(self):
        with self._lock:
            proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        self.state = "off"

    def start(self, file_id, nominal_ctx, threads, gpu_layers=0, parallel=1, physical_batch=2048):
        self.stop()
        self.file = file_id
        self.nominal_ctx = nominal_ctx
        self.eff_ctx = None
        self.error = ""
        self.gpu_layers = int(gpu_layers or 0)
        self.gpu_fell_back = False
        self.parallel = positive(parallel, 1, 1, 16)
        self.physical_batch = positive(physical_batch, 2048, 512, MAX_SERVER_CTX)
        if not (MODELS_DIR / file_id).exists():
            self.state = "missing"
            self.error = f"{file_id} is not in the models folder yet."
            return False
        if not LLAMA_SERVER.exists():
            self.state = "error"
            self.error = f"llama-server binary not found at {LLAMA_SERVER}"
            return False
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOG_DIR / f"{self.kind}-server.log", "ab", buffering=0)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(self.build_args(threads, self.gpu_layers, self.parallel, self.physical_batch),
                stdin=subprocess.DEVNULL, creationflags=flags)
        except Exception as exc:
            self.state = "error"
            self.error = str(exc)
            return False
        self.state = "starting"
        return True

    def start_with_fallback(self, file_id, nominal_ctx, threads, gpu_layers, parallel=1, physical_batch=2048):
        """Start on GPU if requested; if the GPU fails to init, retry on CPU."""
        ok = self.start(file_id, nominal_ctx, threads, gpu_layers, parallel, physical_batch)
        if ok:
            ok = self.wait_ready(120)
        if (not ok) and gpu_layers and gpu_layers > 0:
            self.gpu_fell_back = True
            print(f"[{self.kind}] GPU offload failed to start - falling back to CPU.")
            ok = self.start(file_id, nominal_ctx, threads, 0, parallel, physical_batch)
            if ok:
                ok = self.wait_ready(120)
        return ok

    def health(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.state = "error"
            self.error = self.error or "server process exited unexpectedly (see data/logs)."
            return "off"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                data = json.load(r)
            return "ok" if data.get("status") == "ok" else "loading"
        except urllib.error.HTTPError:
            return "loading"
        except Exception:
            return "off"

    def wait_ready(self, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.health() == "ok":
                self.state = "ok"
                self.eff_ctx = self.read_effective_ctx(self.nominal_ctx)
                return True
            if self.state == "error":
                return False
            time.sleep(0.5)
        self.state = "error"
        self.error = self.error or "timed out waiting for the model to load."
        return False

    def read_effective_ctx(self, fallback):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/props", timeout=3) as r:
                data = json.load(r)
            for src in (data, data.get("default_generation_settings", {})):
                if isinstance(src, dict) and src.get("n_ctx"):
                    return int(src["n_ctx"])
        except Exception:
            pass
        return fallback

    def describe(self):
        return {"kind": self.kind, "port": self.port, "file": self.file,
            "state": self.state, "error": self.error,
            "nominal_ctx": self.nominal_ctx, "effective_ctx": self.eff_ctx,
            "gpu_backend": detect_gpu_backend(),
            "gpu_layers": self.gpu_layers,
            "gpu_fell_back": self.gpu_fell_back, "parallel": self.parallel,
            "physical_batch": self.physical_batch}

EMBED_SERVER = LocalServer("embedding", EMBEDDING_PORT)
RERANK_SERVER = LocalServer("reranker", RERANK_PORT)

def _server_for(kind):
    return EMBED_SERVER if kind == "embedding" else RERANK_SERVER

def _start_one(kind, cfg):
    file_id = cfg.get("embedding_model" if kind == "embedding" else "rerank_model") or \
        (DEFAULT_EMBED_FILE if kind == "embedding" else DEFAULT_RERANK_FILE)
    entry = catalog_entry(file_id) or {"id": file_id, "ctx": 2048}
    srv = _server_for(kind)
    gpu_on = bool(cfg.get("gpu_offload", True)) and detect_gpu_backend() != "cpu"
    layers = positive(cfg.get("gpu_layers"), 99, 0, 999) if gpu_on else 0
    parallel = positive(cfg.get("local_server_parallel"), 1, 1, 16)
    physical_batch = positive(cfg.get("local_server_batch"), 2048, 512, MAX_SERVER_CTX)
    srv.start_with_fallback(entry["id"], entry.get("ctx", 2048), HARDWARE["cores"], layers, parallel, physical_batch)

def _start_local_servers(cfg, wait=False):
    def run():
        _start_one("embedding", cfg)
        _start_one("reranker", cfg)
    if wait:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()

def _shutdown_local_servers():
    EMBED_SERVER.stop()
    RERANK_SERVER.stop()


_FIELD = {
    "departure", "depart", "departs", "arrival", "arrive", "arrives", "date", "time", "datetime",
    "day", "flight", "flightno", "terminal", "gate", "status", "price", "fare", "from", "to",
    "origin", "destination", "pnr", "booking", "reference", "ref", "seat", "class", "duration",
    "carrier", "airline", "vessel", "train", "bus", "route", "scheduled", "etd", "eta",
    "ticket", "confirmation",
}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december",
           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_DOW = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"}

REL_RANK = {"ANSWERS": 3, "PARTIAL": 2, "RELATED": 1, "OFFTOPIC": 0}
REL_LABEL = {
    "ANSWERS":  "directly answers the question",
    "PARTIAL":  "on-topic, but the exact requested detail is partial/absent",
    "RELATED":  "same domain, different entity or direction",
    "OFFTOPIC": "unrelated — kept as a contrast note",
}
_TAG_RE = re.compile(r"\[\s*(ANSWERS|PARTIAL|RELATED|OFF\s*TOPIC|OFF_TOPIC|OFF-TOPIC)\s*\]", re.I)


def _norm_tag(token):
    t = token.upper().replace(" ", "").replace("-", "").replace("_", "")
    return t if t in REL_RANK else "RELATED"


# --------------------------------------------------------------------------- #
#  storage / settings / text helpers
# --------------------------------------------------------------------------- #
def load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def settings():
    config = {**DEFAULTS, **load(SETTINGS_FILE, {})}
    # Repair older settings files where the UI accidentally saved File types as blank.
    if not str(config.get("include_extensions") or "").strip():
        config["include_extensions"] = DEFAULTS["include_extensions"]
    if not config["source_folder"] or not Path(config["source_folder"]).is_dir():
        config["source_folder"] = DEFAULTS["source_folder"]
    if config.get("chat_model") == "auto":
        config["chat_model"] = ""
    if config.get("analysis_model") == "auto":
        config["analysis_model"] = ""
    return config


def _json_lock_for(path):
    key = str(Path(path).resolve()).lower() if os.name == "nt" else str(Path(path).resolve())
    with _JSON_LOCKS_GUARD:
        return _JSON_LOCKS.setdefault(key, threading.RLock())


def save_json(path, value):
    """Safely save JSON even when browser auto-save requests overlap on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_lock = _json_lock_for(path)
    with file_lock:
        # A unique file prevents concurrent requests from writing/renaming the same
        # settings.tmp. Keep it in the target directory so os.replace stays atomic.
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(json_safe(value), stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            last_error = None
            for attempt in range(8):
                try:
                    os.replace(str(tmp), str(path))
                    return
                except PermissionError as exc:
                    last_error = exc
                    # Windows may briefly hold the destination during antivirus,
                    # backup, indexing, or another request's replace operation.
                    if os.name != "nt":
                        raise
                    time.sleep(0.05 * (attempt + 1))
                except OSError as exc:
                    last_error = exc
                    if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33):
                        raise
                    time.sleep(0.05 * (attempt + 1))
            raise last_error or RuntimeError(f"Could not replace {path}")
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def words(value):
    return re.findall(r"[\w'-]+", value.lower(), re.UNICODE)


def positive(value, default, minimum=1, maximum=10000):
    try:
        return max(minimum, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


def file_signature(path):
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def chunk_params_sig(config):
    # File type selection controls which documents are scanned during this run;
    # it does not change how an existing passage was chunked or embedded.
    # Therefore include_extensions must NOT be part of this signature, otherwise
    # changing from .txt to .xlsx incorrectly forces a destructive full rebuild.
    keys = ("chunk_size", "chunk_overlap", "max_row_chars")
    base = "|".join(str(config.get(k)) for k in keys)
    return base + "|emb=" + normalize_embed_id(config.get("embedding_model"))


def json_safe(value):
    """Recursively convert vectors and database scalar types to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, array.array):
        return list(value)
    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "as_py"):
        try:
            return json_safe(value.as_py())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def public(item):
    return json_safe({k: v for k, v in item.items()
                      if k not in ("embedding", "vector", "_vec_buf")})


def hardware_profile():
    cores = os.cpu_count() or 4
    memory_gb = None
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory_gb = round(status.total_physical / 1024 ** 3)
        except Exception:
            pass
    workers = 1 if cores < 4 else min(4, max(2, cores // 2))
    return {
        "cores": cores,
        "memory_gb": memory_gb,
        "embedding_workers": workers,
        "label": f"{cores} logical CPU cores" + (f", {memory_gb} GB RAM" if memory_gb else ""),
    }


HARDWARE = hardware_profile()


# --------------------------------------------------------------------------- #
#  document readers
# --------------------------------------------------------------------------- #
def read_docx(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(n.text or "" for n in p.iter(ns + "t")) for p in root.iter(ns + "p"))


def col_name(n):
    result = ""
    while n:
        n, r = divmod(n - 1, 26)
        result = chr(65 + r) + result
    return result


def read_xlsx(path):
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet in wb.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            headers = [text(v) or col_name(i + 1) for i, v in enumerate(rows[0])]
            for number, row in enumerate(rows[1:], 2):
                values = [text(v) for v in row]
                if any(values):
                    yield sheet.title, number, " | ".join(
                        f"{headers[i]}: {v}" for i, v in enumerate(values) if v
                    )
        return
    except ImportError:
        pass
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = [
                "".join(t.text or "" for t in x.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
                for x in ET.fromstring(z.read("xl/sharedStrings.xml"))
            ]
        sheets = [(n, z.read(n)) for n in z.namelist()
                  if re.match(r"xl/worksheets/sheet\d+\.xml$", n)]
        for index, (_, raw) in enumerate(sheets, 1):
            root = ET.fromstring(raw)
            rows = []
            for row in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
                values = {}
                for cell in row:
                    ref = cell.attrib.get("r", "A1")
                    col = re.match(r"[A-Z]+", ref).group()
                    typ = cell.attrib.get("t")
                    value = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    v = value.text if value is not None else ""
                    try:
                        values[col] = shared[int(v)] if typ == "s" and v else v
                    except (ValueError, IndexError):
                        values[col] = v
                rows.append((int(row.attrib.get("r", len(rows) + 1)), values))
            if not rows:
                continue
            header = rows[0][1]
            for number, row in rows[1:]:
                line = " | ".join(f"{header.get(c, c)}: {v}" for c, v in row.items() if v)
                if line:
                    yield f"Sheet {index}", number, line


def pdf_fallback(path):
    raw = path.read_bytes()
    result = []
    for stream in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            stream = zlib.decompress(stream)
        except zlib.error:
            pass
        for token in re.findall(rb"\((?:\\.|[^\\)])*\)\s*Tj|\[(.*?)\]\s*TJ", stream, re.S):
            if isinstance(token, tuple):
                token = token[0]
            result.extend(re.findall(rb"\((.*?)\)", token) or [token[:-2]])
    decoded = " ".join(x.decode("latin-1", "ignore").replace("\\(", "(").replace("\\)", ")") for x in result)
    return text(decoded)


def read_pdf(path):
    try:
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    except ImportError:
        pass
    try:
        return subprocess.check_output(["pdftotext", str(path), "-"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return pdf_fallback(path)


def read_pptx(path):
    """Extract text from PowerPoint slides without requiring PowerPoint."""
    slide_re = re.compile(r"ppt/slides/slide(\d+)\.xml$")
    with zipfile.ZipFile(path) as z:
        slides = []
        for name in z.namelist():
            match = slide_re.match(name)
            if match:
                slides.append((int(match.group(1)), name))
        for slide_no, name in sorted(slides):
            root = ET.fromstring(z.read(name))
            values = [node.text or "" for node in root.iter()
                      if node.tag.endswith("}t") and (node.text or "").strip()]
            content = "\n".join(values).strip()
            if content:
                yield f"Slide {slide_no}", None, content


def read_json_lines(path):
    """Read JSON/JSONL as searchable, line-addressable text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(raw)
            raw = json.dumps(value, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    for number, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if line:
            yield "JSON", number, line


def read_legacy_doc(path):
    """Extract old binary .doc text using an available local converter."""
    commands = [["antiword", str(path)]]
    if os.name == "nt":
        commands += [["powershell", "-NoProfile", "-Command",
                      "$w=New-Object -ComObject Word.Application; $w.Visible=$false; "
                      f"$d=$w.Documents.Open('{str(path).replace(chr(39), chr(39)*2)}'); "
                      "$t=$d.Content.Text; $d.Close(); $w.Quit(); [Console]::OutputEncoding=[Text.Encoding]::UTF8; $t"]]
    for command in commands:
        try:
            return subprocess.check_output(command, text=True, encoding="utf-8",
                                           errors="replace", stderr=subprocess.DEVNULL,
                                           timeout=120)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError("Legacy .doc requires Microsoft Word or antiword. Convert the file to .docx if extraction fails.")


def read_legacy_xls(path):
    """Extract old .xls workbooks using xlrd when installed."""
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError(
            f"Legacy .xls found but xlrd is not installed. Install with: {sys.executable} -m pip install xlrd"
        ) from exc
    book = xlrd.open_workbook(str(path), on_demand=True)
    try:
        for sheet in book.sheets():
            if sheet.nrows == 0:
                continue
            headers = [text(sheet.cell_value(0, col)) or col_name(col + 1)
                       for col in range(sheet.ncols)]
            for row_no in range(1, sheet.nrows):
                values = [text(sheet.cell_value(row_no, col)) for col in range(sheet.ncols)]
                line = " | ".join(f"{headers[i]}: {value}"
                                  for i, value in enumerate(values) if value)
                if line:
                    yield sheet.name, row_no + 1, line
    finally:
        book.release_resources()


def read_msg(path):
    """Read a standalone Outlook .msg export when extract-msg is installed."""
    try:
        import extract_msg
    except ImportError as exc:
        raise RuntimeError(
            "Outlook .msg file found but extract-msg is not installed. "
            f"Install it with: {sys.executable} -m pip install extract-msg"
        ) from exc
    msg = extract_msg.Message(str(path))
    try:
        parts = [
            f"Subject: {msg.subject or ''}",
            f"From: {msg.sender or ''}",
            f"To: {msg.to or ''}",
            f"CC: {getattr(msg, 'cc', '') or ''}",
            f"Date: {msg.date or ''}",
            "",
            msg.body or "",
        ]
        return "\n".join(parts)
    finally:
        msg.close()

def split_text_chunks(content, size, overlap):
    """Create near-target chunks while preserving text and configured overlap.

    chunk_size is a maximum target, not a guaranteed minimum. Short source units
    stay short. Longer units use natural boundaries and a balanced final pair.
    """
    content = str(content or "").strip()
    if not content:
        return []
    if len(content) <= size:
        return [content]
    overlap = max(0, min(int(overlap), size // 2))
    pieces, start_pos, total = [], 0, len(content)
    while start_pos < total:
        remaining = total - start_pos
        if remaining <= size:
            tail = content[start_pos:].strip()
            if tail:
                pieces.append(tail)
            break
        if remaining <= (2 * size - overlap):
            first_len = min(size, max(overlap + 1, (remaining + overlap + 1) // 2))
            raw_end = start_pos + first_len
        else:
            raw_end = start_pos + size
        window = content[start_pos:raw_end]
        min_boundary = max(1, int(len(window) * 0.72))
        candidates = []
        for pattern in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
            pos = window.rfind(pattern, min_boundary)
            if pos >= 0:
                candidates.append(pos + len(pattern))
        end_pos = start_pos + (max(candidates) if candidates else len(window))
        piece = content[start_pos:end_pos].strip()
        if piece:
            pieces.append(piece)
        start_pos = max(start_pos + 1, end_pos - overlap)
        while start_pos < end_pos and content[start_pos].isspace():
            start_pos += 1
    return pieces


def quick_file_fingerprint(path, block_size=1048576):
    """Efficient move-detection fingerprint: size plus SHA-256 of head and tail."""
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(block_size))
        if stat.st_size > block_size:
            stream.seek(max(0, stat.st_size - block_size))
            digest.update(stream.read(block_size))
    return digest.hexdigest()


def snapshot_document_id(path, fingerprint=None):
    path = Path(path)
    fingerprint = fingerprint or quick_file_fingerprint(path)
    return hashlib.sha256((path.name.lower() + "|" + fingerprint).encode("utf-8", "replace")).hexdigest()


def load_snapshot_manifest():
    value = load(SNAPSHOT_MANIFEST_FILE, {"version": 1, "documents": {}})
    if not isinstance(value, dict):
        value = {"version": 1, "documents": {}}
    value.setdefault("version", 1)
    value.setdefault("documents", {})
    return value


def save_document_snapshot(path, source_root, extracted_text=None, manifest=None):
    """Persist compressed full extracted text and identity metadata for fallback use."""
    path = Path(path).resolve()
    source_root = Path(source_root).resolve()
    stat = path.stat()
    fingerprint = quick_file_fingerprint(path)
    document_id = snapshot_document_id(path, fingerprint)
    snapshot_path = SNAPSHOT_DIR / f"{document_id}.txt.gz"
    if extracted_text is None:
        extracted_text = extract_full_document_text(path)
    extracted_text = str(extracted_text or "").replace("\x00", "").strip()
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", errors="replace") as stream:
        stream.write(extracted_text)
    os.replace(tmp, snapshot_path)
    try:
        relative = str(path.relative_to(source_root))
    except ValueError:
        relative = path.name
    record = {
        "document_id": document_id, "filename": path.name,
        "original_path": str(path), "relative_path": relative,
        "size": stat.st_size, "modified_ns": stat.st_mtime_ns,
        "fingerprint": fingerprint, "snapshot": str(snapshot_path),
        "snapshot_chars": len(extracted_text),
        "snapshot_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_path": str(path), "status": "available",
    }
    if manifest is None:
        with SNAPSHOT_MANIFEST_LOCK:
            manifest = load_snapshot_manifest()
            manifest["documents"][document_id] = record
            save_json(SNAPSHOT_MANIFEST_FILE, manifest)
    else:
        manifest["documents"][document_id] = record
    return record


def read_document_snapshot(document_id):
    with SNAPSHOT_MANIFEST_LOCK:
        record = load_snapshot_manifest().get("documents", {}).get(str(document_id), {})
    if not record:
        raise FileNotFoundError("No extracted-text snapshot is available for this evidence.")
    snapshot = Path(record.get("snapshot", ""))
    if not snapshot.is_file():
        raise FileNotFoundError("The extracted-text snapshot file is missing.")
    with gzip.open(snapshot, "rt", encoding="utf-8", errors="replace") as stream:
        return stream.read(), record


def locate_snapshot_document(document_id, search_root):
    """Search a replacement root and report exact, missing and discrepant structure."""
    search_root = Path(search_root).expanduser().resolve(strict=True)
    if not search_root.is_dir():
        raise ValueError("The replacement path is not a folder.")
    with SNAPSHOT_MANIFEST_LOCK:
        manifest = load_snapshot_manifest()
        record = manifest.get("documents", {}).get(str(document_id), {})
    if not record:
        raise FileNotFoundError("Document identity is not present in the snapshot manifest.")
    expected_relative = Path(record.get("relative_path") or record.get("filename") or "")
    candidates, discrepancies = [], []
    direct = search_root / expected_relative
    paths = []
    if direct.is_file():
        paths.append(direct)
    filename = record.get("filename", "")
    if filename:
        try:
            paths.extend(p for p in search_root.rglob(filename) if p.is_file() and p not in paths)
        except OSError as exc:
            discrepancies.append(f"Folder scan error: {exc}")
    for candidate in paths[:500]:
        try:
            stat = candidate.stat()
            size_match = int(record.get("size", -1)) == stat.st_size
            fingerprint = quick_file_fingerprint(candidate) if size_match else ""
            fingerprint_match = bool(fingerprint and fingerprint == record.get("fingerprint"))
            item = {"path": str(candidate), "relative_path": str(candidate.relative_to(search_root)),
                    "size": stat.st_size, "size_match": size_match,
                    "fingerprint_match": fingerprint_match,
                    "structure_match": candidate == direct}
            candidates.append(item)
            if not fingerprint_match:
                discrepancies.append(f"Different content: {item['relative_path']}")
        except OSError as exc:
            discrepancies.append(f"Unreadable candidate {candidate}: {exc}")
    exact = next((item for item in candidates if item["fingerprint_match"]), None)
    if exact:
        with SNAPSHOT_MANIFEST_LOCK:
            manifest = load_snapshot_manifest()
            current = manifest["documents"].get(str(document_id), record)
            current["current_path"] = exact["path"]
            current["status"] = "relocated"
            current["relocated_root"] = str(search_root)
            current["relocated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            manifest["documents"][str(document_id)] = current
            save_json(SNAPSHOT_MANIFEST_FILE, manifest)
    return {"document_id": document_id, "search_root": str(search_root),
            "expected_relative_path": str(expected_relative), "exact_match": exact,
            "candidates": candidates, "discrepancies": discrepancies,
            "missing_expected_path": not direct.is_file()}


def extract_full_document_text(path):
    """Extract the complete readable text of one source document on demand."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md", ".csv"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in (".json", ".jsonl"):
        return "\n".join(line for _, _, line in read_json_lines(path))
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".doc":
        return read_legacy_doc(path)
    if ext == ".pptx":
        return "\n\n".join(f"[{section}]\n{content}" for section, _, content in read_pptx(path))
    if ext in (".xlsx", ".xlsm"):
        return "\n".join(f"[{sheet} row {row}] {line}" for sheet, row, line in read_xlsx(path))
    if ext == ".xls":
        return "\n".join(f"[{sheet} row {row}] {line}" for sheet, row, line in read_legacy_xls(path))
    if ext == ".msg":
        return read_msg(path)
    raise RuntimeError(f"Full-document text preview is not supported for {ext or 'this file type'}.")


def cached_full_document_text(path):
    """Return full extracted text, cached by absolute path, size and mtime."""
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    key = (str(path).lower() if os.name == "nt" else str(path), stat.st_size, stat.st_mtime_ns)
    with _DOCUMENT_TEXT_CACHE_LOCK:
        cached = _DOCUMENT_TEXT_CACHE.get(key)
        if cached is not None:
            cached["used"] = time.time()
            return cached["text"], True
    extracted = extract_full_document_text(path)
    extracted = str(extracted or "").replace("\x00", "").strip()
    with _DOCUMENT_TEXT_CACHE_LOCK:
        # Remove older signatures for the same file, then evict least recently used.
        for old_key in list(_DOCUMENT_TEXT_CACHE):
            if old_key[0] == key[0] and old_key != key:
                _DOCUMENT_TEXT_CACHE.pop(old_key, None)
        _DOCUMENT_TEXT_CACHE[key] = {"text": extracted, "used": time.time()}
        while len(_DOCUMENT_TEXT_CACHE) > DOCUMENT_TEXT_CACHE_LIMIT:
            oldest = min(_DOCUMENT_TEXT_CACHE, key=lambda k: _DOCUMENT_TEXT_CACHE[k]["used"])
            _DOCUMENT_TEXT_CACHE.pop(oldest, None)
    return extracted, False


def safe_source_document(requested):
    """Resolve a requested source path and restrict it to the configured source tree."""
    target = Path(requested).expanduser().resolve(strict=True)
    source_root = Path(settings()["source_folder"]).expanduser().resolve(strict=True)
    try:
        target.relative_to(source_root)
    except ValueError as exc:
        raise PermissionError("The requested file is outside the configured source folder.") from exc
    if not target.is_file():
        raise FileNotFoundError("Original document no longer exists.")
    return target


def chunks_for(path, config):
    ext = path.suffix.lower()
    source = path.name
    if ext in (".txt", ".md", ".csv"):
        units = [("Text", None, path.read_text(encoding="utf-8", errors="replace"))]
    elif ext in (".json", ".jsonl"):
        units = list(read_json_lines(path))
    elif ext == ".doc":
        units = [("Legacy Word document", None, read_legacy_doc(path))]
    elif ext == ".docx":
        units = [("Document", None, read_docx(path))]
    elif ext == ".pptx":
        units = list(read_pptx(path))
    elif ext == ".msg":
        units = [("Outlook message", None, read_msg(path))]
    elif ext == ".xls":
        raw_units = list(read_legacy_xls(path))
        units = raw_units
    elif ext in (".xlsx", ".xlsm"):
        raw_units = list(read_xlsx(path))
        units = []
        BATCH_SIZE = 15  # Group 15 Excel rows into 1 passage chunk
        current_batch = []
        current_sheet = None

        for sheet, row_num, line in raw_units:
            if current_sheet is not None and (sheet != current_sheet or len(current_batch) >= BATCH_SIZE):
                first_row = current_batch[0][1]
                last_row = current_batch[-1][1]
                row_range = f"Rows {first_row}-{last_row}" if len(current_batch) > 1 else first_row
                units.append((current_sheet, row_range, "\n".join(item[2] for item in current_batch)))
                current_batch = []
            
            current_sheet = sheet
            current_batch.append((sheet, row_num, line))

        if current_batch:
            first_row = current_batch[0][1]
            last_row = current_batch[-1][1]
            row_range = f"Rows {first_row}-{last_row}" if len(current_batch) > 1 else first_row
            units.append((current_sheet, row_range, "\n".join(item[2] for item in current_batch)))
    elif ext == ".pdf":
        units = [("PDF", None, read_pdf(path))]
    else:
        return []
    out = []
    for section, row, content in units:
        content = re.sub(r"[^\S\r\n]+", " ", str(content or ""))
        content = re.sub(r"[ \t]*\r?\n[ \t]*", "\n", content)
        content = re.sub(r"\n{2,}", "\n", content).strip()
        if not content:
            continue
        if row is not None:
            limit = positive(config.get("max_row_chars"), 5000, 100, 50000)
            out.append({
                "id": str(uuid.uuid4()), "source": source, "path": str(path),
                "section": section, "row": row, "chunk": 0,
                "chunk_number": 1, "chunk_count": 1,
                "chunk_chars": len(content[:limit]), "unit_chars": len(content),
                "configured_chunk_size": positive(config.get("chunk_size"), 900, 200, 12000),
                "text": content[:limit],
            })
            continue
        size = positive(config.get("chunk_size"), 900, 200, 12000)
        overlap = min(positive(config.get("chunk_overlap"), 140, 0, size - 1), size // 2)
        pieces = split_text_chunks(content, size, overlap)
        for position, piece in enumerate(pieces):
            out.append({
                "id": str(uuid.uuid4()), "source": source, "path": str(path),
                "section": section, "row": row, "chunk": position,
                "chunk_number": position + 1, "chunk_count": len(pieces),
                "chunk_chars": len(piece), "unit_chars": len(content),
                "configured_chunk_size": size, "text": piece,
            })
    return out


# --------------------------------------------------------------------------- #
#  LM Studio / local model transport
# --------------------------------------------------------------------------- #
def openai_base(config):
    base = str(config.get("lmstudio_url", "")).strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def api(config, endpoint, body=None, timeout=120):
    data = None if body is None else json.dumps(json_safe(body)).encode()
    req = urllib.request.Request(openai_base(config) + endpoint, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def lmstudio_api(config, endpoint, body=None, timeout=5):
    base = openai_base(config)[:-3].rstrip("/")
    data = None if body is None else json.dumps(json_safe(body)).encode()
    req = urllib.request.Request(base + endpoint, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def local_api(url, endpoint, body, timeout=120):
    req = urllib.request.Request(url + endpoint, json.dumps(json_safe(body)).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def resolved(config, purpose):
    """Resolve a model id for embedding or for the final answer (chat)."""
    if purpose == "embedding":
        return config.get("embedding_model") or DEFAULT_EMBED_FILE
    selected = config.get("chat_model", "")
    if selected:
        return selected
    if purpose == "chat":
        raise RuntimeError("Choose a final answer model in Settings, or use Evidence only.")
    try:
        available = lmstudio_api(config, "/api/v1/models", timeout=3).get("models", [])
        for model in available:
            if model.get("type") == "llm" and model.get("loaded_instances"):
                return model["loaded_instances"][0].get("id") or model["key"]
    except Exception:
        pass
    models = api(config, "/models", timeout=5).get("data", [])
    for model in models:
        name = model.get("id", "")
        if (purpose == "embedding") == ("embed" in name.lower()):
            return name
    raise RuntimeError(f"No loaded {purpose} model found in LM Studio.")


def analysis_model_for(config):
    """The model used for rewrite/analyze/assess. Falls back to the answer model."""
    return (config.get("analysis_model") or "").strip() or (config.get("chat_model") or "").strip()


def embed(config, inputs, tolerate_errors=False, on_progress=None):
    """Create normalized vectors with bounded waits and optional live heartbeat events."""
    ctx = EMBED_SERVER.eff_ctx or EMBED_SERVER.nominal_ctx or 2048
    max_chars = max(200, model_max_chars(ctx, "embedding"))
    request_timeout = positive(config.get("embedding_request_timeout"), 90, 15, 600)
    heartbeat_seconds = positive(config.get("embedding_heartbeat_seconds"), 5, 2, 60)

    def notify(event, **detail):
        if on_progress:
            try:
                on_progress({"event": event, **detail})
            except Exception:
                pass

    def request_vector(content, item_number):
        box = {}
        started = time.time()
        def runner():
            try:
                box["value"] = local_api(
                    EMBEDDING_URL, "/embedding",
                    {"content": content, "truncate": True},
                    timeout=request_timeout,
                )
            except BaseException as exc:
                box["error"] = exc
        worker = threading.Thread(target=runner, daemon=True,
                                  name=f"embedding-request-{item_number}")
        worker.start()
        while worker.is_alive():
            worker.join(heartbeat_seconds)
            if worker.is_alive():
                notify("heartbeat", item=item_number, elapsed=int(time.time() - started),
                       backend="CPU" if EMBED_SERVER.gpu_layers == 0 else detect_gpu_backend().upper(),
                       message=f"Embedding passage {item_number} is still running")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def one(number_item):
        number, item = number_item
        item = item[:max_chars]
        cached = _EMBED_CACHE.get(item)
        if cached is not None:
            notify("cached", item=number)
            return cached
        last_exc = None
        attempt = item[:min(len(item), max_chars, max(256, int(ctx * 0.88)))]
        # Only shrink/retry for an actual context/token overflow. Timeouts, refused
        # connections and server failures return immediately during tolerant ingestion.
        for retry in range(4):
            try:
                notify("start", item=number, chars=len(attempt), retry=retry)
                data = request_vector(attempt, number)
                entry = data[0] if isinstance(data, list) else data["value"][0]
                value = entry["embedding"] if isinstance(entry, dict) else entry
                if isinstance(value, list) and value and isinstance(value[0], list):
                    value = value[0]
                vector = [float(x) for x in value.split()] if isinstance(value, str) else value
                if not vector:
                    raise RuntimeError("embedding server returned an empty vector")
                magnitude = math.sqrt(sum(x * x for x in vector)) or 1.0
                vector = [x / magnitude for x in vector]
                if len(_EMBED_CACHE) > 4096:
                    _EMBED_CACHE.clear()
                _EMBED_CACHE[item] = vector
                notify("done", item=number, dimensions=len(vector))
                return vector
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                overflow = any(x in msg for x in ("context", "token", "too long", "exceed", "n_batch"))
                notify("retry" if overflow else "error", item=number, retry=retry,
                       error=f"{type(exc).__name__}: {exc}")
                if not overflow or len(attempt) <= 256:
                    break
                attempt = attempt[:max(256, len(attempt) // 2)]
        if tolerate_errors:
            return None
        raise last_exc or RuntimeError("embedding failed")

    workers = positive(config.get("embedding_workers"), 1, 1, 16)
    workers = min(workers, positive(config.get("local_server_parallel"), 1, 1, 16))
    numbered = list(enumerate(inputs, 1))
    if len(inputs) < 2 or workers == 1:
        return [one(pair) for pair in numbered]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, numbered))


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _variant_safe(variant, q_tokens):
    extra = set(words(variant)) - q_tokens
    if not extra:
        return True
    for tok in extra:
        if re.fullmatch(r"\d+(?:st|nd|rd|th)", tok):
            return False
        if re.fullmatch(r"\d{1,2}(?:[/-]\d{1,2}(?:[/-]\d{2,4})?)?", tok):
            return False
        if re.fullmatch(r"\d{4}", tok):
            return False
        if tok in _MONTHS or tok in _DOW:
            return False
    return True


def rerank_char_limit():
    ctx = RERANK_SERVER.eff_ctx or RERANK_SERVER.nominal_ctx or 8192
    return max(200, model_max_chars(ctx, "reranker"))

def rerank_once(query, items):
    if not items:
        return items
    limit = rerank_char_limit()
    try:
        ranked = local_api(RERANK_URL, "/rerank",
                           {"query": query, "documents": [it["text"][:limit] for it in items]})["results"]
        out = [items[r["index"]] for r in ranked if 0 <= r.get("index", -1) < len(items)]
        return out or items
    except Exception:
        return items


# ---- model introspection for verified loading ----
def _lm_models(config):
    try:
        return lmstudio_api(config, "/api/v1/models", timeout=4).get("models", [])
    except Exception:
        return []


def _candidate_ids(models, key):
    """All identifiers LM Studio knows this model by, plus whether it is loaded."""
    cands, loaded = [], False
    for m in models:
        if m.get("type") != "llm":
            continue
        if key in (m.get("key"), m.get("path"), m.get("display_name")):
            if m.get("key"):
                cands.append(m["key"])
            if m.get("path"):
                cands.append(m["path"])
            loaded = loaded or bool(m.get("loaded_instances"))
    if not cands:
        cands = [key]
    return list(dict.fromkeys(cands)), loaded


def _loaded_set(models):
    s = set()
    for m in models:
        if m.get("type") == "llm" and m.get("loaded_instances"):
            if m.get("key"):
                s.add(m["key"])
            if m.get("path"):
                s.add(m["path"])
    return s


# --------------------------------------------------------------------------- #
#  retrieval
# --------------------------------------------------------------------------- #
def searchable_title(chunk):
    """Return an email/document title, preferring indexed Subject metadata."""
    raw = str(chunk.get("text", ""))[:12000]
    # PST conversation exports contain THREAD SUBJECT; standalone messages use SUBJECT.
    for label in ("THREAD SUBJECT", "SUBJECT"):
        match = re.search(rf"(?im)^\s*{label}\s*:\s*(.+?)\s*$", raw)
        if match and text(match.group(1)):
            return text(match.group(1))[:500]
    return Path(str(chunk.get("source", "") or "Untitled")).stem


def retrieve(query, config, excluded=None, rerank=True):
    # DYNAMIC RAM CACHING: loads instantly using array buffer
    index = get_cached_index()
    corpus = index.get("chunks", [])
    if not corpus:
        return [], "Index is empty. Ingest documents first."
    
    indexed_model = index.get("embedding_model")
    if not indexed_model or index.get("version") != 2:
        return [], "This index was created by an older version. Rebuild the index before searching."
    if normalize_embed_id(indexed_model) != normalize_embed_id(resolved(config, "embedding")):
        return [], "This index was built with a different embedding model. Re-select that model or run a full rebuild."
    
    vector = embed(config, [query])[0]
    
    # Keyword / Lexical Scoring
    lexical_data = index["lexical"]
    lexical = [0.0] * len(corpus)
    for term in set(words(query)):
        postings = lexical_data["postings"].get(term, [])
        df = lexical_data["document_frequency"].get(term, 0)
        idf = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
        for item_index, count in postings:
            length = lexical_data["lengths"][item_index]
            lexical[item_index] += (
                idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / max(lexical_data["average_length"], 1)))
            )
            
    candidates_needed = positive(config.get("candidate_count"), 32, 4, 400)
    top_k = candidates_needed * 5
    
    semantic_rank = []
    semantic_scores = {}
    used_lancedb = False
    
    # 1. LanceDB Vector Search (Sub-millisecond)
    if config.get("use_lancedb") and LANCEDB_AVAILABLE and (DATA / "lancedb").exists():
        try:
            import numpy as np
            db = lancedb.connect(str(DATA / "lancedb"))
            table = db.open_table("chunks")
            query_vec = [float(x) for x in vector]
            results = table.search(query_vec).metric("cosine").limit(top_k).to_list()
            id_to_idx = {c["id"]: i for i, c in enumerate(corpus)}
            for r in results:
                idx = id_to_idx.get(r.get("id"))
                if idx is not None:
                    semantic_rank.append(idx)
                    # cosine distance ∈ [0, 2]; similarity = 1 − distance
                    semantic_scores[idx] = 1.0 - r.get("_distance", 1.0)
            used_lancedb = True
        except Exception:
            pass   # fall through to brute-force array search below

    # 2. Fallback Array Buffer Search (Pure python, super fast)
    if not used_lancedb:
        def _fast_cosine(vec_a, chunk):
            vec_b = chunk.get("_vec_buf")
            if vec_b is not None and len(vec_a) == len(vec_b):
                return sum(x * y for x, y in zip(vec_a, vec_b))
            return cosine(vec_a, chunk.get("embedding", []))
            
        scores = [(_fast_cosine(vector, c), i) for i, c in enumerate(corpus)]
        scores.sort(reverse=True, key=lambda x: x[0])
        semantic_rank = [i for _, i in scores[:top_k]]
        for s, i in scores[:top_k]:
            semantic_scores[i] = s

    # Exclude zero-match passages from lexical rank. Previously they still got
    # keyword rank positions, which diluted short exact searches such as "APAC".
    lexical_rank = [
        i for i in sorted(range(len(corpus)), key=lambda i: lexical[i], reverse=True)
        if lexical[i] > 0
    ][:top_k]

    semantic_weight = max(0.0, float(config.get("semantic_weight", 0.72)))
    keyword_weight = max(0.0, float(config.get("keyword_weight", 0.28)))
    fused = Counter()
    for rank, idx in enumerate(semantic_rank, 1):
        fused[idx] += semantic_weight / (60 + rank)
    for rank, idx in enumerate(lexical_rank, 1):
        fused[idx] += keyword_weight / (60 + rank)

    # Literal coverage boost for acronyms, codes, names and exact phrases.
    # A title/subject hit is deliberately stronger than a body hit, especially
    # for one- or two-keyword enquiries where the title is highly discriminative.
    query_terms = set(words(query))
    if query_terms and lexical_rank and keyword_weight > 0:
        max_lexical = max(lexical[i] for i in lexical_rank) or 1.0
        exact_phrase = text(query).lower()
        short_query_multiplier = 2.4 if len(query_terms) == 1 else (1.7 if len(query_terms) == 2 else 1.0)
        for idx in lexical_rank:
            title = searchable_title(corpus[idx])
            title_text = title.lower()
            body_text = str(corpus[idx].get("text", "")).lower()
            source_text = str(corpus[idx].get("source", "")).lower()
            searchable = body_text + " " + source_text
            title_terms = set(words(title_text))
            body_terms = set(words(searchable))
            body_coverage = len(query_terms & body_terms) / len(query_terms)
            title_coverage = len(query_terms & title_terms) / len(query_terms)
            lexical_strength = lexical[idx] / max_lexical
            body_phrase = 1.0 if exact_phrase and exact_phrase in searchable else 0.0
            title_phrase = 1.0 if exact_phrase and exact_phrase in title_text else 0.0
            fused[idx] += keyword_weight * (
                0.012 * body_coverage +
                0.008 * lexical_strength +
                0.008 * body_phrase +
                short_query_multiplier * (0.055 * title_coverage + 0.035 * title_phrase)
            )

    excluded = excluded or set()
    result = []
    for idx in sorted(fused, key=fused.get, reverse=True):
        chunk = corpus[idx]
        if chunk.get("id") in excluded:
            continue
        item = {k: v for k, v in chunk.items() if k != "_vec_buf"}
        item["_score"] = fused[idx]
        item["_semantic_score"] = semantic_scores.get(idx, 0)
        item["_keyword_score"] = lexical[idx]
        item["title"] = searchable_title(chunk)
        result.append(item)
        if len(result) >= candidates_needed:
            break
            
    if rerank and config["use_llm_rerank"] and result:
        result = rerank_once(query, result)
    return result, None


def retrieve_fused(queries, config, on_variant=None, rerank_query=None, do_rerank=True):
    fused, items = Counter(), {}
    unique = list(dict.fromkeys(queries))
    for index, variant in enumerate(unique, 1):
        found, error = retrieve(variant, config, rerank=False)
        if error:
            return [], error
        if on_variant:
            on_variant(index, len(unique), variant, len(found))
        for rank, item in enumerate(found, 1):
            item_id = item["id"]
            fused[item_id] += 1 / (60 + rank)
            existing = items.get(item_id)
            if not existing or item.get("_semantic_score", 0) > existing.get("_semantic_score", 0):
                items[item_id] = item
    ranked = []
    for item_id in sorted(fused, key=fused.get, reverse=True):
        item = dict(items[item_id])
        item["_score"] = fused[item_id]
        ranked.append(item)
    if do_rerank and config["use_llm_rerank"] and ranked:
        ranked = rerank_once(rerank_query or unique[0], ranked)
    return ranked, None


# --------------------------------------------------------------------------- #
#  chat
# --------------------------------------------------------------------------- #
def chat(config, messages, tokens=None, model=None):
    use = model or resolved(config, "chat")
    response = api(
        config,
        "/chat/completions",
        {
            "model": use,
            "messages": messages,
            "temperature": float(config["answer_temperature"]),
            "max_tokens": tokens or positive(config["answer_tokens"], 900, 64, 8000),
        },
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("LM Studio returned an empty answer. Verify the selected model is loaded and supports chat completions.")
    return content


def json_reply(config, messages, tokens=300, model=None):
    reply = chat(config, messages, tokens, model=model).strip()
    reply = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply, flags=re.I).strip()
    start, end = reply.find("{"), reply.rfind("}")
    if start >= 0 and end >= start:
        try:
            return json.loads(reply[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_raw": reply}


def chat_stream(config, messages, tokens, on_delta, model=None):
    use = model or resolved(config, "chat")
    body = {
        "model": use,
        "messages": messages,
        "temperature": float(config["answer_temperature"]),
        "max_tokens": tokens,
        "stream": True,
    }
    request = urllib.request.Request(
        openai_base(config) + "/chat/completions",
        json.dumps(json_safe(body)).encode(),
        {"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    output = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload).get("choices", [{}])[0].get("delta", {}).get("content", "")
            except (json.JSONDecodeError, IndexError):
                delta = ""
            if delta:
                output.append(delta)
                on_delta(delta)
    return "".join(output).strip()


# --------------------------------------------------------------------------- #
#  understand (query planning)  -- analysis model
# --------------------------------------------------------------------------- #
def understand_query(config, query, model=None):
    q_clean = query.strip().lower()
    q_words = set(words(query))
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "thanks", "thank you"}
    if q_clean in greetings or (len(q_words) <= 3 and bool(q_words & greetings)):
        return {"rewrite": query, "variants": [query], "retrieve": False, "is_greeting": True}
    instruction = (
        "You are a strict search query optimizer. Generate 1 to 3 concise alternative search queries to find relevant passages.\n"
        "CRITICAL RULE: NEVER invent, add, or guess cities, countries, or names that are NOT explicitly typed in the user prompt.\n"
        "Also write a \"hyde\" field: one short hypothetical sentence that might answer the question (used only to improve retrieval).\n"
        "Return JSON ONLY: {\"rewrite\": \"best query\", \"variants\": [\"query 1\", \"query 2\"], \"hyde\": \"hypothetical answer\"}."
    )
    rewrite = query
    variants = [query]
    try:
        data = json_reply(config, [{"role": "system", "content": instruction}, {"role": "user", "content": query}], 300, model=model)
        if isinstance(data, dict):
            r = text(data.get("rewrite"))[:600]
            if r and r.upper() not in ("NONE", "N/A"):
                rewrite = r
            variants = [rewrite]
            v_list = data.get("variants")
            if isinstance(v_list, list):
                for candidate in v_list:
                    c_text = text(candidate)[:600]
                    if c_text and c_text.lower() not in {v.lower() for v in variants}:
                        variants.append(c_text)
            hyde = text(data.get("hyde"))[:1000]
            if hyde and config.get("use_hyde", True):
                variants.append(hyde)
    except Exception:
        pass
    if query.lower() not in {v.lower() for v in variants}:
        variants.append(query)
    q_tokens = set(words(query))
    safe, seen_low = [], set()
    for v in variants:
        low = v.lower()
        if low in seen_low:
            continue
        if v == rewrite or low == query.lower() or _variant_safe(v, q_tokens):
            safe.append(v)
        seen_low.add(low)
    variants = safe or [query]
    return {"rewrite": rewrite, "variants": variants[:5], "retrieve": True, "is_greeting": False}


# --------------------------------------------------------------------------- #
#  ANALYZE  (never rejects)  -- analysis model
# --------------------------------------------------------------------------- #
_ANALYZE_SYSTEM = (
    "You are a meticulous document analyst. You read ONE source passage and describe, "
    "in 1 to 3 plain sentences, EXACTLY what factual content it contains — names, numbers, "
    "routes, dates, times, carriers, terminals, references, prices — whether or not it matches "
    "the question. Be concrete and quote codes/numbers verbatim.\n"
    "Rules:\n"
    "- NEVER refuse and NEVER output only a tag. Always write the descriptive summary first.\n"
    "- If the passage is about a different entity or direction than the question (e.g. a train "
    "when asked about a flight, or a flight FROM X when asked about a flight TO X), still "
    "summarise what it IS about, then mark it accordingly.\n"
    "- After the summary, on a NEW line, output exactly ONE tag in square brackets:\n"
    "  [ANSWERS]  = the passage directly contains the requested fact\n"
    "  [PARTIAL]  = on-topic but the exact requested detail is missing or incomplete\n"
    "  [RELATED]  = same domain but a different entity / direction / record\n"
    "  [OFFTOPIC] = unrelated to the question's domain\n"
    "Output format:\n"
    "<1-3 sentence factual summary>\n"
    "[TAG]"
)


def _parse_analysis(reply):
    raw = text(reply).strip()
    if not raw:
        return "(the model produced no analysis text)", "RELATED"
    tags = list(_TAG_RE.finditer(raw))
    if tags:
        tag = _norm_tag(tags[-1].group(1))
        summary = _TAG_RE.sub("", raw).strip()
    else:
        tag = "RELATED"
        summary = raw
    summary = re.sub(r"\s{2,}", " ", summary).strip("\n\t[]")
    if not summary:
        summary = f"(model returned only the relevance tag [{tag}]; no descriptive summary)"
    return summary[:1800], tag


def analyze_passage(config, query, item, on_delta=None, model=None):
    file_info = f"FILE NAME: {item.get('source', 'Unknown')}"
    messages = [
        {"role": "system", "content": _ANALYZE_SYSTEM},
        {"role": "user", "content": f"{file_info}\nQUESTION: {query}\nSOURCE TEXT:\n{item['text']}"},
    ]
    raw_text = ""
    try:
        if on_delta:
            chunks = []

            def _cap(d):
                chunks.append(d)
                on_delta(d)

            raw_text = chat_stream(config, messages, 300, _cap, model=model)
        else:
            raw_text = chat(config, messages, 300, model=model)
        summary, tag = _parse_analysis(raw_text)
        return summary, tag, (raw_text or "").strip(), ""
    except Exception as exc:
        # Keep transport/model errors out of the evidence preview. When analysis is
        # unavailable, show a clean excerpt of the actual source passage and expose
        # the technical problem separately to Telemetry/UI metadata.
        streamed = (raw_text or "").strip()
        if streamed:
            summary, tag = _parse_analysis(streamed)
        else:
            source_text = text(item.get("text", ""))
            summary = source_text[:600] or "Source passage is empty."
            tag = "RELATED"
        return summary, tag, streamed, str(exc)


# --------------------------------------------------------------------------- #
#  ASSESS  (sufficiency + gap)  -- analysis model
# --------------------------------------------------------------------------- #
_ASSESS_SYSTEM = (
    "You decide whether a set of document NOTES can answer a QUESTION. Each note is "
    "prefixed by a relevance tag. Reply with JSON ONLY, no prose, no markdown:\n"
    "{\"can_answer\": true, \"confidence\": 0.0, \"gap\": \"\"}\n"
    "- can_answer = true ONLY if the notes contain the specific requested fact, OR enough "
    "concrete partial facts to give a genuinely useful, specific answer.\n"
    "- can_answer = false if the notes are off-topic, only tangentially related, or missing "
    "the core requested detail. Then set gap to a SHORT phrase naming exactly what is still "
    "missing (this phrase will be used as a new search query), or '' if nothing more could help.\n"
    "- confidence is your 0.0-1.0 certainty in can_answer."
)


def assess_sufficiency(config, query, analyzed, model=None):
    notes = "\n".join(f"- [{a['relevance']}] {a['summary']}" for a in analyzed)
    messages = [
        {"role": "system", "content": _ASSESS_SYSTEM},
        {"role": "user", "content": f"QUESTION: {query}\nNOTES:\n{notes}"},
    ]
    try:
        data = json_reply(config, messages, 160, model=model)
        if not isinstance(data, dict):
            return False, ""
        conf = float(data.get("confidence", 0) or 0)
        can = bool(data.get("can_answer"))
        gap = text(data.get("gap", ""))[:240]
        sufficient = can or conf >= ASSESS_CONFIDENCE_STOP
        return sufficient, ("" if sufficient else gap)
    except Exception:
        return False, ""


# --------------------------------------------------------------------------- #
#  collation helpers for the ANSWER stage
# --------------------------------------------------------------------------- #
def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def collate_for_answer(analyzed):
    ordered = sorted(analyzed, key=lambda a: (REL_RANK.get(a["relevance"], 0), a.get("semantic", 0)), reverse=True)
    merged = []
    for a in ordered:
        w = set(words(a["summary"]))
        dup = False
        for m in merged:
            if _jaccard(w, set(words(m["summary"]))) >= 0.9:
                dup = True
                break
        if not dup:
            merged.append(a)
    return merged


def evidence_prompt(query, evidence):
    blocks = []
    n_contrib = sum(1 for it in evidence if it.get("relevance") in ("ANSWERS", "PARTIAL"))
    for i, item in enumerate(evidence):
        section = item.get("section") or ("Passage " + str((item.get("chunk") or 0) + 1))
        row = f" (Row {item['row']})" if item.get("row") else ""
        rel = item.get("relevance", "RELATED")
        blocks.append(
            f"[{i + 1}] File: {item.get('source', 'Unknown')} | {section}{row} | relevance: {rel}\n{item['text']}"
        )
    context = "\n".join(blocks)
    if n_contrib > 1:
        synthesis = (
            f"CRITICAL — MULTI-SOURCE SYNTHESIS: the {n_contrib} notes tagged ANSWERS/PARTIAL are "
            "COMPLEMENTARY, not redundant. Each note usually carries DIFFERENT details (one has the "
            "names, another the flight number & fare class, another the price & payment method, etc.). "
            "You MUST read ALL of them and MERGE their distinct facts into ONE complete answer. Do NOT "
            "stop after the first note that looks relevant, and do NOT copy any single note verbatim. "
            "Tag every fact with the exact note number(s) it came from, e.g. [1][2][3]. An answer that "
            "cites only one note is WRONG here — you are expected to draw from several of them."
        )
    else:
        synthesis = "Cite every fact with the note number it came from, like [1]."
    return (
        "Answer the question using ONLY the provided NOTES, but COMBINE them.\n"
        + synthesis + "\n"
        "Notes tagged ANSWERS/PARTIAL carry the facts; RELATED/OFFTOPIC are contrast notes — use them "
        "to explain precisely what the documents DO contain (entity, direction, codes) instead of a "
        "bare refusal.\n"
        "Ignore any trailing '= ...' legend sentence inside a note; it is metadata, not content.\n"
        "Be complete: include every concrete detail the notes provide that bears on the question "
        "(names, codes, dates, times, routes, terminals, fare class, price, payment, duration).\n"
        f"NOTES:\n{context}\n"
        f"QUESTION: {query}"
    )


def context_limited_evidence(config, evidence):
    budget = positive(config.get("context_window"), 8192, 1024, 131072) * 4
    reserved = positive(config.get("answer_tokens"), 900, 64, 8000) * 4 + 1600
    available, used, selected = max(800, budget - reserved), 0, []
    for item in evidence:
        value = item["text"]
        remaining = available - used
        if remaining <= 0:
            break
        if len(value) > remaining:
            value = value[:remaining].rsplit(" ", 1)[0] or value[:remaining]
        selected.append({**item, "text": value})
        used += len(value)
    return selected or evidence[:1]


def compress_memory(config, query, evidence):
    q_terms = set(words(query))
    out = []
    for item in evidence:
        raw = item.get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw) if s.strip()]
        if q_terms and len(sentences) > 2:
            kept = [s for s in sentences if (set(words(s)) & q_terms) or (set(words(s)) & _FIELD)]
            if len(kept) < max(1, len(sentences) // 2):
                kept = sentences
            raw = " ".join(kept)
        out.append({**item, "text": raw[:1600]})
    return out


def faithful(config, query, evidence, answer):
    return bool(re.search(r"\[\d+\]", answer))


def _citation_numbers(answer):
    """Sorted distinct source numbers the answer actually cites, e.g. [1, 3]."""
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})


# --------------------------------------------------------------------------- #
# Outlook PST ingestion (Windows / Outlook COM)
# --------------------------------------------------------------------------- #
PST_JOBS = {}
PST_JOBS_LOCK = threading.Lock()

PST_DEPENDENCIES = {
    "win32com.client": "pywin32",
    "pythoncom": "pywin32",
    "rapidfuzz": "rapidfuzz",
    "extract_msg": "extract-msg",
    "openpyxl": "openpyxl",
    "docx": "python-docx",
    "pypdf": "pypdf",
}

def pst_dependency_report():
    import importlib.util
    missing = []
    for module, package in PST_DEPENDENCIES.items():
        if importlib.util.find_spec(module) is None:
            missing.append({"module": module, "package": package,
                            "install": f"{sys.executable} -m pip install {package}"})
    required = [m for m in missing if m["module"] in ("win32com.client", "pythoncom")]
    return {"platform": sys.platform, "outlook_com_supported": os.name == "nt",
            "missing": missing, "required_missing": required,
            "install_all": (f"{sys.executable} -m pip install " +
                            " ".join(sorted({m['package'] for m in missing}))) if missing else ""}

def _pst_norm_name(filename):
    stem, ext = os.path.splitext(os.path.basename(str(filename or "")))
    stem = re.sub(r"(?i)(?:[ _.-]*(?:copy|duplicate))(?:[ _.-]*\\?\d+\\?)?$", "", stem)
    stem = re.sub(r"(?:[ _.-]*\\(\d+\\)|[ _.-]+\d+)$", "", stem)
    return re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip(), ext.lower()

def _pst_similarity(a, b):
    try:
        from rapidfuzz import fuzz
        return int(fuzz.ratio(a, b))
    except ImportError:
        return int(round(100 * SequenceMatcher(None, a, b).ratio()))

def _pst_safe_name(value, limit=90):
    value = re.sub(r'[\\/:*?"<>|]+', '_', str(value or 'No Subject')).strip(' ._')
    return (value[:limit] or 'No Subject')

def _pst_attachment_text(path):
    ext = path.suffix.lower()
    try:
        if ext == '.pdf': return read_pdf(path)
        if ext == '.docx': return read_docx(path)
        if ext in ('.xlsx', '.xlsm'):
            return '\n'.join(f"{sheet} row {row}: {line}" for sheet, row, line in read_xlsx(path))
        if ext == '.msg':
            try:
                import extract_msg
            except ImportError as exc:
                raise RuntimeError(f"Missing dependency 'extract-msg'. Install with: {sys.executable} -m pip install extract-msg") from exc
            msg = extract_msg.Message(str(path))
            try:
                return f"Subject: {msg.subject or ''}\nFrom: {msg.sender or ''}\nTo: {msg.to or ''}\nDate: {msg.date or ''}\n\n{msg.body or ''}"
            finally:
                msg.close()
    except Exception as exc:
        return f"[Attachment extraction error: {path.name}: {exc}]"
    return ""

def import_pst_to_source(pst_path, config, progress=None):
    if os.name != 'nt':
        raise RuntimeError("PST import through Outlook requires Windows and desktop Outlook.")
    report = pst_dependency_report()
    if report['required_missing']:
        raise RuntimeError("Missing Outlook dependency. Install with: " + report['install_all'])
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency '{exc.name}'. Install with: {sys.executable} -m pip install pywin32") from exc
    pst = Path(pst_path).expanduser().resolve()
    if not pst.is_file() or pst.suffix.lower() != '.pst':
        raise ValueError("Enter an existing .pst file path.")
    threshold = positive(config.get('pst_similarity_threshold'), 90, 50, 100)
    allowed = {x.strip().lower() for x in str(config.get('pst_attachment_extensions','')).split(',') if x.strip()}
    processing_mode = str(config.get('pst_processing_mode', 'emails_only') or 'emails_only').strip().lower()
    if processing_mode not in ('emails_only', 'attachments_only', 'emails_and_attachments'):
        processing_mode = 'emails_only'
    include_email_text = processing_mode in ('emails_only', 'emails_and_attachments')
    extract_atts = processing_mode in ('attachments_only', 'emails_and_attachments') and bool(config.get('pst_extract_attachments', True))
    if progress:
        progress('PST mode: ' + {'emails_only':'emails first (attachments skipped)', 'attachments_only':'attachments only', 'emails_and_attachments':'emails and attachments'}[processing_mode], 2)
    mode_suffix = {'emails_only':'emails', 'attachments_only':'attachments', 'emails_and_attachments':'combined'}[processing_mode]
    output = Path(config['source_folder']) / '.pst_extracted' / (_pst_safe_name(pst.stem) + '_' + hashlib.sha1(str(pst).encode()).hexdigest()[:8] + '_' + mode_suffix)
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if progress: progress('Connecting to Outlook...', 3)
    pythoncom.CoInitialize()
    if progress: progress('COM initialised. Creating Outlook.Application...', 4)
    ns = None
    added = False
    try:
        outlook = win32com.client.Dispatch('Outlook.Application')
        if progress: progress('Outlook.Application connected. Requesting MAPI namespace...', 5)
        ns = outlook.GetNamespace('MAPI')
        if progress: progress('MAPI namespace ready. Enumerating Outlook stores...', 6)
        existing = None
        stores = ns.Stores
        if progress: progress(f'Outlook reports {stores.Count} mounted store(s). Looking for target PST...', 7)
        for store in stores:
            try:
                if store.FilePath and os.path.normcase(os.path.abspath(store.FilePath)) == os.path.normcase(str(pst)):
                    existing = store; break
            except Exception: pass
        if existing is None:
            if progress: progress('Mounting PST in Outlook...', 7)
            if progress: progress(f'Calling Namespace.AddStore for: {pst}', 8)
            ns.AddStore(str(pst)); added = True
            if progress: progress('Namespace.AddStore returned. Locating mounted PST store...', 9)
            for store in ns.Stores:
                try:
                    if store.FilePath and os.path.normcase(os.path.abspath(store.FilePath)) == os.path.normcase(str(pst)):
                        existing = store; break
                except Exception: pass
        if existing is None: raise RuntimeError('Outlook mounted the PST but its store could not be located.')
        if progress: progress('PST mounted. Reading folder tree...', 10)
        root = existing.GetRootFolder()
        conversations = {}
        attachments = {}
        stack=[root]; scanned=0
        folder_count = 0
        while stack:
            folder=stack.pop(); folder_count += 1
            if progress: progress(f"Scanning folder {folder_count}: {getattr(folder, 'Name', 'Unknown')}", min(75, 10 + folder_count))
            try:
                for sub in folder.Folders: stack.append(sub)
            except Exception: pass
            try:
                items = folder.Items
                item_count = int(items.Count or 0)
            except Exception as exc:
                if progress: progress(f"Skipped folder {folder_count}: could not read Items ({exc})", min(75, 10 + folder_count))
                continue
            folder_name = str(getattr(folder, 'Name', 'Unknown') or 'Unknown')
            if progress: progress(f"Folder {folder_count}: {folder_name} contains {item_count} item(s). Starting indexed scan...", min(75, 10 + folder_count))
            # Outlook's Python COM collection iterator can freeze on a large/corrupt PST folder.
            # Indexed Items.Item(n) provides exact progress and identifies the item where COM blocks.
            for item_index in range(1, item_count + 1):
                if progress and (item_index == 1 or item_index % 10 == 0 or item_index == item_count):
                    progress(f"Folder {folder_count}: {folder_name} — opening item {item_index}/{item_count}; total emails {scanned}",
                             min(80, 12 + int(65 * item_index / max(1, item_count))))
                try:
                    mail = items.Item(item_index)
                except Exception as exc:
                    if progress: progress(f"Folder {folder_count}: {folder_name} — skipped unreadable item {item_index}/{item_count}: {exc}",
                                           min(80, 12 + int(65 * item_index / max(1, item_count))))
                    continue
                try:
                    item_class = getattr(mail, 'Class', 0)
                    if item_class != 43:
                        continue
                    subject = str(getattr(mail, 'Subject', '') or 'No Subject')
                    if progress:
                        progress(f"Folder {folder_count}: {folder_name} — analysing email {item_index}/{item_count} | extracted {scanned} | {subject[:100]}",
                                 min(80, 12 + int(65 * item_index / max(1, item_count))))
                    conv = str(getattr(mail, 'ConversationID', '') or '').strip()
                    if not conv:
                        norm = re.sub(r'(?i)^(?:(?:re|fw|fwd):\s*)+', '', subject).strip().lower()
                        conv = 'subject::' + norm
                    received = getattr(mail, 'ReceivedTime', None)
                    received_iso = received.strftime('%Y-%m-%d %H:%M:%S') if received else ''
                    entry = {"subject": subject, "from": str(getattr(mail, 'SenderName', '') or ''),
                             "to": str(getattr(mail, 'To', '') or ''), "cc": str(getattr(mail, 'CC', '') or ''),
                             "received": received_iso, "received_obj": received,
                             "body": str(getattr(mail, 'Body', '') or '') if include_email_text else '',
                             "folder": str(getattr(folder, 'FolderPath', '') or '')}
                    # Conversation metadata is kept in all modes so attachments remain traceable
                    # to the originating email and Outlook ConversationID.
                    conversations.setdefault(conv, []).append(entry)
                    if extract_atts:
                        try: count = mail.Attachments.Count
                        except Exception: count = 0
                        for i in range(1, count + 1):
                            if progress: progress(f"Folder {folder_count}: {folder_name} — email {item_index}/{item_count}, attachment {i}/{count}",
                                                   min(80, 12 + int(65 * item_index / max(1, item_count))))
                            att = mail.Attachments.Item(i)
                            name = str(att.FileName or f'attachment_{i}')
                            stem, ext = _pst_norm_name(name)
                            if ext not in allowed or not stem: continue
                            keylist = attachments.setdefault(conv, []); match = None; best = 0
                            for old in keylist:
                                os_, oe = _pst_norm_name(old['name'])
                                if oe != ext: continue
                                score = _pst_similarity(stem, os_)
                                if score > best: best = score; match = old
                            # Save immediately; do not retain live Outlook COM Attachment proxies.
                            saved = output / ('_pending_' + uuid.uuid4().hex + ext)
                            att.SaveAsFile(str(saved))
                            candidate = {"name": name, "received": received, "score": best,
                                         "saved_path": str(saved), "subject": subject}
                            if match is not None and best >= threshold:
                                if received and (not match['received'] or received > match['received']):
                                    try: Path(match.get('saved_path', '')).unlink(missing_ok=True)
                                    except Exception: pass
                                    keylist[keylist.index(match)] = candidate
                                else:
                                    saved.unlink(missing_ok=True)
                            else:
                                keylist.append(candidate)
                    scanned += 1
                except Exception as exc:
                    print(f"[PST] Item processing error | folder={folder_name} item={item_index}/{item_count} | {type(exc).__name__}: {exc}", flush=True)
                    continue
            if progress:
                progress(f"Completed folder {folder_count}: {folder_name} — inspected {item_count} item(s); total extracted emails {scanned}",
                         min(80, 12 + folder_count))
        if progress: progress(f'Found {scanned} emails in {len(conversations)} conversations. Writing conversation files...', 82)
        written=0; att_count=0
        for conv, emails in conversations.items():
            emails.sort(key=lambda x: x['received_obj'] or datetime.min)
            title=emails[-1]['subject'] if emails else 'No Subject'
            fn=_pst_safe_name(title)+'__'+hashlib.sha1(conv.encode('utf-8','replace')).hexdigest()[:10]+'.txt'
            parts=[f"PST PROCESSING MODE: {processing_mode}",f"CONVERSATION ID: {conv}",f"THREAD SUBJECT: {title}",f"EMAIL COUNT: {len(emails)}",'='*80]
            if include_email_text:
                for idx,e in enumerate(emails,1):
                    parts += ['',f"EMAIL {idx}/{len(emails)}",f"SUBJECT: {e['subject']}",f"FROM: {e['from']}",
                              f"TO: {e['to']}",f"CC: {e['cc']}",f"DATE: {e['received']}",f"SOURCE FOLDER: {e['folder']}",'-'*80,e['body']]
            else:
                parts += ['', 'EMAIL BODY EXTRACTION: skipped (attachments-only mode)',
                          'Attachment records below retain their source email subject and received date.']
            for idx,a in enumerate(attachments.get(conv,[]),1):
                temp = Path(a['saved_path'])
                try:
                    content = _pst_attachment_text(temp)
                    parts += ['',f"ATTACHMENT {idx} (latest fuzzy match)",f"FILENAME: {a['name']}",
                              f"SOURCE EMAIL SUBJECT: {a.get('subject','')}",f"RECEIVED: {a['received']}",f"SIMILARITY SCORE: {a['score']}%",'-'*80,content]
                    att_count += 1
                finally:
                    try: temp.unlink(missing_ok=True)
                    except Exception: pass
            (output/fn).write_text('\n'.join(parts),encoding='utf-8',errors='replace'); written+=1
            if progress and written % 20 == 0: progress(f'Wrote {written}/{len(conversations)} conversations...', min(98, 82 + int(16 * written / max(1, len(conversations)))))
        if progress: progress('PST extraction complete.', 100)
        return {"emails":scanned,"conversations":len(conversations),"attachments":att_count,"files":written,"output":str(output),"threshold":threshold,"processing_mode":processing_mode}
    finally:
        if added and ns is not None:
            try: ns.RemoveStore(existing.GetRootFolder())
            except Exception: pass
        pythoncom.CoUninitialize()

def start_pst_job(pst_path, config):
    job_id = uuid.uuid4().hex
    now = time.time()
    job = {"id": job_id, "state": "queued", "percent": 1,
           "phase": "queued", "message": "Queued PST extraction...",
           "created": now, "updated": now, "history": []}
    pst_log_dir = DATA / "logs"
    pst_log_dir.mkdir(parents=True, exist_ok=True)
    pst_log_path = pst_log_dir / f"pst_{time.strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}.log"
    job["log_file"] = str(pst_log_path)
    with PST_JOBS_LOCK:
        PST_JOBS[job_id] = job

    def write_log(line):
        try:
            with pst_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except Exception:
            pass

    def update(message, percent=None):
        stamp = time.time()
        line = f"[PST {job_id[:8]}] {time.strftime('%H:%M:%S')} | {message}"
        print(line, flush=True)
        write_log(line)
        with PST_JOBS_LOCK:
            current = PST_JOBS.get(job_id, job)
            current["state"] = "running"
            current["phase"] = str(message).split("...", 1)[0]
            current["message"] = str(message)
            current["updated"] = stamp
            current["history"] = (current.get("history", []) + [{"time": time.strftime('%H:%M:%S'), "message": str(message)}])[-250:]
            current["events"] = int(current.get("events", 0)) + 1
            if percent is not None:
                current["percent"] = max(1, min(100, int(percent)))

    def heartbeat():
        warning_after = positive(config.get("pst_stall_warning_seconds"), 30, 10, 600)
        while True:
            time.sleep(5)
            with PST_JOBS_LOCK:
                current = PST_JOBS.get(job_id)
                if not current or current.get("state") in ("done", "error"):
                    return
                idle = int(time.time() - current.get("updated", now))
                elapsed = int(time.time() - current.get("created", now))
                current["elapsed_seconds"] = elapsed
                current["idle_seconds"] = idle
                current["stalled"] = idle >= warning_after
                phase = current.get("message", "unknown phase")
            if idle >= warning_after:
                if idle == warning_after or idle % 30 < 5:
                    warning = f"[PST {job_id[:8]}] WARNING | No phase change for {idle}s. Current COM operation: {phase}"
                    print(warning, flush=True); write_log(warning)
            elif idle % 15 < 5:
                beat = f"[PST {job_id[:8]}] heartbeat | elapsed={elapsed}s idle={idle}s | {phase}"
                print(beat, flush=True); write_log(beat)

    def worker():
        try:
            update("Starting Outlook PST extraction...", 2)
            result = import_pst_to_source(pst_path, config, update)
            with PST_JOBS_LOCK:
                PST_JOBS[job_id].update({"state": "done", "percent": 100,
                                         "updated": time.time(), "stalled": False,
                                         "message": "PST extraction complete.", "result": result})
            print(f"[PST {job_id[:8]}] DONE | {result}", flush=True)
        except Exception as exc:
            report = pst_dependency_report()
            with PST_JOBS_LOCK:
                PST_JOBS[job_id].update({"state": "error", "updated": time.time(),
                                         "message": str(exc), "error": str(exc),
                                         "dependency_help": report.get("install_all", ""),
                                         "missing": report.get("missing", [])})
            print(f"[PST {job_id[:8]}] ERROR | {type(exc).__name__}: {exc}", flush=True)

    threading.Thread(target=heartbeat, daemon=True, name=f"pst-heartbeat-{job_id[:8]}").start()
    threading.Thread(target=worker, daemon=True, name=f"pst-import-{job_id[:8]}").start()
    return job

# --------------------------------------------------------------------------- #
#  HTTP handler
# --------------------------------------------------------------------------- #
class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def end_headers(self):
        if self.path == "/index.html":
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def send_json(self, data, status=200):
        raw = json.dumps(json_safe(data), ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    # ---- SSE transport (fixed framing: blank line terminates each frame) ----
    def send_stream_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        self._gone = False
        try:
            self.wfile.write(b": stream open\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._gone = True

    def send_stream(self, data):
        if getattr(self, "_gone", False):
            return
        try:
            payload = json.dumps(json_safe(data), ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._gone = True

    # ---- live helpers ----
    def _log(self, message, level="info"):
        self.send_stream({"type": "log", "text": message, "level": level, "t": time.strftime("%H:%M:%S")})

    def _stage(self, stage_id, status, elapsed=None):
        """Emit a stage event; elapsed may be supplied by measured fast paths."""
        now = time.time()
        payload = {"type": "stage", "id": stage_id, "status": status}
        if status == "active":
            self._stage_t[stage_id] = now
        elif status == "done":
            if elapsed is not None:
                payload["elapsed"] = round(float(elapsed), 3)
            elif stage_id in self._stage_t:
                payload["elapsed"] = round(now - self._stage_t[stage_id], 2)
        self.send_stream(payload)

    def pulse_while(self, fn, interval=0.7):
        box = {}

        def runner():
            try:
                box["value"] = fn()
            except BaseException as exc:
                box["error"] = exc

        worker = threading.Thread(target=runner, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(interval)
            if worker.is_alive():
                self.send_stream({"type": "pulse"})
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # ---- the pipeline ----
    def _answer_stream(self, c, query, want_answer, evidence_only, adaptive):
        t0 = time.time()
        self._stage_t = {}
        self._gone = False
        stage = "understand"

        answer_model = (c.get("chat_model") or "").strip()
        a_model = analysis_model_for(c)        
        # Query planning and passage analysis are independently controllable.
        # Evidence-only mode must never summarize/check chunks with an LLM.
        can_rewrite = bool(a_model) and bool(c.get("use_query_rewrite", True))
        have_analysis = bool(a_model) and not evidence_only

        self.send_stream({"type": "run_start", "evidence_only": evidence_only,
                          "want_answer": want_answer, "adaptive": adaptive,
                          "analysis_model": a_model, "answer_model": answer_model})
        try:
            # Fast retrieval-only path: no LM Studio, query rewrite, HyDE,
            # passage analysis, sufficiency checks, or answer generation.
            # It uses the same hybrid semantic + BM25 retrieval and optional
            # local BGE reranker as the full pipeline, preserving retrieval quality.
            if evidence_only:
                stage = "retrieve"
                self._stage("understand", "skipped")
                self._stage("retrieve", "active")
                self._stage("analyze", "skipped")
                started = time.perf_counter()
                results, error = retrieve(query, c, rerank=True)
                if error:
                    raise RuntimeError(error)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                limited = results[:positive(c.get("max_candidate_checks"), 24, 4, 200)]
                self.send_stream({"type": "retrieve_progress", "index": 1, "total": 1,
                                  "variant": query, "hits": len(results)})
                for number, item in enumerate(limited, 1):
                    meta = {
                        "number": number, "total": len(limited),
                        "source": item.get("source", ""),
                        "title": item.get("title") or searchable_title(item),
                        "section": item.get("section") or f"Passage {(item.get('chunk') or 0) + 1}",
                        "row": item.get("row"), "path": item.get("path", ""),
                        "document_id": item.get("document_id", ""),
                        "snapshot_available": bool(item.get("snapshot_available") or item.get("document_id")),
                        "chunk_number": item.get("chunk_number", (item.get("chunk") or 0) + 1),
                        "chunk_count": item.get("chunk_count"),
                        "unit_chars": item.get("unit_chars"),
                        "configured_chunk_size": item.get("configured_chunk_size"),
                        "semantic": round(float(item.get("_semantic_score", 0)), 3),
                        "score": round(float(item.get("_score", 0)), 4),
                    }
                    self.send_stream({
                        "type": "verify_done", **meta, "accepted": True,
                        "relevance": "RAW", "verdict": "RAW EVIDENCE - NO LLM / LM STUDIO",
                        "reason": "retrieval only", "fact": item.get("text", ""),
                        "parsed": item.get("text", ""), "raw": "",
                        "full_text": item.get("text", ""), "analysis_error": "",
                        "raw_evidence": True, "elapsed": 0,
                        "accepted_count": number, "target": len(limited),
                    })
                self.send_stream({"type": "sources", "results": [
                    {"number": n, **public(item)}
                    for n, item in enumerate(limited, 1)
                ]})
                self._stage("retrieve", "done", round(elapsed_ms / 1000.0, 3))
                self._stage("answer", "skipped")
                self._log(f"Retrieval-only: {len(limited)} chunks in {elapsed_ms:.1f} ms; no LLM/LM Studio calls.", "ok")
                self.send_stream({"type": "done", "elapsed": round(time.time() - t0, 3),
                                  "retrieval_ms": round(elapsed_ms, 1)})
                return
            checkpoint = positive(c.get("rerank_count"), 4, 1, 30)
            max_checks = positive(c.get("max_candidate_checks"), 24, 4, 200)
            base_candidates = positive(c.get("candidate_count"), 32, 4, 400)

            self._log(f"Models — analysis: {a_model or '(none → raw passages)'} · "
                      f"answer: {answer_model or '(none → evidence only)'}")

            # ---- 01 understand (analysis model) ----
            self._stage("understand", "active")
            if not adaptive or not can_rewrite:
                plan = {"rewrite": query, "variants": [query], "retrieve": True, "is_greeting": False}
                reason = "disabled in Settings" if not c.get("use_query_rewrite", True) else "adaptive planning unavailable"
                self._log(f"LLM query rewrite {reason} — using the exact user question for retrieval.")
            else:
                self._log("Planning retrieval — asking the analysis model for a rewrite + HyDE variants...")
                plan = self.pulse_while(lambda: understand_query(c, query, model=a_model))
            if plan.get("rewrite") and plan["rewrite"] != query:
                self.send_stream({"type": "query_rewrite", "query": plan["rewrite"]})
                self._log(f"Query rewritten to: {plan['rewrite']}")
            self.send_stream({"type": "variants", "variants": plan["variants"]})
            self._log(f"Retrieval plan ready — {len(plan['variants'])} search variant(s).", "ok")
            self._stage("understand", "done")

            if not plan.get("retrieve", True):
                for sid in ("retrieve", "analyze", "answer"):
                    self._stage(sid, "skipped")
                if plan.get("is_greeting"):
                    self._log("Greeting detected — no document search needed.")
                    self.send_stream({"type": "delta", "text": "Hello! Ask a specific question about your documents or search for information."})
                else:
                    self.send_stream({"type": "delta", "text": "Insufficient evidence."})
                self.send_stream({"type": "done", "elapsed": round(time.time() - t0, 1)})
                return

            # ---- 02 retrieve + 03 analyze + 04 assess (corrective loop) ----
            stage = "retrieve"
            self._stage("retrieve", "active")
            base_variants = list(plan["variants"])
            extra_variants = []
            seen, analyzed, checks_done = set(), [], 0
            sufficient, gap, last_gap = False, "", ""
            analyze_started = False

            def on_variant(index, total, variant, hits):
                self.send_stream({"type": "retrieve_progress", "index": index, "total": total,
                                  "variant": variant[:160], "hits": hits})
                self._log(f"  variant {index}/{total} → {hits} hit(s): {variant[:90]}")

            wave = 0
            while not sufficient and checks_done < max_checks and wave < MAX_WAVES and not self._gone:
                wave += 1
                q_list = list(dict.fromkeys(base_variants + extra_variants))
                c2 = dict(c)
                c2["candidate_count"] = min(400, base_candidates + (wave - 1) * WAVE_EXTRA_CANDIDATES)
                if wave == 1:
                    self._log(f"── retrieval wave 1: {len(q_list)} variant(s), up to {c2['candidate_count']} candidates ──")
                else:
                    self.send_stream({"type": "retrieve_wave", "wave": wave, "total": MAX_WAVES,
                                      "have": len(analyzed), "variants": q_list,
                                      "gap": last_gap, "candidates": c2["candidate_count"]})
                    self._log(f"── corrective wave {wave}/{MAX_WAVES}: analysed {len(analyzed)}, "
                              f"not yet sufficient — reformulated query from gap: “{last_gap[:80]}” ──", "warn")
                candidates, error = retrieve_fused(q_list, c2, on_variant, rerank_query=query)
                if error:
                    raise RuntimeError(error)
                new_cands = [x for x in candidates if x["id"] not in seen]
                if not new_cands:
                    self._log(f"Wave {wave} surfaced no unseen passages — corpus exhausted for these queries.")
                    break
                if not analyze_started:
                    stage = "analyze"
                    self._stage("analyze", "active")
                    analyze_started = True
                self.send_stream({"type": "candidates", "wave": wave, "wave_checks": len(new_cands),
                                  "checked": checks_done, "total": max_checks,
                                  "analyzed": len(analyzed), "checkpoint": checkpoint})
                for item in new_cands:
                    if self._gone or checks_done >= max_checks or sufficient:
                        break
                    checks_done += 1
                    seen.add(item["id"])
                    meta = {
                        "number": checks_done, "total": max_checks,
                        "source": item.get("source", ""),
                        "title": item.get("title") or searchable_title(item),
                        "section": item.get("section") or f"Passage {(item.get('chunk') or 0) + 1}",
                        "row": item.get("row"), "path": item.get("path", ""),
                        "document_id": item.get("document_id", ""),
                        "snapshot_available": bool(item.get("snapshot_available") or item.get("document_id")),
                        "chunk_number": item.get("chunk_number", (item.get("chunk") or 0) + 1),
                        "chunk_count": item.get("chunk_count"),
                        "unit_chars": item.get("unit_chars"),
                        "configured_chunk_size": item.get("configured_chunk_size"),
                        "semantic": round(float(item.get("_semantic_score", 0)), 3),
                        "score": round(float(item.get("_score", 0)), 4),
                    }
                    self.send_stream({"type": "verify_start", **meta})
                    t_check = time.time()
                    if not have_analysis:
                        # Return the complete retrieved source chunk directly. No LLM call,
                        # summary, relevance classification, or sufficiency assessment.
                        summary, tag, raw_text, analysis_error = item["text"], "RAW", "", ""
                    else:
                        self._log(f"[{checks_done}] analysing {meta['source']} ({meta['section']})...")
                        n = checks_done
                        summary, tag, raw_text, analysis_error = analyze_passage(
                            c, query, item,
                            lambda d, num=n: self.send_stream({"type": "verify_delta", "number": num, "text": d}),
                            model=a_model)
                        if analysis_error:
                            self._log(f"[{checks_done}] analysis unavailable for {meta['source']}: {analysis_error}", "warn")
                    entry = {**meta, "summary": summary, "raw": raw_text, "relevance": tag,
                             "text": summary, "path": item.get("path", ""),
                             "document_id": item.get("document_id", ""),
                             "snapshot_available": bool(item.get("snapshot_available") or item.get("document_id"))}
                    analyzed.append(entry)
                    verdict = ("RAW EVIDENCE · no LLM analysis" if tag == "RAW"
                               else f"ANALYZED · [{tag}] {REL_LABEL.get(tag, '')}")
                    self._log(
                        f"[{checks_done}] {meta['source']} → kept [{tag}] "
                        f"({round(time.time() - t_check, 1)}s, {len(analyzed)} note(s) held)",
                        "ok",
                    )
                    self.send_stream({
                        "type": "verify_done", **meta,
                        "accepted": True,
                        "relevance": tag,
                        "verdict": verdict,
                        "reason": verdict,
                        "fact": summary,
                        "parsed": summary,
                        "raw": raw_text,
                        "full_text": item.get("text", ""),
                        "analysis_error": analysis_error,
                        "raw_evidence": tag == "RAW",
                        "elapsed": round(time.time() - t_check, 2),
                        "accepted_count": len(analyzed),
                        "target": checkpoint,
                    })
                    if have_analysis and len(analyzed) % checkpoint == 0 and checks_done < max_checks:
                        self._log(f"Assess checkpoint ({len(analyzed)} notes) — can we answer yet?")
                        sufficient, gap = self.pulse_while(lambda: assess_sufficiency(c, query, analyzed, model=a_model))
                        if sufficient:
                            self._log("Assess: SUFFICIENT — proceeding to answer.", "ok")
                            self.send_stream({"type": "assess", "sufficient": True, "gap": "", "notes": len(analyzed)})
                        else:
                            self._log(f"Assess: not yet — gap: “{gap or '(none reported)'}”", "warn")
                            self.send_stream({"type": "assess", "sufficient": False, "gap": gap, "notes": len(analyzed)})
                            if gap and gap.lower() != last_gap.lower():
                                extra_variants.append(gap)
                                last_gap = gap

            self._stage("retrieve", "done")
            self._stage("analyze", "done" if analyze_started else "skipped")

            if not analyzed:
                self._stage("answer", "skipped")
                self._log("Retrieval found nothing at all in the index.", "warn")
                self.send_stream({"type": "delta", "text": "The provided documents do not contain the answer."})
                self.send_stream({"type": "done", "elapsed": round(time.time() - t0, 1)})
                return

            if want_answer and have_analysis and not sufficient and analyzed:
                self._log("Final assess after budget/exhaustion...")
                sufficient, gap = self.pulse_while(lambda: assess_sufficiency(c, query, analyzed, model=a_model))
                self.send_stream({"type": "assess", "sufficient": sufficient, "gap": gap,
                                  "notes": len(analyzed), "final": True})

            analyzed = compress_memory(c, query, analyzed)
            ordered = collate_for_answer(analyzed)
            self.send_stream({"type": "sources", "results": [
                {k: v for k, v in a.items() if k != "embedding"} for a in ordered
            ]})

            if not want_answer:
                self._stage("answer", "skipped")
                self._log(f"Evidence-only mode — {len(ordered)} note(s) returned, no answer generated.", "ok")
                self.send_stream({"type": "done", "elapsed": round(time.time() - t0, 1)})
                return

            # ---- 05 answer (answer model) ----
            stage = "answer"
            self._stage("answer", "active")
            limited = context_limited_evidence(c, ordered)
            used_chars = sum(len(x["text"]) for x in limited)
            self._log(f"Composing cited answer with {answer_model} from {len(limited)} note(s), "
                      f"{used_chars} chars (sufficient={sufficient})...")
            prompt = evidence_prompt(query, limited)
            tokens = positive(c["answer_tokens"], 900, 64, 8000)
            if sufficient:
                sys_msg = (
                    "You are a precise evidence-only document assistant. The NOTES are complementary "
                    "fragments of the answer — your job is to MERGE them. Pull the distinct details from "
                    "EVERY ANSWERS/PARTIAL note that bears on the question (names, codes, dates, times, "
                    "routes, terminals, fare class, price, payment, duration) and weave them into one "
                    "complete answer, tagging each fact with its [n]. A single-note answer is incomplete "
                    "whenever another note adds a relevant detail. Never invent facts not in the notes."
                )
            else:
                sys_msg = (
                    "You are a precise evidence-only document assistant. The NOTES are complementary "
                    "fragments and do NOT fully answer the question. First MERGE every distinct detail "
                    "the ANSWERS/PARTIAL notes provide into a partial answer, tagging each fact with its "
                    "[n] (a one-note answer is incomplete if other notes add details). Then use the "
                    "RELATED/OFFTOPIC contrast notes to state exactly what the documents DO contain "
                    "(entity, direction, codes) and which requested detail is absent. Never output a "
                    "bare 'no information' refusal while related notes exist."
                )
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ]
            answer = chat_stream(c, messages, tokens,
                                 lambda d: self.send_stream({"type": "delta", "text": d}),
                                 model=answer_model)
            cited = _citation_numbers(answer)
            contrib = [i + 1 for i, it in enumerate(limited)
                       if it.get("relevance") in ("ANSWERS", "PARTIAL")]
            need_combine = len(contrib) > 1 and len(set(cited) & set(contrib)) < 2
            if not faithful(c, query, limited, answer) or need_combine:
                if need_combine:
                    self._log(
                        f"Answer anchored to a single source (cited {sorted(set(cited) & set(contrib)) or cited}); "
                        f"{len(contrib)} notes contribute ({contrib}) — regenerating to COMBINE them.",
                        "warn",
                    )
                else:
                    self._log("No [n] citations detected — requesting a cited regeneration.", "warn")
                self.send_stream({"type": "answer_reset"})
                regen = [
                    {"role": "system", "content": (
                        "Your previous answer was REJECTED because it did not combine the sources. "
                        f"{len(contrib)} notes are tagged ANSWERS/PARTIAL — note numbers {contrib}. You "
                        "MUST merge the DISTINCT facts from at least two of these notes into one complete "
                        "answer, tagging each fact with the note it came from (e.g. [1][2][3]). Do not copy "
                        "a single note and do not stop at the first relevant note. Use RELATED/OFFTOPIC "
                        "notes only to explain gaps. Cite every claim with [n]."
                    )},
                    {"role": "user", "content": prompt},
                ]
                answer = chat_stream(c, regen, tokens,
                                     lambda d: self.send_stream({"type": "delta", "text": d}))

            approx_tokens = max(1, len(answer) // 4)
            self.send_stream({"type": "answer_meta", "tokens": approx_tokens, "chars": len(answer),
                              "sufficient": sufficient, "notes": len(ordered)})
            self._stage("answer", "done")
            self._log(f"Answer complete — ~{approx_tokens} tokens, {round(time.time() - t0, 1)}s total.", "ok")
            self.send_stream({"type": "done", "elapsed": round(time.time() - t0, 1)})
        except Exception as e:
            self._log(f"{stage.upper()} failed — {e}", "error")
            self._stage(stage, "error")
            self.send_stream({"type": "error", "error": str(e), "stage": stage})
            self.send_stream({"type": "done", "elapsed": round(time.time() - t0, 1), "aborted": True})

    # ---- download model SSE ----
    def _download_model_stream(self, file_id):
        self.send_stream_start()
        entry = catalog_entry(file_id)
        if not entry:
            self.send_stream({"type": "error", "error": f"Unknown model id: {file_id}"})
            self.send_stream({"type": "done", "aborted": True})
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / file_id
        tmp = dest.with_suffix(dest.suffix + ".part")
        last_err = ""
        for url in (entry.get("urls") or []):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "OfflineRAG/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    total = int(resp.headers.get("Content-Length") or 0)
                    done = 0
                    with open(tmp, "wb") as out:
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                            done += len(chunk)
                            if total:
                                self.send_stream({"type": "progress",
                                                  "percent": round(100 * done / total),
                                                  "mb_done": round(done / 1048576, 1),
                                                  "mb_total": round(total / 1048576, 1)})
                tmp.replace(dest)
                self.send_stream({"type": "done", "id": file_id})
                return
            except Exception as exc:
                last_err = str(exc)
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
                self._log(f"Download source failed: {exc}", "warn")
        self.send_stream({"type": "error", "error": f"Download failed: {last_err}"})
        self.send_stream({"type": "done", "aborted": True})


    # ---- ingest (streamed) ----
    def _ingest_stream(self, c, body):
        folder = Path(c["source_folder"])
        allowed_value = str(c.get("include_extensions") or DEFAULTS["include_extensions"])
        allowed = {x.strip().lower() for x in allowed_value.split(",") if x.strip()}
        mode = str(body.get("mode", "incremental")).lower()
        self._gone = False
        self.send_stream_start()
        self.send_stream({"type": "progress", "phase": "starting", "percent": 1,
                          "text": "Backend connected. Reading settings and preparing folder scan..."})
        try:
            existing = load(INDEX_FILE, {"chunks": [], "files": {}, "chunk_params": None})
            params = chunk_params_sig(c)
            if mode == "full":
                existing = {"chunks": [], "files": {}, "chunk_params": None}
                _EMBED_CACHE.clear()
                with _DOCUMENT_TEXT_CACHE_LOCK:
                    _DOCUMENT_TEXT_CACHE.clear()
                invalidate_index_cache()
                self._log("Full rebuild: cleared RAM index and embedding caches; all vectors will be regenerated.")
            if mode != "full" and existing.get("chunk_params") != params:
                mode = "full"
                self._log("Chunking parameters changed — forcing a full rebuild.", "warn")

            self._log(f"Ingest started ({mode}) — scanning {folder} ...")
            self.send_stream({
                "type": "progress", "phase": "ingest", "percent": 2,
                "text": ("Full rebuild requested. Scanning document folder..." if mode == "full"
                         else "Scanning for new or changed documents (saved vectors are reused)..."),
            })

            if not folder.is_dir():
                raise RuntimeError(f"Source folder does not exist or is not accessible: {folder}")
            all_files, scan_errors = [], []
            try:
                scanned_entries = 0
                last_scan_update = time.time()
                for p in folder.rglob("*"):
                    scanned_entries += 1
                    try:
                        if p.is_file() and DATA not in p.parents:
                            all_files.append(p)
                    except OSError as exc:
                        scan_errors.append(f"{p}: {exc}")
                    if scanned_entries % 250 == 0 or time.time() - last_scan_update >= 2:
                        self.send_stream({"type": "progress", "phase": "scan", "percent": 3,
                                          "text": f"Scanning folder... {scanned_entries:,} entries checked; {len(all_files):,} files found."})
                        last_scan_update = time.time()
            except OSError as exc:
                raise RuntimeError(f"Unable to scan source folder {folder}: {exc}") from exc
            ext_counts = Counter((p.suffix.lower() or "[no extension]") for p in all_files)
            files = [p for p in all_files if p.suffix.lower() in allowed]
            current_sigs = {str(p): file_signature(p) for p in files}
            top_types = ", ".join(f"{ext}={count}" for ext, count in ext_counts.most_common(12)) or "none"
            self._log(f"Folder scan complete: {len(all_files)} total file(s); {len(files)} supported. Types: {top_types}")
            if scan_errors:
                self._log(f"Folder scan had {len(scan_errors)} inaccessible path(s); first: {scan_errors[0]}", "warn")
            if not files:
                allowed_text = ", ".join(sorted(allowed))
                found_text = top_types
                message = (f"No supported documents found. Scanned {len(all_files)} file(s) under {folder}. "
                           f"Allowed types: {allowed_text}. Found types: {found_text}.")
                if ext_counts.get(".msg") and ".msg" not in allowed:
                    message += " Add .msg to File types, then retry."
                self._log(message, "error")
                self.send_stream({"type": "error", "error": message})
                return

            reused, changed = [], files
            if mode != "full":
                old_sigs = existing.get("files", {})

                # ADD / UPDATE IS APPEND-PRESERVING:
                # Keep every previously indexed document that is absent from the
                # current scan. This includes files excluded by File types and files
                # later removed from the source folder. A same-path file that still
                # exists but has changed is replaced by its newly parsed passages.
                # Only Rebuild from scratch or Delete index removes retained content.
                retained_absent = {
                    old_path: old_sig for old_path, old_sig in old_sigs.items()
                    if old_path not in current_sigs
                }
                if retained_absent:
                    current_sigs.update(retained_absent)
                    self._log(
                        f"Retaining {len(retained_absent)} previously indexed document(s) "
                        "not present in the current scan.", "ok"
                    )

                reused = [
                    ch for ch in existing.get("chunks", [])
                    if isinstance(ch, dict) and ch.get("embedding")
                    and ch.get("path") in current_sigs
                    and old_sigs.get(ch.get("path")) == current_sigs.get(ch.get("path"))
                ]
                reused_paths = {ch.get("path") for ch in reused}
                changed = [p for p in files if str(p) not in reused_paths]
                self.send_stream({
                    "type": "progress", "phase": "ingest", "percent": 8,
                    "text": (f"Reusing {len(reused)} saved passage(s). "
                             f"Updating {len(changed)} changed/new document(s). "
                             f"Retaining {len(retained_absent)} document(s) absent from this scan."),
                })
                self._log(f"{len(files)} file(s) found — {len(reused)} passage(s) reusable, "
                          f"{len(changed)} to process, {len(retained_absent)} retained absent document(s).")
            else:
                self.send_stream({
                    "type": "progress", "phase": "ingest", "percent": 8,
                    "text": f"Found {len(files)} supported document(s). Creating passages...",
                })
                self._log(f"{len(files)} supported document(s) found.")

            new_chunks, errors = [], []
            with SNAPSHOT_MANIFEST_LOCK:
                snapshot_manifest = load_snapshot_manifest()
            for number, p in enumerate(changed, 1):
                if self._gone:
                    return
                try:
                    added = chunks_for(p, c)
                    snapshot_record = save_document_snapshot(p, folder, manifest=snapshot_manifest)
                    for chunk in added:
                        chunk["document_id"] = snapshot_record["document_id"]
                        chunk["snapshot_available"] = True
                    new_chunks.extend(added)
                    self._log(f"Parsed and snapshotted {p.name} — {len(added)} passage(s), {snapshot_record['snapshot_chars']:,} extracted characters retained.")
                except Exception as e:
                    errors.append(f"{p.name}: {e}")
                    self._log(f"Failed to parse {p.name} — {e}", "error")
                self.send_stream({
                    "type": "progress", "phase": "ingest",
                    "percent": 8 + round(20 * number / max(len(changed), 1)),
                    "text": f"Processed {number} of {len(changed)} document(s): {p.name}",
                })

            # Keep snapshots for current indexed files and save the manifest atomically.
            current_document_ids = {chunk.get("document_id") for chunk in (reused + new_chunks) if chunk.get("document_id")}
            snapshot_manifest["documents"] = {doc_id: rec for doc_id, rec in snapshot_manifest.get("documents", {}).items()
                                                if doc_id in current_document_ids or Path(rec.get("snapshot", "")).is_file()}
            with SNAPSHOT_MANIFEST_LOCK:
                save_json(SNAPSHOT_MANIFEST_FILE, snapshot_manifest)

            if not new_chunks and not reused:
                self._log("No supported text found in the selected folder.", "error")
                self.send_stream({"type": "error",
                                  "error": "No supported text was found in the selected folder."})
                return

            embedding_model = resolved(c, "embedding")

            if new_chunks:
                self.send_stream({
                    "type": "progress", "phase": "ingest", "percent": 30,
                    "text": (f"Creating semantic vectors for {len(new_chunks)} new/changed "
                             f"passage(s) using {HARDWARE['label']}..."),
                })
                self._log(f"Embedding {len(new_chunks)} passage(s) with {embedding_model}...")
                workers = positive(c.get("embedding_workers"), 1, 1, 16)
                batch_size = max(1, workers)
                backend = "CPU" if EMBED_SERVER.gpu_layers == 0 else detect_gpu_backend().upper()
                self._log(f"Embedding backend: {backend}; workers={workers}; request timeout={positive(c.get('embedding_request_timeout'), 90, 15, 600)}s.")
                for start in range(0, len(new_chunks), batch_size):
                    if self._gone:
                        return
                    batch = new_chunks[start:start + batch_size]
                    def embedding_event(ev, base=start, total=len(new_chunks)):
                        absolute = min(total, base + int(ev.get("item", 1)))
                        kind = ev.get("event")
                        if kind == "heartbeat":
                            msg = (f"Heartbeat: {backend} embedding passage {absolute}/{total} "
                                   f"still active ({ev.get('elapsed', 0)}s in current request).")
                            self.send_stream({"type": "heartbeat", "phase": "embedding",
                                              "percent": 30 + round(50 * max(0, absolute - 1) / total),
                                              "text": msg, "elapsed": ev.get("elapsed", 0),
                                              "current": absolute, "total": total, "backend": backend})
                            self._log(msg)
                        elif kind == "retry":
                            self._log(f"Passage {absolute}/{total}: context overflow; retrying with shorter text. {ev.get('error','')}", "warn")
                        elif kind == "error":
                            self._log(f"Passage {absolute}/{total}: embedding request failed without repeated retries. {ev.get('error','')}", "warn")
                    vectors = embed(c, [x["text"] for x in batch], tolerate_errors=True,
                                    on_progress=embedding_event)
                    failed = 0
                    for item, vector in zip(batch, vectors):
                        if vector is None:
                            failed += 1
                            errors.append(f"{item.get('source', 'passage')}: embedding failed")
                            self._log(f"Skipped one passage from {item.get('source', 'unknown')} after an embedding error.", "warn")
                        else:
                            item["embedding"] = vector
                    if failed:
                        self._log(f"Embedding batch completed with {failed} skipped passage(s); ingestion will continue.", "warn")
                    done_count = min(start + len(batch), len(new_chunks))
                    self._log(f"Vectorized {done_count}/{len(new_chunks)} passage(s).")
                    self.send_stream({
                        "type": "progress", "phase": "ingest",
                        "percent": 30 + round(50 * done_count / len(new_chunks)),
                        "text": f"Vectorized {done_count} of {len(new_chunks)} passage(s)",
                    })
                new_chunks = [x for x in new_chunks if x.get("embedding")]
                if not new_chunks and not reused:
                    raise RuntimeError("All passages failed to embed; check the embedding server log and context setting.")
            else:
                self._log("No new passages to embed — reusing all saved vectors.", "ok")
                self.send_stream({"type": "progress", "phase": "ingest", "percent": 80,
                                  "text": "No new passages to embed. Reusing all saved vectors..."})

            docs = [d for d in (reused + new_chunks) if d.get("path") in current_sigs]
            if not docs:
                self.send_stream({"type": "error",
                                  "error": "No supported text was found in the selected folder."})
                return

            self._log(f"Building exact-keyword index over {len(docs)} passage(s)...")
            self.send_stream({"type": "progress", "phase": "ingest", "percent": 87,
                              "text": f"Building exact-keyword index over {len(docs)} passage(s)..."})

            lengths, document_frequency, postings = [], Counter(), {}
            for item_index, item in enumerate(docs):
                counts = Counter(words(item["text"]))
                lengths.append(sum(counts.values()))
                for term, count in counts.items():
                    document_frequency[term] += 1
                    postings.setdefault(term, []).append([item_index, count])
            lexical = {
                "lengths": lengths,
                "average_length": sum(lengths) / len(lengths),
                "document_frequency": document_frequency,
                "postings": postings,
            }

            self.send_stream({"type": "progress", "phase": "ingest", "percent": 96,
                              "text": "Saving portable index to data/index.json..."})

            with lock:
                save_json(INDEX_FILE, {
                    "version": 2, "chunks": docs, "lexical": lexical,
                    "embedding_model": embedding_model, "files": current_sigs,
                    "chunk_params": params, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

            invalidate_index_cache()

            lancedb_ok = False
            if c.get("use_lancedb") and LANCEDB_AVAILABLE:
                try:
                    import numpy as np
                    self._log("Initializing LanceDB vector database...", "info")
                    self.send_stream({"type": "progress", "phase": "ingest", "percent": 97,
                                      "text": "Preparing LanceDB vector database..."})

                    db = lancedb.connect(str(DATA / "lancedb"))
                    lancedb_data = [
                        {"id": d["id"],
                         "vector": np.array(d["embedding"], dtype=np.float32)}
                        for d in docs if d.get("embedding")
                    ]

                    if lancedb_data:
                        ldb_batch = 2000
                        total_batches = (len(lancedb_data) + ldb_batch - 1) // ldb_batch
                        for i in range(total_batches):
                            batch = lancedb_data[i * ldb_batch : (i + 1) * ldb_batch]
                            if i == 0:
                                db.create_table("chunks", data=batch, mode="overwrite")
                            else:
                                tbl = db.open_table("chunks")
                                tbl.add(batch)
                            done_count = min((i + 1) * ldb_batch, len(lancedb_data))
                            pct = 97 + max(1, int(3 * done_count / len(lancedb_data)))
                            self.send_stream({
                                "type": "progress", "phase": "ingest",
                                "percent": min(pct, 99),
                                "text": f"LanceDB: {done_count:,} / {len(lancedb_data):,} vectors written..."
                            })
                            self._log(f"LanceDB: {done_count:,}/{len(lancedb_data):,} vectors saved.")

                    lancedb_ok = True
                    self._log("LanceDB vector index ready.", "ok")

                except Exception as ldb_exc:
                    self._log(f"LanceDB write failed (non-fatal): {ldb_exc}", "warn")
                    ldb_path = DATA / "lancedb"
                    if ldb_path.exists():
                        try:
                            shutil.rmtree(ldb_path)
                        except OSError:
                            pass
            elif c.get("use_lancedb") and not LANCEDB_AVAILABLE:
                self._log("LanceDB not installed — run Install_LanceDB.bat. Skipping vector DB.", "warn")

            # ---- final done event ----
            self._log(f"Index saved — {len(docs)} passages, "
                      f"{len({x['path'] for x in docs})} documents.", "ok")
            self.send_stream({
                "type": "done",
                "documents": len({x["path"] for x in docs}),
                "chunks": len(docs),
                "reused": len(reused),
                "new": len(new_chunks),
                "mode": mode,
                "errors": errors,
                "lancedb": lancedb_ok,
                "performance": HARDWARE["label"],
            })

        except Exception as e:
            self._log(f"Ingest failed — {e}", "error")
            try:
                self.send_stream({"type": "error", "error": str(e)})
            except Exception:
                pass

    # ---- routes ----
    def do_GET(self):
        if self.path == "/api/local-models":
            cfg = settings()
            cat = [{**m,
                    "downloaded": (MODELS_DIR / m["id"]).exists(),
                    "max_chars": model_max_chars(m.get("ctx", 2048), m.get("kind", "embedding"))}
                   for m in load_catalog()]
            servers = {}
            for kind in ("embedding", "reranker"):
                srv = _server_for(kind)
                srv.health()
                d = srv.describe()
                ctx = d.get("effective_ctx") or d.get("nominal_ctx") or 2048
                d["max_chars"] = model_max_chars(ctx, kind)
                servers[kind] = d
            return self.send_json({
                "catalog": cat,
                "selection": {"embedding": cfg.get("embedding_model"), "reranker": cfg.get("rerank_model")},
                "servers": servers,
                "constants": {"est_chars_per_token": EST_CHARS_PER_TOKEN, "ctx_safety": CTX_SAFETY,
                    "rerank_query_reserve": RERANK_QUERY_RESERVE},
                "threads": HARDWARE["cores"],
                "gpu_backend": detect_gpu_backend(),
            })

        if self.path == "/api/local-model-status":
            out = {}
            for kind in ("embedding", "reranker"):
                srv = _server_for(kind)
                srv.health()
                d = srv.describe()
                ctx = d.get("effective_ctx") or d.get("nominal_ctx") or 2048
                d["max_chars"] = model_max_chars(ctx, kind)
                out[kind] = d
            return self.send_json({"servers": out})

        if self.path == "/api/pst/dependencies":
            return self.send_json(pst_dependency_report())
        if self.path.startswith("/api/pst/status"):
            job_id = parse_qs(urlparse(self.path).query).get("job", [""])[0]
            with PST_JOBS_LOCK:
                job = dict(PST_JOBS.get(job_id, {}))
            return self.send_json(job if job else {"error": "PST job not found."}, 200 if job else 404)


        if self.path.startswith("/api/document-text"):
            try:
                params = parse_qs(urlparse(self.path).query)
                requested = params.get("path", [""])[0]
                document_id = params.get("document_id", [""])[0]
                offset = max(0, int(params.get("offset", ["0"])[0] or 0))
                requested_limit = int(params.get("limit", [str(DOCUMENT_VIEW_PAGE_CHARS)])[0] or DOCUMENT_VIEW_PAGE_CHARS)
                limit = max(1000, min(requested_limit, DOCUMENT_VIEW_PAGE_CHARS))
                content, source_kind, active_path, record, cache_hit = "", "original", "", {}, False
                try:
                    target = safe_source_document(requested)
                    content, cache_hit = cached_full_document_text(target)
                    active_path = str(target)
                except (FileNotFoundError, PermissionError, OSError):
                    if document_id:
                        with SNAPSHOT_MANIFEST_LOCK:
                            record = load_snapshot_manifest().get("documents", {}).get(document_id, {})
                        relocated = record.get("current_path", "")
                        if relocated and Path(relocated).is_file() and relocated != requested:
                            try:
                                target = Path(relocated).resolve(strict=True)
                                content, cache_hit = cached_full_document_text(target)
                                active_path, source_kind = str(target), "relocated"
                            except Exception:
                                content = ""
                        if not content:
                            content, record = read_document_snapshot(document_id)
                            active_path, source_kind, cache_hit = record.get("snapshot", ""), "snapshot", True
                    else:
                        raise FileNotFoundError("Original document is missing and no snapshot identity was supplied.")
                total = len(content)
                page = content[offset:offset + limit]
                next_offset = offset + len(page)
                return self.send_json({"success": True, "path": active_path, "original_path": requested,
                    "document_id": document_id, "text": page, "offset": offset,
                    "returned": len(page), "total_chars": total,
                    "next_offset": next_offset if next_offset < total else None,
                    "complete": next_offset >= total, "cache_hit": cache_hit,
                    "source_kind": source_kind, "using_fallback": source_kind == "snapshot",
                    "original_missing": source_kind in ("snapshot", "relocated"),
                    "snapshot_created": record.get("snapshot_created", ""),
                    "expected_relative_path": record.get("relative_path", "")})
            except ValueError as exc:
                return self.send_json({"success": False, "error": f"Invalid document request: {exc}"}, 400)
            except Exception as exc:
                print(f"[document-text] {type(exc).__name__}: {exc}", flush=True)
                return self.send_json({"success": False, "error": f"Document retrieval failed: {type(exc).__name__}: {exc}"}, 500)

        if self.path.startswith("/api/open-path"):
            params = parse_qs(urlparse(self.path).query)
            requested = params.get("path", [""])[0]
            reveal = params.get("reveal", ["0"])[0] == "1"
            try:
                target = safe_source_document(requested)
                if os.name == "nt":
                    if reveal:
                        subprocess.Popen(["explorer", "/select,", str(target)])
                    else:
                        os.startfile(str(target))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-R" if reveal else str(target)] + ([str(target)] if reveal else []))
                else:
                    subprocess.Popen(["xdg-open", str(target.parent if reveal else target)])
                return self.send_json({"success": True, "path": str(target), "reveal": reveal})
            except FileNotFoundError:
                return self.send_json({"success": False, "error": "Original document no longer exists."}, 404)
            except Exception as exc:
                return self.send_json({"success": False, "error": str(exc)}, 500)

        if self.path == "/api/settings":
            return self.send_json(settings())

        if self.path == "/api/options":
            base = ROOT.parent.parent
            folders = [ROOT.parent, base]
            try:
                folders += [p for p in base.iterdir() if p.is_dir() and p.name not in ("OfflineRAG",)]
            except OSError:
                pass
            return self.send_json({
                "folders": [{"value": str(p), "label": "Ingest folder (recommended)" if p == ROOT.parent else p.name}
                            for p in dict.fromkeys(folders)]
            })
        if self.path == "/api/status":
            idx = load(INDEX_FILE, {"chunks": [], "updated": None})
            return self.send_json({"chunks": len(idx.get("chunks", [])), "updated": idx.get("updated"),
                "index_exists": INDEX_FILE.exists(), "lancedb_exists": (DATA / "lancedb").exists(),
                "embedding_cache_entries": len(_EMBED_CACHE),
                "hardware": HARDWARE["label"], "gpu_backend": detect_gpu_backend()})


        if self.path == "/api/models":
            state, detail = "connected", ""
            try:
                config = settings()
                available = lmstudio_api(config, "/api/v1/models", timeout=2).get("models", [])
                models = [
                    {"id": item["key"], "label": item.get("display_name", item["key"]),
                     "loaded": bool(item.get("loaded_instances")), "type": item.get("type", "llm")}
                    for item in available if item.get("type") == "llm"
                ]
            except Exception:
                try:
                    models = [{"id": item["id"], "label": item["id"], "loaded": True, "type": "llm"}
                              for item in api(settings(), "/models", timeout=2).get("data", [])]
                except Exception:
                    models, state = [], "unavailable"
            detail = (f"LM Studio is not reachable at {openai_base(settings())}. Start its local server, then refresh models.")
            return self.send_json({
                "data": [
                    {"id": DEFAULT_EMBED_FILE, "label": "Local Embedding (llama-server)", "type": "bundled", "loaded": True},
                    {"id": DEFAULT_RERANK_FILE, "label": "Local Reranker (llama-server)", "type": "bundled", "loaded": True},
                ] + models,
                "state": state,
                "detail": detail if state != "connected" else "",
            })
        # Serve the UI explicitly. This avoids directory redirects, stale cached
        # responses and browser confusion if the working directory changes.
        clean_path = self.path.split("?", 1)[0]
        if clean_path in ("/", "/index.html"):
            try:
                raw = (ROOT / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(raw)
                return
            except Exception as exc:
                return self.send_json({"error": f"Unable to serve index.html: {exc}"}, 500)
        return super().do_GET()

    def do_POST(self):
        try:
            body = self.body()
            if self.path == "/api/select-local-model":
                kind = (body.get("kind") or "").strip()
                file_id = (body.get("id") or "").strip()
                if kind not in ("embedding", "reranker") or not file_id:
                    return self.send_json({"success": False, "error": "Bad request."}, 400)
                entry = catalog_entry(file_id)
                if entry and entry.get("kind") != kind:
                    return self.send_json({"success": False, "error": "Model kind mismatch."}, 400)
                if not (MODELS_DIR / file_id).exists():
                    return self.send_json({"success": False, "error": f"{file_id} is not downloaded yet."}, 400)
                cfg = settings()
                cfg["embedding_model" if kind == "embedding" else "rerank_model"] = file_id
                save_json(SETTINGS_FILE, cfg)
                srv = _server_for(kind)
                srv.start(file_id, (entry or {}).get("ctx", 2048), HARDWARE["cores"])
                ok = srv.wait_ready(120)
                return self.send_json({
                    "success": ok, "kind": kind, "id": file_id,
                    "state": srv.state, "error": srv.error,
                    "effective_ctx": srv.eff_ctx,
                    "reindex_required": kind == "embedding",
                    "server": srv.describe(),
                }, 200 if ok else 500)

            if self.path == "/api/ingest-stream":
                c = settings()
                self._ingest_stream(c, body)
                return




            if self.path == "/api/pst/import":
                c = settings()
                pst_path = str(body.get("pst_path") or c.get("pst_path") or "").strip()
                try:
                    if not pst_path or not Path(pst_path).expanduser().is_file():
                        return self.send_json({"success": False, "error": "Enter an existing PST file path."}, 400)
                    job = start_pst_job(pst_path, c)
                    return self.send_json({"success": True, "job": job["id"], "state": job["state"]}, 202)
                except Exception as exc:
                    return self.send_json({"success": False, "error": str(exc)}, 400)

            if self.path == "/api/document-relocate":
                document_id = str(body.get("document_id") or "").strip()
                folder = str(body.get("folder") or "").strip()
                if not document_id or not folder:
                    return self.send_json({"success": False, "error": "Document identity and replacement folder are required."}, 400)
                try:
                    report = locate_snapshot_document(document_id, folder)
                    return self.send_json({"success": True, **report})
                except Exception as exc:
                    return self.send_json({"success": False, "error": str(exc)}, 400)

            if self.path == "/api/settings":
                allowed = set(DEFAULTS)
                updated = {key: value for key, value in body.items() if key in allowed}
                if updated.get("source_folder") and not Path(updated["source_folder"]).is_dir():
                    return self.send_json({"error": "Enter an existing local document folder path."}, 400)
                for key in ("use_llm_rerank", "adaptive_rag", "use_hyde", "use_lancedb", "gpu_offload", "pst_extract_attachments"):
                    if key in updated:
                        updated[key] = str(updated[key]).lower() == "true"
                gpu_changed = any(k in updated for k in ("gpu_offload", "gpu_layers", "local_server_parallel", "local_server_batch"))
                # Hold the same per-file lock across read + merge + write so two
                # simultaneous auto-saves cannot overwrite each other's fields.
                with _json_lock_for(SETTINGS_FILE):
                    current = settings()
                    save_json(SETTINGS_FILE, {**current, **updated})
                    saved = settings()
                if gpu_changed:   # apply GPU offload change without a full app restart
                    threading.Thread(target=lambda: _start_local_servers(settings(), wait=True), daemon=True).start()
                # Retrieval weights are loaded for every enquiry; no restart or re-index.
                saved["retrieval_balance_applies"] = "next_query"
                return self.send_json(saved)

            if self.path == "/api/load-model":
                key = (body.get("model") or "").strip()
                if not key:
                    return self.send_json({"success": False, "loaded": False,
                                           "error": "Choose a model to load."}, 400)
                c = settings()
                try:
                    models = _lm_models(c)
                    if not models:
                        return self.send_json({"success": False, "loaded": False, "model": key,
                                               "error": "LM Studio returned no models. Is the local server running?"})
                    cands, already = _candidate_ids(models, key)
                    if already:
                        return self.send_json({"success": True, "loaded": True, "model": key,
                                               "note": "already loaded"})
                    before = _loaded_set(models)
                    last_err = ""
                    for ident in cands:
                        try:
                            lmstudio_api(c, "/api/v1/models/load", {"model": ident}, timeout=240)
                        except urllib.error.HTTPError as he:
                            try:
                                last_err = f"HTTP {he.code}: {he.read().decode('utf-8', 'replace')[:240]}"
                            except Exception:
                                last_err = f"HTTP {he.code}"
                            continue
                        except Exception as ex:
                            last_err = str(ex)
                            continue
                        for _ in range(24):
                            time.sleep(0.5)
                            now_models = _lm_models(c)
                            _, loaded = _candidate_ids(now_models, key)
                            if loaded:
                                return self.send_json({"success": True, "loaded": True, "model": key})
                            after = _loaded_set(now_models)
                            cands2, _ = _candidate_ids(now_models, key)
                            if (after - before) and (set(cands2) & after):
                                return self.send_json({"success": True, "loaded": True, "model": key})
                    return self.send_json({
                        "success": False, "loaded": False, "model": key,
                        "error": last_err or "LM Studio accepted the request but the model did not appear as loaded.",
                    })
                except Exception as ex:
                    return self.send_json({"success": False, "loaded": False, "model": key, "error": str(ex)})
            if self.path == "/api/ingest":
                c = settings()
                folder = Path(c["source_folder"])
                allowed_value = str(c.get("include_extensions") or DEFAULTS["include_extensions"])
                allowed = {x.strip().lower() for x in allowed_value.split(",") if x.strip()}
                docs, errors = [], []
                for p in folder.rglob("*"):
                    if p.is_file() and p.suffix.lower() in allowed and DATA not in p.parents:
                        try:
                            docs.extend(chunks_for(p, c))
                        except Exception as e:
                            errors.append(f"{p.name}: {e}")
                if not docs:
                    return self.send_json({"documents": 0, "chunks": 0,
                                           "errors": errors + ["No supported text was found in the selected folder."]}, 400)
                embedding_model = resolved(c, "embedding")
                batch_size = max(16, HARDWARE["embedding_workers"] * 16)
                for start in range(0, len(docs), batch_size):
                    batch = docs[start:start + batch_size]
                    for item, vector in zip(batch, embed(c, [x["text"] for x in batch])):
                        item["embedding"] = vector
                lengths, document_frequency, postings = [], Counter(), {}
                for item_index, item in enumerate(docs):
                    counts = Counter(words(item["text"]))
                    lengths.append(sum(counts.values()))
                    for term, count in counts.items():
                        document_frequency[term] += 1
                        postings.setdefault(term, []).append([item_index, count])
                lexical = {"lengths": lengths, "average_length": sum(lengths) / len(lengths),
                           "document_frequency": document_frequency, "postings": postings}
                current_sigs = {str(p): file_signature(p) for p in folder.rglob("*")
                                if p.is_file() and p.suffix.lower() in allowed and DATA not in p.parents}
                with lock:
                    save_json(INDEX_FILE, {"version": 2, "chunks": docs, "lexical": lexical,
                        "embedding_model": embedding_model, "files": current_sigs,
                        "chunk_params": chunk_params_sig(c),
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
                invalidate_index_cache()

                # LanceDB (non-fatal — must not abort the response)
                lancedb_ok = False
                if c.get("use_lancedb") and LANCEDB_AVAILABLE:
                    try:
                        import numpy as np
                        db = lancedb.connect(str(DATA / "lancedb"))
                        db.create_table("chunks", data=[
                            {"id": d["id"],
                             "vector": np.array(d["embedding"], dtype=np.float32)}
                            for d in docs if d.get("embedding")
                        ], mode="overwrite")
                        lancedb_ok = True
                    except Exception:
                        ldb_path = DATA / "lancedb"
                        if ldb_path.exists():
                            try:
                                shutil.rmtree(ldb_path)
                            except OSError:
                                pass

                return self.send_json({"documents": len({x["path"] for x in docs}),
                                       "chunks": len(docs),
                                       "errors": errors,
                                       "lancedb": lancedb_ok,
                                       "performance": HARDWARE["label"]})
            
            if self.path == "/api/search":
                c = settings()
                query = body.get("query", "").strip()
                if not query:
                    return self.send_json({"error": "Enter a question to search your documents."}, 400)
                results, error = retrieve(query, c)
                if error:
                    return self.send_json({"error": error}, 400)
                answer = ""
                if body.get("answer", True) and c["chat_model"]:
                    answer = chat(c, [
                        {"role": "system", "content": "You are a precise document question-answering assistant. Follow the evidence and citation requirements exactly."},
                        {"role": "user", "content": evidence_prompt(query, context_limited_evidence(c, results))},
                    ], model=c["chat_model"])
                return self.send_json({"answer": answer, "results": [public(r) for r in results]})
            
            if self.path == "/api/answer-stream":
                c = settings()
                query = text(body.get("query"))
                if not query:
                    self.send_stream_start()
                    self.send_stream({"type": "error", "error": "Enter a question to search your documents."})
                    self.send_stream({"type": "done", "aborted": True})
                    return
                # The request toggle is authoritative. A missing answer model also
                # forces evidence-only mode.
                retrieval_only = bool(body.get("retrieval_only", False))
                evidence_only = retrieval_only or (not bool(body.get("answer", True))) or (not c["chat_model"])
                want_answer = not evidence_only
                self.send_stream_start()
                self._answer_stream(c, query, want_answer, evidence_only, bool(body.get("adaptive", True)))
                return
            
            if self.path == "/api/index/clear":
                removed = []
                candidates = [INDEX_FILE, INDEX_FILE.with_suffix(".tmp")]
                candidates.extend(DATA.glob("index.json.*.tmp"))
                candidates.extend(DATA.glob("index.*.tmp"))
                for candidate in dict.fromkeys(candidates):
                    try:
                        if candidate.exists():
                            candidate.unlink()
                            removed.append(str(candidate))
                    except OSError as exc:
                        return self.send_json({"cleared": False, "error": f"Could not delete {candidate}: {exc}"}, 500)
                db_path = DATA / "lancedb"
                if db_path.exists():
                    try:
                        shutil.rmtree(db_path)
                        removed.append(str(db_path))
                    except OSError as exc:
                        return self.send_json({"cleared": False, "error": f"Could not delete LanceDB: {exc}"}, 500)
                invalidate_index_cache()
                with _DOCUMENT_TEXT_CACHE_LOCK:
                    _DOCUMENT_TEXT_CACHE.clear()
                try:
                    SNAPSHOT_MANIFEST_FILE.unlink(missing_ok=True)
                    if SNAPSHOT_DIR.exists():
                        shutil.rmtree(SNAPSHOT_DIR)
                    SNAPSHOT_DIR.mkdir(exist_ok=True)
                except OSError as exc:
                    return self.send_json({"cleared": False, "error": f"Could not clear document snapshots: {exc}"}, 500)
                _EMBED_CACHE.clear()
                empty = not INDEX_FILE.exists() and not db_path.exists()
                return self.send_json({"cleared": empty, "chunks": 0, "removed": removed}, 200 if empty else 500)
            self.send_json({"error": "Not found"}, 404)
        except Exception as e:
            try:
                self.send_json({"error": str(e)}, 500)
            except Exception:
                pass


if __name__ == "__main__":
    os.chdir(ROOT)
    # Port 8765 is the actual Offline RAG application. Ports 8787 and 8788
    # are internal llama-server model APIs and their built-in diagnostic UIs.
    try:
        web_server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    except OSError as exc:
        print(f"ERROR: could not start Offline RAG UI on port 8765: {exc}", flush=True)
        print("Close older Offline RAG/Python processes and run the launcher again.", flush=True)
        raise
    print(f"GPU backend: {detect_gpu_backend().upper()}", flush=True)
    print("OFFLINE RAG UI READY: http://127.0.0.1:8765/", flush=True)
    print("Internal APIs only: embedding=8787, reranker=8788", flush=True)
    _start_local_servers(settings(), wait=False)
    atexit.register(_shutdown_local_servers)
    web_server.serve_forever()
