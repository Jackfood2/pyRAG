#!/usr/bin/env python3

import atexit
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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

MAX_WAVES = 5                 # corrective-retrieval waves before we answer anyway
WAVE_EXTRA_CANDIDATES = 8     # candidate budget grows by this each wave
ASSESS_CONFIDENCE_STOP = 0.8  # assess confidence at/above this stops early

DEFAULTS = {
    "source_folder": str(ROOT.parent),
    "lmstudio_url": "http://127.0.0.1:1234/v1",
    "embedding_model": DEFAULT_EMBED_FILE,
    "analysis_model": "",          # small/fast model: rewrite + analyze + assess
    "chat_model": "",              # larger model: final cited answer only
    "rerank_model": DEFAULT_RERANK_FILE,
    "chunk_size": 900,
    "chunk_overlap": 140,
    "candidate_count": 32,
    "rerank_count": 4,                 # repurposed: assess-checkpoint interval
    "max_candidate_checks": 24,        # hard cap on passages analysed
    "semantic_weight": 0.72,
    "keyword_weight": 0.28,
    "use_llm_rerank": True,
    "answer_temperature": 0.1,
    "answer_tokens": 900,
    "include_extensions": ".txt,.md,.csv,.pdf,.docx,.xlsx,.xlsm",
    "max_row_chars": 5000,
    "adaptive_rag": True,
    "use_hyde": True,
    "fast_path_score": 0.82,           # legacy, unused by v5
    "memory_fact_limit": 18,
    "context_window": 8192,
}

lock = threading.Lock()
_EMBED_CACHE = {}

# --------------------------------------------------------------------------- #
#  local model catalog + llama-server process management
# --------------------------------------------------------------------------- #
RUNTIME_DIR = ROOT / "runtime"
LLAMA_SERVER = RUNTIME_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")
MODELS_DIR = ROOT / "models"
LOG_DIR = DATA / "logs"

EMBEDDING_PORT = 8787
RERANK_PORT = 8788

EST_CHARS_PER_TOKEN = 2.5     # conservative for dense/technical text
CTX_SAFETY = 0.9              # headroom for special tokens / chat template
RERANK_QUERY_RESERVE = 384    # tokens reserved for the query in a rerank call
MAX_SERVER_CTX = 8192

# verified == URL confirmed working. For others, fix URLs in models/catalog.json.
BUILTIN_CATALOG = [
    {"id": "nomic-embed-text-v1.5.Q4_K_M.gguf", "kind": "embedding",
     "name": "Nomic Embed v1.5 (Q4)", "ctx": 8192, "verified": True,
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
    """Largest passage (in chars) that safely fits a model's context."""
    usable = max(64.0, float(ctx) * CTX_SAFETY)
    if kind == "reranker":
        usable = max(64.0, usable - RERANK_QUERY_RESERVE)
    return int(usable * EST_CHARS_PER_TOKEN)

class LocalServer:
    """Owns one bundled llama-server process (embedding or reranker)."""

    def __init__(self, kind, port):
        self.kind = kind
        self.port = port
        self.proc = None
        self.file = None
        self.nominal_ctx = None
        self.eff_ctx = None
        self.state = "off"          # off | missing | starting | ok | error
        self.error = ""
        self._lock = threading.Lock()

    def build_args(self, threads):
        ctx = min(int(self.nominal_ctx or 2048), MAX_SERVER_CTX)
        args = [str(LLAMA_SERVER), "-m", str(MODELS_DIR / self.file),
                "--host", "127.0.0.1", "--port", str(self.port),
                "-t", str(threads), "-c", str(ctx), "-b", str(ctx), "-ub", str(ctx)]
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

    def start(self, file_id, nominal_ctx, threads):
        self.stop()
        self.file = file_id
        self.nominal_ctx = nominal_ctx
        self.eff_ctx = None
        self.error = ""
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
            self.proc = subprocess.Popen(self.build_args(threads),
                                         stdin=subprocess.DEVNULL, creationflags=flags)
        except Exception as exc:
            self.state = "error"
            self.error = str(exc)
            return False
        self.state = "starting"
        return True

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
                "nominal_ctx": self.nominal_ctx, "effective_ctx": self.eff_ctx}

EMBED_SERVER = LocalServer("embedding", EMBEDDING_PORT)
RERANK_SERVER = LocalServer("reranker", RERANK_PORT)

def _server_for(kind):
    return EMBED_SERVER if kind == "embedding" else RERANK_SERVER

def _start_one(kind, cfg):
    file_id = cfg.get("embedding_model" if kind == "embedding" else "rerank_model") or \
              (DEFAULT_EMBED_FILE if kind == "embedding" else DEFAULT_RERANK_FILE)
    entry = catalog_entry(file_id) or {"id": file_id, "ctx": 2048}
    srv = _server_for(kind)
    if srv.start(entry["id"], entry.get("ctx", 2048), HARDWARE["cores"]):
        srv.wait_ready(120)

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
    if not config["source_folder"] or not Path(config["source_folder"]).is_dir():
        config["source_folder"] = DEFAULTS["source_folder"]
    if config.get("chat_model") == "auto":
        config["chat_model"] = ""
    if config.get("analysis_model") == "auto":
        config["analysis_model"] = ""
    return config


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


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
    keys = ("chunk_size", "chunk_overlap", "max_row_chars", "include_extensions")
    base = "|".join(str(config.get(k)) for k in keys)
    return base + "|emb=" + normalize_embed_id(config.get("embedding_model"))


def public(item):
    return {k: v for k, v in item.items() if k != "embedding"}


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


def chunks_for(path, config):
    ext = path.suffix.lower()
    source = path.name
    if ext in (".txt", ".md", ".csv"):
        units = [("Text", None, path.read_text(encoding="utf-8", errors="replace"))]
    elif ext == ".docx":
        units = [("Document", None, read_docx(path))]
    elif ext in (".xlsx", ".xlsm"):
        units = list(read_xlsx(path))
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
                "section": section, "row": row, "text": content[:limit],
            })
            continue
        size = positive(config.get("chunk_size"), 900, 200, 4000)
        overlap = min(positive(config.get("chunk_overlap"), 140, 0, size - 1), size // 2)
        sentences = re.split(r"\n{2,}|(?<=[.!?])\s+", content)
        pieces, current = [], ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > size:
                pieces.append(current)
                current = (current[-overlap:] + " " if overlap else "") + sentence
            elif len(sentence) > size:
                if current:
                    pieces.append(current)
                    current = ""
                for start in range(0, len(sentence), size - overlap or size):
                    piece = sentence[start:start + size]
                    if piece:
                        pieces.append(piece)
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)
        for position, piece in enumerate(pieces):
            out.append({
                "id": str(uuid.uuid4()), "source": source, "path": str(path),
                "section": section, "row": None, "chunk": position, "text": piece,
            })
    return out


# --------------------------------------------------------------------------- #
#  LM Studio / local model transport
# --------------------------------------------------------------------------- #
def openai_base(config):
    base = str(config.get("lmstudio_url", "")).strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def api(config, endpoint, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(openai_base(config) + endpoint, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def lmstudio_api(config, endpoint, body=None, timeout=5):
    base = openai_base(config)[:-3].rstrip("/")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + endpoint, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def local_api(url, endpoint, body):
    req = urllib.request.Request(url + endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
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


def embed(config, inputs):
    def one(item):
        cached = _EMBED_CACHE.get(item)
        if cached is not None:
            return cached
        data = local_api(EMBEDDING_URL, "/embedding", {"content": item, "truncate": True})
        entry = data[0] if isinstance(data, list) else data["value"][0]
        value = entry["embedding"] if isinstance(entry, dict) else entry
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        vector = [float(x) for x in value.split()] if isinstance(value, str) else value
        magnitude = math.sqrt(sum(x * x for x in vector)) or 1.0
        vector = [x / magnitude for x in vector]
        if len(_EMBED_CACHE) > 4096:
            _EMBED_CACHE.clear()
        _EMBED_CACHE[item] = vector
        return vector

    if len(inputs) < 2:
        return [one(item) for item in inputs]
    with ThreadPoolExecutor(max_workers=1) as pool:
        return list(pool.map(one, inputs))


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
def retrieve(query, config, excluded=None, rerank=True):
    index = load(INDEX_FILE, {"chunks": []})
    corpus = index["chunks"]
    if not corpus:
        return [], "Index is empty. Ingest documents first."
    indexed_model = index.get("embedding_model")
    if not indexed_model or index.get("version") != 2:
        return [], "This index was created by an older version. Rebuild the index before searching."
    if normalize_embed_id(indexed_model) != normalize_embed_id(resolved(config, "embedding")):
        return [], "This index was built with a different embedding model. Re-select that model or run a full rebuild."
    vector = embed(config, [query])[0]
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
    semantic_rank = sorted(range(len(corpus)), key=lambda i: cosine(vector, corpus[i].get("embedding", [])), reverse=True)
    lexical_rank = sorted(range(len(corpus)), key=lambda i: lexical[i], reverse=True)
    fused = Counter()
    for rank, idx in enumerate(semantic_rank, 1):
        fused[idx] += float(config.get("semantic_weight", 0.72)) / (60 + rank)
    for rank, idx in enumerate(lexical_rank, 1):
        fused[idx] += float(config.get("keyword_weight", 0.28)) / (60 + rank)
    candidates = positive(config.get("candidate_count"), 32, 4, 400)
    excluded = excluded or set()
    result = [
        dict(corpus[idx], _score=fused[idx], _semantic_score=cosine(vector, corpus[idx].get("embedding", [])))
        for idx in sorted(fused, key=fused.get, reverse=True)
        if corpus[idx].get("id") not in excluded
    ][:candidates]
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
        json.dumps(body).encode(),
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
        return summary, tag, (raw_text or "").strip()
    except Exception as exc:
        streamed = (raw_text or "").strip()
        fallback = streamed or item.get("text", "")[:600]
        if streamed:
            summary, tag = _parse_analysis(fallback)
        else:
            summary = f"(analysis call failed: {exc}) passage begins: {item.get('text', '')[:300]}"
            tag = "RELATED"
        return summary, tag, streamed


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
        raw = json.dumps(data, ensure_ascii=False).encode()
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
            self.wfile.write(b": stream open\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._gone = True

    def send_stream(self, data):
        if getattr(self, "_gone", False):
            return
        try:
            payload = json.dumps(data, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._gone = True

    # ---- live helpers ----
    def _log(self, message, level="info"):
        self.send_stream({"type": "log", "text": message, "level": level, "t": time.strftime("%H:%M:%S")})

    def _stage(self, stage_id, status):
        now = time.time()
        payload = {"type": "stage", "id": stage_id, "status": status}
        if status == "active":
            self._stage_t[stage_id] = now
        elif status == "done" and stage_id in self._stage_t:
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
        a_model = analysis_model_for(c)        # analysis model (falls back to answer model)
        have_analysis = bool(a_model)

        self.send_stream({"type": "run_start", "evidence_only": evidence_only,
                          "want_answer": want_answer, "adaptive": adaptive,
                          "analysis_model": a_model, "answer_model": answer_model})
        try:
            checkpoint = positive(c.get("rerank_count"), 4, 1, 30)
            max_checks = positive(c.get("max_candidate_checks"), 24, 4, 200)
            base_candidates = positive(c.get("candidate_count"), 32, 4, 400)

            self._log(f"Models — analysis: {a_model or '(none → raw passages)'} · "
                      f"answer: {answer_model or '(none → evidence only)'}")

            # ---- 01 understand (analysis model) ----
            self._stage("understand", "active")
            if not adaptive or not have_analysis:
                plan = {"rewrite": query, "variants": [query], "retrieve": True, "is_greeting": False}
                self._log("Adaptive planning off — using the raw question for retrieval.")
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
                        "section": item.get("section") or f"Passage {(item.get('chunk') or 0) + 1}",
                        "row": item.get("row"),
                        "semantic": round(float(item.get("_semantic_score", 0)), 3),
                        "score": round(float(item.get("_score", 0)), 4),
                    }
                    self.send_stream({"type": "verify_start", **meta})
                    t_check = time.time()
                    if not have_analysis:
                        summary, tag, raw_text = item["text"][:1600], "RELATED", item["text"][:1600]
                    else:
                        self._log(f"[{checks_done}] analysing {meta['source']} ({meta['section']})...")
                        n = checks_done
                        summary, tag, raw_text = analyze_passage(
                            c, query, item,
                            lambda d, num=n: self.send_stream({"type": "verify_delta", "number": num, "text": d}),
                            model=a_model)
                    entry = {**meta, "summary": summary, "raw": raw_text, "relevance": tag,
                             "text": summary, "path": item.get("path", "")}
                    analyzed.append(entry)
                    verdict = f"ANALYZED · [{tag}] {REL_LABEL.get(tag, '')}"
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
                        "elapsed": round(time.time() - t_check, 2),
                        "accepted_count": len(analyzed),
                        "target": checkpoint,
                    })
                    # ---- 04 assess at each checkpoint (analysis model) ----
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

            # final assess only matters when we will actually compose an answer
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
        allowed = {x.strip().lower() for x in c["include_extensions"].split(",") if x.strip()}
        mode = str(body.get("mode", "incremental")).lower()
        self._gone = False
        self.send_stream_start()
        try:
            existing = load(INDEX_FILE, {"chunks": [], "files": {}, "chunk_params": None})
            params = chunk_params_sig(c)
            if mode != "full" and existing.get("chunk_params") != params:
                mode = "full"
                self._log("Chunking parameters changed — forcing a full rebuild.", "warn")
            self._log(f"Ingest started ({mode}) — scanning {folder} ...")
            self.send_stream({
                "type": "progress", "phase": "ingest", "percent": 2,
                "text": ("Full rebuild requested. Scanning document folder..." if mode == "full"
                         else "Scanning for new or changed documents (saved vectors are reused)..."),
            })
            files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed and DATA not in p.parents]
            current_sigs = {str(p): file_signature(p) for p in files}
            reused, changed = [], files
            if mode != "full":
                old_sigs = existing.get("files", {})
                reused = [
                    ch for ch in existing.get("chunks", [])
                    if isinstance(ch, dict) and ch.get("embedding") and ch.get("path") in current_sigs
                    and old_sigs.get(ch.get("path")) == current_sigs[ch.get("path")]
                ]
                reused_paths = {ch.get("path") for ch in reused}
                changed = [p for p in files if str(p) not in reused_paths]
                dropped = len([s for s in old_sigs if s not in current_sigs])
                self.send_stream({
                    "type": "progress", "phase": "ingest", "percent": 8,
                    "text": (f"Reusing {len(reused)} saved passage(s) from {len(files) - len(changed)} unchanged document(s). "
                             f"Updating {len(changed)} changed/new document(s)"
                             + (f" and dropping {dropped} removed file(s)." if dropped else ".")),
                })
                self._log(f"{len(files)} file(s) found — {len(reused)} passage(s) reusable, {len(changed)} to process.")
            else:
                self.send_stream({
                    "type": "progress", "phase": "ingest", "percent": 8,
                    "text": f"Found {len(files)} supported document(s). Creating passages...",
                })
                self._log(f"{len(files)} supported document(s) found.")
            new_chunks, errors = [], []
            for number, p in enumerate(changed, 1):
                if self._gone:
                    return
                try:
                    added = chunks_for(p, c)
                    new_chunks.extend(added)
                    self._log(f"Parsed {p.name} — {len(added)} passage(s).")
                except Exception as e:
                    errors.append(f"{p.name}: {e}")
                    self._log(f"Failed to parse {p.name} — {e}", "error")
                self.send_stream({
                    "type": "progress", "phase": "ingest",
                    "percent": 8 + round(20 * number / max(len(changed), 1)),
                    "text": f"Processed {number} of {len(changed)} document(s): {p.name}",
                })
            if not new_chunks and not reused:
                self._log("No supported text found in the selected folder.", "error")
                self.send_stream({"type": "error", "error": "No supported text was found in the selected folder."})
                return
            embedding_model = resolved(c, "embedding")
            if new_chunks:
                self.send_stream({
                    "type": "progress", "phase": "ingest", "percent": 30,
                    "text": f"Creating semantic vectors for {len(new_chunks)} new/changed passage(s) using {HARDWARE['label']}...",
                })
                self._log(f"Embedding {len(new_chunks)} passage(s) with {embedding_model}...")
                batch_size = max(16, HARDWARE["embedding_workers"] * 16)
                for start in range(0, len(new_chunks), batch_size):
                    if self._gone:
                        return
                    batch = new_chunks[start:start + batch_size]
                    for item, vector in zip(batch, embed(c, [x["text"] for x in batch])):
                        item["embedding"] = vector
                    done_count = min(start + len(batch), len(new_chunks))
                    self._log(f"Vectorized {done_count}/{len(new_chunks)} passage(s).")
                    self.send_stream({
                        "type": "progress", "phase": "ingest",
                        "percent": 30 + round(50 * done_count / len(new_chunks)),
                        "text": f"Vectorized {done_count} of {len(new_chunks)} passage(s)",
                    })
            else:
                self._log("No new passages to embed — reusing all saved vectors.", "ok")
                self.send_stream({"type": "progress", "phase": "ingest", "percent": 80,
                                  "text": "No new passages to embed. Reusing all saved vectors..."})
            docs = [d for d in (reused + new_chunks) if d.get("path") in current_sigs]
            if not docs:
                self.send_stream({"type": "error", "error": "No supported text was found in the selected folder."})
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
            self._log(f"Index saved — {len(docs)} passages, {len({x['path'] for x in docs})} documents.", "ok")
            self.send_stream({
                "type": "done", "documents": len({x["path"] for x in docs}), "chunks": len(docs),
                "reused": len(reused), "new": len(new_chunks), "mode": mode, "errors": errors,
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
            return self.send_json({"chunks": len(idx["chunks"]), "updated": idx.get("updated"), "hardware": HARDWARE["label"]})
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
        if self.path == "/":
            self.path = "/index.html"
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

            if self.path == "/api/download-model":
                self._download_model_stream((body.get("id") or "").strip())
                return

            if self.path == "/api/settings":
                current = settings()
                allowed = set(DEFAULTS)
                updated = {key: value for key, value in body.items() if key in allowed}
                if updated.get("source_folder") and not Path(updated["source_folder"]).is_dir():
                    return self.send_json({"error": "Enter an existing local document folder path."}, 400)
                for key in ("use_llm_rerank", "adaptive_rag", "use_hyde"):
                    if key in updated:
                        updated[key] = str(updated[key]).lower() == "true"
                save_json(SETTINGS_FILE, {**current, **updated})
                return self.send_json(settings())
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
                allowed = {x.strip().lower() for x in c["include_extensions"].split(",") if x.strip()}
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
                return self.send_json({"documents": len({x["path"] for x in docs}), "chunks": len(docs),
                                       "errors": errors, "performance": HARDWARE["label"]})
            if self.path == "/api/ingest-stream":
                self._ingest_stream(settings(), body)
                return
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
                evidence_only = not c["chat_model"]
                want_answer = bool(body.get("answer", True)) and not evidence_only
                self.send_stream_start()
                self._answer_stream(c, query, want_answer, evidence_only, bool(body.get("adaptive", True)))
                return
            if self.path == "/api/index/clear":
                for candidate in (INDEX_FILE, INDEX_FILE.with_suffix(".tmp")):
                    try:
                        candidate.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
                return self.send_json({"cleared": True})
            self.send_json({"error": "Not found"}, 404)
        except Exception as e:
            try:
                self.send_json({"error": str(e)}, 500)
            except Exception:
                pass


if __name__ == "__main__":
    os.chdir(ROOT)
    _start_local_servers(settings(), wait=False)
    atexit.register(_shutdown_local_servers)
    print("Offline RAG: http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()