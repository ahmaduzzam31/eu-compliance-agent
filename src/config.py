"""All settings in one place.

Values come from .env (copy .env.example to .env). Changing LLM_PROVIDER and LLM_MODEL
swaps the chat model, for example to OpenAI, Anthropic or Google, with no code changes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Models
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))  # Ollama context window in tokens
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Splitting
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# Paths
RAW_DIR = ROOT / "data" / "raw"
SOURCES_CSV = ROOT / "data" / "sources.csv"
INDEX_DIR = ROOT / "data" / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
CHROMA_DIR = ROOT / "chroma_db"
CHROMA_COLLECTION = "eu_compliance"
RESULTS_DIR = ROOT / "results"

# Short names used in citations and for routing
DOC_LABELS = {
    "ai_act": "EU AI Act",
    "gdpr": "GDPR",
    "cbi_outsourcing": "CBI Outsourcing Guidance",
    "cbi_op_resilience": "CBI Operational Resilience Guidance",
}
