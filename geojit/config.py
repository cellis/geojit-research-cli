import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    data_dir: Path
    qdrant_collection: str
    qdrant_url: str | None
    qdrant_api_key: str | None
    database_url: str | None
    openai_api_key: str | None
    openai_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def _load_env_key(path: Path) -> str | None:
    # Supports either KEY=VALUE lines or a single line with the key
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text:
        return None
    if "=" not in text:
        return text
    for line in text.splitlines():
        if not line or line.strip().startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            if key.strip().upper() in {"OPENAI_API_KEY", "API_KEY", "OPENAI"}:
                return val.strip()
    return None


def load_settings() -> Settings:
    cwd = Path.cwd()
    data_dir = Path(os.getenv("GEOJIT_DATA_DIR", cwd / "Financial_Research_Agent_Files")).expanduser()

    # OpenAI key can come from env or .env file with a raw key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        env_key = _load_env_key(cwd / ".env")
        openai_api_key = env_key or None

    return Settings(
        data_dir=data_dir,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "geojit"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        database_url=os.getenv("DATABASE_URL", "postgresql://localhost/geojit"),
        openai_api_key=openai_api_key,
        openai_model=os.getenv("GEOJIT_MODEL", "gpt-5"),
        embedding_model=os.getenv("GEOJIT_EMBED_MODEL", "text-embedding-3-large"),
        chunk_size=int(os.getenv("GEOJIT_CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("GEOJIT_CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("GEOJIT_TOP_K", "6")),
    )

