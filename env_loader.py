# -*- coding: utf-8 -*-
"""
Load .env file and resolve config paths from environment variables.

Priority: ENV vars > .env file > config.json > defaults
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_dotenv():
    """Parse .env file (key=value) and set as env vars (won't overwrite existing)."""
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            # Only set if not already defined (real env vars take priority)
            os.environ.setdefault(key, value)


def resolve_config(user_cfg: dict) -> dict:
    """Merge env vars into config, filling in path-related fields."""
    # Data directory
    data_dir = os.environ.get("AKARI_MEM_DATA_DIR", user_cfg.get("data_dir", ""))
    if not data_dir:
        data_dir = os.path.join(_PROJECT_ROOT, "data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_PROJECT_ROOT, data_dir)
    user_cfg["data_dir"] = data_dir

    # Model cache directory (embedding / rerank)
    cache_dir = os.environ.get("HF_HOME", "")
    if cache_dir:
        emb = user_cfg.get("embedding", {})
        if "cache_dir" not in emb:
            emb["cache_dir"] = cache_dir
            user_cfg["embedding"] = emb
        rnk = user_cfg.get("rerank", {})
        if "cache_dir" not in rnk:
            rnk["cache_dir"] = cache_dir
            user_cfg["rerank"] = rnk

    return user_cfg


def setup():
    """One-call init: load .env, setup extra lib path."""
    import sys
    load_dotenv()

    # Extra Python libs path
    extra_lib = os.environ.get("AKARI_MEM_LIBS", "")
    if extra_lib and os.path.isdir(extra_lib) and extra_lib not in sys.path:
        sys.path.append(extra_lib)

    # Project root on path
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
