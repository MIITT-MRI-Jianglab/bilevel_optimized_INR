# inr/config.py
from pathlib import Path
import commentjson as json

def load_config(path: str | Path) -> dict:
    """Read a JSON or JSON-with-comments file and return the dict."""
    with open(path, "r") as f:
        return json.load(f)