from pathlib import Path
from typing import Union
import commentjson as json

def load_config(path: Union[str, Path]) -> dict:
    """Load a JSON or JSON-with-comments config file."""
    with open(path, "r") as f:
        return json.load(f)