"""Read this Streamlit application's local .env without third-party packages."""
from pathlib import Path
import os
import re


def load_local_env():
    env_file = Path(__file__).resolve().parent / '.env'
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$', line)
        if not match:
            continue
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        else:
            value = re.sub(r'\s+#.*$', '', value).strip()
        os.environ.setdefault(name, value)
