"""Read-only filesystem helpers for monitor sources."""

import hashlib
import json
import stat
from functools import lru_cache

from .history import json_safe


def identifier(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:24]


def stamp(path):
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            return None
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns
    except FileNotFoundError:
        return None


@lru_cache(maxsize=96)
def _document(path, signature):
    if signature is None:
        return {}
    try:
        data = json.loads(path.read_bytes())
        return json_safe(data) if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, RecursionError):
        return {}


def document(path):
    return _document(path, stamp(path))


def directories(root):
    try:
        return [p for p in root.iterdir() if not p.is_symlink() and p.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        return []
