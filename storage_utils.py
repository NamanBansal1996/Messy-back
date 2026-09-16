import json
import os
import threading

_locks = {}
_locks_guard = threading.Lock()


def _lock_for(path):
    """One lock per file path, so concurrent requests touching different
    JSON stores don't block each other, but concurrent writes to the same
    store are serialized."""
    with _locks_guard:
        if path not in _locks:
            _locks[path] = threading.Lock()
        return _locks[path]


def read_json(path, default):
    lock = _lock_for(path)
    with lock:
        if not os.path.exists(path):
            return default
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default


def write_json_atomic(path, data):
    """Write via a temp file + os.replace so a crash or a concurrent read
    never observes a partially-written file."""
    lock = _lock_for(path)
    with lock:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, path)
