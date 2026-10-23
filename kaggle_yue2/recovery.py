"""Small, dependency-free artifact and recovery primitives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

ARTIFACT_VERSION = 1


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_child(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Artifact path must be inside its run directory")
    return path


def save_phase(path, signature, tokens, truncated):
    payload = {
        "version": ARTIFACT_VERSION, "signature": signature,
        "tokens": list(tokens), "truncated": bool(truncated),
    }
    atomic_json(path, {**payload, "sha256": digest(payload)})


def load_phase(path, signature, phase):
    record = read_json(path)
    checksum = record.pop("sha256", None)
    if checksum != digest(record) or record.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"Corrupt or unsupported {phase} recovery artifact")
    if record.get("signature") != signature:
        raise ValueError(f"Saved {phase} inputs do not match this request/runtime")
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
        raise ValueError(f"Invalid {phase} recovery tokens")
    low, high = (0, 151643) if phase == "abc" else (151853, 184621)
    if any(not low <= t < high for t in tokens) or type(record.get("truncated")) is not bool:
        raise ValueError(f"Invalid {phase} recovery vocabulary or ending status")
    return tokens, record["truncated"]


def acoustic_ranges(frames, prefix_tokens, context=24576):
    size = (context - prefix_tokens - 3) // 2
    if frames < 1 or size < 1:
        raise ValueError("Prompt leaves no acoustic context")
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]
