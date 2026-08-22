from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelCheck:
    name: str
    role: str
    path: Path
    exists: bool
    size_ok: bool
    sha256_ok: bool
    actual_bytes: int | None
    actual_sha256: str | None

    @property
    def ok(self) -> bool:
        return self.exists and self.size_ok and self.sha256_ok


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest().upper()


def load_model_manifest(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    path = root / "configs" / "model_manifest.json"

    if not path.exists():
        raise FileNotFoundError(f"Model manifest missing: {path}")

    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_models(project_root: str | Path) -> list[ModelCheck]:
    root = Path(project_root).resolve()
    manifest = load_model_manifest(root)

    checks: list[ModelCheck] = []

    for model in manifest.get("models", []):
        path = root / "models" / model["name"]
        exists = path.exists()

        actual_bytes = path.stat().st_size if exists else None
        size_ok = (
            exists
            and actual_bytes == int(model["bytes"])
        )

        actual_sha256 = sha256_file(path) if exists else None
        sha256_ok = (
            exists
            and actual_sha256.upper() == str(model["sha256"]).upper()
        )

        checks.append(
            ModelCheck(
                name=str(model["name"]),
                role=str(model.get("role", "")),
                path=path,
                exists=exists,
                size_ok=size_ok,
                sha256_ok=sha256_ok,
                actual_bytes=actual_bytes,
                actual_sha256=actual_sha256,
            )
        )

    return checks
