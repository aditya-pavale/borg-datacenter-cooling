"""Raw-data immutability guard.

Every downstream script that reads data/raw/ should call
`assert_raw_data_unchanged()` (or run this file directly) before doing
anything else. It compares the current size + MD5 of each raw file
against the baseline recorded in docs/raw_data_manifest.json and raises
if anything differs. It never writes to data/raw/.

Baseline note: the manifest was created during Phase 1 (first
inspection of the dataset), not at download time, because no hash was
captured before that point. This script can therefore only prove "no
change since we started working with it," not "no change since
download" -- that distinction is documented in docs/dataset_audit.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "docs" / "raw_data_manifest.json"


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_baseline_manifest() -> dict:
    """Create the manifest if it does not already exist. Never overwrites."""
    if MANIFEST_PATH.exists():
        raise RuntimeError(
            f"{MANIFEST_PATH} already exists. Refusing to overwrite an "
            "existing integrity baseline. Delete it manually first if you "
            "are certain a new baseline is intended."
        )
    files = {}
    for p in sorted(RAW_DIR.glob("*")):
        if p.is_file():
            files[p.name] = {"size_bytes": p.stat().st_size, "md5": _md5(p)}
    manifest = {"raw_dir": str(RAW_DIR.relative_to(REPO_ROOT)), "files": files}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def assert_raw_data_unchanged() -> None:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"No integrity baseline at {MANIFEST_PATH}. Run "
            "`python src/data/verify_raw_integrity.py --init` once to create it."
        )
    manifest = json.loads(MANIFEST_PATH.read_text())
    problems = []
    for fname, info in manifest["files"].items():
        p = RAW_DIR / fname
        if not p.exists():
            problems.append(f"MISSING: {fname}")
            continue
        actual_size = p.stat().st_size
        if actual_size != info["size_bytes"]:
            problems.append(
                f"SIZE MISMATCH: {fname} expected {info['size_bytes']} got {actual_size}"
            )
            continue
        actual_md5 = _md5(p)
        if actual_md5 != info["md5"]:
            problems.append(
                f"MD5 MISMATCH: {fname} expected {info['md5']} got {actual_md5}"
            )
    if problems:
        raise RuntimeError(
            "Raw data integrity check FAILED:\n" + "\n".join(problems)
        )


if __name__ == "__main__":
    import sys

    if "--init" in sys.argv:
        m = write_baseline_manifest()
        print(f"Wrote baseline manifest to {MANIFEST_PATH}")
        print(json.dumps(m, indent=2))
    else:
        assert_raw_data_unchanged()
        print("Raw data integrity OK: matches", MANIFEST_PATH)
