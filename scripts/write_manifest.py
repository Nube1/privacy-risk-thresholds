"""Write a SHA-256 manifest for a set of evidence files.

    python scripts/write_manifest.py OUTPUT FILE [FILE ...]

The manifest detects accidental change. It is not tamper evidence against someone
who can rewrite the manifest too; for that its hash must be published or time-stamped
outside the operator's control.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    output, files = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
