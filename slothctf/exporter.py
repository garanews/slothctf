from __future__ import annotations

import json
from pathlib import Path


def export_challenge(challenge_dir: Path) -> Path:
    challenge_dir = Path(challenge_dir)
    manifest = {
        "generated_at": "local",
        "files": [
            "README.md",
            "challenge.json",
            "solution.md",
            "src/challenge.c",
            "organizer/encoder.c",
            "bin/challenge",
        ],
    }

    manifest_path = challenge_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
