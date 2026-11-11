"""Download MIND-small (train + dev) into ./data.

Microsoft's original blob storage (mind201910small.blob.core.windows.net) no longer allows public access,
so this pulls the identical TSVs from a Hugging Face mirror. Override with MIND_MIRROR=<base url>.
"""
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys.config import DATA_DIR, MIND_FILES, MIND_MIRROR  # noqa: E402


def main() -> None:
    base = os.environ.get("MIND_MIRROR", MIND_MIRROR).rstrip("/")
    for split, files in MIND_FILES.items():
        out_dir = DATA_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, remote in files.items():
            dest = out_dir / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"✓ {dest.relative_to(DATA_DIR.parent)} (cached)")
                continue
            print(f"↓ {base}/{remote}")
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(f"{base}/{remote}", tmp)
            tmp.rename(dest)
    print("done")


if __name__ == "__main__":
    main()
