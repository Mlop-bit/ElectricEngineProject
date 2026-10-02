"""Download the IDMT-ISA Electric Engine dataset from Zenodo.

Run with ``python scripts/download_dataset.py``. The archive is about 1.5 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path


DOWNLOAD_URL = (
    "https://zenodo.org/records/7551261/files/"
    "IDMT-ISA-ELECTRIC-ENGINE.zip?download=1"
)
EXPECTED_MD5 = "88e30729a12090b5df38198795518ea5"
CHUNK_SIZE = 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download(destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"File already exists: {destination} (use --force to replace it)")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.md5()
    try:
        request = urllib.request.Request(
            DOWNLOAD_URL, headers={"User-Agent": "ElectricEngineProject/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(
                        f"\rDownloaded {downloaded / (1024**3):.2f} / "
                        f"{total / (1024**3):.2f} GiB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )

        print()
        actual_md5 = digest.hexdigest()
        if actual_md5 != EXPECTED_MD5:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 check failed: expected {EXPECTED_MD5}, got {actual_md5}"
            )

        partial.replace(destination)
        print(f"Download complete and verified: {destination}")
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        if partial.exists():
            print(f"Incomplete file kept at {partial}", file=sys.stderr)
        raise SystemExit(1) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "IDMT-ISA-ELECTRIC-ENGINE.zip",
        help="where to save the ZIP (default: data/IDMT-ISA-ELECTRIC-ENGINE.zip)",
    )
    parser.add_argument(
        "--force", action="store_true", help="download again if the file exists"
    )
    args = parser.parse_args()
    download(args.output, force=args.force)


if __name__ == "__main__":
    main()
