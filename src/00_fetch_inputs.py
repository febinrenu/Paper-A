"""Fetch the public inputs released by Spracklen et al. (USENIX Security 2025, MIT licence).

- Prompt sets and early-2024 registry snapshots from GitHub (Spracks/PackageHallucination)
- The full npm snapshot CSV (56 MB) extracted from the Zenodo artifact zip via HTTP range
  requests, so the 9.9 GB archive is never downloaded in full.
Records SHA-256 of every file in data/MANIFEST.json.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

from common import SPRACKLEN, client, get_with_retry, sha256, update_manifest

GH = "https://raw.githubusercontent.com/Spracks/PackageHallucination/main/"
GH_FILES = [
    *(f"Data/{lang}/{s}.json" for lang in ("Python", "Javascript") for s in ("SO_AT", "SO_LY", "LLM_AT", "LLM_LY")),
    "Data/Python/pypi_package_names.csv",
    "Data/Python/false_positive_packages.csv",
    "Data/Javascript/false_positive_packages.csv",
    "Data/Javascript/core_modules.csv",
    "Data/Javascript/options.csv",
    "custom_parse_python.py",
    "custom_parse_javascript.py",
    "package_detection.py",
    "generate_code.py",
    "generate_package_names.py",
    "aggregate_results.py",
    "LICENSE",
]
ZENODO = "https://zenodo.org/records/14676377/files/Artifacts_Submission.zip?download=1"
ZIP_MEMBERS = {"Artifacts_Submission/Data/Javascript/npm_package_names.csv": "Data/Javascript/npm_package_names.csv"}


class RangeReader(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        d = urllib.request.urlopen(req).read()
        b[: len(d)] = d
        self.pos += len(d)
        return len(d)


def main() -> None:
    hashes = {}
    with client() as c:
        for rel in GH_FILES:
            dst = SPRACKLEN / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                r = get_with_retry(c, GH + rel)
                if r.status_code == 404:
                    print("missing upstream:", rel)
                    continue
                r.raise_for_status()
                dst.write_bytes(r.content)
            hashes[f"spracklen/{rel}"] = sha256(dst)
            print("ok", rel)

    z = zipfile.ZipFile(io.BufferedReader(RangeReader(ZENODO), buffer_size=1 << 22))
    for member, rel in ZIP_MEMBERS.items():
        dst = SPRACKLEN / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(member))
        hashes[f"spracklen/{rel}"] = sha256(dst)
        print("ok (zenodo)", rel)

    update_manifest(hashes)


if __name__ == "__main__":
    main()
