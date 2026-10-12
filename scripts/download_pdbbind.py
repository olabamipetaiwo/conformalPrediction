"""Download and extract the PDBbind v2020 structure archives for LP-PDBBind.

LP-PDBBind ships only meta-information; the prepared structure files (protein
``.pdb`` + ligand ``.sdf``/``.mol2``) come from the PDBbind website, which is
behind a login. Rather than reverse-engineer that site's auth, this script
authenticates with a session cookie you copy from your logged-in browser.

Setup (one time):
  1. Log into https://www.pdbbind-plus.org.cn in your browser.
  2. Copy the request Cookie header for a download (DevTools > Network > a
     download request > Request Headers > Cookie) into ``.env`` as
     ``PDBBIND_COOKIE=...``.
  3. Put the archive download URLs in ``.env`` as ``PDBBIND_URLS=`` (comma- or
     whitespace-separated), or accept the documented v2020 general-set defaults
     below and confirm they resolve.

Run:
  python scripts/download_pdbbind.py            # download + extract + verify
  python scripts/download_pdbbind.py --verify-only

The v2020 general set (refined + other_PL, ~19,443 complexes) covers every
LP-PDBBind train/val/test id. Each complex extracts to
``data/raw/pdbbind/<pdbid>/<pdbid>_protein.pdb`` and ``<pdbid>_ligand.sdf``.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from cap6545.data.splits import load_lp_pdbbind_splits
from cap6545.data.structures import index_structures
from cap6545.utils.paths import RAW_DIR

load_dotenv()

PDBBIND_DIR = RAW_DIR / "pdbbind"

# Documented historical URLs for the v2020 general set. Verify these resolve for
# your account, or override with PDBBIND_URLS in .env.
DEFAULT_URLS = (
    "http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz",
    "http://www.pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz",
)

_GZIP_MAGIC = b"\x1f\x8b"


def _configured_urls() -> list[str]:
    raw = os.environ.get("PDBBIND_URLS", "").replace(",", " ").split()
    return raw or list(DEFAULT_URLS)


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == _GZIP_MAGIC


def download(url: str, dest_dir: Path, cookie: str) -> Path:
    """Stream one archive to ``dest_dir``; fail loudly if auth returned non-gzip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / url.rsplit("/", 1)[-1]
    if target.exists() and _is_gzip(target):
        print(f"already present: {target.name}")
        return target

    print(f"downloading {url}")
    headers = {"Cookie": cookie, "User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    if not _is_gzip(target):
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"{url} did not return a gzip archive (likely an auth/login page). "
            "Check PDBBIND_COOKIE is current and the URL is correct."
        )
    print(f"  wrote {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def extract(archive: Path, dest_dir: Path) -> None:
    print(f"extracting {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")


def verify_coverage(root: Path) -> tuple[int, list[str]]:
    """Return (#ids checked, missing ids) across train/val/test."""
    splits, _ = load_lp_pdbbind_splits()
    ids = set()
    for name in ("train", "val", "test"):
        ids |= set(splits[name].index)
    index = index_structures(root)
    missing = sorted(pid for pid in ids if pid.lower() not in index)
    return len(ids), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        cookie = os.environ.get("PDBBIND_COOKIE")
        if not cookie:
            raise SystemExit(
                "PDBBIND_COOKIE not set. Copy it into .env (see this file's docstring)."
            )
        for url in _configured_urls():
            archive = download(url, PDBBIND_DIR, cookie)
            extract(archive, PDBBIND_DIR)

    n, missing = verify_coverage(PDBBIND_DIR)
    print(f"coverage: {n - len(missing)}/{n} train+val+test complexes present")
    if missing:
        preview = ", ".join(missing[:10])
        print(f"WARNING: {len(missing)} missing (e.g. {preview}).")
        raise SystemExit(1)
    print("all train/val/test structures present.")


if __name__ == "__main__":
    main()
