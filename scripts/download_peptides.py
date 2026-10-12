"""Download the Peptides-Holdout CSV from Zenodo record 8410136.

The source ZIP (toolboxsf_benchmarks.zip) is ~500 MB and we only need one small CSV,
so this uses HTTP range requests via ``remotezip`` to extract a single member
instead of downloading the whole archive.
"""

from __future__ import annotations

from pathlib import Path

from remotezip import RemoteZip

from cap6545.utils.paths import RAW_DIR

ZENODO_URL = "https://zenodo.org/records/8410136/files/toolboxsf_benchmarks.zip"
MEMBER_NAME = "toolboxsf_benchmarks/csv_files/peptides_holdout_test.csv"
TARGET_NAME = "peptides_holdout_test.csv"


def main() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / TARGET_NAME
    if target.exists():
        print(f"already present: {target}")
        return target

    print(f"extracting {MEMBER_NAME} from {ZENODO_URL} (range requests only)")
    with RemoteZip(ZENODO_URL) as zf, zf.open(MEMBER_NAME) as src, target.open("wb") as dst:
        dst.write(src.read())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return target


if __name__ == "__main__":
    main()
