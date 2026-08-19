#!/usr/bin/env python3
"""Add a "language" field to every record in manifest.json (derived from the
sprog= query parameter) and fix the one record whose "level" is null.

Usage:
    python3 update_manifest.py [MANIFEST]              # dry run
    python3 update_manifest.py [MANIFEST] --apply       # write changes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, TypeAdapter

Level = Literal["bachelor", "kandidat", "tilvalg", "master"]
Language = Literal["da", "en"]

LEVEL_FIX_DOK_ORDNING_ID = "13731"
LEVEL_FIX_EXPECTED_TITLE = "Studieordning for Uddannelsen i konferencetolkning (2018)"
LEVEL_FIX_VALUE: Level = "master"


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    filename: str
    dokOrdningId: str
    url: str
    title: str
    level: Optional[Level]
    year: Optional[int]
    programme: str
    language: Optional[Language] = None


RECORD_LIST_ADAPTER: TypeAdapter[list[Record]] = TypeAdapter(list[Record])


def derive_language(url: str) -> Optional[Language]:
    query = urlsplit(url).query
    values = parse_qs(query).get("sprog")
    if not values:
        return None
    value = values[0]
    if value == "da":
        return "da"
    if value == "en":
        return "en"
    return None


def load_records(path: Path) -> list[Record]:
    try:
        return RECORD_LIST_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"error: {path} is malformed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def apply_updates(
    records: list[Record],
) -> tuple[list[Record], Counter[Optional[str]], list[Record], Optional[Record]]:
    """Returns (updated_records, language_counts, unrecognized_sprog_records,
    the record whose level was fixed, or None if not found)."""
    language_counts: Counter[Optional[str]] = Counter()
    unrecognized: list[Record] = []
    level_fixed: Optional[Record] = None

    for record in records:
        language = derive_language(record.url)
        record.language = language
        language_counts[language] += 1
        if language is None:
            unrecognized.append(record)

        if record.dokOrdningId == LEVEL_FIX_DOK_ORDNING_ID:
            if record.title != LEVEL_FIX_EXPECTED_TITLE:
                print(
                    f"warning: dokOrdningId {LEVEL_FIX_DOK_ORDNING_ID} has "
                    f"unexpected title {record.title!r}; setting level anyway",
                    file=sys.stderr,
                )
            record.level = LEVEL_FIX_VALUE
            level_fixed = record

    return records, language_counts, unrecognized, level_fixed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, nargs="?", default=Path("manifest.json")
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the updated manifest (default: dry run)",
    )
    args = parser.parse_args()

    manifest_path: Path = args.manifest
    if not manifest_path.is_file():
        print(f"error: {manifest_path} not found", file=sys.stderr)
        return 1

    records = load_records(manifest_path)
    expected_count = len(records)

    records, language_counts, unrecognized, level_fixed = apply_updates(records)

    print(f"{expected_count} record(s) loaded.")
    print("Language counts:")
    for language, count in sorted(
        language_counts.items(), key=lambda kv: (kv[0] is None, kv[0])
    ):
        print(f"  {language!r}: {count}")

    if unrecognized:
        print(f"\n{len(unrecognized)} record(s) with missing/unrecognized sprog=:")
        for record in unrecognized:
            print(f"  dokOrdningId={record.dokOrdningId!r} url={record.url!r}")

    if level_fixed is not None:
        print(
            f"\nSet level={LEVEL_FIX_VALUE!r} for dokOrdningId="
            f"{level_fixed.dokOrdningId!r} ({level_fixed.title!r})"
        )
    else:
        print(
            f"\nwarning: no record found with dokOrdningId="
            f"{LEVEL_FIX_DOK_ORDNING_ID!r}; level not fixed",
            file=sys.stderr,
        )

    if not args.apply:
        print("\nDry run only; pass --apply to write changes.")
        return 0

    output = [r.model_dump(mode="json") for r in records]
    serialized = json.dumps(output, ensure_ascii=False, indent=2)

    round_tripped = RECORD_LIST_ADAPTER.validate_json(serialized)
    if len(round_tripped) != expected_count:
        print(
            f"error: round-trip check failed, expected {expected_count} records, "
            f"got {len(round_tripped)}",
            file=sys.stderr,
        )
        return 1
    unique_ids = {r.dokOrdningId for r in round_tripped}
    if len(unique_ids) != expected_count:
        print(
            f"error: round-trip check failed, expected {expected_count} unique "
            f"dokOrdningId values, got {len(unique_ids)}",
            file=sys.stderr,
        )
        return 1

    fd, tmp_name = tempfile.mkstemp(
        dir=manifest_path.parent, prefix=manifest_path.name, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.replace(tmp_name, manifest_path)
    except BaseException:
        os.unlink(tmp_name)
        raise

    print(f"\nWrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
