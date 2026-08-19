#!/usr/bin/env python3
"""Extract metadata from Aarhus University EDDI studieordning HTML files,
write a manifest.json, and rename each file to <dokOrdningId>.html.

Usage:
    python3 rename_studieordninger.py FOLDER              # dry run
    python3 rename_studieordninger.py FOLDER --apply       # write manifest + rename
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional

from selectolax.parser import HTMLParser

Level = Literal["bachelor", "kandidat", "tilvalg", "master"]

# Chrome writes: <!-- saved from url=(NNNN)https://... -->
SAVED_FROM_RE = re.compile(
    r"saved from url=\(\d+\)(?P<url>\S+)", re.IGNORECASE
)
DOK_ORDNING_ID_RE = re.compile(r"[?&]dokOrdningId=(?P<id>\d+)")
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
DA_PREFIX = "Studieordning for "
EN_PREFIX = "Academic regulations for "


@dataclass
class StudieordningRecord:
    filename: str
    dokOrdningId: str
    url: str
    title: str
    level: Optional[Level]
    year: Optional[int]
    programme: str


@dataclass
class ExtractionFailure:
    filename: str
    reason: str


def find_saved_from_url(raw_html: str) -> Optional[str]:
    # Chrome writes this comment near the top of the file; scanning the
    # first few KB is enough and avoids running the regex over huge files.
    head = raw_html[:4096]
    match = SAVED_FROM_RE.search(head)
    if match is None:
        match = SAVED_FROM_RE.search(raw_html)
    if match is None:
        return None
    return match.group("url")


def extract_title(raw_html: str) -> Optional[str]:
    tree = HTMLParser(raw_html)
    h1 = tree.css_first("h1")
    if h1 is None:
        return None
    text = h1.text(strip=True)
    return text or None


def derive_level(title: str) -> Optional[Level]:
    lower = title.lower()
    if "tilvalg" in lower or "supplementary subject" in lower:
        return "tilvalg"
    if "kandidat" in lower or "master's degree programme" in lower:
        return "kandidat"
    if "master" in lower:
        return "master"
    if "bachelor" in lower:
        return "bachelor"
    return None


def derive_year(title: str) -> Optional[int]:
    match = YEAR_RE.search(title.strip())
    if match is None:
        return None
    return int(match.group(1))


def derive_programme(title: str) -> str:
    programme = title.strip()
    if programme.startswith(DA_PREFIX):
        programme = programme[len(DA_PREFIX):]
    elif programme.startswith(EN_PREFIX):
        programme = programme[len(EN_PREFIX):]
    programme = YEAR_RE.sub("", programme).strip()
    return programme


def read_html(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1252", errors="replace")


def process_file(
    path: Path,
) -> tuple[Optional[StudieordningRecord], Optional[ExtractionFailure]]:
    raw_html = read_html(path)

    url = find_saved_from_url(raw_html)
    if url is None:
        return None, ExtractionFailure(
            filename=path.name,
            reason="could not find 'saved from url=(...)' comment",
        )

    id_match = DOK_ORDNING_ID_RE.search(url)
    if id_match is None:
        return None, ExtractionFailure(
            filename=path.name,
            reason=f"could not find dokOrdningId in url: {url}",
        )
    dok_ordning_id = id_match.group("id")

    title = extract_title(raw_html)
    if title is None:
        return None, ExtractionFailure(
            filename=path.name, reason="no <h1> found in document"
        )

    record = StudieordningRecord(
        filename=path.name,
        dokOrdningId=dok_ordning_id,
        url=url,
        title=title,
        level=derive_level(title),
        year=derive_year(title),
        programme=derive_programme(title),
    )
    return record, None


def plan_renames(
    records: list[StudieordningRecord], folder: Path
) -> tuple[dict[str, str], list[str]]:
    """Map original filename -> target filename, skipping collisions.

    A collision is either: two records resolving to the same target id, or
    a target filename that already exists on disk (e.g. an unrelated file,
    or a file that failed extraction and so isn't being renamed away).
    Returns (renames, collision_messages).
    """
    targets_by_name: dict[str, list[str]] = {}
    for record in records:
        target = f"{record.dokOrdningId}.html"
        targets_by_name.setdefault(target, []).append(record.filename)

    renames: dict[str, str] = {}
    collisions: list[str] = []

    for target, sources in targets_by_name.items():
        if len(sources) > 1:
            collisions.append(
                f"{sources!r} all resolve to {target!r}; skipping all of them"
            )
            continue

        source = sources[0]
        if target == source:
            continue  # already correctly named, nothing to do

        if (folder / target).exists():
            collisions.append(
                f"target {target!r} for {source!r} already exists on disk; skipping"
            )
            continue

        renames[source] = target

    return renames, collisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder containing *.html files")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write manifest.json and rename files (default: dry run)",
    )
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"error: {folder} is not a directory", file=sys.stderr)
        return 1

    html_files = sorted(folder.glob("*.html"))
    if not html_files:
        print(f"error: no *.html files found in {folder}", file=sys.stderr)
        return 1

    records: list[StudieordningRecord] = []
    failures: list[ExtractionFailure] = []

    for path in html_files:
        record, failure = process_file(path)
        if record is not None:
            records.append(record)
        if failure is not None:
            failures.append(failure)

    renames, collisions = plan_renames(records, folder)

    manifest = [asdict(r) for r in records]
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\n{len(records)} record(s) extracted, {len(failures)} failure(s).",
          file=sys.stderr)

    if failures:
        print("\nExtraction failures (not renamed):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure.filename}: {failure.reason}", file=sys.stderr)

    if collisions:
        print("\nCollisions (not renamed):", file=sys.stderr)
        for message in collisions:
            print(f"  {message}", file=sys.stderr)

    print(f"\nPlanned renames ({len(renames)}):", file=sys.stderr)
    for source, target in sorted(renames.items()):
        print(f"  {source!r} -> {target!r}", file=sys.stderr)

    if not args.apply:
        print("\nDry run only; pass --apply to write manifest.json and rename files.",
              file=sys.stderr)
        return 0

    manifest_path = folder / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {manifest_path}", file=sys.stderr)

    for source, target in renames.items():
        (folder / source).rename(folder / target)
    print(f"Renamed {len(renames)} file(s).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
