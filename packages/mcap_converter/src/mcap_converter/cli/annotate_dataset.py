"""Write session_metadata.json labels into an already-converted dataset.

The converter learned to carry recorder labels into ``meta/episodes`` as it
writes (see ``core.episode_labels``), but that only helps datasets converted
from now on.  This tool covers the ones already on disk: it reads the audited
``session_metadata.json`` sitting next to a converted session and writes its
per-episode fields as columns, without reconverting anything.

Alignment is checked rather than assumed.  The metadata lists one entry per
episode with an explicit ``episode_index``; those indices must cover the
dataset's episodes exactly, or nothing is written.  That is the whole point —
an audit that silently disagrees with the dataset is worse than no audit.

Annotate each session *before* merging.  Session-level indices line up with the
session's own dataset one-to-one, and ``merge-datasets`` then offsets
``episode_index`` while leaving unknown columns attached to the right rows.
Annotating after a merge means redoing that offset arithmetic by hand.

Usage:
    annotate-dataset <dataset_root> [options]

Examples:
    annotate-dataset /data/2026-08-03/s01
    annotate-dataset /data/2026-08-03/s01 --dry-run
    annotate-dataset /data/2026-08-03/s01 --include-derived
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fields that record something observed. `date`/`session`/`source_episode_number`
# are provenance: after a merge they are the only way back to the recording.
DEFAULT_COLUMNS = (
    "envelope_size",
    "envelope_facing_side",
    "date",
    "session",
    "source_episode_number",
)

# Fields computed from envelope_size by whoever wrote the metadata, kept out by
# default: destination_basket_side follows sorting_rules, and arm was derived
# from destination_basket_side rather than observed. Storing them duplicates
# envelope_size under two other names.
DERIVED_COLUMNS = ("destination_basket_side", "arm")

MISSING = ""


def load_episode_entries(metadata_path: Path) -> tuple[list[dict], dict]:
    """Return the per-episode entries and the whole document.

    Accepts the entries under ``loop.episodes`` (the audit format) or a
    top-level ``episodes`` list (what ``label-session`` writes).
    """
    doc = json.loads(metadata_path.read_text())
    entries = doc.get("loop", {}).get("episodes") or doc.get("episodes")
    if not entries:
        raise ValueError(
            f"{metadata_path}: no per-episode entries found "
            "(expected loop.episodes or episodes)"
        )
    return entries, doc


def index_entries(entries: list[dict], metadata_path: Path) -> dict[int, dict]:
    """Key the entries by episode_index, rejecting duplicates."""
    by_index: dict[int, dict] = {}
    for entry in entries:
        if "episode_index" not in entry:
            raise ValueError(
                f"{metadata_path}: an entry has no episode_index "
                f"(keys: {sorted(entry)}). Cannot place it without one."
            )
        idx = int(entry["episode_index"])
        if idx in by_index:
            raise ValueError(f"{metadata_path}: episode_index {idx} appears more than once")
        by_index[idx] = entry
    return by_index


def check_alignment(by_index: dict[int, dict], total_episodes: int) -> None:
    """Refuse to write when the audit and the dataset disagree about episodes."""
    expected = set(range(total_episodes))
    got = set(by_index)

    extra = sorted(got - expected)
    missing = sorted(expected - got)
    if not extra and not missing:
        return

    lines = [
        f"metadata does not line up with the dataset ({total_episodes} episodes):"
    ]
    if missing:
        lines.append(
            f"  {len(missing)} dataset episode(s) have no entry: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
    if extra:
        lines.append(
            f"  {len(extra)} entry/entries point outside the dataset: {extra[:10]}"
            f"{' ...' if len(extra) > 10 else ''}"
        )
    lines.append(
        "  Indices must refer to THIS dataset's episodes. If the metadata was written "
        "against the raw recordings, remember the converter skips flagged episodes; if "
        "against a merged dataset, annotate that dataset instead of this session."
    )
    raise ValueError("\n".join(lines))


def resolve_columns(by_index: dict[int, dict], columns: tuple[str, ...]) -> tuple[str, ...]:
    """Keep only the requested columns that at least one entry actually has."""
    present = set()
    for entry in by_index.values():
        present.update(k for k in columns if entry.get(k) not in (None, ""))
    return tuple(c for c in columns if c in present)


def read_episode_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {dataset_root / 'meta' / 'episodes'}")
    return files


def total_episodes(dataset_root: Path) -> int:
    info = dataset_root / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(f"not a LeRobot dataset (no {info})")
    return int(json.loads(info.read_text())["total_episodes"])


def find_conflicts(
    dataset_root: Path, by_index: dict[int, dict], columns: tuple[str, ...]
) -> list[str]:
    """Episodes whose existing column values differ from what we would write."""
    import pandas as pd

    conflicts: list[str] = []
    for path in read_episode_files(dataset_root):
        df = pd.read_parquet(path)
        for col in columns:
            if col not in df.columns:
                continue
            for _, row in df.iterrows():
                idx = int(row["episode_index"])
                new = str(by_index[idx].get(col, MISSING))
                old = "" if row[col] is None else str(row[col])
                if old and old != new:
                    conflicts.append(f"episode {idx}: {col}={old!r} would become {new!r}")
    return conflicts


def annotate(
    dataset_root: Path, by_index: dict[int, dict], columns: tuple[str, ...]
) -> int:
    """Write the columns into every episode-metadata parquet. Returns rows written."""
    import pandas as pd

    written = 0
    for path in read_episode_files(dataset_root):
        df = pd.read_parquet(path)
        for col in columns:
            df[col] = [
                str(by_index[int(i)].get(col, MISSING)) for i in df["episode_index"]
            ]
        df.to_parquet(path)
        written += len(df)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotate-dataset",
        description=(
            "Write session_metadata.json per-episode labels into a converted "
            "dataset's meta/episodes columns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", metavar="PATH", help="Converted LeRobot dataset (one session)")
    parser.add_argument(
        "--metadata", metavar="PATH",
        help="Session metadata file (default: <dataset>/session_metadata.json)",
    )
    parser.add_argument(
        "--columns", metavar="C1,C2",
        help=f"Fields to write (default: {','.join(DEFAULT_COLUMNS)})",
    )
    parser.add_argument(
        "--include-derived", action="store_true",
        help=f"Also write {DERIVED_COLUMNS} — values computed from envelope_size, "
             "not independently observed",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing column values that differ",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.dataset).resolve()
    meta_path = Path(args.metadata).resolve() if args.metadata else root / "session_metadata.json"

    columns = (
        tuple(c.strip() for c in args.columns.split(",") if c.strip())
        if args.columns
        else DEFAULT_COLUMNS + (DERIVED_COLUMNS if args.include_derived else ())
    )

    try:
        if not meta_path.exists():
            raise FileNotFoundError(f"no session metadata at {meta_path}")
        n_eps = total_episodes(root)
        entries, _doc = load_episode_entries(meta_path)
        by_index = index_entries(entries, meta_path)

        print(f"Dataset {root}")
        print(f"  {n_eps} episode(s) | {len(by_index)} metadata entry/entries")

        check_alignment(by_index, n_eps)
        print("  alignment OK — every episode has exactly one entry")

        columns = resolve_columns(by_index, columns)
        if not columns:
            raise ValueError("none of the requested fields are present in the metadata")
        print(f"  columns: {', '.join(columns)}")

        conflicts = find_conflicts(root, by_index, columns)
        if conflicts and not args.force:
            print(f"\nError: {len(conflicts)} existing value(s) would change:", file=sys.stderr)
            for c in conflicts[:10]:
                print(f"  - {c}", file=sys.stderr)
            if len(conflicts) > 10:
                print(f"  ... and {len(conflicts) - 10} more", file=sys.stderr)
            print("\nRe-run with --force if that is intended.", file=sys.stderr)
            sys.exit(1)
        if conflicts:
            print(f"  --force: overwriting {len(conflicts)} differing value(s)")

        if args.dry_run:
            sample = by_index[0]
            print("\n--dry-run: nothing written. Episode 0 would get:")
            for col in columns:
                print(f"    {col:24} {sample.get(col, MISSING)!r}")
            return

        written = annotate(root, by_index, columns)
        print(f"\nAnnotated {written} episode row(s) in {root.name}")

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
