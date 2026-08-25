"""Build a stratified train/val/test episode split for an envelope dataset.

The default trainer split (``anvil_shared.splits.compute_split_episodes``)
shuffles every episode together, so the share of, say, small face-down envelopes
in the test set is only right on average — and a small stratum can vanish from
val or test entirely.  This tool splits *within* each envelope group instead:

    size (big | medium | small)  x  face (face_up | face_down)  =  6 strata

Each of the 6 strata is split 80/10/10 (or whatever ``--split-ratio`` says), so
every split mirrors the dataset composition.  The three splits are disjoint: one
episode carries exactly one (size, face) label and lands in exactly one split.

Output is a ``split_info.json``-shaped file, the same contract the trainer writes
on checkpoint and ``anvil_eval`` reads back.  Feed it to training with
``--split-file=PATH``; without that flag training keeps its current random split.

Usage:
    uv run stratified-split <dataset_root> [--labels PATH] [options]

Values are normalised, so the recording protocol's own vocabulary works as-is:
``upside``/``downside`` are accepted alongside ``face_up``/``face_down`` (also
``up``/``down``, ``cima``/``baixo``), and ``big``/``large``/``grande`` all mean
big.  Column names are auto-detected — ``envelope_facing_side`` and
``envelope_face`` are both recognised; override with --size-column/--face-column.

Examples:
    # labels already in meta/episodes/*.parquet
    # (envelope_size + envelope_face or envelope_facing_side)
    uv run stratified-split /data/merged-dev4only

    # labels supplied by hand, grouped by size then face
    uv run stratified-split /data/merged-dev4only --labels envelopes.json

    # inspect without writing anything
    uv run stratified-split /data/merged-dev4only --labels envelopes.json --dry-run

Label file formats (--labels), all keyed by episode_index:

  1. grouped JSON — the shape you get from listing episodes per group:
     {
       "big":    {"upside": [0, 3, 7], "downside": [1, 2]},
       "medium": {"upside": [4, 5],    "downside": [6]},
       "small":  {"upside": [8],       "downside": [9]}
     }

  2. flat JSON — one entry per episode:
     {"0": {"envelope_size": "big", "envelope_facing_side": "upside"},
      "1": ["big", "downside"],
      "2": "small|face_up"}

  3. CSV — a header plus one row per episode:
     episode_index,envelope_size,envelope_facing_side
     0,big,upside
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from anvil_shared.stratified import (
    compute_stratified_split_episodes,
    summarize_strata,
    validate_split_info,
)

SIZES = ("big", "medium", "small")
FACES = ("face_up", "face_down")

_SIZE_ALIASES = {
    "big": "big", "large": "big", "l": "big", "b": "big", "grande": "big",
    "medium": "medium", "med": "medium", "m": "medium", "medio": "medium", "médio": "medium",
    "small": "small", "s": "small", "pequeno": "small",
}
_FACE_ALIASES = {
    "face_up": "face_up", "faceup": "face_up", "face-up": "face_up", "up": "face_up",
    "upside": "face_up", "u": "face_up", "cima": "face_up", "true": "face_up",
    "face_down": "face_down", "facedown": "face_down", "face-down": "face_down",
    "down": "face_down", "downside": "face_down", "d": "face_down",
    "baixo": "face_down", "false": "face_down",
}

# Column names looked for in meta/episodes/*.parquet, in order of preference.
SIZE_COLUMNS = ("envelope_size", "size", "envelope")
FACE_COLUMNS = (
    "envelope_face", "envelope_facing_side", "facing_side",
    "face", "orientation", "envelope_orientation",
)


def stratum_key(size: str, face: str) -> str:
    return f"{size}|{face}"


def _norm(raw, aliases: dict[str, str], what: str, ep: int | None = None) -> str:
    key = str(raw).strip().lower().replace(" ", "_")
    if key in aliases:
        return aliases[key]
    where = f" for episode {ep}" if ep is not None else ""
    raise ValueError(
        f"unrecognised {what} value {raw!r}{where}. "
        f"Accepted: {sorted(set(aliases))}"
    )


def _record(labels: dict[int, str], ep: int, size: str, face: str, source: str) -> None:
    """Assign one episode's stratum, refusing to silently overwrite a conflict."""
    key = stratum_key(size, face)
    prev = labels.get(ep)
    if prev is not None and prev != key:
        raise ValueError(
            f"episode {ep} is labelled both {prev!r} and {key!r} in {source}. "
            "Each episode must have exactly one (size, face) label."
        )
    labels[ep] = key


def _labels_from_grouped_json(data: dict, source: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    for raw_size, faces in data.items():
        size = _norm(raw_size, _SIZE_ALIASES, "size")
        if not isinstance(faces, dict):
            raise ValueError(
                f"{source}: expected a mapping of face -> [episodes] under {raw_size!r}, "
                f"got {type(faces).__name__}"
            )
        for raw_face, eps in faces.items():
            face = _norm(raw_face, _FACE_ALIASES, "face")
            if not isinstance(eps, (list, tuple)):
                raise ValueError(
                    f"{source}: expected a list of episode indices under "
                    f"{raw_size!r}/{raw_face!r}, got {type(eps).__name__}"
                )
            for ep in eps:
                _record(labels, int(ep), size, face, source)
    return labels


def _labels_from_flat_json(data: dict, source: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    for raw_ep, value in data.items():
        ep = int(raw_ep)
        if isinstance(value, dict):
            lowered = {str(k).lower(): v for k, v in value.items()}
            try:
                raw_size = next(lowered[k] for k in ("size", "envelope_size") if k in lowered)
                raw_face = next(
                    lowered[k]
                    for k in (
                        "face", "envelope_face", "envelope_facing_side",
                        "facing_side", "orientation",
                    )
                    if k in lowered
                )
            except StopIteration:
                raise ValueError(
                    f"{source}: episode {ep} needs both a size and a face key, got {sorted(value)}"
                ) from None
        elif isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(
                    f"{source}: episode {ep} needs exactly [size, face], got {list(value)}"
                )
            raw_size, raw_face = value
        elif isinstance(value, str):
            parts = value.replace("/", "|").split("|")
            if len(parts) != 2:
                raise ValueError(
                    f"{source}: episode {ep} string label must look like 'big|face_up', got {value!r}"
                )
            raw_size, raw_face = parts
        else:
            raise ValueError(f"{source}: unsupported label for episode {ep}: {value!r}")

        _record(
            labels,
            ep,
            _norm(raw_size, _SIZE_ALIASES, "size", ep),
            _norm(raw_face, _FACE_ALIASES, "face", ep),
            source,
        )
    return labels


def _labels_from_csv(path: Path) -> dict[int, str]:
    source = str(path)
    labels: dict[int, str] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: file is empty")
        cols = {name.strip().lower(): name for name in reader.fieldnames}

        def pick(candidates: tuple[str, ...], what: str) -> str:
            for c in candidates:
                if c in cols:
                    return cols[c]
            raise ValueError(
                f"{source}: no {what} column found. Looked for {list(candidates)}, "
                f"file has {reader.fieldnames}"
            )

        ep_col = pick(("episode_index", "episode", "ep"), "episode index")
        size_col = pick(SIZE_COLUMNS, "size")
        face_col = pick(FACE_COLUMNS, "face")

        for row in reader:
            if row[ep_col] is None or str(row[ep_col]).strip() == "":
                continue
            ep = int(str(row[ep_col]).strip())
            _record(
                labels,
                ep,
                _norm(row[size_col], _SIZE_ALIASES, "size", ep),
                _norm(row[face_col], _FACE_ALIASES, "face", ep),
                source,
            )
    return labels


def load_labels_from_file(path: Path) -> dict[int, str]:
    """Read episode labels from a JSON (grouped or flat) or CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"labels file not found: {path}")

    if path.suffix.lower() == ".csv":
        return _labels_from_csv(path)

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")
    if not data:
        raise ValueError(f"{path}: no labels in file")

    # Grouped form iff every top-level key names a size and maps to a dict.
    looks_grouped = all(
        str(k).strip().lower() in _SIZE_ALIASES and isinstance(v, dict) for k, v in data.items()
    )
    return (
        _labels_from_grouped_json(data, str(path))
        if looks_grouped
        else _labels_from_flat_json(data, str(path))
    )


def load_labels_from_meta(root: Path, size_col: str | None, face_col: str | None) -> dict[int, str]:
    """Read episode labels from extra columns in ``meta/episodes/*.parquet``.

    Custom columns survive ``merge-datasets``: LeRobot's aggregation rewrites only
    the columns it knows about and offsets ``episode_index``, so per-session labels
    stay attached to the right episodes in a merged dataset.
    """
    import pandas as pd

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def resolve(explicit: str | None, candidates: tuple[str, ...], what: str) -> str:
        if explicit:
            if explicit not in df.columns:
                raise ValueError(f"--{what}-column={explicit!r} not found in episode metadata")
            return explicit
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"no {what} column in episode metadata (looked for {list(candidates)}).\n"
            "The colleagues' annotation has probably not landed yet — pass --labels PATH "
            "to supply the labels from a file instead."
        )

    sc = resolve(size_col, SIZE_COLUMNS, "size")
    fc = resolve(face_col, FACE_COLUMNS, "face")
    print(f"  reading labels from meta columns: {sc!r}, {fc!r}")

    labels: dict[int, str] = {}
    for row in df[["episode_index", sc, fc]].itertuples(index=False):
        ep = int(row[0])
        # mcap-convert writes an empty string for episodes the recorder never
        # labelled. Leave them out so check_coverage reports them as unlabelled
        # (and points at --allow-partial) instead of failing on a blank value.
        if row[1] is None or row[2] is None or str(row[1]).strip() == "" or str(row[2]).strip() == "":
            continue
        _record(
            labels,
            ep,
            _norm(row[1], _SIZE_ALIASES, "size", ep),
            _norm(row[2], _FACE_ALIASES, "face", ep),
            "episode metadata",
        )
    return labels


def read_total_episodes(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"not a LeRobot dataset (no {info_path})")
    return int(json.loads(info_path.read_text())["total_episodes"])


def check_coverage(labels: dict[int, str], total: int, allow_partial: bool) -> None:
    """Refuse to build a split from labels that do not line up with the dataset."""
    out_of_range = sorted(ep for ep in labels if ep < 0 or ep >= total)
    if out_of_range:
        raise ValueError(
            f"{len(out_of_range)} labelled episode(s) fall outside the dataset's "
            f"0..{total - 1} range: {out_of_range[:10]}"
            f"{' ...' if len(out_of_range) > 10 else ''}\n"
            "Labels are keyed by the episode indices of THIS dataset. If they were "
            "written against the per-session datasets, remember merge-datasets "
            "renumbers episodes."
        )

    missing = sorted(set(range(total)) - set(labels))
    if missing:
        msg = (
            f"{len(missing)} of {total} episode(s) have no label: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
        if not allow_partial:
            raise ValueError(
                msg + "\nLabel them, or pass --allow-partial to split only the labelled ones "
                "(unlabelled episodes are then used by nothing at all)."
            )
        print(f"  WARNING: {msg}")
        print("  --allow-partial given: these episodes will not appear in any split")


def print_report(labels: dict[int, str], splits: dict[str, list[int]], total: int) -> None:
    table = summarize_strata(labels, splits)
    width = max(len("stratum"), *(len(k.replace("|", " / ")) for k in table))

    print(f"\n{'stratum'.ljust(width)}  {'total':>6}{'train':>8}{'val':>6}{'test':>6}")
    print("-" * (width + 28))
    for stratum, row in table.items():
        print(
            f"{stratum.replace('|', ' / ').ljust(width)}  {row['total']:>6}"
            f"{row['train']:>8}{row['val']:>6}{row['test']:>6}"
        )
    print("-" * (width + 28))
    n = {k: len(v) for k, v in splits.items()}
    labelled = len(labels)
    print(
        f"{'all'.ljust(width)}  {labelled:>6}{n['train']:>8}{n['val']:>6}{n['test']:>6}"
    )
    if labelled:
        print(
            f"{''.ljust(width)}  {'':>6}"
            f"{100 * n['train'] / labelled:>7.1f}%{100 * n['val'] / labelled:>5.0f}%"
            f"{100 * n['test'] / labelled:>5.0f}%"
        )

    empty = [s for s, r in table.items() if r["val"] == 0 or r["test"] == 0]
    if empty:
        print(
            "\nWARNING: stratum(s) too small to reach every split: "
            + ", ".join(s.replace("|", " / ") for s in empty)
        )

    absent = [f"{s} / {f}" for s in SIZES for f in FACES if stratum_key(s, f) not in table]
    if absent:
        print("\nNote: no episodes labelled for: " + ", ".join(absent))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stratified-split",
        description=(
            "Build a train/val/test episode split stratified by envelope size "
            "(big/medium/small) x face (face_up/face_down)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", metavar="PATH", help="Root of the LeRobot dataset to split")
    parser.add_argument(
        "--labels",
        metavar="PATH",
        help="JSON/CSV file with per-episode size+face labels. When omitted, labels are read "
             "from extra columns in the dataset's meta/episodes/*.parquet.",
    )
    parser.add_argument(
        "--split-ratio",
        default="8,1,1",
        metavar="TRAIN,VAL,TEST",
        help="Ratio applied WITHIN each stratum (default: 8,1,1). Two values mean no test set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Base seed; each stratum derives its own from it (default: 42)",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Where to write the split file (default: <dataset>/stratified_split_info.json)",
    )
    parser.add_argument(
        "--size-column", metavar="NAME",
        help=f"Episode-metadata column holding the size (default: first of {list(SIZE_COLUMNS)})",
    )
    parser.add_argument(
        "--face-column", metavar="NAME",
        help=f"Episode-metadata column holding the face (default: first of {list(FACE_COLUMNS)})",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Permit episodes with no label; they are excluded from all three splits",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the split and exit without writing"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        ratio = [float(x) for x in args.split_ratio.split(",")]
    except ValueError:
        print(f"Error: --split-ratio={args.split_ratio!r} is not a comma-separated number list",
              file=sys.stderr)
        sys.exit(1)
    if len(ratio) == 2:
        ratio.append(0.0)
    if len(ratio) != 3:
        print(f"Error: --split-ratio needs 2 or 3 values, got {len(ratio)}", file=sys.stderr)
        sys.exit(1)

    root = Path(args.dataset).resolve()
    try:
        total = read_total_episodes(root)
        print(f"Dataset {root}\n  {total} episodes")

        if args.labels:
            labels = load_labels_from_file(Path(args.labels).resolve())
            source = str(Path(args.labels).resolve())
            print(f"  read {len(labels)} label(s) from {source}")
        else:
            labels = load_labels_from_meta(root, args.size_column, args.face_column)
            source = "meta/episodes"
            print(f"  read {len(labels)} label(s) from episode metadata")

        check_coverage(labels, total, args.allow_partial)
        splits = compute_stratified_split_episodes(labels, ratio, seed=args.seed)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_report(labels, splits, total)

    split_info = {
        "strategy": "stratified",
        "stratified_by": ["size", "face"],
        "split_ratio": ratio,
        "seed": args.seed,
        "total_episodes": total,
        "labels_source": source,
        "train_episodes": splits["train"],
        "val_episodes": splits["val"],
        "test_episodes": splits["test"],
        "strata": summarize_strata(labels, splits),
    }

    problems = validate_split_info(split_info, total)
    if problems:
        print("\nError: the computed split is not self-consistent:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    out_path = Path(args.output).resolve() if args.output else root / "stratified_split_info.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split_info, indent=2) + "\n")

    print(f"\nWrote {out_path}")
    print("Use it for training with:")
    print(f"  --split-file={out_path}")


if __name__ == "__main__":
    main()
