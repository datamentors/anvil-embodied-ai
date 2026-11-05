"""session-valid: check one converted session against its raw input.

Runs after ``mcap-convert``, on the raw session it converted (after ``mcap-exclude``
moved episodes out) and the dataset it wrote. Every check reports pass, warn or fail:

* ``episode_accounting`` — raw MCAPs = dataset episodes + excluded (``--exclusions``)
  + skipped by the converter (above ``--include-flagged`` in the mcap-valid report).
  With the ``source_episode_dir``/``source_mcap`` columns the converter writes, an
  episode it dropped on its own (corrupt or empty frames) is named and warned about.
* ``info_json`` — ``meta/info.json`` exists, codebase_version is v3.0 and fps is a
  positive integer, equal to ``--fps`` when given.
* ``episode_lengths`` — ``meta/episodes`` has one row per episode, none empty, and
  the lengths add up to ``total_frames``.
* ``data_rows`` — the data parquet rows and episodes match ``meta/info.json``.
* ``action_state`` — share of each episode's frames whose action equals the measured
  state (missing commands): warn above ``--action-state-warn``, fail above
  ``--action-state-fail``.
* ``frames_vs_duration`` — episode length / (recorded duration × fps) outside
  [``--frames-ratio-min``, ``--frames-ratio-max``] warns.
* ``videos`` — video features with non-empty MP4s == ``--expected-cameras``.
* ``labels`` — label columns in ``meta/episodes`` are filled in and equal each raw
  episode's ``metadata.json``. Missing labels fail with ``--require-labels`` and warn
  otherwise; values that differ from the raw always fail.
* ``provenance`` — ``conversion_config.yaml`` exists and records the code version.

``--out`` gets ``{"status": "pass|warn|fail", "checks": [{"name", "status", "detail"}],
"episodes": {"raw", "dataset", "excluded", "skipped"}}``.

Exit codes: 0 for pass or warn, 1 for fail.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mcap_converter.cli.convert import _SEVERITY_ORDER, resolve_quality_skip_paths
from mcap_converter.cli.mcap_valid import default_report_paths
from mcap_converter.core.episode_labels import ENVELOPE_KEYS, read_sidecar
from mcap_converter.core.quality import SEVERITY_WARNING

console = Console()

PASS, WARN, FAIL = "pass", "warn", "fail"
_STATUS_COLOR = {PASS: "green", WARN: "yellow", FAIL: "red"}
# Keys anvil_shared.git_provenance() appends to conversion_config.yaml.
_PROVENANCE_KEYS = ("code_commit", "code_tag", "code_version")
_MAX_LISTED = 5


@dataclass
class Check:
    """One named check; its status is the worst of the problems recorded on it."""

    name: str
    fails: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    ok: str = ""

    @property
    def status(self) -> str:
        return FAIL if self.fails else WARN if self.warns else PASS

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": "; ".join(self.fails + self.warns) or self.ok,
        }


@dataclass
class DataSummary:
    """What the data parquet holds, accumulated file by file."""

    rows: int = 0
    episodes: set[int] = field(default_factory=set)
    equal: dict[int, list[int]] = field(default_factory=dict)  # episode -> [action==state, frames]
    error: str | None = None


def _listed(items: list) -> str:
    shown = ", ".join(str(i) for i in items[:_MAX_LISTED])
    return shown + (f" (+{len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else "")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _matrix(column) -> np.ndarray:
    """A list or fixed-size-list parquet column as a (rows, dims) float array."""
    array = column.combine_chunks()
    return array.flatten().to_numpy(zero_copy_only=False).reshape(len(array), -1)


def read_episode_rows(dataset: Path) -> tuple[list[dict], set[str], str | None]:
    """``meta/episodes`` rows by episode_index without stats columns, all column names, error."""
    rows: list[dict] = []
    names: set[str] = set()
    files = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return rows, names, "meta/episodes parquet missing"
    try:
        for f in files:
            schema = pq.read_schema(f).names
            names.update(schema)
            wanted = [
                c for c in schema if not c.startswith(("stats/", "videos/", "data/", "meta/"))
            ]
            rows.extend(pq.read_table(f, columns=wanted).to_pylist())
    except (OSError, ValueError, pa.ArrowException) as exc:
        return [], set(), f"meta/episodes unreadable: {exc}"
    return sorted(rows, key=lambda r: r.get("episode_index") or 0), names, None


def read_data(dataset: Path) -> DataSummary:
    """Row count, episodes and per-episode action==state counts of the data parquet."""
    summary = DataSummary()
    files = sorted((dataset / "data").rglob("*.parquet"))
    if not files:
        summary.error = "data parquet missing"
        return summary
    for f in files:
        try:
            names = pq.read_schema(f).names
            missing = [
                c for c in ("episode_index", "action", "observation.state") if c not in names
            ]
            if missing:
                summary.error = f"{f.name} lacks {missing}"
                return summary
            table = pq.read_table(f, columns=["episode_index", "action", "observation.state"])
            episode = table.column("episode_index").to_numpy()
            action, state = (
                _matrix(table.column("action")),
                _matrix(table.column("observation.state")),
            )
        except (OSError, ValueError, pa.ArrowException) as exc:
            summary.error = f"{f.name} unreadable: {exc}"
            return summary
        if action.shape != state.shape:
            summary.error = f"action shape {action.shape} != observation.state shape {state.shape}"
            return summary
        equal = np.all(np.abs(action - state) < 1e-6, axis=1)
        summary.rows += table.num_rows
        for e in np.unique(episode):
            mask = episode == e
            counts = summary.equal.setdefault(int(e), [0, 0])
            counts[0] += int(equal[mask].sum())
            counts[1] += int(mask.sum())
        summary.episodes.update(int(e) for e in np.unique(episode))
    return summary


def check_accounting(
    raw_mcaps: list[Path],
    report: dict[str, dict] | None,
    skip_paths: dict,
    exclusions: dict | None,
    info: dict,
    rows: list[dict],
    columns: set[str],
) -> tuple[Check, dict, dict[int, Path]]:
    """Episode accounting, the episodes summary block and the dataset-episode -> raw-MCAP map."""
    check = Check("episode_accounting")
    expected = [m for m in raw_mcaps if str(m.resolve()) not in skip_paths]
    skipped = len(raw_mcaps) - len(expected)
    dataset_count = info.get("total_episodes", len(rows)) if info else len(rows)

    total, kept = len(raw_mcaps), len(raw_mcaps)
    if not isinstance(exclusions, dict) or not {"total", "kept", "excluded"} <= exclusions.keys():
        check.fails.append("exclusions JSON missing or unreadable")
    else:
        total, kept = exclusions["total"], exclusions["kept"]
        if kept != len(raw_mcaps):
            check.fails.append(
                f"raw session holds {len(raw_mcaps)} MCAP(s) but exclusions kept {kept}"
            )
    if report is None:
        check.fails.append("mcap-valid report missing or unreadable")
    else:
        unreported = [m.parent.name for m in raw_mcaps if str(m.resolve()) not in report]
        if unreported:
            check.warns.append(
                f"{len(unreported)} MCAP(s) not in the report: {_listed(unreported)}"
            )

    by_key = {(m.parent.name, m.name): m for m in raw_mcaps}
    episode_to_mcap: dict[int, Path] = {}
    if rows and {"episode_index", "source_episode_dir", "source_mcap"} <= columns:
        keys = [(r.get("source_episode_dir"), r.get("source_mcap")) for r in rows]
        unknown = [f"{d}/{m}" for d, m in keys if (d, m) not in by_key]
        duplicates = [f"{d}/{m}" for (d, m), n in Counter(keys).items() if n > 1]
        converted = {by_key[k] for k in keys if k in by_key}
        not_skipped = sorted(m.parent.name for m in converted - set(expected))
        dropped = [m.parent.name for m in expected if m not in converted]
        if unknown:
            check.fails.append(
                f"{len(unknown)} dataset episode(s) not in the raw session: {_listed(unknown)}"
            )
        if duplicates:
            check.fails.append(f"raw episode(s) converted twice: {_listed(duplicates)}")
        if not_skipped:
            check.fails.append(
                f"{len(not_skipped)} episode(s) above --include-flagged were converted: "
                f"{_listed(not_skipped)}"
            )
        if dropped:
            check.warns.append(
                f"converter dropped {len(dropped)} episode(s) on its own (corrupt or empty "
                f"frames): {_listed(dropped)}"
            )
        episode_to_mcap = {r["episode_index"]: by_key[k] for r, k in zip(rows, keys) if k in by_key}
    else:
        if dataset_count != len(expected):
            check.fails.append(
                f"dataset has {dataset_count} episode(s), expected {len(expected)} "
                f"({len(raw_mcaps)} raw MCAP(s) - {skipped} skipped)"
            )
        if rows and "episode_index" in columns and len(rows) == len(expected):
            episode_to_mcap = {r["episode_index"]: m for r, m in zip(rows, expected)}

    check.ok = (
        f"raw {total} = dataset {dataset_count} + excluded {total - kept} + skipped {skipped}"
    )
    episodes = {
        "raw": total,
        "dataset": dataset_count,
        "excluded": total - kept,
        "skipped": skipped,
    }
    return check, episodes, episode_to_mcap


def check_info(info: dict | None, fps: int | None) -> Check:
    check = Check("info_json")
    if info is None:
        check.fails.append("meta/info.json missing or unreadable")
        return check
    if info.get("codebase_version") != "v3.0":
        check.fails.append(f"codebase_version={info.get('codebase_version')}, expected v3.0")
    info_fps = info.get("fps")
    if not isinstance(info_fps, int) or info_fps <= 0:
        check.fails.append(f"fps={info_fps} is not a positive integer")
    elif fps is not None and info_fps != fps:
        check.fails.append(f"fps={info_fps}, expected {fps}")
    check.ok = (
        f"{info.get('codebase_version')}, {info_fps} fps, {info.get('total_episodes')} episodes, "
        f"{info.get('total_frames')} frames"
    )
    return check


def check_lengths(info: dict, rows: list[dict], columns: set[str], error: str | None) -> Check:
    check = Check("episode_lengths")
    if error:
        check.fails.append(error)
        return check
    if not {"episode_index", "length"} <= columns:
        check.fails.append("meta/episodes lacks episode_index or length")
        return check
    lengths = [r["length"] for r in rows]
    if len(rows) != info.get("total_episodes"):
        check.fails.append(
            f"{len(rows)} rows, info.json total_episodes={info.get('total_episodes')}"
        )
    if [r["episode_index"] for r in rows] != list(range(len(rows))):
        check.fails.append("episode_index is not 0..N-1")
    empty = [r["episode_index"] for r in rows if not r["length"] or r["length"] <= 0]
    if empty:
        check.fails.append(f"episode(s) with length 0: {_listed(empty)}")
    if sum(lengths) != info.get("total_frames"):
        check.fails.append(
            f"sum(length)={sum(lengths)}, info.json total_frames={info.get('total_frames')}"
        )
    if lengths:
        check.ok = f"{len(rows)} episodes, {min(lengths)}-{max(lengths)} frames each"
    return check


def check_data(info: dict, data: DataSummary) -> Check:
    check = Check("data_rows")
    if data.error:
        check.fails.append(data.error)
        return check
    if data.rows != info.get("total_frames"):
        check.fails.append(f"{data.rows} rows, info.json total_frames={info.get('total_frames')}")
    if len(data.episodes) != info.get("total_episodes"):
        check.fails.append(
            f"{len(data.episodes)} episodes in data, "
            f"info.json total_episodes={info.get('total_episodes')}"
        )
    check.ok = f"{data.rows} rows over {len(data.episodes)} episodes"
    return check


def _episode_label(index: int, episode_to_mcap: dict[int, Path]) -> str:
    mcap = episode_to_mcap.get(index)
    return f"{index} ({mcap.parent.name})" if mcap else str(index)


def check_action_state(
    data: DataSummary, episode_to_mcap: dict[int, Path], warn_at: float, fail_at: float
) -> Check:
    check = Check("action_state")
    if data.error:
        check.fails.append(f"not checked: {data.error}")
        return check
    fractions = {e: equal / frames for e, (equal, frames) in data.equal.items() if frames}
    above_fail = [e for e, f in sorted(fractions.items()) if f > fail_at]
    above_warn = [e for e, f in sorted(fractions.items()) if warn_at < f <= fail_at]
    for bucket, threshold, target in (
        (above_fail, fail_at, check.fails),
        (above_warn, warn_at, check.warns),
    ):
        if bucket:
            shown = [f"{_episode_label(e, episode_to_mcap)} {fractions[e]:.0%}" for e in bucket]
            target.append(
                f"{len(bucket)} episode(s) with more than {threshold:.0%} of actions equal to the "
                f"measured state (missing commands?): {_listed(shown)}"
            )
    worst = max(fractions.values(), default=0.0)
    check.ok = f"at most {worst:.0%} of an episode's actions equal the measured state"
    return check


def check_frames_vs_duration(
    info: dict,
    rows: list[dict],
    episode_to_mcap: dict[int, Path],
    report: dict[str, dict] | None,
    fps: int | None,
    ratio_min: float,
    ratio_max: float,
) -> Check:
    check = Check("frames_vs_duration")
    info_fps = info.get("fps")
    fps = info_fps if isinstance(info_fps, int) and info_fps > 0 else fps
    if not fps or report is None or not episode_to_mcap:
        check.warns.append("not checked: no fps, no report or episodes not mapped to raw MCAPs")
        return check
    outside, checked = [], 0
    for row in rows:
        mcap = episode_to_mcap.get(row.get("episode_index"))
        duration = (report.get(str(mcap.resolve())) or {}).get("duration_s") if mcap else None
        if not duration or not row.get("length"):
            continue
        checked += 1
        ratio = row["length"] / (duration * fps)
        if not ratio_min <= ratio <= ratio_max:
            outside.append(
                f"{_episode_label(row['episode_index'], episode_to_mcap)} {row['length']} frames "
                f"vs {duration:.1f}s ({ratio:.2f})"
            )
    if outside:
        check.warns.append(
            f"{len(outside)} episode(s) outside [{ratio_min}, {ratio_max}] "
            f"of duration x {fps} fps: {_listed(outside)}"
        )
    check.ok = f"{checked} episode(s) within [{ratio_min}, {ratio_max}] of duration x {fps} fps"
    return check


def check_videos(dataset: Path, info: dict, expected: int) -> Check:
    check = Check("videos")
    keys = [k for k, f in (info.get("features") or {}).items() if f.get("dtype") == "video"]
    empty = []
    for key in keys:
        mp4s = list((dataset / "videos" / key).rglob("*.mp4"))
        if not mp4s or any(f.stat().st_size == 0 for f in mp4s):
            empty.append(key)
    non_empty = [k for k in keys if k not in empty]
    if empty:
        check.fails.append(f"videos missing or empty for {_listed(empty)}")
    if len(non_empty) != expected:
        check.fails.append(f"{len(non_empty)} non-empty video feature(s), expected {expected}")
    check.ok = f"{len(non_empty)} video features: {', '.join(non_empty)}"
    return check


def check_labels(
    rows: list[dict],
    columns: set[str],
    episode_to_mcap: dict[int, Path],
    keys: tuple[str, ...],
    explicit: bool,
    required: bool,
) -> Check:
    check = Check("labels")
    missing_target = check.fails if required else check.warns
    sidecars = {
        r["episode_index"]: read_sidecar(episode_to_mcap[r["episode_index"]])
        for r in rows
        if r.get("episode_index") in episode_to_mcap
    }
    in_raw = {k for k in keys for s in sidecars.values() if s.get(k) not in (None, "")}
    labels = list(keys) if explicit else [k for k in keys if k in columns or k in in_raw]
    if not labels:
        missing_target.append(f"no label columns in meta/episodes (looked for {', '.join(keys)})")
        return check

    absent = [k for k in labels if k not in columns]
    if absent:
        missing_target.append(f"meta/episodes lacks {', '.join(absent)}")
    unlabelled = [
        r.get("episode_index") for r in rows if any(r.get(k) in (None, "") for k in labels)
    ]
    if unlabelled:
        missing_target.append(
            f"{len(unlabelled)} episode(s) with an empty label: {_listed(unlabelled)}"
        )
    differ = []
    for row in rows:
        sidecar = sidecars.get(row.get("episode_index"))
        for k in labels:
            want = "" if sidecar is None or sidecar.get(k) is None else str(sidecar[k])
            got = "" if row.get(k) is None else str(row[k])
            if want and got != want:
                episode = _episode_label(row["episode_index"], episode_to_mcap)
                differ.append(f"{episode} {k}={got!r} vs raw {want!r}")
    if differ:
        check.fails.append(
            f"{len(differ)} label(s) differ from raw metadata.json: {_listed(differ)}"
        )
    if not sidecars:
        check.warns.append(
            "labels not compared with raw metadata.json: episodes not mapped to raw MCAPs"
        )
    check.ok = f"{', '.join(labels)} on all {len(rows)} episodes, equal to raw metadata.json"
    return check


def check_provenance(dataset: Path) -> Check:
    check = Check("provenance")
    path = dataset / "conversion_config.yaml"
    if not path.is_file():
        check.fails.append("conversion_config.yaml missing")
        return check
    try:
        config = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        check.fails.append(f"conversion_config.yaml unreadable: {exc}")
        return check
    found = {k: config[k] for k in _PROVENANCE_KEYS if isinstance(config, dict) and config.get(k)}
    if not found:
        check.warns.append(f"conversion_config.yaml records no {'/'.join(_PROVENANCE_KEYS)}")
    check.ok = ", ".join(f"{k}={v}" for k, v in found.items())
    return check


def validate_session(opts: argparse.Namespace) -> dict:
    """Run every check and return the result document written to ``--out``."""
    raw, dataset = Path(opts.raw), Path(opts.dataset)
    raw_mcaps = sorted(raw.rglob("*.mcap")) if raw.is_dir() else []
    report_path = Path(opts.report) if opts.report else default_report_paths(raw)[0]
    payload = _load_json(report_path)
    report, skip_paths = None, {}
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        try:
            report = {str(Path(e["path"]).resolve()): e for e in payload["episodes"]}
            skip_paths = resolve_quality_skip_paths(str(report_path), opts.include_flagged)
        except (KeyError, TypeError, ValueError):
            report, skip_paths = None, {}
    exclusions = _load_json(Path(opts.exclusions))
    info = _load_json(dataset / "meta" / "info.json")
    if not isinstance(info, dict):
        info = None
    rows, columns, rows_error = read_episode_rows(dataset)
    data = read_data(dataset)

    accounting, episodes, episode_to_mcap = check_accounting(
        raw_mcaps, report, skip_paths, exclusions, info or {}, rows, columns
    )
    label_keys = (
        tuple(k.strip() for k in opts.labels.split(",") if k.strip())
        if opts.labels
        else ENVELOPE_KEYS
    )
    checks = [
        accounting,
        check_info(info, opts.fps),
        check_lengths(info or {}, rows, columns, rows_error),
        check_data(info or {}, data),
        check_action_state(data, episode_to_mcap, opts.action_state_warn, opts.action_state_fail),
        check_frames_vs_duration(
            info or {},
            rows,
            episode_to_mcap,
            report,
            opts.fps,
            opts.frames_ratio_min,
            opts.frames_ratio_max,
        ),
        check_videos(dataset, info or {}, opts.expected_cameras),
        check_labels(
            rows, columns, episode_to_mcap, label_keys, bool(opts.labels), opts.require_labels
        ),
        check_provenance(dataset),
    ]
    statuses = {c.status for c in checks}
    status = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    return {"status": status, "checks": [c.to_dict() for c in checks], "episodes": episodes}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check one converted session against its raw input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  session-valid --raw data/raw/my-session --dataset data/datasets/my-session \\
    --exclusions data/excluded/my-session.json --out data/datasets/my-session.valid.json
  session-valid ... --require-labels --expected-cameras 2
""",
    )
    parser.add_argument("--raw", required=True, help="raw session directory mcap-convert read")
    parser.add_argument("--dataset", required=True, help="dataset directory mcap-convert wrote")
    parser.add_argument(
        "--exclusions", required=True, help="exclusions JSON written by mcap-exclude"
    )
    parser.add_argument("--out", required=True, help="where to write the result JSON")
    parser.add_argument(
        "--expected-cameras",
        type=int,
        default=3,
        help="non-empty video features expected (default: 3)",
    )
    parser.add_argument(
        "--require-labels",
        action="store_true",
        help="fail, not warn, when episode labels are missing",
    )
    parser.add_argument(
        "--labels",
        default=None,
        metavar="K1,K2",
        help="label columns to check (default: the keys mcap-convert lifts from metadata.json: "
        f"{','.join(ENVELOPE_KEYS)})",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="mcap-valid JSON report (default: <raw>/mcap_valid_reports/report.json)",
    )
    parser.add_argument(
        "--include-flagged",
        choices=_SEVERITY_ORDER,
        default=SEVERITY_WARNING,
        help="the --include-flagged mcap-convert ran with (default: warning)",
    )
    parser.add_argument("--fps", type=int, default=None, help="fps meta/info.json must declare")
    parser.add_argument("--action-state-warn", type=float, default=0.2)
    parser.add_argument("--action-state-fail", type=float, default=0.5)
    parser.add_argument("--frames-ratio-min", type=float, default=0.9)
    parser.add_argument("--frames-ratio-max", type=float, default=1.05)
    return parser


def main(args: list[str] | None = None) -> int:
    opts = _build_parser().parse_args(args)
    result = validate_session(opts)
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    table = Table(title="session-valid")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for c in result["checks"]:
        color = _STATUS_COLOR[c["status"]]
        table.add_row(c["name"], f"[{color}]{c['status']}[/{color}]", escape(c["detail"]))
    console.print(table)
    counts = result["episodes"]
    color = _STATUS_COLOR[result["status"]]
    console.print(
        f"[{color}]{result['status'].upper()}[/{color}]  raw {counts['raw']}, "
        f"dataset {counts['dataset']}, excluded {counts['excluded']}, "
        f"skipped {counts['skipped']} → [dim]{escape(str(out))}[/dim]"
    )
    return 1 if result["status"] == FAIL else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
