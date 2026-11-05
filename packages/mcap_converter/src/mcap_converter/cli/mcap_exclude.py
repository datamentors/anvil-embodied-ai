"""mcap-exclude: move out of a session the episodes that must not be converted.

Runs between ``mcap-valid`` and ``mcap-convert``. An episode folder is moved from
the session into ``--excluded-dir`` when any of these hold:

* ``low_fps`` — its reference camera (the first ``camera_topic_mapping`` entry, the
  one ``mcap-convert`` detects fps from) snaps below ``--fps``. ``mcap-convert``
  checks ``--fps`` against the slowest MCAP in the session, skipped ones included,
  so a single slow episode would otherwise abort the whole conversion.
* ``no_commands_<arm>`` — the mcap-valid report shows zero messages on that arm's
  command topic (``action_topics``). mcap-valid rates this only ``warning``, so the
  converter would keep the episode and fill its actions with measured positions.
  ``--no-commands any`` (default) moves an episode when a single arm is silent;
  ``--no-commands all`` only when every arm is, so one-handed episodes stay. Not
  applied to ``action_from_observation`` configs, which never read the commands.
* ``manual`` — listed in ``--manual`` (``episode<TAB>reason`` per line).

Critical episodes stay in place: ``mcap-convert`` skips them itself.

``--out`` gets ``{"excluded": [{"episode", "reasons", "fps", "manual_reason"?}],
"kept": <int>, "total": <int>}``, where ``total`` and ``kept`` count the MCAP files
(one per episode) before and after the move.

Exit codes: 0 on success, also when every episode is excluded; 2 on bad input
(missing report or config, an MCAP missing from the report, unexpected layout).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from rich.console import Console
from rich.markup import escape

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.core.reader import McapReader, snap_fps

console = Console()
_err_console = Console(stderr=True)

EXIT_BAD_INPUT = 2


class BadInputError(Exception):
    """Input this command cannot work with; reported with exit code 2."""


def load_manual(path: Path) -> dict[str, str]:
    """``episode<TAB>reason`` lines as {episode: reason}; blank and ``#`` lines are skipped."""
    try:
        text = path.read_text()
    except OSError as exc:
        raise BadInputError(f"cannot read --manual {path}: {exc}") from exc
    manual: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        episode, _, reason = line.partition("\t")
        manual[episode.strip()] = reason.strip()
    return manual


def load_report(path: Path) -> dict[Path, dict]:
    """mcap-valid report entries keyed by resolved MCAP path, as ``mcap-convert`` matches them."""
    try:
        payload = json.loads(path.read_text())
        return {Path(entry["path"]).resolve(): entry for entry in payload["episodes"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BadInputError(f"cannot read mcap-valid report {path}: {exc}") from exc


def load_topics(config_path: Path) -> tuple[str, dict[str, str], bool]:
    """Reference camera topic, {arm: command topic} and whether actions come from observations."""
    try:
        config = ConfigLoader.from_yaml(str(config_path))
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
        raise BadInputError(f"cannot read --config {config_path}: {exc}") from exc
    if not config.camera_topic_mapping:
        raise BadInputError(f"{config_path} has no camera_topic_mapping")
    ref_topic = next(iter(config.camera_topic_mapping))
    commands = {spec.arm or topic: topic for topic, spec in config.action_topics.items()}
    return ref_topic, commands, bool(config.action_from_observation)


def episode_reasons(
    mcap: Path,
    entry: dict,
    *,
    ref_topic: str,
    commands: dict[str, str],
    fps: int,
    no_commands: str,
) -> tuple[list[str], int | None]:
    """Rule-based reasons (low_fps, no_commands_<arm>) for one MCAP, plus its snapped fps."""
    reasons = []
    raw_fps = McapReader(str(mcap)).estimate_fps(ref_topic)
    snapped = snap_fps(raw_fps) if raw_fps else None
    if snapped is None or snapped < fps:
        reasons.append("low_fps")
    counts = {t.get("topic"): t.get("message_count") or 0 for t in entry.get("topics") or []}
    silent = [arm for arm, topic in commands.items() if counts.get(topic, 0) == 0]
    if silent and (no_commands == "any" or len(silent) == len(commands)):
        reasons.extend(f"no_commands_{arm}" for arm in silent)
    return reasons, snapped


def plan_exclusions(
    session: Path,
    report: dict[Path, dict],
    *,
    ref_topic: str,
    commands: dict[str, str],
    fps: int,
    no_commands: str,
    manual: dict[str, str],
) -> tuple[list[Path], list[dict]]:
    """All MCAPs in the session and the episode folders to move out, with their reasons."""
    mcaps = sorted(session.rglob("*.mcap"))
    missing = [str(m) for m in mcaps if m.resolve() not in report]
    if missing:
        raise BadInputError(
            f"{len(missing)} MCAP(s) not in the mcap-valid report (stale report?), "
            f"e.g. {missing[:3]}"
        )

    by_folder: dict[Path, dict] = {}
    for mcap in mcaps:
        reasons, snapped = episode_reasons(
            mcap,
            report[mcap.resolve()],
            ref_topic=ref_topic,
            commands=commands,
            fps=fps,
            no_commands=no_commands,
        )
        if mcap.parent.name in manual:
            reasons.append("manual")
        if not reasons:
            continue
        item = by_folder.setdefault(
            mcap.parent, {"episode": mcap.parent.name, "reasons": [], "fps": snapped}
        )
        item["reasons"].extend(r for r in reasons if r not in item["reasons"])
        if "manual" in reasons:
            item["manual_reason"] = manual[mcap.parent.name]

    for folder in by_folder:
        if folder.parent.resolve() != session.resolve():
            raise BadInputError(
                f"{folder} is not an episode folder directly under the session; cannot move it"
            )
    return mcaps, list(by_folder.values())


def move_out(session: Path, excluded: list[dict], dest: Path) -> None:
    """Move every excluded episode folder into ``dest``, refusing to overwrite anything there."""
    clashes = [e["episode"] for e in excluded if (dest / e["episode"]).exists()]
    if clashes:
        raise BadInputError(f"already present in --excluded-dir {dest}: {clashes}")
    for item in excluded:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(session / item["episode"]), str(dest / item["episode"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move out of a session the episodes that must not be converted.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  mcap-exclude -i data/raw/my-session \\
    --report data/raw/my-session/mcap_valid_reports/report.json \\
    --config configs/mcap_converter/openarm_bimanual_quest.yaml --fps 30 \\
    --excluded-dir data/excluded/my-session --out data/excluded/my-session.json
  mcap-exclude ... --no-commands all          # keep one-handed episodes
  mcap-exclude ... --manual reviewed.tsv      # also move these out (episode<TAB>reason)
""",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="session directory (one folder per episode)"
    )
    parser.add_argument("--report", required=True, help="mcap-valid JSON report for this session")
    parser.add_argument("--config", required=True, help="mcap-convert config YAML")
    parser.add_argument(
        "--fps", type=int, required=True, help="output fps mcap-convert will run at"
    )
    parser.add_argument(
        "--excluded-dir",
        required=True,
        help="directory the excluded episode folders are moved into",
    )
    parser.add_argument("--out", required=True, help="where to write the exclusions JSON")
    parser.add_argument(
        "--no-commands",
        choices=("any", "all"),
        default="any",
        help="exclude when any arm's command topic is silent, or only when all are (default: any)",
    )
    parser.add_argument(
        "--manual", default=None, help="TSV of episodes to exclude by hand: episode<TAB>reason"
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parsed = _build_parser().parse_args(args)
    session = Path(parsed.input)
    dest = Path(parsed.excluded_dir)
    try:
        if not session.is_dir():
            raise BadInputError(f"session directory not found: {session}")
        if dest.resolve().is_relative_to(session.resolve()):
            raise BadInputError("--excluded-dir must be outside the session")
        report = load_report(Path(parsed.report))
        ref_topic, commands, from_observation = load_topics(Path(parsed.config))
        manual = load_manual(Path(parsed.manual)) if parsed.manual else {}
        mcaps, excluded = plan_exclusions(
            session,
            report,
            ref_topic=ref_topic,
            commands={} if from_observation else commands,
            fps=parsed.fps,
            no_commands=parsed.no_commands,
            manual=manual,
        )
        move_out(session, excluded, dest)
    except BadInputError as exc:
        _err_console.print(f"[red]✗ {escape(str(exc))}[/red]")
        return EXIT_BAD_INPUT

    kept = len(list(session.rglob("*.mcap")))
    result = {"excluded": excluded, "kept": kept, "total": len(mcaps)}
    out = Path(parsed.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    console.print(
        f"[bold]{len(excluded)}[/bold] episode folder(s) excluded, "
        f"[bold]{kept}[/bold]/{len(mcaps)} MCAP(s) kept → [dim]{escape(str(out))}[/dim]"
    )
    for item in excluded:
        note = f" ({item['manual_reason']})" if item.get("manual_reason") else ""
        reasons = ", ".join(item["reasons"])
        console.print(f"  [yellow]↷[/yellow] {item['episode']}: {reasons}{note}")
    unknown = sorted(set(manual) - {m.parent.name for m in mcaps})
    if unknown:
        console.print(
            f"  [yellow]--manual episodes not in the session: {escape(', '.join(unknown))}[/yellow]"
        )
    if from_observation and commands:
        console.print(
            "  [dim]no_commands not applied: config derives actions from observations[/dim]"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
