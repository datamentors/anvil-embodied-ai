"""Label a recorded session with the envelope characteristics it was recorded with.

Written for the data-collection team.  The workflow it assumes: one session =
one envelope configuration.  Every episode in the folder was recorded with the
same envelope size and the same facing side, so the operator answers two
questions once and every episode in the session gets stamped.

    label-session /data/raw/2026-08-24-s01

    Envelope size?
      1) small   2) medium   3) big
    > 3
    Which side is the envelope facing?
      1) upside (face up)   2) downside (face down)
    > 1

Two things are written:

  * ``<session>/session_metadata.json`` — the session-level record, in the same
    shape as the recording protocol file, with one entry per episode found.
  * ``<episode>/metadata.json`` — the envelope keys are merged into each
    episode's existing recorder sidecar, preserving whatever is already there
    (``status``, ``note``, ``duration``, ...).

Why both: the per-episode copy is the one that survives everything.  The
converter skips episodes flagged critical — including the ones the recorder
marked ``status="aborted"`` — so raw folder numbering and LeRobot episode
indices drift apart.  A label glued to its own episode directory cannot drift;
a label addressed by index can.  The session file is for humans opening the
folder, and for anything that wants the summary without walking every episode.

Re-running is safe: the tool updates in place and refuses to silently overwrite
a session that was already labelled differently (use --force for that).

With ``--to-nas`` the labelling flows straight into the NAS upload step: once the
labels are on disk, ``fill_dataset_taxonomy_nas.py --from-label <session_dir>``
runs in the same command and asks the taxonomy questions
(use_case/task/embodiment/setup_id), prefilled from what was just written --
episode count, date and session come from the label run and the folder name.  It
stops after generating the ``_generated/<slug>_nas.env`` config and prints the
copy command; moving files to the NAS stays a separate, deliberate step.

Usage:
    label-session <session_dir> [--size SIZE] [--face FACE] [options]

Examples:
    label-session /data/raw/2026-08-24-s01                  # asks interactively
    label-session /data/raw/2026-08-24-s01 --size big --face upside
    label-session /data/raw/2026-08-24-s01 --size small --face down --arm left
    label-session /data/raw/2026-08-24-s01 --size big --face upside --dry-run
    label-session /data/raw/2026-08-24-s01 --size big --face upside --to-nas
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcap_converter.cli.stratified_split import _FACE_ALIASES, _SIZE_ALIASES, _norm

# The vocabulary written to disk — kept identical to the recording protocol file
# so the two describe envelopes with the same words.
SIZES = ("small", "medium", "big")
FACES = ("upside", "downside")

# stratified-split normalises to face_up/face_down; map back to the protocol's words.
_CANONICAL_FACE = {"face_up": "upside", "face_down": "downside"}

DEFAULT_TASK = {
    "id": "sort_envelopes",
    "instruction": "Pick up the envelope and place it in the correct basket",
    "embodiment": "anvil_openarm",
    "operation": "pick_and_place",
}
DEFAULT_SORTING_RULES = {"small": "right", "medium": "left", "big": "left"}

ENVELOPE_KEYS = ("envelope_size", "envelope_facing_side", "destination_basket_side", "arm")

# --to-nas hands off to the data-collection scripts, which live outside this
# package (next to the _registry/ and _generated/ folders they read and write).
NAS_TAXONOMY_SCRIPT = "fill_dataset_taxonomy_nas.py"
NAS_TOOLS_ENV = "ANVIL_NAS_TOOLS"


# =============================================================================
# discovery
# =============================================================================


def find_episodes(session_dir: Path) -> list[Path]:
    """Return the episode directories inside a session, in recording order.

    An episode directory is any immediate subdirectory holding at least one
    ``.mcap`` file — the layout ``ros2 bag record`` produces, one directory per
    episode (``0001/0001_0.mcap`` alongside ``0001/metadata.json``).
    """
    if not session_dir.exists():
        raise FileNotFoundError(f"session folder not found: {session_dir}")
    if not session_dir.is_dir():
        raise NotADirectoryError(f"not a folder: {session_dir}")

    episodes = [d for d in sorted(session_dir.iterdir()) if d.is_dir() and any(d.glob("*.mcap"))]
    if not episodes:
        raise ValueError(
            f"no episodes found in {session_dir}.\n"
            "Expected one subdirectory per episode, each containing a .mcap file "
            "(e.g. 0001/0001_0.mcap). Is this the right folder?"
        )
    return episodes


def read_sidecar(episode_dir: Path) -> dict:
    """Read an episode's recorder metadata.json, or {} when absent/unreadable."""
    path = episode_dir / "metadata.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# =============================================================================
# interactive prompts
# =============================================================================


def _choose(question: str, options: tuple[str, ...], hints: dict[str, str] | None = None) -> str:
    """Ask the operator to pick one option, by number or by name."""
    hints = hints or {}
    while True:
        print(f"\n{question}")
        for i, opt in enumerate(options, 1):
            hint = f"  ({hints[opt]})" if opt in hints else ""
            print(f"  {i}) {opt}{hint}")
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled — nothing was written.", file=sys.stderr)
            sys.exit(130)

        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw.lower() in options:
            return raw.lower()
        print(f"  '{raw}' is not one of the options — type a number 1..{len(options)} or the name.")


def ask_size() -> str:
    return _choose("Envelope size?", SIZES)


def ask_face() -> str:
    return _choose(
        "Which side is the envelope facing?",
        FACES,
        {"upside": "face up", "downside": "face down"},
    )


# =============================================================================
# building and writing
# =============================================================================


def build_session_metadata(
    session_dir: Path,
    episodes: list[Path],
    size: str,
    face: str,
    arm: str | None,
    task: dict,
    sorting_rules: dict,
) -> dict:
    """Assemble the session record, with one explicit entry per episode found.

    Every episode carries its own copy of the envelope fields rather than a rule
    to derive them from position. A reader never has to know how many episodes a
    cycle had, nor which ones were dropped.
    """
    basket = sorting_rules.get(size)
    envelope = {
        "envelope_size": size,
        "envelope_facing_side": face,
        "destination_basket_side": basket,
    }
    if arm:
        envelope["arm"] = arm

    episode_entries = []
    for ep_dir in episodes:
        sidecar = read_sidecar(ep_dir)
        entry = {"episode_dir": ep_dir.name, **envelope}
        # Carry the recorder's own verdict through, so the session file shows at a
        # glance which episodes the converter will drop.
        if "status" in sidecar:
            entry["recorder_status"] = sidecar["status"]
        if sidecar.get("note"):
            entry["recorder_note"] = sidecar["note"]
        episode_entries.append(entry)

    aborted = [e["episode_dir"] for e in episode_entries if e.get("recorder_status") == "aborted"]

    return {
        "schema_version": 1,
        "metadata_status": "recorded",
        "session": {
            "name": session_dir.name,
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "episode_count": len(episode_entries),
            "aborted_count": len(aborted),
            "aborted_episodes": aborted,
        },
        "task": task,
        "sorting_rules": sorting_rules,
        "envelope": envelope,
        "episodes": episode_entries,
    }


def find_conflicts(episodes: list[Path], envelope: dict) -> list[str]:
    """Episodes already labelled with something different from what we're writing."""
    conflicts = []
    for ep_dir in episodes:
        sidecar = read_sidecar(ep_dir)
        differing = [
            f"{k}={sidecar[k]!r} (would become {envelope[k]!r})"
            for k in ENVELOPE_KEYS
            if k in sidecar and k in envelope and sidecar[k] != envelope[k]
        ]
        if differing:
            conflicts.append(f"{ep_dir.name}: " + ", ".join(differing))
    return conflicts


def stamp_episode(episode_dir: Path, envelope: dict) -> None:
    """Merge the envelope fields into an episode's metadata.json, keeping the rest.

    A missing sidecar is created; an existing one keeps every key it already had
    (``status``, ``note``, ``duration``, ...) — the recorder's verdict is not ours
    to rewrite.
    """
    path = episode_dir / "metadata.json"
    data = read_sidecar(episode_dir)
    data.update(envelope)
    path.write_text(json.dumps(data, indent=2) + "\n")


# =============================================================================
# handoff to the NAS upload flow
# =============================================================================


def find_nas_taxonomy_script(explicit_dir: str | None) -> Path:
    """Locate fill_dataset_taxonomy_nas.py, or explain where to point us.

    A folder named explicitly (--nas-tools, then $ANVIL_NAS_TOOLS) is taken at its
    word: if the script is not there, that is an error rather than a reason to go
    looking elsewhere -- silently running a different copy of it is worse than
    stopping. Only when neither is given do we fall back to the home directory,
    where the data-collection scripts sit alongside _registry/ and _generated/.
    """
    explicit = None
    if explicit_dir:
        explicit = ("--nas-tools", Path(explicit_dir).expanduser())
    elif os.environ.get(NAS_TOOLS_ENV):
        explicit = (f"${NAS_TOOLS_ENV}", Path(os.environ[NAS_TOOLS_ENV]).expanduser())

    source, directory = explicit or ("home directory", Path.home())
    script = directory / NAS_TAXONOMY_SCRIPT
    if script.is_file():
        return script.resolve()

    hint = (
        f"Point at the folder holding it with --nas-tools DIR or {NAS_TOOLS_ENV}=DIR."
        if explicit is None else
        f"Check the path given via {source}."
    )
    raise FileNotFoundError(
        f"--to-nas: no {NAS_TAXONOMY_SCRIPT} in {directory} (from {source}).\n{hint}"
    )


def run_nas_taxonomy(script: Path, session_dir: Path) -> int:
    """Run the taxonomy fill-in for this session, inheriting the terminal.

    It is interactive by design (it asks for use_case/task/setup and confirms),
    so stdin/stdout are passed straight through and its exit code becomes ours.
    """
    print(f"\n=== Continuing into the NAS flow: {script.name} ===")
    # Our own output is block-buffered when piped; flush it so the two scripts'
    # messages stay in the order they happened.
    sys.stdout.flush()
    return subprocess.run([sys.executable, str(script), "--from-label", str(session_dir)]).returncode


# =============================================================================
# CLI
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label-session",
        description=(
            "Stamp a recorded session with its envelope size and facing side. "
            "One session = one envelope configuration; every episode gets the same label."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("session", metavar="PATH", help="Session folder (contains one dir per episode)")
    parser.add_argument(
        "--size", metavar="SIZE",
        help=f"Envelope size {SIZES}; asked interactively when omitted",
    )
    parser.add_argument(
        "--face", metavar="FACE",
        help=f"Facing side {FACES} (up/down also accepted); asked interactively when omitted",
    )
    parser.add_argument(
        "--arm", choices=("left", "right"),
        help="Arm used, when the whole session used one arm (omitted from the record otherwise)",
    )
    parser.add_argument(
        "--task-id", default=DEFAULT_TASK["id"], metavar="ID",
        help=f"Task identifier written to the session record (default: {DEFAULT_TASK['id']})",
    )
    parser.add_argument(
        "--instruction", default=DEFAULT_TASK["instruction"], metavar="TEXT",
        help="Task instruction written to the session record",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite episodes already labelled with a different envelope",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written without touching any file",
    )
    parser.add_argument(
        "--to-nas", action="store_true",
        help=(
            f"After labelling, run {NAS_TAXONOMY_SCRIPT} on this session to build the "
            "NAS taxonomy path and upload config (stops before copying any data)"
        ),
    )
    parser.add_argument(
        "--nas-tools", metavar="DIR",
        help=(
            f"Folder holding {NAS_TAXONOMY_SCRIPT} for --to-nas "
            f"(default: ${NAS_TOOLS_ENV}, else the home directory)"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    session_dir = Path(args.session).resolve()

    # Resolved up front: finding out the script is missing after the operator has
    # answered every prompt would be a poor trade.
    nas_script = None
    if args.to_nas:
        try:
            nas_script = find_nas_taxonomy_script(args.nas_tools)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        episodes = find_episodes(session_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Session {session_dir}")
    print(f"  {len(episodes)} episode(s): {', '.join(e.name for e in episodes[:8])}"
          f"{' ...' if len(episodes) > 8 else ''}")

    # Non-interactive contexts must pass the flags rather than hang on input().
    interactive = sys.stdin.isatty()
    if (args.size is None or args.face is None) and not interactive:
        print(
            "Error: no terminal to ask on — pass --size and --face explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        size = _norm(args.size, _SIZE_ALIASES, "size") if args.size else ask_size()
        raw_face = _norm(args.face, _FACE_ALIASES, "face") if args.face else None
        face = _CANONICAL_FACE[raw_face] if raw_face else ask_face()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    task = {**DEFAULT_TASK, "id": args.task_id, "instruction": args.instruction}
    record = build_session_metadata(
        session_dir, episodes, size, face, args.arm, task, DEFAULT_SORTING_RULES
    )
    envelope = record["envelope"]

    print(f"\n  envelope size          : {size}")
    print(f"  facing side            : {face}")
    print(f"  destination basket     : {envelope['destination_basket_side']}")
    if args.arm:
        print(f"  arm                    : {args.arm}")
    if record["session"]["aborted_count"]:
        print(
            f"  recorder marked aborted: {record['session']['aborted_count']} "
            f"({', '.join(record['session']['aborted_episodes'])}) "
            "— labelled anyway; the converter skips them by default"
        )

    conflicts = find_conflicts(episodes, envelope)
    if conflicts and not args.force:
        print(
            f"\nError: {len(conflicts)} episode(s) already carry a different label:",
            file=sys.stderr,
        )
        for c in conflicts[:10]:
            print(f"  - {c}", file=sys.stderr)
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more", file=sys.stderr)
        print(
            "\nThis session looks like it was labelled before with something else. "
            "Check it is the folder you meant, then re-run with --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)
    if conflicts:
        print(f"\n  --force: overwriting {len(conflicts)} differing label(s)")

    session_file = session_dir / "session_metadata.json"
    if args.dry_run:
        print(f"\n--dry-run: would write {session_file}")
        print(f"--dry-run: would stamp {len(episodes)} episode metadata.json file(s)")
        if nas_script:
            print(f"--dry-run: would then run {nas_script} --from-label {session_dir}")
        print("\nSession record that would be written:\n")
        print(json.dumps(record, indent=2))
        return

    session_file.write_text(json.dumps(record, indent=2) + "\n")
    for ep_dir in episodes:
        stamp_episode(ep_dir, envelope)

    print(f"\nWrote {session_file}")
    print(f"Stamped {len(episodes)} episode metadata.json file(s)")

    if nas_script:
        sys.exit(run_nas_taxonomy(nas_script, session_dir))


if __name__ == "__main__":
    main()
