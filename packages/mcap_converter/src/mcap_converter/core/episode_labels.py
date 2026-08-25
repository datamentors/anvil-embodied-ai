"""Carry per-episode recorder labels through conversion into the dataset.

The recorder writes a sidecar ``metadata.json`` next to each episode's MCAP.
``label-session`` adds the envelope characteristics to it.  This module lifts
those keys into the converted dataset's ``meta/episodes/*.parquet`` as ordinary
columns, so a training-time question like "which episodes hold big face-up
envelopes?" is answered from the dataset itself.

Why columns and not a sidecar keyed by index: the converter skips episodes
flagged critical (aborted recordings among them), so raw folder numbering and
LeRobot episode indices drift apart.  A column is written on the episode's own
row and cannot drift.  It also survives ``merge-datasets`` — LeRobot's
aggregation rewrites only the columns it knows and offsets ``episode_index``,
leaving unknown columns attached to the right episode.

Two provenance columns are added alongside the labels (``source_episode_dir``,
``source_mcap``).  The converter already knew which MCAP produced which episode
but only printed it in a summary table; recording it makes the mapping
recoverable from the dataset afterwards.

**Every episode in a run must receive the same set of keys.**  LeRobot buffers
episode metadata and flushes it through ``pa.Table.from_pydict``; a key present
for some episodes and absent for others raises ``ArrowInvalid: Column ... expected
length N but got length M`` at flush time.  Missing values are therefore written
as empty strings, and :func:`resolve_label_keys` fixes the key set once per run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Keys lifted from the recorder sidecar when present. Anything else in
# metadata.json (status, note, duration, ...) stays out of the dataset.
ENVELOPE_KEYS = (
    "envelope_size",
    "envelope_facing_side",
    "destination_basket_side",
    "arm",
)

# Always recorded, so the converted episode can be traced back to its recording.
SOURCE_KEYS = ("source_episode_dir", "source_mcap")

# Written where an episode has no value for a key that other episodes do have.
MISSING = ""


def read_sidecar(mcap_path: str | Path) -> dict:
    """Read the recorder's metadata.json for an episode, or {} when unusable.

    A missing or corrupt sidecar is not an error: plenty of valid recordings
    have none. Mirrors ``quality._check_recorder_aborted_status``.
    """
    path = Path(mcap_path).parent / "metadata.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def source_columns(mcap_path: str | Path) -> dict[str, str]:
    """Provenance for one episode: the directory it came from and its MCAP name."""
    p = Path(mcap_path)
    return {"source_episode_dir": p.parent.name, "source_mcap": p.name}


def read_episode_labels(mcap_path: str | Path, keys: tuple[str, ...]) -> dict[str, str]:
    """Values for ``keys`` for one episode, as strings, missing ones omitted."""
    sidecar = read_sidecar(mcap_path)
    source = source_columns(mcap_path)
    out: dict[str, str] = {}
    for key in keys:
        if key in source:
            out[key] = source[key]
        elif sidecar.get(key) is not None:
            out[key] = str(sidecar[key])
    return out


def _existing_label_keys(dataset_root: str | Path, candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Label columns already present in a dataset being resumed.

    Resuming appends to the same parquet, so the key set has to match what the
    earlier run wrote — including the case of a dataset converted before this
    feature existed, which has none.
    """
    episodes_dir = Path(dataset_root) / "meta" / "episodes"
    files = sorted(episodes_dir.rglob("*.parquet"))
    if not files:
        return ()
    try:
        import pyarrow.parquet as pq

        existing = set(pq.read_schema(files[0]).names)
    except Exception as e:  # pragma: no cover — unreadable schema, fall back to none
        log.warning("[episode-labels] could not read existing schema (%s); adding no columns", e)
        return ()
    return tuple(k for k in candidates if k in existing)


def resolve_label_keys(
    mcap_files: list,
    dataset_root: str | Path,
    resume_from: int = 0,
    extra_keys: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Decide, once per run, which extra columns every episode will carry.

    On resume the answer comes from the dataset already on disk. Otherwise it is
    the provenance columns plus whichever label keys at least one episode
    actually has — a session with no labels gets no empty columns.
    """
    candidates = SOURCE_KEYS + ENVELOPE_KEYS + tuple(extra_keys)

    if resume_from > 0:
        keys = _existing_label_keys(dataset_root, candidates)
        log.info("[episode-labels] resuming: reusing existing columns %s", list(keys))
        return keys

    present = {k for k in SOURCE_KEYS}
    for mcap_path in mcap_files:
        sidecar = read_sidecar(mcap_path)
        present.update(k for k in candidates if sidecar.get(k) is not None)

    return tuple(k for k in candidates if k in present)


def build_label_table(
    mcap_files: list,
    keys: tuple[str, ...],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Per-MCAP label dicts, every one carrying exactly ``keys``.

    Returns the table (indexed by ``str(path)``) and the list of episodes that
    were missing at least one label, so the caller can warn about a session that
    was only partly labelled.
    """
    table: dict[str, dict[str, str]] = {}
    incomplete: list[str] = []

    label_keys = [k for k in keys if k not in SOURCE_KEYS]
    for mcap_path in mcap_files:
        found = read_episode_labels(mcap_path, keys)
        row = {k: found.get(k, MISSING) for k in keys}
        if any(row[k] == MISSING for k in label_keys):
            incomplete.append(Path(mcap_path).parent.name)
        table[str(mcap_path)] = row

    return table, incomplete


def install_episode_metadata_injector(dataset) -> dict[str, str]:
    """Make extra keys land in ``meta/episodes`` and return the dict that holds them.

    LeRobot merges the ``episode_metadata`` argument straight into the episode
    row (``episode_dict.update(episode_metadata)``), so wrapping that one call is
    enough — no changes to the library. The patch is set on the metadata instance
    rather than the class, so it dies with this dataset object.

    Mutate the returned dict before each ``save_episode()``; whatever it holds at
    that moment is written on that episode's row.
    """
    meta = dataset.meta
    original = meta.save_episode
    extra: dict[str, str] = {}

    def save_episode_with_labels(
        episode_index, episode_length, episode_tasks, episode_stats, episode_metadata
    ):
        return original(
            episode_index,
            episode_length,
            episode_tasks,
            episode_stats,
            {**episode_metadata, **extra},
        )

    meta.save_episode = save_episode_with_labels
    return extra
