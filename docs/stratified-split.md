[← Back to README](../README.md)

# Stratified Episode Split

`stratified-split` builds a train/val/test split **within each envelope group** — size (`big` / `medium` / `small`) × face (`face_up` / `face_down`) — instead of shuffling every episode together. The result is a `split_info.json`-shaped file that `anvil-trainer` consumes with `--split-file`.

## Contents

- [Why](#why)
- [How it works](#how-it-works)
- [Labels](#labels)
  - [From episode metadata](#from-episode-metadata)
  - [From a file](#from-a-file)
  - [Accepted values](#accepted-values)
  - [Coverage and unlabelled episodes](#coverage-and-unlabelled-episodes)
- [Usage](#usage)
- [Output file](#output-file)
- [Training with the split](#training-with-the-split)
- [Datasets without labels in metadata](#datasets-without-labels-in-metadata)
- [Worked example: envelope-all](#worked-example-envelope-all)
- [Status and known gaps](#status-and-known-gaps)

---

## Why

The trainer's default split (`anvil_shared.splits.compute_split_episodes`) shuffles all episodes at once. The share of any sub-population in val and test — say, small face-down envelopes — is then only right *on average*, and a small stratum can be missing from val or test entirely.

Splitting inside each stratum makes every split mirror the dataset's composition, so val/test losses are comparable across strata and across runs.

## How it works

The logic lives in `anvil_shared.stratified`; the CLI is `mcap_converter.cli.stratified_split`.

1. **6 strata.** Each episode gets exactly one key, `"<size>|<face>"`, e.g. `big|face_up`. An episode labelled twice with different values is rejected.
2. **Ratio inside each stratum.** The default is `8,1,1`. For a stratum of `n` episodes: `n_test = round(n·test/Σ)`, `n_val = round(n·val/Σ)`, and train gets the remainder. This is the same rounding as the default split, so the two strategies stay comparable. Two values (`8,2`) mean no test set.
3. **One seed per stratum.** Each stratum is shuffled with its own RNG seeded by `sha256(f"{seed}:{stratum}")`. Adding episodes to one stratum only reshuffles *that* stratum; the others keep their assignment. `hashlib` is used rather than `hash()`, which is salted per process.
4. **Disjoint by construction.** Strata are disjoint and each episode lands in exactly one split. Output lists are sorted.
5. **Small strata are reported.** If a stratum is too small to reach val or test, the tool warns and the report flags it.

Same labels + same ratio + same seed → the same split, byte for byte.

## Labels

### From episode metadata

By default the labels are read from extra columns in `meta/episodes/*.parquet`. The first match wins:

| | Columns looked for, in order |
|---|---|
| size | `envelope_size`, `size`, `envelope` |
| face | `envelope_face`, `envelope_facing_side`, `facing_side`, `face`, `orientation`, `envelope_orientation` |

Override with `--size-column` / `--face-column`.

These columns survive `merge-datasets`: LeRobot's aggregation rewrites only the columns it knows about and offsets `episode_index`, so per-session labels stay attached to the right episodes. Episodes with an empty value (never labelled by the recorder) are treated as unlabelled.

### From a file

`--labels PATH` supplies labels by hand. All formats are keyed by the `episode_index` **of the dataset being split** — remember that `merge-datasets` renumbers episodes.

**Grouped JSON** — size → face → episode list:

```json
{
  "big":    {"upside": [0, 3, 7], "downside": [1, 2]},
  "medium": {"upside": [4, 5],    "downside": [6]},
  "small":  {"upside": [8],       "downside": [9]}
}
```

**Flat JSON** — one entry per episode, in any of three shapes:

```json
{
  "0": {"envelope_size": "big", "envelope_facing_side": "upside"},
  "1": ["big", "downside"],
  "2": "small|face_up"
}
```

**CSV** — a header plus one row per episode:

```csv
episode_index,envelope_size,envelope_facing_side
0,big,upside
```

A JSON file is read as grouped only if every top-level key is a size and maps to an object; otherwise it is read as flat.

### Accepted values

Values are case-insensitive and normalised, so the recording protocol's vocabulary works as-is:

| Canonical | Also accepted |
|---|---|
| `big` | `large`, `l`, `b`, `grande` |
| `medium` | `med`, `m`, `medio`, `médio` |
| `small` | `s`, `pequeno` |
| `face_up` | `upside`, `up`, `faceup`, `face-up`, `u`, `cima`, `true` |
| `face_down` | `downside`, `down`, `facedown`, `face-down`, `d`, `baixo`, `false` |

Anything else is an error that lists the accepted values.

### Coverage and unlabelled episodes

Before splitting, the tool checks the labels against `meta/info.json`:

- A labelled index outside `0..total_episodes-1` is an **error**. This usually means the labels were written against the per-session datasets rather than the merged one.
- Unlabelled episodes are an **error**, unless you pass `--allow-partial`. With it, those episodes are left out of all three splits and are used by nothing at all.

## Usage

```bash
# labels already in meta/episodes/*.parquet
uv run stratified-split /path/to/dataset

# preview the split without writing anything
uv run stratified-split /path/to/dataset --dry-run

# labels from a file; some episodes unlabelled
uv run stratified-split /path/to/dataset --labels envelopes.json --allow-partial
```

| Option | Default | |
|---|---|---|
| `PATH` (positional) | — | Root of the LeRobot dataset |
| `--labels PATH` | read from `meta/episodes` | JSON or CSV with per-episode size + face |
| `--split-ratio` | `8,1,1` | Ratio applied *within* each stratum; two values = no test set |
| `--seed N` | `42` | Base seed; each stratum derives its own |
| `--output PATH` | `<dataset>/stratified_split_info.json` | Where to write the split |
| `--size-column NAME` | auto | Metadata column holding the size |
| `--face-column NAME` | auto | Metadata column holding the face |
| `--allow-partial` | off | Allow unlabelled episodes; they are excluded from every split |
| `--dry-run` | off | Print the split and exit without writing |

The CLI prints a per-stratum table (total / train / val / test, plus overall percentages) and ends with the `--split-file=…` flag to pass to training.

Tests:

```bash
uv run pytest tests/unit/anvil_shared/test_stratified.py tests/unit/mcap_converter/test_stratified_split_cli.py
```

## Output file

```json
{
  "strategy": "stratified",
  "stratified_by": ["size", "face"],
  "split_ratio": [8.0, 1.0, 1.0],
  "seed": 42,
  "total_episodes": 5494,
  "labels_source": "meta/episodes",
  "train_episodes": [0, 1, 3, "…"],
  "val_episodes": ["…"],
  "test_episodes": ["…"],
  "strata": {
    "big|face_down": {"total": 896, "train": 716, "val": 90, "test": 90},
    "…": {}
  }
}
```

`labels_source` is `meta/episodes` or the absolute path of the `--labels` file. Before writing, the tool re-validates the split: indices are integers, in range, disjoint, and train is non-empty.

## Training with the split

```bash
uv run anvil-trainer \
  --dataset.root=/path/to/dataset \
  --split-file=/path/to/dataset/stratified_split_info.json \
  ...
```

- `--split-file` replaces the random split; `--split-ratio` and `--max-episodes` are then ignored.
- The file is **validated against the dataset** before use. A stale split from a smaller dataset is rejected instead of being silently misapplied.
- On `--resume`, the checkpoint's own split wins. Swapping splits mid-run would leak held-out episodes into training.
- Each checkpoint saves the split it trained on as `pretrained_model/split_info.json`.

**Order with `prepare_trainready_dataset.py`.** Write the split *into the source dataset before* running the train-ready preparation. The preparation copies the whole dataset and certifies every non-statistics file as immutable, so `stratified_split_info.json` travels inside the certified `-trainready` copy. The Pi0.5 run scripts look for it there by default (`SPLIT_FILE=${DATASET_ROOT}/stratified_split_info.json`).

## Datasets without labels in metadata

Some datasets are converted without the envelope columns. The teleop causal-command dataset (2457 episodes, September 2026) is one of them: its labels were transferred from the AFO conversion of the same recordings, keyed by `session#episode`, not by merged `episode_index`.

That run used a one-off script, `make_split_teleop.py`, that lived outside the repo. It reads the transferred labels, proves the merged dataset is the exact concatenation of its sessions in merge order, and writes a file with the same schema.

⚠️ **It is not interchangeable with `stratified-split`.** It shuffles all strata with a single `Random(42)` in sequence, instead of one seed per stratum. The same labels give a *different* split from the CLI, and adding episodes to one stratum reshuffles all the strata after it.

For new datasets in this situation, the recommended route is to produce a flat `--labels` JSON keyed by the merged `episode_index` and run `stratified-split` on it. That keeps one splitting algorithm and its guarantees.

## Worked example: envelope-all

`envelope-all`: 5494 episodes, 5482 labelled, 12 left out with `--allow-partial`. Labels come from `meta/episodes` (`envelope_size`, `envelope_facing_side`). Ratio `8,1,1`, seed 42.

| Stratum | Total | Train | Val | Test |
|---|---|---|---|---|
| big / face_down | 896 | 716 | 90 | 90 |
| big / face_up | 1042 | 834 | 104 | 104 |
| medium / face_down | 813 | 651 | 81 | 81 |
| medium / face_up | 868 | 694 | 87 | 87 |
| small / face_down | 820 | 656 | 82 | 82 |
| small / face_up | 1043 | 835 | 104 | 104 |
| **all** | **5482** | **4386** | **548** | **548** |

This split fed the `strat5482` and `strat5482-aug` Pi0.5 runs. Re-running the algorithm on the same dataset reproduces all three episode lists exactly.

## Status and known gaps

- **Not on `main` yet.** The CLI and `anvil_shared.stratified` arrived in `0cf1b70`, and the trainer's `--split-file` in `e56a391`. Both are only on `feat/envelope-episode-labels`. They need to land together: without `--split-file`, `main` cannot consume the file.
- **Possible duplicate flag.** `e56a391` notes that a `--split-manifest` flag doing the same job exists on another branch, and that the Pi0.5 recipes used that name. Pick one before merging.
- **Augmentation still reaches val/test.** Image augmentation is configured once, for all three splits. The runs above used an uncommitted `patches.py` guard (`ANVIL_AUG_TRAIN_ONLY=1`) that turns it off for val and test. Without it, a stratified split still gets jittered evaluation frames.
