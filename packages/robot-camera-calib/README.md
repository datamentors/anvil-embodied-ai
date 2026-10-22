# robot-camera-calib

Camera **intrinsics** calibration (fx, fy, cx, cy, k1, k2, p1, p2, k3) for any robot
described by a YAML file in [`robots/`](robots/). Supported today:

| Robot | Config | Frames from | Result format |
|---|---|---|---|
| AgiBot G2 (head, hand_left, hand_right) | [`robots/g2.yaml`](robots/g2.yaml) | AgiBot GDK | AgiBot `intrinsic_*.json` |
| Anvil workcell (cam_chest, cam_wrist_l, cam_wrist_r) | [`robots/anvil.yaml`](robots/anvil.yaml) | ROS 2 via rosbridge | ROS `camera_info` YAML |

Adding another robot is a YAML file: see
[`robot_camera_calib/README.md`](robot_camera_calib/README.md).

- **Read-only.** The tool only reads frames and joint/TF state. It never commands
  motion, never changes camera or control settings and never writes the robot's
  calibration, so it is safe while someone teleoperates.
- **Results are candidates.** They are written into the run folder only.
  Installing them is a separate, manual decision (see the end).

## Where calibration files live

| What | Default location | Format |
|---|---|---|
| Capture runs | `runs_dir` of the robot's YAML: G2 `/data/dercio-build/g2_camera_calib/runs/<run>/`, Anvil `runs/<run>/` in this package | images + `meta.json` per sample |
| New calibration (solve) | `<run>/results/` | G2 `intrinsic_*.json`, ROS `<cam>.yaml` |
| Kept calibrations | [`calibrations/<robot serial>_<date>_<run>/`](calibrations/) in this package | as `results/` |
| Robot's live calibration (G2) | `/data/parameters/sensor/intrinsic_*.json` on the robot | AgiBot JSON |
| Raw datasets (S3) | `s3://ardia-ai-datasets-dev/raw/<use_case>/<task>/<embodiment>/<setup>/<date>/<session>/<episode>/files/data/parameters/sensor/intrinsic_*.json` | AgiBot JSON, as recorded |
| Processed datasets (NAS) | `/mnt/nas/processed/<use_case>/<task>/<embodiment>/<setup>/<date>/<session>/meta/camera_calibration.json` | `{video_key: 3x3 matrix}` |

Example `<use_case>/<task>/<embodiment>/<setup>/<date>/<session>`:
`cloth_folding/fold_towel/agibot_g2/g2a0004-infpose-headset/2026-10-21/s01`. Full file
examples: [Calibration in recorded datasets](#calibration-in-recorded-datasets).

## Requirements

Python 3.10+, OpenCV >= 4.7 with `aruco` (`CharucoDetector`), NumPy, PyYAML, and
`websockets` for rosbridge robots. The G2's system `python3` already has them. On
any other host, `./run.sh` pulls them through `uv` (nothing is installed system-wide).
No `uv` yet: `curl -LsSf https://astral.sh/uv/install.sh | sh` (installs to
`~/.local/bin/uv`, which `run.sh` also looks for). This package is not a member of the
repo's uv workspace; `uv sync` at the repo root does not install it.

## 1. Board

Print a board at **100% / actual size**, glue it flat onto something rigid (a wavy
board is the biggest error source), then measure one square edge.

```bash
python3 -m robot_camera_calib.capture --print-board board.png --board charuco:10x8:0.030:0.015:4x4_50:legacy
```

Spec format, lengths in metres:
- `charuco:<squares_x>x<squares_y>:<square>:<marker>:<dict>[:legacy]`
- `chessboard:<inner_x>x<inner_y>:<square>`: every corner must be visible in every frame.

The default is the lab board: 10x8 squares, `DICT_4X4_50`, marker = half the square,
**legacy layout** (calib.io style). Without `:legacy` that board is not detected at all.

## 2. Capture (on the robot)

```bash
./sync.sh push                     # from your machine (TARGET=anvil for Anvil)
ssh <robot> && cd ~/anvil-embodied-ai/packages/robot-camera-calib   # where push put it

python3 -m robot_camera_calib.capture --robot g2 --run intr01 --board <measured spec> --web
./run.sh capture --robot anvil --run anvil01 --board <measured spec> --web     # ROS hosts
```

Move the board in front of the cameras and **pause about 1 s at each position**. The
tool takes the photos itself. Every tick (0.3 s) it saves an image for a camera when:
- the robot did not move over the last two ticks,
- the board did not move since the last tick (no blur), and
- the board is somewhere new for that camera (>= 3 cm or >= 7 degrees from saved views).

For good coverage, get the board into every part of each image (corners, edges,
centre), near and far, tilted 20-45 degrees in different directions. Aim for 20-40
photos per camera; solve needs >= 12 views and >= 40% coverage.

Stop with Ctrl-C, `--duration-s` or `--max-samples`. If you get `board_moving` a lot,
pause longer or use `--board-still-px 3`. Too few photos: lower `--min-move-m` or
`--min-rot-deg`. `--cams head,hand_left` captures a subset.

### Live view

`--web` serves a live page on the robot's localhost. Open it from your machine with
`./sync.sh view` (SSH tunnel to http://localhost:9701). Each tile shows the detected
corners, the image areas already covered (green cells) and a border colour: green
saved, yellow same view, orange board moving, red robot moving, grey no board.

## 3. Solve (robot or your machine)

```bash
python3 -m robot_camera_calib.solve <runs_dir>/intr01            # on the robot
./sync.sh pull intr01 && ./run.sh solve runs/intr01               # on your machine
#   runs_dir: robots/<robot>.yaml (relative = inside this package)
#   --board SPEC   override the board size stored at capture time
#   --visual       also write <cam>_visual.jpg check pictures
```

`<run>/results/` gets `report.md`, `report.json` and per camera either
`intrinsic_*.json` (AgiBot format and names) or `<cam>.yaml` (ROS `camera_info`).

Reading the report, per camera:
- **Reprojection RMS on these frames:** how well each set of values explains your
  photos. `new` is fitted on them, so it always wins a little.
- **Hold-out** (>= 24 views): fit on even photos, test on odd ones. This is the fair
  comparison. Factory within about 0.1-0.2 px of new means the factory values are fine.
- **Coverage** below 40%: the new values are not trustworthy (`recommended: NO`).

## Self-test (no robot needed)

```bash
python3 -m robot_camera_calib.selftest      # or ./run.sh selftest
```

## Robot notes

- **G2:** run with the system `python3` (GDK is linked at start-up). Runs go to
  `/data/dercio-build/g2_camera_calib/runs`. `--factory-only` just records the live
  factory values. The kept calibration is in [`calibrations/`](calibrations/).
- **Anvil:** run on the host, not in a container. Runs go to `runs/` in this package. `usb_cam` publishes a dummy
  `camera_info`, so there is no factory comparison. Intrinsics are only valid at the
  focus they were taken at; `run.json` records each camera's v4l2 focus, and capture
  warns if autofocus is on.

## Installing results (manual, on purpose)

- **G2:** GDK reads `/data/parameters/sensor/intrinsic_*.json` (root-owned, written by
  AgiBot's calibration app). Back up the directory, confirm with AgiBot that a hot swap
  is supported, and replace only cameras whose hold-out shows a real improvement. The
  safer default is to have our own code load `results/*.json` directly.
- **ROS robots:** copy `results/<cam>.yaml` where the camera driver can read it and
  point its `camera_info_url` at it (`file:///...`).

## Calibration in recorded datasets

### Raw (S3, as recorded)

The robot's own calibration files travel with every episode, unchanged. For the G2 the
recorder snapshots `/data/parameters/sensor` into each episode, and `g2_ingest` uploads
the episode byte-for-byte:

```
s3://ardia-ai-datasets-dev/raw/<use_case>/<task>/<embodiment>/<setup>/<date>/<session>/<episode>/files/data/parameters/sensor/
    intrinsic_head_front_rgb.json     {"Fx", "Fy", "Cx", "Cy", "k1", "k2", "p1", "p2", "k3", "SN"}
    intrinsic_hand_left_rgb.json
    intrinsic_hand_right_rgb.json
    ...                               (the robot's other sensor files)
```

Example prefix: `raw/cloth_folding/fold_towel/agibot_g2/g2a0004-infpose-headset/2026-10-21/s01/`.
Raw frames are the camera's real (distorted) images, so they go with these K and D.
A complete `intrinsic_hand_left_rgb.json` (G2A0004 factory values):

```json
{"Fx": 479.7045091050678, "Fy": 479.4579738879909, "Cx": 640.495468237256, "Cy": 524.096396833031,
 "k1": -0.04784519575146247, "k2": -0.010293807152412444, "p1": -9.44150220740944e-05,
 "p2": 6.984914892529563e-05, "k3": 0.0019575327156301317, "SN": "FD63711400004af3"}
```

### Processed (LeRobot datasets)

Processed frames are **undistorted**: each frame is remapped with its episode's K and D
to an ideal pinhole camera, so the dataset only needs that camera's matrix, with zero
distortion. It goes in one file per dataset, next to LeRobot's `meta/info.json`:

```
<dataset>/meta/camera_calibration.json
```

For the G2 the dataset root is the same path on the NAS (`//192.168.1.3/Datasets`,
mounted at `/mnt/nas` on the workstation):

```
/mnt/nas/processed/<use_case>/<task>/<embodiment>/<setup>/<date>/<session>/meta/camera_calibration.json
```

The file maps each video key to its 3x3 camera matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
in pixels, and nothing else. A complete file, AgiBot G2 from its factory calibration
(`/mnt/nas/processed/cloth_folding/fold_towel/agibot_g2/g2a0004-infpose-headset/2026-10-21/s01/meta/camera_calibration.json`):

```json
{
  "observation.images.top_head":   [[302.33, 0.0, 316.46], [0.0, 302.59, 203.54], [0.0, 0.0, 1.0]],
  "observation.images.hand_left":  [[407.46, 0.0, 641.44], [0.0, 438.84, 522.97], [0.0, 0.0, 1.0]],
  "observation.images.hand_right": [[411.26, 0.0, 644.18], [0.0, 441.83, 530.21], [0.0, 0.0, 1.0]]
}
```

```python
import json, numpy as np
K = np.array(json.load(open("<dataset>/meta/camera_calibration.json"))["observation.images.hand_left"])
```

How the stored frames were made from the raw ones:
`K_new = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)` (same size, no black
border), then `cv2.undistort(raw, K, D, newCameraMatrix=K_new)`; `K_new` is what the file
stores. A policy trained on these frames needs the same remap on the robot's live images.

One matrix per camera holds for the whole dataset: if the robot's calibration changed
between episodes, the converter stops and that session has to be split. Datasets with
distorted frames get no `camera_calibration.json`.

**Another robot:** undistort with its own K and D (for a ROS camera, `camera_matrix.data`
and `distortion_coefficients.data` of its `camera_info`) and write this same file.

## Files

| Path | What |
|---|---|
| `robot_camera_calib/` | the tool (see its README for the code layout) |
| `robots/*.yaml` | one config per robot |
| `calibrations/` | kept results |
| `run.sh` | runs the tool with uv-provided dependencies |
| `sync.sh` | push code to a robot (`~/<repo>/<path in repo>`, here `~/anvil-embodied-ai/packages/robot-camera-calib`), pull runs, tunnel the live view |
| `ruff.toml` | lint config (repo rules + Google docstrings) |
