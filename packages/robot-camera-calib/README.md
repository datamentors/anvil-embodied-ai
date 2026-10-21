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

## Calibration in processed datasets

Raw recordings keep the robot's own calibration files next to the images. When a
recording is converted to a LeRobot dataset with **undistorted** frames, the dataset
must say exactly which calibration was used. Every robot's converter writes the same
file for that:

```
<dataset>/meta/camera_calibration.json
```

It sits next to LeRobot's own `meta/info.json`. Matrices are OpenCV-shaped (3x3 `K`,
5-element `D` = `[k1, k2, p1, p2, k3]`), so they load straight into `np.array(...)`.

**Per camera** (keyed by the dataset's video key, e.g. `observation.images.hand_left`):

| Field | Meaning |
|---|---|
| `camera_matrix` | K of the **raw** camera: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, pixels |
| `dist_coeffs` | D of the **raw** camera: `[k1, k2, p1, p2, k3]` (OpenCV plumb_bob) |
| `rectified_camera_matrix` | K of the **stored, undistorted** frames; their distortion is zero |
| `width`, `height` | frame size in pixels; raw and undistorted are the same size |
| `source_camera` | the robot's camera name (G2: `hand_left_color`) |
| `vendor_file` | the robot calibration file it was read from (G2: `intrinsic_hand_left_rgb.json`) |
| `serial` | camera serial number from that file, if it has one |
| `vendor_values` | that file, copied verbatim |
| `sha256` | hash of that file, proving exactly which calibration was used |

**Top level:**

| Field | Meaning |
|---|---|
| `format` | layout version, currently `g2_camera_calibration/v1` |
| `frames` | `undistorted ...` or `distorted ...`: which K applies to the stored videos |
| `model`, `source`, `usage` | plain-text notes: the lens model, where the values came from, how to reproduce the frames |
| `cameras` | the per-camera block above, when every episode used the same calibration (the normal case); `null` otherwise |
| `calibrations` | every distinct calibration set, keyed by a short id (one entry normally, equal to `cameras`) |
| `episodes` | per dataset episode: `episode_index`, `source_episode` (raw recording name) and the `calibration` id it used |

How the stored frames were made, for every pixel:
`cv2.undistort(raw, camera_matrix, dist_coeffs, newCameraMatrix=rectified_camera_matrix)`,
with `rectified_camera_matrix = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)` (same
size, no black border).

Example: AgiBot G2, one camera shown, numbers rounded. In the G2 pipeline the values come
from `<episode>/files/data/parameters/sensor/intrinsic_*.json`, the snapshot of
`/data/parameters/sensor` the recorder stores with every episode.

```json
{
  "format": "g2_camera_calibration/v1",
  "frames": "undistorted -- ... use rectified_camera_matrix with zero distortion for these frames",
  "cameras": {
    "observation.images.hand_left": {
      "source_camera": "hand_left_color",
      "vendor_file": "intrinsic_hand_left_rgb.json",
      "serial": "FD63711400004af3",
      "width": 1280, "height": 1056,
      "camera_matrix": [[479.70, 0.0, 640.50], [0.0, 479.46, 524.10], [0.0, 0.0, 1.0]],
      "dist_coeffs": [-0.04785, -0.01029, -0.0000944, 0.0000698, 0.00196],
      "rectified_camera_matrix": [[407.46, 0.0, 641.44], [0.0, 438.84, 522.97], [0.0, 0.0, 1.0]],
      "vendor_values": {"Fx": 479.70, "Fy": 479.46, "Cx": 640.50, "Cy": 524.10, "k1": -0.04785, "...": "...", "SN": "FD63711400004af3"},
      "sha256": "3b5726da..."
    }
  },
  "calibrations": {"26801bfc28a9": "<same as cameras>"},
  "episodes": [{"episode_index": 0, "source_episode": "folding_towels(infPose)014_001", "calibration": "26801bfc28a9"}]
}
```

Reading it back:

```python
import json, numpy as np
cam = json.load(open("<dataset>/meta/camera_calibration.json"))["cameras"]["observation.images.hand_left"]
K_raw, D_raw = np.array(cam["camera_matrix"]), np.array(cam["dist_coeffs"])  # raw images
K = np.array(cam["rectified_camera_matrix"])                                 # dataset frames, no distortion
```

If `cameras` is `null` (the robot was recalibrated mid-session), look up the episode in
`episodes` and read `calibrations[<id>]` instead.

Adding a robot: map each converted camera to its calibration file and read K and D from
it. A ROS `camera_info` YAML gives `K` (`camera_matrix.data`) and `D` (`distortion_coefficients.data`)
directly. Then write this same file. Keep `frames` truthful: a dataset whose videos were
not undistorted says `distorted` and is used with `camera_matrix` + `dist_coeffs`.

## Files

| Path | What |
|---|---|
| `robot_camera_calib/` | the tool (see its README for the code layout) |
| `robots/*.yaml` | one config per robot |
| `calibrations/` | kept results |
| `run.sh` | runs the tool with uv-provided dependencies |
| `sync.sh` | push code to a robot (`~/<repo>/<path in repo>`, here `~/anvil-embodied-ai/packages/robot-camera-calib`), pull runs, tunnel the live view |
| `ruff.toml` | lint config (repo rules + Google docstrings) |
