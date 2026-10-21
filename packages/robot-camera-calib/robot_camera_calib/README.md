# Adding a robot

Everything robot-specific lives in one YAML file under [`../robots/`](../robots/): which
topics to read, how to tell the robot is still, where to write runs and in which format.
The capture loop, the solver and the live view are shared, so a new robot normally
needs **no code**.

## Can my robot use the existing sources?

| Your robot has | Use |
|---|---|
| ROS 2 with `rosbridge_server` running | `source.type: rosbridge`, copy [`robots/anvil.yaml`](../robots/anvil.yaml) |
| AgiBot GDK (G2 family) | `source.type: gdk`, copy [`robots/g2.yaml`](../robots/g2.yaml) |
| Anything else (a vendor SDK, no rosbridge) | write a new source, see [the end](#adding-a-new-source-type) |

For ROS 2, **topic names are free**: you write them in the config. What is fixed is the
message **type**, because the source has to decode it:

- each camera topic must publish `sensor_msgs/msg/CompressedImage` with JPEG data
  (`image_transport`'s `.../compressed` topics do). Raw `sensor_msgs/msg/Image` is not
  supported.
- the stillness topic, if you use one, must publish `sensor_msgs/msg/JointState`.

No rosbridge yet? Start it next to the robot's ROS stack:
`ros2 launch rosbridge_server rosbridge_websocket_launch.xml` (port 9090).

### Why a joint-state topic?

It is the "robot is still" gate. A camera on an arm or head moves with it, and a frame
grabbed while it moves is blurred and taken mid-motion, so capture only saves when no
joint moved over the last two ticks. For cameras that never move (fixed to a frame or
a table), or a robot with no joint-state topic, set `stillness: {type: none}`: the
board-still check (corners must not move between ticks) still rejects blurred frames.

## 1. Write the config

`robots/<name>.yaml`. Every key, for a ROS 2 robot:

```yaml
robot: myrobot                      # name shown in reports
source:
  type: rosbridge                   # gdk | rosbridge
  url: ws://127.0.0.1:9090          # rosbridge websocket, as seen from where capture runs
  throttle_ms: 200                  # optional: at most one frame per camera every N ms
stillness:                          # how "robot is still" is judged
  type: joint_states                # joint_states | none   (gdk robots: tf | none)
  topic: /my_robot/joint_states     # any name; type sensor_msgs/msg/JointState
  still_rad: 0.002                  # max joint change between two ticks
output: ros_camera_info             # ros_camera_info -> results/<cam>.yaml
                                    # agibot_json     -> results/<vendor_intrinsic>
factory_note: "no factory intrinsics"   # optional, recorded in factory.json
runs_dir: runs                      # relative = inside this package; or an absolute path
max_samples: 120                    # optional (default 80)

cameras:                            # name -> settings; the name is used in file names
  front:
    topic: /front_camera/image_raw/compressed   # any name; type CompressedImage (JPEG)
    v4l2_device: /dev/video0        # optional: record focus/exposure, warn on autofocus
  wrist:
    topic: /arm/wrist_cam/compressed
```

Without a joint-state topic the stillness block is just `stillness: {type: none}`.

GDK robots use instead, per camera: `gdk` (the `CameraType` enum name), `parent`
(the TF frame the camera rides on; stillness watches it) and `vendor_intrinsic` (the
vendor file name, also used for the result). The source takes `image_timeout_ms` and
`vendor_param_dir`; `tf` stillness takes `still_mm` and `still_deg`.

To find your topics: `ros2 topic list -t | grep -E 'CompressedImage|JointState'`, and
check a camera sends JPEG with `ros2 topic echo --once <topic> --field format`.

## 2. Check it

```bash
./run.sh selftest                     # every robots/*.yaml must load
```

A missing or wrong key fails with a message naming it, before capture starts.

## 3. Run the calibration

The package goes on the robot where the repo would be, `~/<repo>/<path in the repo>`
(here `~/anvil-embodied-ai/packages/robot-camera-calib`). `sync.sh` works that out from
git; set `REMOTE_REPO` if the repo has another name on the robot, or `REMOTE_CODE` for
any other path under the robot user's home.

```bash
# from your machine
TARGET=myrobot ROBOT_HOST=user@robot-ip ./sync.sh push

# on the robot
cd ~/anvil-embodied-ai/packages/robot-camera-calib
./run.sh capture --robot myrobot --run first01 --board <measured spec> --web
./run.sh solve runs/first01

# from your machine: live view while capturing, results afterwards
TARGET=myrobot ROBOT_HOST=user@robot-ip ./sync.sh view
TARGET=myrobot ROBOT_HOST=user@robot-ip ./sync.sh pull first01
```

`--robot` also takes a path (`--robot /path/to/myrobot.yaml`), so you can try a config
before adding it to `robots/`. Board handling and how to read the report are in the
[package README](../README.md).

First run checklist:
- The log says `rosbridge connected`, and the status line shows `no_data` staying at 0.
- With the robot still and no board, ticks count as `no_board`, not `moving`. Mostly
  `moving` means `still_rad` is too tight for your joint noise.
- Show the board: the live view tile turns orange, then green when it saves.

## Code layout

| File | Role |
|---|---|
| `capture.py` | CLI and the capture loop (the three save gates) |
| `sources.py` | `Source` interface, `GdkSource`, `RosbridgeSource` |
| `config.py` | loads and validates `robots/*.yaml` |
| `solve.py` | intrinsics fit, factory comparison, results and report |
| `common.py` | board detection, pose maths, file formats |
| `web.py` | live view served during capture |
| `gdk_env.py` | AgiBot GDK library bootstrap (G2 only) |
| `selftest.py` | synthetic tests, no robot needed |

Data flow: `capture` saves `samples/NNNN/<cam>.jpg` + `meta.json`, plus
`factory.json` and `run.json` (which stores the full robot config). `solve` reads them
back with the same config and writes `results/`.

## Adding a new source type

1. In `sources.py`, write a class with the methods of the `Source` protocol:
   `write_factory`, `run_info`, `read`, `still`, `K_D`, `sample_meta`, `close`, and a
   `last_error` attribute. `read()` returns a `Tick` (a `Frame` per camera, holding the
   grayscale image and the JPEG bytes to save) or None when there is no fresh data.
   Keep it read-only.
2. Return it from `make_source()` for your `source.type`.
3. In `config.py`, add the type to `SOURCES` (its stillness types and required camera
   keys) and, for a new stillness type, its keys to `STILLNESS`.
4. Add a config to `robots/` and run `./run.sh selftest`.
