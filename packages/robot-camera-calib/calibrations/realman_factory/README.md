# RealMan RMC-AIDAL-III — factory intrinsics (3x RealSense D457)

`factory.json` = the driver's `camera_info` (D457 factory calibration, read from each camera's
EEPROM by `realsense2_camera`), 640x480 color, read 2026-10-23. Same layout as the G2's
`factory.json`; copy it into a run as `factory.json` before solve to compare against it.

| camera | serial | fx | fy | cx | cy |
|---|---|---|---|---|---|
| camera1 (left) | 035622250076 | 382.50 | 382.19 | 314.40 | 244.95 |
| camera2 (centre) | 343622300400 | 386.70 | 386.49 | 325.63 | 242.89 |
| camera3 (right) | 215122250915 | 381.49 | 381.28 | 324.74 | 238.21 |

Checked against a board calibration (2026-10-26, run `test03`), hold-out factory / new:
camera1 0.329 / 0.322 px, camera3 0.306 / 0.293 px, so the factory values are fine.

**camera2:** `run.sh` rotates its image 180 deg (`rotation_filter`) but the driver does not
rotate its `camera_info`, so these values do not fit the published frames (hold-out 0.610 /
0.306 px). With the filter off (run `test04_cam2_norot`) they agree: 0.233 / 0.229 px. For the
rotated frames use cx' = 639 - cx, cy' = 479 - cy and flip the sign of p1, p2.
