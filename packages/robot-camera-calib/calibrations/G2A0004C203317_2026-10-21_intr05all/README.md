# G2A0004C203317 — intrinsics, 2026-10-21 (run intr05all)

Board: charuco:10x8:0.030:0.015:4x4_50:legacy. Captured with robot_camera_calib.
Files `intrinsic_*.json` use AgiBot's format/names (as in /data/parameters/sensor). NOT installed on the robot.

Reprojection error with these values (per board corner, all saved views):

| camera | corners | RMS | mean | median | p95 | max | hold-out factory / new |
|---|---|---|---|---|---|---|---|
| head (640x400) | 522 | 0.357 | 0.292 | 0.238 | 0.688 | 2.095 px | n/a (13 views) |
| hand_left (1280x1056) | 1523 | 0.909 | 0.746 | 0.628 | 1.736 | 3.690 px | 0.913 / 0.885 px |
| hand_right (1280x1056) | 1427 | 0.844 | 0.714 | 0.627 | 1.535 | 3.099 px | 0.951 / 0.901 px |

Source run: robot /data/dercio-build/g2_camera_calib/runs/intr05all (68 samples).
`factory.json` = the vendor values that were live on the robot at capture time.
