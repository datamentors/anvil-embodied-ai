# G2 camera calibration report

- run: `/data/dercio-build/g2_camera_calib/runs/intr05all`
- board: `charuco:10x8:0.030:0.015:4x4_50:legacy`

## head

Board views: 13

### Intrinsics

| | fx | fy | cx | cy |
|---|---|---|---|---|
| factory | 305.02 | 305.01 | 316.53 | 203.46 |
| new | 301.85 | 301.03 | 312.09 | 200.72 |
| delta px | -3.17 | -3.98 | -4.43 | -2.73 |

- reprojection RMS on these frames: factory 0.372 px, new (in-sample fit) 0.357 px
- image coverage: 100% (need 40%)
- recommended: yes

## hand_left

Board views: 40

### Intrinsics

| | fx | fy | cx | cy |
|---|---|---|---|---|
| factory | 479.70 | 479.46 | 640.50 | 524.10 |
| new | 478.64 | 477.59 | 643.48 | 519.74 |
| delta px | -1.06 | -1.87 | +2.99 | -4.36 |

- reprojection RMS on these frames: factory 0.975 px, new (in-sample fit) 0.909 px
- hold-out (fit on even frames, test on odd): factory 0.913 px, new 0.885 px
- image coverage: 98% (need 40%)
- recommended: yes

## hand_right

Board views: 26

### Intrinsics

| | fx | fy | cx | cy |
|---|---|---|---|---|
| factory | 484.36 | 483.73 | 642.84 | 529.54 |
| new | 485.87 | 486.39 | 644.34 | 520.55 |
| delta px | +1.51 | +2.66 | +1.50 | -8.98 |

- reprojection RMS on these frames: factory 0.921 px, new (in-sample fit) 0.844 px
- hold-out (fit on even frames, test on odd): factory 0.951 px, new 0.901 px
- image coverage: 100% (need 40%)
- recommended: yes
