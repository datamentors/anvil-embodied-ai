# mcap-convert-gpu

GPU-optimized sibling of `mcap-converter`: same MCAP -> LeRobot v3.0 conversion
pipeline, but with NVENC streaming video encode (via PyAV) and multi-GPU
episode-shard parallelism instead of CPU-only libx264/libsvtav1 encoding.

Kept as a fully separate installable package (not merged into
`mcap-converter`) so the default CPU path stays untouched and this can be
swapped in deliberately.

## Usage

Use all visible GPUs, auto-sharding episodes across them:

```bash
mcap-convert-gpu -i data/raw/my-session -o data/datasets --config configs/mcap_converter/openarm_bimanual.yaml
```

Pin one whole conversion to a single physical GPU (for running N independent
conversions across N GPUs at once — launch one process per GPU, each on a
different input dir):

```bash
mcap-convert-gpu -i data/raw/session-a -o data/datasets --gpu-id 0 &
mcap-convert-gpu -i data/raw/session-b -o data/datasets --gpu-id 1 &
mcap-convert-gpu -i data/raw/session-c -o data/datasets --gpu-id 2 &
mcap-convert-gpu -i data/raw/session-d -o data/datasets --gpu-id 3 &
wait
```

Combine both: pin to one GPU but still shard multiple episodes within it:

```bash
mcap-convert-gpu -i data/raw/session-a -o data/datasets --gpu-id 0 --parallel-episodes 2
```

Use all 4 GPUs for one single conversion, sharding its episodes across them:

```bash
mcap-convert-gpu -i data/raw/big-session -o data/datasets --parallel-episodes 4
```
