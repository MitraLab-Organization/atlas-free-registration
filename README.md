# Atlas-free registration

Headless Linux wrapper of atlas-free slice alignment. Jenkins runs the image with job environment variables.

Supported slice formats: PNG, TIFF (`.tif`, `.tiff`), JPEG 2000 (`.jp2`). Images are loaded with Pillow, converted to RGB, and scaled to 0–1, including 16-bit TIFF/JP2.

## Image

Default build is CPU torch. If the Jenkins agent has CUDA, build the CUDA wheels instead and pass the GPU in at run time.

```bash
docker build --platform linux/amd64 -t atlas-free-registration:cpu .

docker build --platform linux/amd64 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t atlas-free-registration:cuda .
```

`DEVICE` defaults to `auto`: CUDA if that torch build can see a GPU, otherwise CPU. The CPU image cannot use a host GPU.

If Jenkins already has a CUDA torch environment, skip the image and run `src/slice_alignment.py` in that env.

## Jenkins run

CPU:

```bash
docker run --rm \
  -e OUTDIR=/data/out \
  -e INPUT_DIR=/data/in \
  -v "$SLICES:/data/in:ro" \
  -v "$OUT:/data/out" \
  atlas-free-registration:cpu
```

CUDA:

```bash
docker run --rm --gpus all \
  -e OUTDIR=/data/out \
  -e INPUT_DIR=/data/in \
  -e DEVICE=cuda \
  -v "$SLICES:/data/in:ro" \
  -v "$OUT:/data/out" \
  atlas-free-registration:cuda
```

| Env | Meaning |
|---|---|
| `OUTDIR` or `OUTPUT_DIR` | Output directory inside the container |
| `INPUT_DIR` | Directory of png/tif/tiff/jp2 slices |
| `FNAMES` | Space-separated file list, if not using `INPUT_DIR` |
| `DEVICE` | `auto` (default), `cpu`, or `cuda` |
| `REMOVE_ARTIFACTS` | `1`/`true` to mask near-white rows/columns |
| `SAVE_ALL_FIGS` | `1`/`true` to write diagnostic PNGs |
| `NITER_BIG_LOOP` | Default `400` |

Outputs in `OUTDIR`: `A.npz`, `v.npz`, `Esave.npz`, `RphiI.npz`, `phiiRiJ.npz`, `Wshow.npz`, `W_robust_loss.npz`.
