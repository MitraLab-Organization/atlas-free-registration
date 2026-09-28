# Atlas-free registration

Headless Linux wrapper of atlas-free slice alignment. Jenkins runs the image with job environment variables; it does not need to build it.

Supported slice formats: PNG, TIFF (`.tif`, `.tiff`), JPEG 2000 (`.jp2`). Images are loaded with Pillow, converted to RGB, and scaled to 0–1, including 16-bit TIFF/JP2.

## Image

Build the Linux image wherever you publish images from, then make it available to the Jenkins agent:

```bash
docker build --platform linux/amd64 -t atlas-free-registration .
```

The image is CPU PyTorch, `MPLBACKEND=Agg`, no notebooks.

## Jenkins run

The job only needs to set envs and mount the slice directory and output directory:

```bash
docker run --rm \
  -e OUTDIR=/data/out \
  -e INPUT_DIR=/data/in \
  -e REMOVE_ARTIFACTS \
  -e NITER_BIG_LOOP \
  -e DEVICE \
  -v "$SLICES:/data/in:ro" \
  -v "$OUT:/data/out" \
  atlas-free-registration
```

Alignment order is name-sorted files in `INPUT_DIR`, or the explicit list in `FNAMES`.

| Env | Meaning |
|---|---|
| `OUTDIR` or `OUTPUT_DIR` | Output directory inside the container |
| `INPUT_DIR` | Directory of png/tif/tiff/jp2 slices |
| `FNAMES` | Space-separated file list, if not using `INPUT_DIR` |
| `DEVICE` | `cpu` (default) |
| `REMOVE_ARTIFACTS` | `1`/`true` to mask near-white rows/columns |
| `SAVE_ALL_FIGS` | `1`/`true` to write diagnostic PNGs |
| `NITER_BIG_LOOP` | Default `400` |
| `NITER_REG` | Default `5` |
| `NITER_ATLAS` | Default `5` |

CLI flags still work and override the envs:

```bash
docker run --rm \
  -v /path/to/slices:/data/in:ro \
  -v /path/to/outputs:/data/out \
  atlas-free-registration \
  /data/out --input-dir /data/in --remove_artifacts
```

Outputs in `OUTDIR`:

- `A.npz`, `v.npz`, `Esave.npz`
- `RphiI.npz`, `phiiRiJ.npz`
- `Wshow.npz`, `W_robust_loss.npz`
