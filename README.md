# Atlas-free registration

Headless Linux wrapper of the v03 atlas-free slice alignment pipeline from `atlas_free_alignment_v03.ipynb`. Jenkins runs the image with job environment variables.

The algorithm is coarse-to-fine **rigid** alignment plus atlas estimation (`atlas_free_alignment.py`). Slice order follows the integer at the end of the filename (`..._001.tif`, `..._002.tif`). Missing numbers are filled with blank slices so z spacing matches slice number.

Supported formats: PNG, TIFF, JPEG 2000.

## Image

Images are on GitHub Container Registry:

```bash
docker pull ghcr.io/mitralab-organization/atlas-free-registration:cpu
docker pull ghcr.io/mitralab-organization/atlas-free-registration:cuda
docker pull ghcr.io/mitralab-organization/atlas-free-registration:latest
```

`:latest` is the CPU image. Anonymous pull works only after the GitHub package (and linked repo) are public; until then use `docker login ghcr.io`.

Build locally:

```bash
docker build --platform linux/amd64 -t atlas-free-registration:cpu .

docker build --platform linux/amd64 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t atlas-free-registration:cuda .
```

## Jenkins run

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
| `OUTDIR` | Output directory |
| `INPUT_DIR` | Directory of slices |
| `DEVICE` | `auto`, `cpu`, or `cuda` |
| `DX` | In-plane pixel size before `--ndown` (default `14.72`) |
| `DZ` | Spacing between consecutive slice numbers (default `40`) |
| `NDOWN` | Load-time in-plane downsample `2**NDOWN` (default `2`) |
| `NITER0` `NITER1` `NITER2` `NITER3` | Coarse-to-fine iterations (default 2000, 1000, 500, 200) |
| `NITER_ATLAS` | Atlas updates per iteration (default `5`) |
| `SAVE_ALL_FIGS` | `1` to write diagnostic PNGs |

Set an `NITER*` value to `0` to skip that level. Smoke test: `NITER0=1 NITER1=0 NITER2=0 NITER3=0`.

Outputs: `R.npy`, `v03_rigid_*.npy`, `atlas.nii.gz`, `I.npz`.
