import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import nibabel as nib
import numpy as np
import tifffile
import torch
from PIL import Image
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

import atlas_free_alignment as afa

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jp2", ".j2k", ".jpx", ".jpg", ".jpeg")
TIFF_EXTENSIONS = {".tif", ".tiff"}
DTYPE_MAP = {
    "float": torch.float64,
    "float64": torch.float64,
    "double": torch.float64,
    "float32": torch.float32,
}


def _first_env(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _env_flag(*names):
    value = _first_env(*names)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(*names):
    value = _first_env(*names)
    if value is None:
        return None
    return value.split()


def resolve_device(value):
    requested = str(value or "auto").strip().lower()
    if requested in {"auto", ""}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"DEVICE={device} but this PyTorch build cannot see a GPU. "
            "Build with TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 and run with --gpus all, "
            "or run inside an environment that already has CUDA torch."
        )
    return device


def resolve_dtype(value):
    if value is float:
        return torch.float64
    key = str(value).lower()
    if key not in DTYPE_MAP:
        raise SystemExit(f"Unsupported dtype {value!r}. Use one of: {', '.join(sorted(DTYPE_MAP))}")
    return DTYPE_MAP[key]


def slice_index(path):
    stem = Path(path).stem
    tail = stem.split("_")[-1]
    try:
        return int(tail)
    except ValueError:
        match = re.search(r"(\d+)$", stem)
        if match:
            return int(match.group(1))
        raise ValueError(f"No slice number in {path}")


def collect_slice_paths(fnames, input_dir):
    paths = []
    if input_dir is not None:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")
        paths.extend(
            str(path)
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if fnames:
        paths.extend(fnames)
    if not paths:
        raise SystemExit("Provide -fnames and/or --input-dir with png, tif, tiff, or jp2 files")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))
    try:
        paths.sort(key=slice_index)
    except ValueError:
        paths.sort()
    return paths


def load_slice(path):
    suffix = Path(path).suffix.lower()
    if suffix in TIFF_EXTENSIONS:
        array = np.asarray(tifffile.imread(path))
    else:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)

    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64, copy=False)
        peak = np.nanmax(array) if array.size else 0.0
        if peak > 1.5:
            array = array / (255.0 if peak <= 255.0 else 65535.0)
    elif array.dtype == np.uint8:
        array = array.astype(np.float64) / 255.0
    elif array.dtype == np.uint16:
        array = array.astype(np.float64) / 65535.0
    else:
        info = np.iinfo(array.dtype)
        array = array.astype(np.float64) / float(info.max)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] >= 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[-1] != 3:
        raise SystemExit(f"Unsupported image shape {array.shape} for {path}")
    return array


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def blur_downsampled(Jd, W0d, blur, dtype, device):
    Jd_np = to_numpy(Jd * W0d)
    W_np = to_numpy(W0d)
    Jd_np = gaussian_filter(Jd_np, blur)
    W_blur = gaussian_filter(W_np, blur[1:])
    Jd_np = Jd_np / (W_blur + 1e-6)
    return (
        torch.tensor(Jd_np, dtype=dtype, device=device),
        torch.tensor(W_blur, dtype=dtype, device=device),
    )


def save_nifti(path, volume, xI):
    dI = np.array([to_numpy(x[1] - x[0]).item() if torch.is_tensor(x[1] - x[0]) else float(x[1] - x[0]) for x in xI])
    affine = np.diag(dI.tolist() + [1])
    affine[:3, -1] = np.array([float(to_numpy(x[0])) for x in xI])
    array = (to_numpy(volume.permute(1, 2, 3, 0)) * 256).clip(0, 255).astype(np.uint8)
    nib.save(nib.Nifti1Image(array, affine), path)


def main():
    parser = argparse.ArgumentParser(description="Atlas-free slice alignment (v03 coarse-to-fine rigid pipeline)")
    parser.add_argument("outdir", nargs="?", default=_first_env("OUTDIR", "OUTPUT_DIR", "AFR_OUTDIR"), type=Path)
    parser.add_argument("-fnames", nargs="+", default=_env_list("FNAMES", "AFR_FNAMES"), type=str)
    parser.add_argument("--input-dir", default=_first_env("INPUT_DIR", "AFR_INPUT_DIR"), type=Path)
    parser.add_argument("-device", default=_first_env("DEVICE", "AFR_DEVICE", default="auto"))
    parser.add_argument("-dtype", default=_first_env("DTYPE", "AFR_DTYPE", default="float64"))
    parser.add_argument("--dx", default=float(_first_env("DX", default=0.46 * 32)), type=float, help="In-plane pixel size before ndown")
    parser.add_argument("--dz", default=float(_first_env("DZ", default=40)), type=float, help="Spacing between consecutive slice numbers")
    parser.add_argument("--ndown", default=int(_first_env("NDOWN", default=2)), type=int, help="In-plane downsample of 2**ndown after load")
    parser.add_argument("-niter0", default=int(_first_env("NITER0", default=2000)), type=int)
    parser.add_argument("-niter1", default=int(_first_env("NITER1", default=1000)), type=int)
    parser.add_argument("-niter2", default=int(_first_env("NITER2", default=500)), type=int)
    parser.add_argument("-niter3", default=int(_first_env("NITER3", default=200)), type=int)
    parser.add_argument("-niter_atlas", default=int(_first_env("NITER_ATLAS", default=5)), type=int)
    parser.add_argument("-c", default=float(_first_env("C", default=3 * 0.25**2)), type=float)
    parser.add_argument("-p", default=float(_first_env("P", default=1.0)), type=float)
    parser.add_argument("--no-fill-missing", action="store_false", dest="fill_missing", help="Do not insert blank slices for missing slice numbers")
    parser.set_defaults(fill_missing=True)
    parser.add_argument("--remove_artifacts", action="store_true")
    parser.add_argument("--saveAllFigs", action="store_true")
    args = parser.parse_args()

    if args.outdir is None:
        raise SystemExit("Provide outdir as a positional argument or set OUTDIR")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    save_figs = args.saveAllFigs or _env_flag("SAVE_ALL_FIGS", "AFR_SAVE_ALL_FIGS")
    ndraw = 100 if save_figs else 0
    dx = args.dx
    dz = args.dz
    ndown = args.ndown
    c = args.c
    p = args.p
    niter_atlas = args.niter_atlas

    paths = collect_slice_paths(args.fnames, args.input_dir)
    print(f"Found {len(paths)} files on device={device} dtype={dtype}", flush=True)

    indexed = []
    use_index = True
    for path in paths:
        try:
            indexed.append((slice_index(path), path))
        except ValueError:
            use_index = False
            break

    if use_index and args.fill_missing:
        locations = np.array([item[0] for item in indexed])
        path_by_loc = {item[0]: item[1] for item in indexed}
        all_locations = np.arange(np.min(locations), np.max(locations) + 1)
    else:
        all_locations = np.arange(len(paths))
        path_by_loc = {i: path for i, path in enumerate(paths)}

    J_ = []
    W_ = []
    xJ_ = []
    nJ_ = []
    for loc in all_locations:
        path = path_by_loc.get(int(loc))
        if path is not None:
            print(f"  {int(loc)} {path}", flush=True)
            Ji = load_slice(path)
            if args.remove_artifacts:
                rowones = np.all(Ji >= 0.95, (0, -1))
                colones = np.all(Ji >= 0.95, (1, -1))
                Wi = ((1 - rowones[None, :]) * (1 - colones[:, None])).astype(np.float64)
                Wi = Wi * (1.0 - np.all(Ji == 1.0, axis=-1))
            else:
                Wi = (1.0 - np.all(Ji == 1.0, axis=-1)).astype(np.float64)
        else:
            print(f"  {int(loc)} missing, inserting blank", flush=True)
            Ji = np.zeros((11, 11, 3), dtype=np.float64)
            Wi = np.zeros((11, 11), dtype=np.float64)

        nJi = np.array(Wi.shape)
        xJi = [np.arange(n) * dx - (n - 1) * dx / 2 for n in nJi]
        xJi, Ji, Wi = afa.downsample_image_domain(xJi, Ji.transpose(-1, 0, 1), [2 ** ndown] * 2, Wi)
        Wi[np.isnan(Wi)] = 0
        Ji[np.isnan(Ji)] = 1
        try:
            q = np.quantile(Ji[:, Wi == 1], 0.99, axis=(1,))
            Ji = np.clip(Ji / q[..., None, None], 0, 1)
        except Exception:
            pass
        nJ_.append(np.array(Wi.shape))
        xJ_.append(xJi)
        J_.append(Ji)
        W_.append(Wi)

    q = np.ceil(np.quantile(nJ_, [0.95], 0) * 1.5).astype(int)
    nJ = np.concatenate([[len(all_locations)], q.squeeze()])
    dJ = np.array([dz, dx * 2 ** ndown, dx * 2 ** ndown])
    xJ = [np.arange(n) * d - (n - 1) * d / 2 for n, d in zip(nJ, dJ)]
    XJ2d = np.stack(np.meshgrid(*xJ[1:], indexing="ij"), -1)
    J__ = []
    W__ = []
    for i, Ji in enumerate(J_):
        Jout = interpn(xJ_[i], (Ji * W_[i]).transpose(1, 2, 0), XJ2d, bounds_error=False, fill_value=1).transpose(-1, 0, 1)
        Wout = interpn(xJ_[i], W_[i], XJ2d, bounds_error=False, fill_value=0, method="nearest")
        Jout = Jout / (Wout + 1e-6)
        Jout[np.isnan(Jout)] = 1
        Jout[np.isinf(Jout)] = 1
        Wout = (Wout == 1.0).astype(float)
        J__.append(Jout)
        W__.append(Wout)

    J = torch.tensor(np.stack(J__, 1), dtype=dtype, device=device).clip(0, 1)
    W0 = torch.tensor(np.stack(W__, 0), dtype=dtype, device=device)
    xJ = [torch.tensor(x, dtype=dtype, device=device) for x in xJ]
    I = torch.ones(J.shape, dtype=dtype, device=device) * 0.5
    xI = [x.clone() for x in xJ]
    R = torch.eye(3, dtype=dtype, device=device)[None].repeat(J.shape[1], 1, 1)

    eL0 = 1e-5 * 5
    eT0 = 5e2 * 2
    a = torch.tensor([2e2 * 5, 2e2, 2e2], dtype=dtype, device=device)
    blur = (0, 0, 1, 1)
    levels = [
        {"name": "0", "down": [8, 8], "niter": args.niter0, "eL": eL0, "eT": eT0, "init_ones": True, "a_div": 1},
        {"name": "1", "down": [4, 4], "niter": args.niter1, "eL": eL0 / 4, "eT": eT0 / 4, "init_ones": False, "a_div": 2},
        {"name": "2", "down": [2, 2], "niter": args.niter2, "eL": eL0 / 100, "eT": eT0 / 100, "init_ones": False, "a_div": 2},
        {"name": "3", "down": [1, 1], "niter": args.niter3, "eL": eL0 / 500, "eT": eT0 / 500, "init_ones": False, "a_div": 1},
    ]

    I_full = I
    Rout = R
    Idout = I
    RiJ = RiW = RI = None
    for level in levels:
        if level["niter"] <= 0:
            continue
        a = a / level["a_div"]
        down = level["down"]
        print(f"Level {level['name']} down={down} niter={level['niter']}", flush=True)
        xId, Id = afa.downsample_image_domain(xI, I_full, [1, down[0], down[1]])
        xJd, Jd, W0d = afa.downsample_image_domain(xJ, J, [1, down[0], down[1]], W0)
        Jd, W0d = blur_downsampled(Jd, W0d, blur, dtype, device)
        Id = Id.to(device=device, dtype=dtype)
        xId = [x.to(device=device, dtype=dtype) if torch.is_tensor(x) else torch.tensor(x, dtype=dtype, device=device) for x in xId]
        xJd = [x.to(device=device, dtype=dtype) if torch.is_tensor(x) else torch.tensor(x, dtype=dtype, device=device) for x in xJd]
        if level["init_ones"]:
            Id = Id * 0 + 1
        dI = np.array([float(to_numpy(x[1] - x[0])) for x in xId])
        _, _, Kblurhatd = afa.get_frequency_operators(Id.shape[1:], dI, a, p, device=device, dtype=dtype)
        figprefix = os.path.join(outdir, f"v03_it_{level['name']}")
        Rout, Idout, RiJ, RiW, RI = afa.atlas_free_alignment(
            xId, Id, xJd, Jd, W0d, Kblurhatd, Rout,
            level["niter"], niter_atlas, level["eL"], level["eT"],
            c=c, figprefix=figprefix, ndraw=ndraw,
        )
        np.save(outdir / f"v03_rigid_{level['name']}.npy", Rout.detach().cpu().numpy())
        I_full = afa.sinc_upsample(Idout, I.shape[-3:])

    np.save(outdir / "R.npy", Rout.detach().cpu().numpy())
    save_nifti(outdir / "atlas.nii.gz", Idout, xI)
    np.savez(outdir / "I.npz", data=I_full.detach().cpu())
    if RiJ is not None:
        np.savez(outdir / "RiJ.npz", data=RiJ.detach().cpu())
        np.savez(outdir / "RiW.npz", data=RiW.detach().cpu())
        np.savez(outdir / "RI.npz", data=RI.detach().cpu())
    print(f"Wrote outputs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
