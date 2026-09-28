import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interpn

from slice_alignment_help import *

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jp2", ".j2k", ".jpx", ".jpg", ".jpeg")
DTYPE_MAP = {
    "float": float,
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
    def update(self, *args, **kwargs):
        pass


try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        return _NullDisplay()


def resolve_dtype(value):
    if value is float:
        return float
    key = str(value).lower()
    if key not in DTYPE_MAP:
        raise SystemExit(f"Unsupported dtype {value!r}. Use one of: {', '.join(sorted(DTYPE_MAP))}")
    return DTYPE_MAP[key]


def collect_slice_paths(fnames, input_dir):
    paths = []
    if input_dir is not None:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")
        paths.extend(
            str(path)
            for path in sorted(input_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if fnames:
        paths.extend(fnames)
    if not paths:
        raise SystemExit("Provide -fnames and/or --input-dir with png, tif, tiff, or jp2 files")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))
    return paths


def load_slice(path):
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


def save_figure(outdir, name, plot_fn):
    fig, ax = plt.subplots()
    plot_fn(ax)
    fig.savefig(os.path.join(outdir, name))
    plt.close(fig)


def main():
    """
    Perform atlas-free slice alignment of a set of images

    Parameters:
    ===========
    outdir : str
        The directory where all intermediate and final outputs should be saved
    -fnames : list of str
        The list of all files to be registered in alignment order
    --input-dir : str
        Directory of png/tif/tiff/jp2 slices, used in name-sorted order
    -npad : int
        TODO
    -down : int
        TODO
    -a : float
        TODO
    -p : float
        TODO
    -niter_big_loop : int
        TODO
    -niter_reg : int
        TODO
    -niter_atlas : int
        TODO
    -asquare : float
        TODO
    -asquare0 : float
        TODO
    -anisotropy_factor : float
        TODO
    -epT : float
        TODO
    -epL : float
        TODO
    -epv : float
        TODO
    -c : float
        TODO
    -sigmaM : float
        TODO
    -sigmaR : float
        TODO
    -a_reg : float
        TODO
    -device : str
        Default - cpu; The device where PyTorch computations should be performed
    -dtype : str
        Default - float; The dtype used for PyTorch computations
    --enable_deformation : bool
        TODO: NOT CURRENTLY SUPPORTED
    --remove_artifacts : bool
        If present, remove rows or columns containing exclusively 1s. This type of artifact is common in certain use cases.
    --saveAllFigs : bool
        If present, save all potential figures into 'outdir'
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "outdir",
        nargs="?",
        default=_first_env("OUTDIR", "OUTPUT_DIR", "AFR_OUTDIR"),
        type=Path,
        help="Output directory. Also accepted as OUTDIR / OUTPUT_DIR.",
    )
    parser.add_argument(
        "-fnames",
        nargs="+",
        default=_env_list("FNAMES", "AFR_FNAMES"),
        type=str,
        help="List of file names in alignment order. Also accepted as FNAMES.",
    )
    parser.add_argument(
        "--input-dir",
        default=_first_env("INPUT_DIR", "AFR_INPUT_DIR"),
        type=Path,
        help="Directory of slices in name-sorted order (png, tif, tiff, jp2). Also accepted as INPUT_DIR.",
    )
    parser.add_argument("-npad", default=0, type=int, help="Default - 0; The number of slices to be padded")
    parser.add_argument("-down", default=1, type=int, help="Default - 1; TODO - Add description")
    parser.add_argument("-a", default=10.0, type=float, help="Default - 10.0; Highpass operator for 2D registration")
    parser.add_argument("-p", default=2.0, type=float, help="Default - 2.0; Lowpass operator for 2D registration")
    parser.add_argument("-niter_big_loop", default=int(_first_env("NITER_BIG_LOOP", default=400)), type=int, help="Default - 400; The number of iterations to perform registration of the whole dataset")
    parser.add_argument("-niter_reg", default=int(_first_env("NITER_REG", default=5)), type=int, help="Default - 5; The number of iterations to perform registration on a subproblem")
    parser.add_argument("-niter_atlas", default=int(_first_env("NITER_ATLAS", default=5)), type=int, help="Default - 5; The number of iterations to perform registration on a subproblem")
    parser.add_argument("-asquare", default=0.25**2, type=float, help="Default - 0.25**2; A scalar used for registration")
    parser.add_argument("-asquare0", default=3.5**2, type=float, help="Default - 3.5**2; A scalar used for registration")
    parser.add_argument("-anisotropy_factor", default=0.01, type=float, help="Default - 0.01; A scalar used during registration")
    parser.add_argument("-epT", default=0.0, type=float, help="Default - 0; A scalar used during registration")
    parser.add_argument("-epL", default=0.0, type=float, help="Default - 0; A scalar used during registration")
    parser.add_argument("-epv", default=500.0, type=float, help="Default - 500; A scalar used during registration")
    parser.add_argument("-c", default=2.0, type=float, help="Default - 2; A scalar used during registration")
    parser.add_argument("-sigmaM", default=1.0, type=float, help="Default - 1; A scalar used during registration")
    parser.add_argument("-sigmaR", default=500.0, type=float, help="Default - 500; A scalar used during registration")
    parser.add_argument("-a_reg", default=6.0, type=float, help="Default - 6; A scalar used during registration")
    parser.add_argument("-device", default=_first_env("DEVICE", "AFR_DEVICE", default="cpu"), help="Default - cpu; The device where PyTorch computations should occur during registration")
    parser.add_argument("-dtype", default=_first_env("DTYPE", "AFR_DTYPE", default="float"), help="Default - float (float64); The dtype to be used during PyTorch computation")
    parser.add_argument("--enable_deformation", action="store_true", help="TODO: NOT CURRENTLY SUPPORTED")
    parser.add_argument("--remove_artifacts", action="store_true", help="Remove rows or columns containing exclusively 1s. This type of artifact is common in certain use cases.")
    parser.add_argument("--saveAllFigs", action="store_true", help="If present, save all potential figures into 'outdir'")

    args = parser.parse_args()

    if args.outdir is None:
        raise SystemExit("Provide outdir as a positional argument or set OUTDIR")

    outdir = args.outdir
    fnames = collect_slice_paths(args.fnames, args.input_dir)
    npad = args.npad
    down = args.down
    a = args.a
    p = args.p
    niter_big_loop = args.niter_big_loop
    niter_reg = args.niter_reg
    niter_atlas = args.niter_atlas
    asquare = args.asquare
    asquare0 = args.asquare0
    anisotropy_factor = args.anisotropy_factor
    epT = args.epT
    epL = args.epL
    epv = args.epv
    c = args.c
    sigmaM = args.sigmaM
    sigmaR = args.sigmaR
    a_reg = args.a_reg
    device = args.device
    dtype = resolve_dtype(args.dtype)
    enable_deformation = args.enable_deformation
    remove_artifacts = args.remove_artifacts or _env_flag("REMOVE_ARTIFACTS", "AFR_REMOVE_ARTIFACTS")
    saveAllFigs = args.saveAllFigs or _env_flag("SAVE_ALL_FIGS", "AFR_SAVE_ALL_FIGS")
    draw_every = 5 if saveAllFigs else 0
    draw_atlas = bool(saveAllFigs)

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print(f"Loaded {len(fnames)} slices on device={device} dtype={dtype}", flush=True)
    for fname in fnames:
        print(f"  {fname}", flush=True)

    J_ = []
    W_ = []
    for fname in fnames:
        Ji = load_slice(fname)

        if remove_artifacts:
            rowones = np.all(Ji >= 0.95, (0, -1))
            colones = np.all(Ji >= 0.95, (1, -1))
            Wi = (1 - rowones[None, :]) * (1 - colones[:, None])
        else:
            Wi = np.ones(Ji.shape[:2])

        J_.append(Ji)
        W_.append(Wi)

    nJ = [Ji.shape for Ji in J_]
    nJ = np.max(nJ, 0)
    nJ = [len(J_), nJ[0], nJ[1]]
    x2d = [np.arange(n) * down - (n - 1) * down / 2 for n in nJ[1:]]
    X2d = np.stack(np.meshgrid(*x2d, indexing="ij"), -1)
    J__ = []
    W__ = []
    for Ji, Wi in zip(J_, W_):
        x = [np.arange(n) * down - (n - 1) * down / 2 for n in Ji.shape[:2]]
        Ji_ = interpn(x, Ji, X2d, bounds_error=False, method="nearest")
        Wi_ = interpn(x, Wi, X2d, bounds_error=False, method="nearest")

        Wi_ = (1.0 - np.isnan(Ji_[..., 0])) * Wi_
        Ji_[np.isnan(Ji_)] = 0
        Wi_[np.isnan(Wi_)] = 0
        J__.append(Ji_)
        W__.append(Wi_)

    J = np.stack(J__, 0).transpose(-1, 0, 1, 2)
    W = np.stack(W__)
    xJ = [np.arange(nJ[0]) - (nJ[0] - 1) / 2, x2d[0], x2d[1]]

    if npad == 0:
        J = np.pad(J, ((0, 0), (npad, npad), (0, 0), (0, 0)), mode="reflect")
        W = np.pad(W, ((npad, npad), (0, 0), (0, 0)), mode="reflect")
        nJ = J.shape[1:]
        xJ = [np.arange(nJ[0]) - (nJ[0] - 1) / 2, x2d[0], x2d[1]]

    J = torch.tensor(J, dtype=dtype, device=device)
    W = torch.tensor(W, dtype=dtype, device=device)
    xJ = [torch.tensor(x, dtype=dtype, device=device) for x in xJ]

    XJ = torch.stack(torch.meshgrid(xJ, indexing="ij"), -1)
    A = torch.eye(3)
    A = A[None].repeat(nJ[0], 1, 1)
    A = A2DtoA3D(A)
    Ai = torch.linalg.inv(A)

    Ai = Ai.to(dtype=dtype, device=device)
    XJ = XJ.to(dtype=dtype, device=device)
    Xs = AX(Ai, XJ)
    AJ = interp(xJ, J, XJ)

    if saveAllFigs:
        save_figure(outdir, "fig0.png", lambda ax: ax.imshow(J[:, AJ.shape[1] // 2].permute(1, 2, 0)))
        save_figure(outdir, "fig1.png", lambda ax: ax.imshow(AJ[:, AJ.shape[1] // 2].permute(1, 2, 0)))

    AJ = interp(xJ, J, Xs)
    if saveAllFigs:
        save_figure(outdir, "fig2.png", lambda ax: ax.imshow(AJ[:, AJ.shape[1] // 2].permute(1, 2, 0)))
        save_figure(outdir, "fig3.png", lambda ax: ax.imshow(AJ[:, :, AJ.shape[2] // 2].permute(1, 2, 0), aspect="auto"))

    extendv = 1.1
    dv = down * 2
    vmin1 = torch.amin(xJ[1])
    vmin2 = torch.amin(xJ[2])
    vmax1 = torch.amax(xJ[1])
    vmax2 = torch.amax(xJ[2])
    vc1 = (vmin1 + vmax1) / 2
    vc2 = (vmin2 + vmax2) / 2
    vr1 = (vmax1 - vmin1) / 2 * extendv
    vr2 = (vmax2 - vmin2) / 2 * extendv
    v1 = torch.arange(vc1 - vr1, vc1 + vr1, dv, device=device, dtype=dtype)
    v2 = torch.arange(vc2 - vr2, vc2 + vr2, dv, device=device, dtype=dtype)
    xv = [xJ[0], v1, v2]
    XV = torch.stack(torch.meshgrid(*xv, indexing="ij"), -1)
    XV2d = XV[..., 1:]
    v2d = torch.zeros_like(XV2d)
    v2d = torch.randn(v2d.shape, dtype=v2d.dtype)

    L = L_from_xv_a_p(xv, a, p)
    LL = L**2
    K = 1.0 / LL

    if saveAllFigs:
        save_figure(outdir, "fig4_L.png", lambda ax: ax.imshow(L))
        save_figure(outdir, "fig5_K.png", lambda ax: ax.imshow(K))

    v2d = torch.fft.ifftn(torch.fft.fftn(v2d, dim=(1, 2)) * K[..., None], dim=(1, 2)).real
    v3d = v2DToV3D(v2d)
    v3d /= torch.std(v3d)
    v3d *= 20
    v = v2d

    phi = exp(xv, v3d)
    if saveAllFigs:
        def _deform(ax):
            ax.contour(xv[2], xv[1], phi[phi.shape[0] // 2, ..., 1])
            ax.contour(xv[2], xv[1], phi[phi.shape[0] // 2, ..., 2])
            ax.set_title("Example deformation")
        save_figure(outdir, "fig6_exdef.png", _deform)

    I = (torch.sum(J * W, 1, keepdims=True) / (1e-6 + torch.sum(W, 0, keepdims=True))).repeat(1, J.shape[1], 1, 1)
    xI = [x.clone() for x in xJ]
    if saveAllFigs:
        save_figure(outdir, "fig7.png", lambda ax: ax.imshow(I[:, I.shape[1] // 2].permute(1, 2, 0)))
        save_figure(outdir, "fig8.png", lambda ax: ax.imshow(I[:, :, I.shape[2] // 2].permute(1, 2, 0), aspect="auto"))

    phiI = interp(xI, I, phi)
    if saveAllFigs:
        save_figure(outdir, "fig9.png", lambda ax: ax.imshow(phiI[:, AJ.shape[1] // 2].permute(1, 2, 0)))
        save_figure(outdir, "fig10.png", lambda ax: ax.imshow(phiI[:, :, phiI.shape[2] // 2].permute(1, 2, 0), aspect="auto"))

    A_temp = torch.eye(3)[None].repeat(J.shape[1], 1, 1)
    v_temp = torch.zeros_like(v)
    Anew, vnew, Eregistration, Ereg = weighted_see_registration(
        xI, I, xJ, J, W, xv, v_temp, A_temp, a, p, sigmaM=1.0, sigmaR=1e5, niter=10, epT=1e-2, epL=1e-6, epv=1e1, draw=draw_every
    )

    Wdetjac = detjac(xv, vnew)
    if saveAllFigs:
        def _jac(ax):
            mappable = ax.imshow(Wdetjac[Wdetjac.shape[0] // 2])
            plt.colorbar(mappable, ax=ax)
        save_figure(outdir, "fig11.png", _jac)
        save_figure(outdir, "fig12.png", lambda ax: ax.imshow(W[W.shape[0] // 2], interpolation="none"))

    RphiI = transform_image(xI, I, xv, vnew, Anew, xJ)
    if saveAllFigs:
        save_figure(outdir, "fig13.png", lambda ax: ax.imshow(RphiI[:, :, RphiI.shape[2] // 2].permute(1, 2, 0), aspect="auto", interpolation="none"))

    L, WR = robust_loss(RphiI, xJ, J, W, c, return_weights=True)
    if saveAllFigs:
        save_figure(outdir, "fig14.png", lambda ax: ax.imshow(WR[WR.shape[0] // 2]))

    phiiRiJ = inverse_transform_image(xJ, J, xv, vnew, Anew, xI, padding_mode="border")
    phiiRiW = inverse_transform_image(xJ, W[None] * WR, xv, vnew, Anew, xI, padding_mode="zeros", mode="nearest")[0]
    XI = torch.stack(torch.meshgrid(xI, indexing="ij"), -1)
    Wdetjacs = interp(xv, Wdetjac[None], XI)[0]

    if saveAllFigs:
        save_figure(outdir, "fig15.png", lambda ax: ax.imshow((phiiRiJ)[:, I.shape[1] // 2].permute(1, 2, 0)))
        save_figure(outdir, "fig16.png", lambda ax: ax.imshow((phiiRiW)[I.shape[1] // 2]))

    Inew = I.clone()
    Inew, Eat, ERat = atlas_from_aligned_slices_and_weights(
        xI, Inew * 0, dtype, device, phiiRiJ, phiiRiW * Wdetjacs, asquare=2.0**2, niter=2, draw=draw_atlas, anisotropy_factor=1.0
    )

    if saveAllFigs:
        save_figure(outdir, "fig17.png", lambda ax: ax.imshow(Inew[:, I.shape[1] // 2].permute(1, 2, 0)))
        save_figure(outdir, "fig18.png", lambda ax: ax.imshow(Inew[:, :, I.shape[2] // 2].permute(1, 2, 0), aspect="auto"))

    if saveAllFigs:
        fig_at, ax_at = plt.subplots(2, 3)
        ax_at = ax_at.ravel()
        hfig_at = display(fig_at, display_id=True)
        fig_at_estimate = plt.figure()
        hfig_at_estimate = display(fig_at_estimate, display_id=True)
        fig_reg = plt.figure()
        hfig_reg = display(fig_reg, display_id=True)
        fig_E, ax_E = plt.subplots(1, 1)
        if isinstance(ax_E, np.ndarray):
            ax_E = ax_E.ravel()
        else:
            ax_E = [ax_E]
        hfig_E = display(fig_E, display_id=True)
    else:
        fig_at = ax_at = fig_at_estimate = fig_reg = fig_E = ax_E = None
        hfig_at = hfig_at_estimate = hfig_reg = hfig_E = None

    bigger = 20
    x2dI = [torch.arange(n + bigger, dtype=dtype) * down - (n + bigger - 1) * down / 2 for n in nJ[1:]]
    xI = [torch.arange(nJ[0], dtype=dtype) - (nJ[0] - 1) / 2, x2dI[0], x2dI[1]]
    XI = torch.stack(torch.meshgrid(xI, indexing="ij"), -1)

    Esave = []
    v = torch.zeros_like(v)
    A = torch.eye(3)
    A = A[None].repeat(nJ[0], 1, 1)

    I = torch.zeros((J.shape[0], XI.shape[0], XI.shape[1], XI.shape[2])) + (torch.sum(J * W, dim=(1, 2, 3)) / torch.sum(W, dim=(0, 1, 2)))[..., None, None, None]
    RphiI = transform_image(xI, I, xv, v, A, xJ)
    rloss, W_robust_loss = robust_loss(RphiI, xJ, J, W, c, return_weights=True)

    for it_big_loop in range(niter_big_loop):
        if it_big_loop == 0:
            asquare = 4.0**2 * asquare0
        elif it_big_loop == 20:
            asquare = 2.0**2 * asquare0
        elif it_big_loop == 40:
            asquare = 1.0**2 * asquare0
        asquare = asquare0

        if it_big_loop > 0:
            A, v, Eregistration, Ereg = weighted_see_registration(
                xI, I, xJ, J, W * W_robust_loss, xv, v, A, a_reg, p, sigmaM, sigmaR, niter_reg, epT, epL, epv,
                draw=draw_every, fig=fig_reg, hfig=hfig_reg
            )
        else:
            Ereg = 0.0

        Wdetjac = detjac(xv, v)

        phiiRiJ = inverse_transform_image(xJ, J, xv, v, A, xI, padding_mode="border", mode="nearest")
        phiiRiW = inverse_transform_image(xJ, W[None] * W_robust_loss, xv, v, A, xI, padding_mode="zeros", mode="nearest")[0]
        Wdetjacs = interp(xv, Wdetjac[None], XI)[0]
        I, Eat, ERat = atlas_from_aligned_slices_and_weights(
            xI, I, dtype, device, phiiRiJ, phiiRiW * Wdetjacs, asquare, niter=niter_atlas,
            fig=fig_at_estimate, hfig=hfig_at_estimate, draw=draw_atlas, anisotropy_factor=anisotropy_factor
        )

        RphiI = transform_image(xI, I, xv, v, A, xJ)
        rloss, W_robust_loss = robust_loss(RphiI, xJ, J, W, c, return_weights=True)
        Wshow = phiiRiW * Wdetjacs

        if saveAllFigs:
            ax_at[0].cla()
            ax_at[0].imshow(I[:, I.shape[1] // 2].permute(1, 2, 0))
            ax_at[1].cla()
            ax_at[1].imshow(I[:, :, I.shape[2] // 2].permute(1, 2, 0), aspect="auto", interpolation="none")
            ax_at[2].cla()
            ax_at[2].imshow(I[:, :, :, I.shape[3] // 2].permute(1, 2, 0), aspect="auto", interpolation="none")
            ax_at[3].cla()
            ax_at[3].imshow(Wshow[I.shape[1] // 2])
            ax_at[4].cla()
            ax_at[4].imshow(Wshow[:, I.shape[2] // 2], aspect="auto", interpolation="none")
            ax_at[5].cla()
            ax_at[5].imshow(Wshow[:, :, I.shape[3] // 2], aspect="auto", interpolation="none")

        Esave.append([rloss.item() + Ereg + ERat, rloss.item(), Ereg, ERat])
        if saveAllFigs:
            ax_E[0].cla()
            ax_E[0].plot(Esave)
            ax_E[0].legend(["total", "robust matching", "registration reg", "atlas reg"])
            fig_at_estimate.savefig(os.path.join(outdir, f"atlas_{it_big_loop:06d}.png"))

        print(
            f"iter {it_big_loop + 1}/{niter_big_loop} "
            f"total={Esave[-1][0]:.6g} match={Esave[-1][1]:.6g} "
            f"reg={Esave[-1][2]:.6g} atlas={Esave[-1][3]:.6g}",
            flush=True,
        )

    np.savez(os.path.join(outdir, "A.npz"), data=A)
    np.savez(os.path.join(outdir, "v.npz"), data=v)
    np.savez(os.path.join(outdir, "Esave.npz"), data=Esave)
    np.savez(os.path.join(outdir, "RphiI.npz"), data=RphiI)
    np.savez(os.path.join(outdir, "phiiRiJ.npz"), data=phiiRiJ)
    np.savez(os.path.join(outdir, "Wshow.npz"), data=Wshow)
    np.savez(os.path.join(outdir, "W_robust_loss.npz"), data=W_robust_loss)
    plt.close("all")
    print(f"Wrote outputs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
