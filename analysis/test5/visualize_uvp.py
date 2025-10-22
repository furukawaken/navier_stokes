# save_triptych_frames.py
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import sys
from netCDF4 import Dataset

# =======================
# CONFIG
# =======================
NC_PATH   = Path(__file__).resolve().parent / "../../output/test5/ns.nc"
OUTDIR    = Path(__file__).resolve().parent / "./fig/uvp"
DPI       = 200
CMAP     = "viridis"
EVERY_N  = None       # 例: 5 にすると 0,5,10,... 時刻のみ出力。None なら全フレーム。
USE_QUANTILE = False  # True にすると 1–99% 分位でカラースケール固定（全データ走査します）
# =======================

def compute_limits(nc):
    # Reading variables
    u_var = nc.variables["u"]   # (time, y, xi)
    v_var = nc.variables["v"]   # (time, eta, x)
    p_var = nc.variables["p"]   # (time, y, x)
    nt = u_var.shape[0]

    # Scanning maximums of u, v, p
    if USE_QUANTILE:
        u = u_var[:]                           # (t,y,xi)
        u_c = 0.5*(u + np.roll(u, 1, axis=2))  # (t,y,xi) ~ (t,y,x)

        v = v_var[:]                           # (t,eta,x)
        v_c = 0.5*(v + np.roll(v, 1, axis=1))  # (t,eta,x) ~ (t,y,x)

        p = p_var[:]                           # (t,y,x)

        u_vmin, u_vmax = np.quantile(u_c, [0.01, 0.99])
        v_vmin, v_vmax = np.quantile(v_c, [0.01, 0.99])
        p_vmin, p_vmax = np.quantile(p,   [0.01, 0.99])
    else:
        u_vmin = v_vmin = p_vmin = np.inf
        u_vmax = v_vmax = p_vmax = -np.inf
        for k in range(nt):
            u_slice = u_var[k, :, :]                       # (y,xi)
            u_c = 0.5*(u_slice + np.roll(u_slice, 1, axis=1))
            u_vmin = min(u_vmin, np.min(u_c))
            u_vmax = max(u_vmax, np.max(u_c))

            v_slice = v_var[k, :, :]                       # (eta,x)
            v_c = 0.5*(v_slice + np.roll(v_slice, 1, axis=0))
            v_vmin = min(v_vmin, np.min(v_c))
            v_vmax = max(v_vmax, np.max(v_c))

            p_slice = p_var[k, :, :]                       # (y,x)
            p_vmin = min(p_vmin, np.min(p_slice))
            p_vmax = max(p_vmax, np.max(p_slice))

    return (float(u_vmin), float(u_vmax)), (float(v_vmin), float(v_vmax)), (float(p_vmin), float(p_vmax))

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with Dataset(str(NC_PATH), "r") as nc:
        time = nc.variables["time"][:]   # (t,)
        x    = nc.variables["x"][:]      # (x,)
        y    = nc.variables["y"][:]      # (y,)
        u    = nc.variables["u"]         # (t,y,xi)
        v    = nc.variables["v"]         # (t,eta,x)
        p    = nc.variables["p"]         # (t,y,x)

        (u_lims, v_lims, p_lims) = compute_limits(nc)
        (u_vmin, u_vmax) = u_lims
        (v_vmin, v_vmax) = v_lims
        (p_vmin, p_vmax) = p_lims

        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

        t_indices = list(range(len(time)))
        if EVERY_N is not None and EVERY_N > 1:
            t_indices = [i for i in t_indices if i % EVERY_N == 0]

        for k in t_indices:
            t = float(time[k])

            # u_centered:
            u_slice = u[k, :, :]                               # (y,xi)
            u_c = 0.5*(u_slice + np.roll(u_slice, 1, axis=1))  # (y,x)

            # v_centered:
            v_slice = v[k, :, :]                               # (eta,x)
            v_c = 0.5*(v_slice + np.roll(v_slice, 1, axis=0))  # (y,x)

            # p
            p_slice = p[k, :, :]                               # (y,x)

            #fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
            fig, axes = plt.subplots(
                1, 3, figsize=(15, 5.0),
                gridspec_kw={
                    "top":    0.80,
                    "bottom": 0.10,
                    "left":   0.04,
                    "right":  0.96,
                    "wspace": 0.30
                }
            )

            # Left
            im0 = axes[0].imshow(
                u_c, 
                origin="lower",
                extent=extent,
                vmin=u_vmin,
                vmax=u_vmax,
                cmap=CMAP,
                aspect="equal"
            )
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            axes[0].set_title("u (cell-centered)", fontsize=16)
            # Middle
            im1 = axes[1].imshow(
                v_c,
                origin="lower",
                extent=extent,
                vmin=v_vmin,
                vmax=v_vmax,
                cmap=CMAP,
                aspect="equal"
            )
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].set_title("v (cell-centered)", fontsize=16)
            #Right
            im2 = axes[2].imshow(
                p_slice,
                origin="lower",
                extent=extent,
                vmin=p_vmin,
                vmax=p_vmax,
                cmap=CMAP,
                aspect="equal"
            )
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            axes[2].set_title("p (cell-centered)", fontsize=16)

            for ax in axes:
                ax.set_xlabel("x", fontsize=14)
                ax.set_ylabel("y", fontsize=14)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

            fig.suptitle(f"Heatmap of (u, v, p) at t = {t:.6f}", fontsize=22, y=0.96)
            out_path = OUTDIR / f"snapshot_{k:05d}.png"
            fig.savefig(out_path, dpi=DPI)
            plt.close(fig)

    print(f"Saved PNGs to: {OUTDIR}/")

if __name__ == "__main__":
    main()