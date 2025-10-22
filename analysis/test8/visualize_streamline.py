# save_triptych_frames.py
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import sys
from netCDF4 import Dataset
from typing import Tuple, Literal

# =======================
# CONFIG
# =======================
NC_PATH   = Path(__file__).resolve().parent / "../../output/test8/ns.nc"
OUTDIR    = Path(__file__).resolve().parent / "./fig/streamline"
DPI       = 200
CMAP     = "viridis"
EVERY_N  = None       # 例: 5 にすると 0,5,10,... 時刻のみ出力。None なら全フレーム。
USE_QUANTILE = False  # True にすると 1–99% 分位でカラースケール固定（全データ走査します）
Q_LOW, Q_HIGH = 0.01, 0.99
# =======================
def _to_ndarray(a):
    return a.filled(np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)

def compute_limits(nc):
    # Reading variables
    u_var = nc.variables["u"]   # (time, y, xi)
    v_var = nc.variables["v"]   # (time, eta, x)
    p_var = nc.variables["p"]   # (time, y, x)
    nt = u_var.shape[0]

    # Scanning maximums of u, v, p,
    if USE_QUANTILE:
        u = u_var[:]                           # (t,y,xi)
        u_c = 0.5*(u + np.roll(u, 1, axis=2))  # (t,y,xi) ~ (t,y,x)

        v = v_var[:]                           # (t,eta,x)
        v_c = 0.5*(v + np.roll(v, 1, axis=1))  # (t,eta,x) ~ (t,y,x)

        p = p_var[:]                           # (t,y,x)

        u_vmin, u_vmax = np.quantile(u_c, [0.01, 0.99])
        v_vmin, v_vmax = np.quantile(v_c, [0.01, 0.99])

        speed = np.hypot(u_c, v_c)                 # (t, y, x)
        s_vmin, s_vmax = np.nanquantile(speed, [Q_LOW, Q_HIGH])

    else:
        u_vmin = v_vmin = p_vmin = s_vmin =  np.inf
        u_vmax = v_vmax = p_vmax = s_vmax = -np.inf
        for k in range(nt):
            u_slice = _to_ndarray(u_var[k, :, :])          # (y, xi)
            v_slice = _to_ndarray(v_var[k, :, :])          # (eta, x)
            p_slice = _to_ndarray(p_var[k, :, :])          # (y, x)

            u_c = 0.5*(u_slice + np.roll(u_slice, 1, axis=1))  # (y, x)
            v_c = 0.5*(v_slice + np.roll(v_slice, 1, axis=0))  # (y, x)
            speed = np.hypot(u_c, v_c)

            u_vmin = min(u_vmin, np.nanmin(u_c));     u_vmax = max(u_vmax, np.nanmax(u_c))
            v_vmin = min(v_vmin, np.nanmin(v_c));     v_vmax = max(v_vmax, np.nanmax(v_c))
            p_vmin = min(p_vmin, np.nanmin(p_slice)); p_vmax = max(p_vmax, np.nanmax(p_slice))
            s_vmin = min(s_vmin, np.nanmin(speed));   s_vmax = max(s_vmax, np.nanmax(speed))

    return (
        (float(u_vmin), float(u_vmax)),
        (float(v_vmin), float(v_vmax)),
        (float(p_vmin), float(p_vmax)),
        (float(s_vmin), float(s_vmax)),
    )

def curl_from_cell_centered(u_c: np.ndarray, v_c: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dvdx = (np.roll(v_c, -1, axis=1) - np.roll(v_c, 1, axis=1)) / (2.0 * dx)
    dudy = (np.roll(u_c, -1, axis=0) - np.roll(u_c, 1, axis=0)) / (2.0 * dy)

    return dvdx - dudy

def vorticity_minmax_per_frame(u_c: np.ndarray, v_c: np.ndarray, dx: float, dy: float) -> Tuple[float, float]:
    w = curl_from_cell_centered(u_c, v_c, dx, dy)

    return float(np.nanmin(w)), float(np.nanmax(w))

def vorticity_range_quantile_per_frame(
    u_c: np.ndarray, v_c: np.ndarray, dx: float, dy: float,
    q_low: float = 0.01, q_high: float = 0.99
) -> Tuple[float, float]:
    w = curl_from_cell_centered(u_c, v_c, dx, dy)

    return float(np.nanquantile(w, q_low)), float(np.nanquantile(w, q_high))

def to_cell_centered(u_face: np.ndarray, v_face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u_c = 0.5 * (u_face + np.roll(u_face, 1, axis=1))
    v_c = 0.5 * (v_face + np.roll(v_face, 1, axis=0))
    return u_c, v_c

def vorticity_limits_over_time(
    u_var, v_var, dx: float, dy: float,
    method: Literal["minmax", "quantile"] = "quantile",
    q_low: float = 0.01, q_high: float = 0.99
) -> Tuple[float, float]:
    # Initialize
    vmin = np.inf
    vmax = -np.inf

    nt = u_var.shape[0]
    for k in range(nt):
        u = np.asarray(u_var[k, :, :])
        v = np.asarray(v_var[k, :, :])

        u_c, v_c = to_cell_centered(u, v)

        if method == "minmax":
            lo, hi = vorticity_minmax_per_frame(u_c, v_c, dx, dy)
        else:  # "quantile"
            lo, hi = vorticity_range_quantile_per_frame(u_c, v_c, dx, dy, q_low, q_high)

        if lo < vmin: vmin = lo
        if hi > vmax: vmax = hi

    return float(vmin), float(vmax)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with Dataset(str(NC_PATH), "r") as nc:
        dt  = float(nc.getncattr("dt"))        # or: float(nc.dt)
        dx  = float(nc.getncattr("dx"))
        dy  = float(nc.getncattr("dy"))
        time  = nc.variables["time"][:]         # (nt,)
        x     = nc.variables["x"][:]            # (nx,)
        y     = nc.variables["y"][:]            # (ny,)
        u_var = nc.variables["u"]               # (nt, ny, xi)
        v_var = nc.variables["v"]               # (nt, eta, nx)
        p_var = nc.variables["p"]               # (nt, ny, nx)
        u_da  = nc.variables["optional"]
        v_da  = nc.variables["optional2"]
        u_obs = nc.variables["optional3"]
        v_obs = nc.variables["optional4"]
        nt    = time.shape[0]

        (u_lims, v_lims, p_lims, s_lims) = compute_limits(nc)
        (u_vmin, u_vmax) = u_lims
        (v_vmin, v_vmax) = v_lims
        (p_vmin, p_vmax) = p_lims
        (s_vmin, s_vmax) = s_lims
        rot_uv_lims  = vorticity_limits_over_time(u_var, v_var, dx, dy)
        (rot_uv_vmin, rot_uv_vmax) = rot_uv_lims


        extent_img = [
            x[0]  - 0.5 * dx,
            x[-1] + 0.5 * dx,
            y[0]  - 0.5 * dy,
            y[-1] + 0.5 * dy
        ]
        X, Y = np.meshgrid(x, y)

        for k in range(nt):
            t = float(time[k])
            u = u_var[k, :, :]                                  # (ny, xi)
            v = v_var[k, :, :]                                  # (eta, nx)
            u_c = 0.5 * (u + np.roll(u, 1, axis=1))             # (ny, nx)
            v_c = 0.5 * (v + np.roll(v, 1, axis=0))             # (ny, nx)
            speed  = np.hypot(u_c, v_c)                         # sqrt(u_c^2 + v_c^2)

            fig, axes = plt.subplots(
                #1, 3, figsize=(15, 5.0),
                1, 2, figsize=(10, 5.0),
                gridspec_kw={
                    "top":    0.80,
                    "bottom": 0.10,
                    "left":   0.04,
                    "right":  0.96,
                    "wspace": 0.30
                }
            )

            # Left
            # Plotting |(u, v)| as wallpaper
            im0 = axes[0].imshow(
                speed,
                origin="lower",
                extent=extent_img,
                cmap=CMAP,
                vmin=0.0,
                vmax=s_vmax,
                aspect="equal"
            )
            # Plotting streamline
            norm = matplotlib.colors.Normalize(
                vmin=float(speed.min()),
                vmax=float(speed.max())
            )
            axes[0].streamplot(
                x,
                y,
                u_c,
                v_c,
                density=1.2,
                linewidth=1.0,
                color=speed,
                cmap="plasma",
                norm=norm,
                arrowsize=1.2
            )
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            axes[0].set_title("Stream Line and |(u, v)|", fontsize=16)

            # Right
            rot_uv = curl_from_cell_centered(u_c, v_c, dx, dy)
            im1 =axes[1].imshow(
                rot_uv,
                origin="lower",
                extent=extent_img,
                cmap=CMAP,
                vmin=rot_uv_vmin,
                vmax=rot_uv_vmax,
                aspect="equal"
            )
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].set_title("Rotation", fontsize=16)
            
            for ax in axes:
                ax.set_xlabel("x", fontsize=14)
                ax.set_ylabel("y", fontsize=14)
                ax.set_xlim(extent_img[0], extent_img[1])
                ax.set_ylim(extent_img[2], extent_img[3])

            fig.suptitle(f"Flow and Rotation filed at t = {t:.6f}", fontsize=22, y=0.96)
            fig.savefig(OUTDIR / f"snapshot_{k:05d}.png", dpi=200)
            plt.close(fig)
            #sys.exit()

    print(f"Saved PNGs to: {OUTDIR}/")

if __name__ == "__main__":
    main()