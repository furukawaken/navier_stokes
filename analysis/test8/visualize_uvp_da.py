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
NC_PATH   = Path(__file__).resolve().parent / "../../output/test8/ns.nc"
OUTDIR    = Path(__file__).resolve().parent / "./fig/uvp_da"
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
        u_da  = nc.variables["optional"]
        v_da  = nc.variables["optional2"]
        u_obs = nc.variables["optional3"]
        v_obs = nc.variables["optional4"]

        # For rmse
        rmse_u_true_da_list = []
        rmse_v_true_da_list = []
        rmse_vec_true_da_list = []
        rmse_vec_true_obs_list = []
        time_list = []

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

            # u_centered:
            u_da_slice = u_da[k, :, :]                               # (y,xi)
            u_da_c = 0.5*(u_da_slice + np.roll(u_da_slice, 1, axis=1))  # (y,x)

            # v_centered:
            v_da_slice = v_da[k, :, :]                               # (eta,x)
            v_da_c = 0.5*(v_da_slice + np.roll(v_da_slice, 1, axis=0))  # (y,x)

            # u_centered:
            u_obs_slice = u_obs[k, :, :]                               # (y,xi)
            u_obs_c = 0.5*(u_obs_slice + np.roll(u_obs_slice, 1, axis=1))  # (y,x)

            # v_centered:
            v_obs_slice = v_obs[k, :, :]                               # (eta,x)
            v_obs_c = 0.5*(v_obs_slice + np.roll(v_obs_slice, 1, axis=0))  # (y,x)

            # p
            #p_slice = p[k, :, :]                               # (y,x)

            fig, axes = plt.subplots(
                2, 3, figsize=(15, 10.0),
                gridspec_kw={
                    "top":    0.88,  # suptitle とかぶらないよう少し広げた
                    "bottom": 0.10,
                    "left":   0.06,
                    "right":  0.98,
                    "wspace": 0.30
                }
            )

            # ---- 上段: u ----
            im0 = axes[0, 0].imshow(
                u_c, origin="lower", extent=extent,
                vmin=u_vmin, vmax=u_vmax, cmap=CMAP, aspect="equal"
            )
            fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
            axes[0, 0].set_title("u_true", fontsize=16)

            im1 = axes[0, 1].imshow(
                u_obs_c, origin="lower", extent=extent,
                vmin=u_vmin, vmax=u_vmax, cmap=CMAP, aspect="equal"
            )
            fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            axes[0, 1].set_title("u_obs", fontsize=16)

            im2 = axes[0, 2].imshow(
                u_da_c, origin="lower", extent=extent,
                vmin=u_vmin, vmax=u_vmax, cmap=CMAP, aspect="equal"
            )
            fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
            axes[0, 2].set_title("u_da", fontsize=16)

            # ---- 下段: v ----
            im3 = axes[1, 0].imshow(
                v_c, origin="lower", extent=extent,
                vmin=v_vmin, vmax=v_vmax, cmap=CMAP, aspect="equal"
            )
            fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
            axes[1, 0].set_title("v_true", fontsize=16)

            im4 = axes[1, 1].imshow(
                v_obs_c, origin="lower", extent=extent,
                vmin=v_vmin, vmax=v_vmax, cmap=CMAP, aspect="equal"
            )
            fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
            axes[1, 1].set_title("v_obs", fontsize=16)

            im5 = axes[1, 2].imshow(
                v_da_c, origin="lower", extent=extent,
                vmin=v_vmin, vmax=v_vmax, cmap=CMAP, aspect="equal"
            )
            fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
            axes[1, 2].set_title("v_da", fontsize=16)

            for ax in axes.flat:  # ← .flat で全6枚を走査
                ax.set_xlabel("x", fontsize=14)
                ax.set_ylabel("y", fontsize=14)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

            fig.suptitle(f"Prediction by DA at t = {t:.6f}", fontsize=22, y=0.95)
            out_path = OUTDIR / f"snapshot_{k:05d}.png"
            fig.savefig(out_path, dpi=DPI)
            plt.close(fig)

            # For rmse
            du_true_da = u_da_c - u_c
            dv_true_da = v_da_c - v_c
            du_true_obs = u_obs_c - u_c
            dv_true_obs = v_obs_c - v_c

            rmse_u_true_da    = float(np.sqrt(np.nanmean(du_true_da**2)))
            rmse_v_true_da    = float(np.sqrt(np.nanmean(dv_true_da**2)))
            rmse_vec_true_da  = float(np.sqrt(np.nanmean((du_true_da**2 + dv_true_da**2) * 0.5)))  # ベクトルRMSE
            rmse_vec_true_obs = float(np.sqrt(np.nanmean((du_true_obs**2 + dv_true_obs**2) * 0.5)))  # ベクトルRMSE

            rmse_u_true_da_list.append(rmse_u_true_da)
            rmse_v_true_da_list.append(rmse_v_true_da)
            rmse_vec_true_da_list.append(rmse_vec_true_da)
            rmse_vec_true_obs_list.append(rmse_vec_true_obs)
            time_list.append(t)

        fig_rmse, ax = plt.subplots(figsize=(8, 4.5))
        #ax.plot(time_list, rmse_u_true_da_list, label="RMSE(u_true, u_da)", lw=2)
        #ax.plot(time_list, rmse_v_true_da_list, label="RMSE(v_true, v_da)", lw=2)
        ax.plot(time_list, rmse_vec_true_da_list, label="RMS(U_true - U_da)", lw=2, alpha=0.8)
        ax.plot(time_list, rmse_vec_true_obs_list, label="RMS(U_true - U_obs)", lw=2, alpha=0.8)

        ax.set_xlabel("time")
        ax.set_ylabel("RMSE")
        ax.set_ylim(0.0, 2.5)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend()
        fig_rmse.tight_layout()

        out_path_rmse = "./fig/rmse/rmse_timeseries.png"
        fig_rmse.savefig(out_path_rmse, dpi=DPI)
        plt.close(fig_rmse)

        # csv
        import csv
        with open("./fig/rmse/rmse_timeseries.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "rmse_u", "rmse_v", "rmse_vector"])
            for T, ru, rv, rvv in zip(time_list, rmse_u_true_da_list, rmse_v_true_da_list, rmse_vec_true_da_list):
                w.writerow([f"{T:.10g}", f"{ru:.10g}", f"{rv:.10g}", f"{rvv:.10g}"])

    print(f"Saved PNGs to: {OUTDIR}/")

if __name__ == "__main__":
    main()