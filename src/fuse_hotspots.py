# -*- coding: utf-8 -*-
"""
融合熱點:巡邏優先度 = w_ais·AIS 船舶密度(合法交通) + w_dark·SAR 暗船(無 AIS)
================================================================================
兩層皆為已正規化([0,1])之真實熱點(S21 / S22)。加權和後重新正規化到 [0,1],
海域 clip,輸出單一「巡邏優先度」熱點 data/scenario_TWreal_priority_v1.npy。

用法:
    python fuse_hotspots.py            # 預設 0.6 AIS / 0.4 暗船
    python fuse_hotspots.py 0.5 0.5
權重若不合 1 會自動正規化。
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import MOGA_GPSIFF_patrol_clean as M

DATA = os.path.join(M.SCRIPT_DIR, "..", "data")
FIG = os.path.join(M.SCRIPT_DIR, "..", "figures")


def main():
    w_ais = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    w_dark = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    s = w_ais + w_dark
    w_ais, w_dark = w_ais / s, w_dark / s          # 正規化權重

    ais = np.load(os.path.join(DATA, "scenario_TWreal_ais_v1.npy"))
    dark = np.load(os.path.join(DATA, "scenario_TWreal_sar_dark_v1.npy"))
    assert ais.shape == dark.shape == M.no_go_zone.shape

    fused = w_ais * ais + w_dark * dark
    fused[M.no_go_zone != 0] = 0.0                 # 海域 clip
    if fused.max() > 0:
        fused = fused / fused.max()                # 重新正規化到 [0,1]
    np.save(os.path.join(DATA, "scenario_TWreal_priority_v1.npy"), fused)
    print(f"巡邏優先度:w_AIS={w_ais:.2f} w_dark={w_dark:.2f};非零海域格={int((fused>0).sum())}")

    no_go = M.no_go_zone
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(np.where(no_go == 1, 1, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=1.5)
    im = ax.imshow(np.where(no_go == 0, fused, np.nan), origin="lower", cmap="hot_r", vmin=0, vmax=1, alpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="patrol priority (norm.)")
    for (x, y) in M.base_ports:
        ax.plot(x, y, "*", ms=9, mfc="gold", mec="k", mew=0.6)
    ax.set_title(f"TWreal patrol-priority hotspot  ({w_ais:.1f}·AIS + {w_dark:.1f}·dark)", fontsize=11)
    ax.set_xlim(0, no_go.shape[1]); ax.set_ylim(0, no_go.shape[0]); ax.set_aspect("equal")
    out = os.path.join(FIG, "twreal_priority_hotspot.png")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("saved:", os.path.join(DATA, "scenario_TWreal_priority_v1.npy"), "and", out)


if __name__ == "__main__":
    main()
