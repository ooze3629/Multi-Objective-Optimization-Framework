# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC","Noto Sans CJK JP","Microsoft JhengHei","DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import MOGA_GPSIFF_patrol_clean as M
from compare_sel import run_cmp

def draw_path(ax, ch):
    ax.imshow(M.weight_map, cmap="YlOrRd", origin="lower", alpha=0.55, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(M.no_go_zone == 0, M.no_go_zone), cmap="Greys", origin="lower", alpha=0.5)
    colors = cm.get_cmap("tab20", M.NUM_BASES)
    for v in range(M.NUM_VESSELS):
        verts = ch["routes"][v]
        ax.plot([p[0] for p in verts], [p[1] for p in verts], color=colors(v // 2),
                lw=1.2, ls=("-" if v % 2 == 0 else "--"), alpha=0.85)
        pts = [M.Patrol_Point[i] for i in ch["assignment"][v]]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=colors(v // 2), s=10, zorder=3)
    for name, (bx, by) in M.base.items():
        ax.scatter(bx, by, c="red", s=55, marker="*", zorder=5, edgecolors="k", linewidths=0.4)
    ax.set_xlim(0, M.MAP_W); ax.set_ylim(0, M.MAP_H); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

def montage(pop, label, save):
    b1 = max(pop, key=lambda c: c["F1"]); b2 = min(pop, key=lambda c: c["F2"]); b3 = min(pop, key=lambda c: c["F3"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
    for ax, (ch, lab) in zip(axes, [(b1, "F1 最佳(最大覆蓋)"), (b2, "F2 最佳(最低成本)"), (b3, "F3 最佳(最小協同距離)")]):
        draw_path(ax, ch)
        ax.set_title("%s\nF1=%.0f  F2=%.0f  F3=%.0f" % (lab, ch["F1"], ch["F2"], ch["F3"]), fontsize=11)
    fig.suptitle(label, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig(save, dpi=110); plt.close(fig)


if __name__ == "__main__":
    t = time.time()
    popA = run_cmp("nd_first", 42, 100, 8000)[0]
    print("nd_first done %.0fs" % (time.time() - t)); t = time.time()
    popB = run_cmp("three_tier", 42, 100, 8000)[0]
    print("three_tier done %.0fs" % (time.time() - t))
    montage(popA, "nd_first(非支配優先):各目標最佳解之巡邏路徑", "path_nd.png")
    montage(popB, "three_tier(極端值優先):各目標最佳解之巡邏路徑", "path_tt.png")
    from PIL import Image
    print("path_nd", Image.open("path_nd.png").size, "path_tt", Image.open("path_tt.png").size)
