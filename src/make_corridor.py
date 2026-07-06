# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from scipy.ndimage import binary_dilation
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC","Noto Sans CJK JP","Microsoft JhengHei","DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

W, H = 72, 48
xx, yy = np.meshgrid(np.arange(W), np.arange(H))
weight = np.full((H, W), 0.15)
for (cx, cy, a) in [(28, 34, 1.0), (54, 16, 0.85)]:
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    weight = np.maximum(weight, a * np.exp(-(d / 8.5) ** 2))
nogo = np.zeros((H, W), bool); nogo[19:29, 34:45] = True
weight[nogo] = 0; legal = ~nogo

pts = [(10, 9), (24, 39), (46, 35), (62, 14), (10, 9)]
def line_cells(p1, p2):
    n = max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]))
    if n == 0: return [p1]
    return [(int(round(p1[0]+(p2[0]-p1[0])*t/n)), int(round(p1[1]+(p2[1]-p1[1])*t/n))) for t in range(n+1)]
routemask = np.zeros((H, W), bool)
for a in range(len(pts)-1):
    for (x, y) in line_cells(pts[a], pts[a+1]):
        if 0 <= x < W and 0 <= y < H: routemask[y, x] = True
R = 4
yk, xk = np.ogrid[-R:R+1, -R:R+1]; disk = (xk*xk + yk*yk) <= R*R
corridor = binary_dilation(routemask, structure=disk) & legal

fig, ax = plt.subplots(figsize=(8.4, 5.6))
ax.imshow(weight, cmap="YlOrRd", origin="lower", alpha=0.75, vmin=0, vmax=1)
ax.imshow(np.ma.masked_where(~nogo, nogo), cmap=ListedColormap(["#888888"]), origin="lower", alpha=0.7)
ax.imshow(np.ma.masked_where(~corridor, corridor), cmap=ListedColormap(["#2E75B6"]), origin="lower", alpha=0.32)
rx = [p[0] for p in pts]; ry = [p[1] for p in pts]
ax.plot(rx, ry, color="#111111", lw=2.2, zorder=5)
ax.scatter(rx[1:-1], ry[1:-1], color="#111111", s=46, zorder=6)
ax.scatter([rx[0]], [ry[0]], color="red", marker="*", s=200, zorder=7, edgecolors="k", linewidths=0.5)
for (sx, sy) in [(24, 39), (46, 35), (62, 14)]:
    ax.add_patch(Circle((sx, sy), R, fill=False, ls="--", ec="#1F4E79", lw=1.4, zorder=6))

ax.annotate("基地(起訖)", (rx[0], ry[0]), xytext=(13, 4), textcoords="data", fontsize=11, color="red")
ax.annotate("巡邏點", (rx[2], ry[2]), xytext=(48, 41), textcoords="data", fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#111"))
ax.annotate("感測半徑 r\n(圓盤足跡)", (62+R*0.7, 14+R*0.7), xytext=(50, 4), textcoords="data", fontsize=10.5, color="#1F4E79",
            arrowprops=dict(arrowstyle="->", color="#1F4E79"))
ax.annotate("覆蓋走廊\n(航跡 ⊕ 圓盤 ∩ 合法海域)", (34, 26), xytext=(2, 2), textcoords="data", fontsize=10.5, color="#1F4E79")
ax.annotate("禁航區(不計覆蓋)", (39, 24), xytext=(30, 44.5), textcoords="data", fontsize=10.5, color="#333",
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("高權重海域", (28, 34), xytext=(13, 30), textcoords="data", fontsize=10.5, color="#8B0000")

ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("圖 2-1　F1 走廊覆蓋示意圖", fontsize=13, fontweight="bold", color="#1F4E79")
fig.tight_layout(); fig.savefig("corridor.png", dpi=140, bbox_inches="tight")
from PIL import Image; print("corridor", Image.open("corridor.png").size)
