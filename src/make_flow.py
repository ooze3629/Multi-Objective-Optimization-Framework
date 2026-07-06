# -*- coding: utf-8 -*-
"""圖 3-1:MOGA + GPSIFF 演算法流程圖。英文標籤;輸出彩色 flowchart.png 與黑白 flowchart_bw.png。"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

STEPS = [
    "Initialize population; evaluate F1, F2, F3",
    "GPSIFF scoring (p - q + c)",
    "Binary tournament selection",
    "Uniform crossover",
    "Point-wise mutation",
    "Repair: point-level + route detour (avoid no-go)",
    "Evaluate offspring (F1, F2, F3)",
    "(mu+lambda) three-tier environmental selection\n(extremes -> remaining nondominated -> GPSIFF fill)",
]


def build(bw):
    # 配色:彩色 vs 灰階
    if bw:
        PROC, PROCE, SE, SEE = "#E6E6E6", "#000000", "#C8C8C8", "#000000"
        DIA, DIAE, LOOP, YES = "#DDDDDD", "#000000", "#000000", "#000000"
        ARR = "#000000"
    else:
        PROC, PROCE, SE, SEE = "#DCE9F5", "#2E75B6", "#D9EAD3", "#5B8C3E"
        DIA, DIAE, LOOP, YES = "#FCE5CD", "#C77F1B", "#C0392B", "#5B8C3E"
        ARR = "#444444"

    fig, ax = plt.subplots(figsize=(6.4, 9.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 15); ax.axis("off")
    BOXW = 5.0; CX = 4.6; LOOPX = 8.7

    def box(y, text, fc, ec, h=0.95):
        ax.add_patch(FancyBboxPatch((CX-BOXW/2, y-h/2), BOXW, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.6, zorder=2))
        ax.text(CX, y, text, ha="center", va="center", fontsize=9.5, zorder=3)

    def diamond(y, text, w=3.6, h=1.3):
        ax.add_patch(Polygon([(CX, y+h/2), (CX+w/2, y), (CX, y-h/2), (CX-w/2, y)],
                             closed=True, fc=DIA, ec=DIAE, lw=1.6, zorder=2))
        ax.text(CX, y, text, ha="center", va="center", fontsize=9.5, zorder=3)

    def arrow(y1, y2):
        ax.add_patch(FancyArrowPatch((CX, y1), (CX, y2), arrowstyle="-|>", mutation_scale=15, lw=1.5, color=ARR, zorder=1))

    ys = [13.6, 12.2, 11.0, 9.8, 8.6, 7.4, 6.2, 4.8]
    box(ys[0], STEPS[0], SE, SEE)
    for i in range(1, 8):
        box(ys[i], STEPS[i], PROC, PROCE, h=(1.15 if i == 7 else 0.95))
    for i in range(len(ys)-1):
        arrow(ys[i]-0.48, ys[i+1] + (0.58 if i+1 == 7 else 0.48))

    dy = 3.1
    diamond(dy, "Reach FEs budget?")
    arrow(ys[7]-0.58, dy+0.65)
    oy = 1.2
    box(oy, "Output Pareto front", SE, SEE)
    arrow(dy-0.65, oy+0.48)
    ax.text(CX+0.18, (dy+oy)/2+0.1, "Yes", fontsize=10.5, color=YES, zorder=3)

    # 回饋線(No):菱形右頂點 → 右側垂直上行 → 回到 GPSIFF 框右緣
    dright = CX + 1.8
    ax.add_patch(FancyArrowPatch((dright, dy), (LOOPX, dy), arrowstyle="-", lw=1.5, color=LOOP, zorder=1))
    ax.add_patch(FancyArrowPatch((LOOPX, dy), (LOOPX, ys[1]), arrowstyle="-", lw=1.5, color=LOOP, zorder=1))
    ax.add_patch(FancyArrowPatch((LOOPX, ys[1]), (CX+BOXW/2, ys[1]), arrowstyle="-|>", mutation_scale=15, lw=1.5, color=LOOP, zorder=1))
    ax.text(LOOPX+0.15, (dy+ys[1])/2, "No (not reached)", fontsize=10, color=LOOP, rotation=90, va="center")

    ax.text(CX, 14.5, "Fig. 3-1  MOGA + GPSIFF Algorithm Flow", ha="center", fontsize=12.5,
            fontweight="bold", color=("#000000" if bw else "#1F4E79"))
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from PIL import Image
    for bw, name in [(False, "flowchart.png"), (True, "flowchart_bw.png")]:
        fig = build(bw); fig.savefig(name, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(("彩色" if not bw else "黑白") + "流程圖存檔:", name, Image.open(name).size)
