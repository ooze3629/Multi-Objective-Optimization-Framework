# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import pickle, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC","Noto Sans CJK JP","Microsoft JhengHei","DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import moea_metrics as MM
import baselines as B


if __name__ == "__main__":
    d = pickle.load(open("formal_results.pkl", "rb"))
    seeds = d["meta"]["seeds"]; scns = list(d["runs"].keys()); ALG = B.ALGOS
    OBJS = ("F1", "F2", "F3")

    def extent(arc, lo, hi):
        if not arc: return 0.0
        vals = {o: [a[o] for a in arc] for o in OBJS}
        rng = []
        for o in OBJS:
            denom = (hi[o] - lo[o]) or 1.0
            rng.append((max(vals[o]) - min(vals[o])) / denom)
        return sum(rng) / 3.0

    agg = {s: {a: {"HV": [], "IGD+": [], "extent": [], "n": []} for a in ALG} for s in scns}
    cself = {s: {a: [] for a in ["ES", "隨機", "貪婪"]} for s in scns}   # C(本方法, a)
    for s in scns:
        g = d["greedy"][s]
        for sd in seeds:
            rec = dict(d["runs"][s][sd]); rec["貪婪"] = g
            rec = {a: MM.nondominated(rec[a]) for a in ALG}
            allarc = [x for a in ALG for x in rec[a]]
            lo, hi = MM._bounds(allarc); ref = MM.nondominated(allarc)
            for a in ALG:
                agg[s][a]["HV"].append(MM.hv(rec[a], lo, hi))
                agg[s][a]["IGD+"].append(MM.igd_plus(rec[a], ref, lo, hi))
                agg[s][a]["extent"].append(extent(rec[a], lo, hi))
                agg[s][a]["n"].append(len(rec[a]))
            for a in ["ES", "隨機", "貪婪"]:
                cself[s][a].append(MM.c_metric(rec["本方法"], rec[a]))

    def ms(x): return st.mean(x), (st.pstdev(x) if len(x) > 1 else 0.0)

    # ---- 表格數值 ----
    print("=== 比對表(平均±標準差,n=10):HV↑ / IGD+↓ / 前緣延展↑ / 非支配解數 ===")
    for s in scns:
        print(f"\n[{s}]")
        for a in ALG:
            hv = ms(agg[s][a]["HV"]); ig = ms(agg[s][a]["IGD+"]); ex = ms(agg[s][a]["extent"]); n = ms(agg[s][a]["n"])
            print(f"  {a:4s}  HV {hv[0]:.3f}±{hv[1]:.3f}  IGD+ {ig[0]:.3f}±{ig[1]:.3f}  "
                  f"延展 {ex[0]:.3f}±{ex[1]:.3f}  解數 {n[0]:.0f}±{n[1]:.0f}")

    # ---- C-metric 盒鬚圖(1x3,每海域 3 盒:C(本方法, ES/隨機/貪婪)) ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, s in zip(axes, scns):
        data = [cself[s]["ES"], cself[s]["隨機"], cself[s]["貪婪"]]
        try:
            bp = ax.boxplot(data, tick_labels=["vs ES", "vs 隨機", "vs 貪婪"], patch_artist=True)
        except TypeError:
            bp = ax.boxplot(data, labels=["vs ES", "vs 隨機", "vs 貪婪"], patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#2E9E5B", "#E08A1E", "#999999"]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_title(s, fontsize=12); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("C(本方法, ·)  越高代表本方法支配越多")
    fig.suptitle("本方法對各基準之覆蓋率 C-metric(10 種子分布)", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"cmetric_boxplot_{B.M.DRONE_TIER}.png", dpi=130); plt.close(fig)
    from PIL import Image
    print("\ncmetric_boxplot.png", Image.open("cmetric_boxplot.png").size)
