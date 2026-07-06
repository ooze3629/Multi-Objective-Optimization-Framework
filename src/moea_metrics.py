# -*- coding: utf-8 -*-
"""
多目標演算法評估指標:C-metric(兩集合覆蓋)、IGD⁺、Hypervolume,
以及多種子重複的統計(平均 ± 標準差 + Wilcoxon 配對檢定)。

比較對象:環境選擇策略 nd_first(非支配優先) vs three_tier(極端值優先三層)。
每個種子各跑兩版,指標以「該種子兩版聯集」做共同正規化與參考前緣(配對比較)。
"""
import warnings; warnings.filterwarnings("ignore")
import math
import statistics as stats
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC",
                                          "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import MOGA_GPSIFF_patrol_clean as M
from compare_sel import run_cmp, _hv3d
from scipy.stats import wilcoxon

OBJS = ("F1", "F2", "F3")


# ---------------- 支配關係 ----------------
def weakly_dominates(a, b):
    """a 覆蓋 b(各目標皆不劣):F1 取大、F2/F3 取小。"""
    return a["F1"] >= b["F1"] and a["F2"] <= b["F2"] and a["F3"] <= b["F3"]


def nondominated(arc):
    nd = []
    for i, p in enumerate(arc):
        if not any(j != i and M.dominates(arc[j], p) for j in range(len(arc))):
            nd.append(p)
    return nd


# ---------------- C-metric ----------------
def c_metric(A, B):
    """C(A,B):B 中被 A 某解覆蓋(弱支配)的比例,∈[0,1],越大代表 A 越優於 B。"""
    if not B:
        return 0.0
    cov = sum(1 for b in B if any(weakly_dominates(a, b) for a in A))
    return cov / len(B)


# ---------------- 正規化(共同:聯集範圍;F1 轉最小化,0=最佳)----------------
def _bounds(*archives):
    allp = [p for arc in archives for p in arc]
    lo = {f: min(p[f] for p in allp) for f in OBJS}
    hi = {f: max(p[f] for p in allp) for f in OBJS}
    return lo, hi


def _norm_pts(arc, lo, hi):
    rng = {f: (hi[f] - lo[f]) or 1.0 for f in OBJS}
    return [((hi["F1"] - p["F1"]) / rng["F1"],
            (p["F2"] - lo["F2"]) / rng["F2"],
            (p["F3"] - lo["F3"]) / rng["F3"]) for p in arc]


# ---------------- HV / IGD⁺ ----------------
def hv(arc, lo, hi, margin=1.1):
    return _hv3d(_norm_pts(arc, lo, hi), (margin, margin, margin))


def igd_plus(arc, ref, lo, hi):
    """
    IGD⁺(最小化、正規化空間):每個參考點到 obtained set 的 d⁺ 平均,越小越好。
    d⁺(z,a) = sqrt( Σ max(a_i - z_i, 0)^2 )(僅計 a 比 z 差的分量,Pareto 相容)。
    """
    A = _norm_pts(arc, lo, hi)
    Z = _norm_pts(ref, lo, hi)
    if not A or not Z:
        return float("nan")
    total = 0.0
    for z in Z:
        total += min(math.sqrt(sum(max(a[d] - z[d], 0.0) ** 2 for d in range(3))) for a in A)
    return total / len(Z)


# ---------------- 多種子實驗 ----------------
def run_multiseed(seeds, pop_size=100, max_fes=3000):
    rec = {"nd_first": {"HV": [], "IGD+": [], "n": []},
           "three_tier": {"HV": [], "IGD+": [], "n": []},
           "C_tt_nd": [], "C_nd_tt": []}
    for s in seeds:
        _, arcA, _, _ = run_cmp("nd_first", s, pop_size, max_fes)     # A = nd_first
        _, arcB, _, _ = run_cmp("three_tier", s, pop_size, max_fes)   # B = three_tier
        lo, hi = _bounds(arcA, arcB)
        ref = nondominated(arcA + arcB)                               # 共同參考前緣
        rec["nd_first"]["HV"].append(hv(arcA, lo, hi))
        rec["three_tier"]["HV"].append(hv(arcB, lo, hi))
        rec["nd_first"]["IGD+"].append(igd_plus(arcA, ref, lo, hi))
        rec["three_tier"]["IGD+"].append(igd_plus(arcB, ref, lo, hi))
        rec["nd_first"]["n"].append(len(arcA))
        rec["three_tier"]["n"].append(len(arcB))
        rec["C_tt_nd"].append(c_metric(arcB, arcA))                   # three_tier 覆蓋 nd_first
        rec["C_nd_tt"].append(c_metric(arcA, arcB))                   # nd_first 覆蓋 three_tier
        print(f"  seed {s}: HV nd={rec['nd_first']['HV'][-1]:.4f} tt={rec['three_tier']['HV'][-1]:.4f} "
              f"| IGD+ nd={rec['nd_first']['IGD+'][-1]:.4f} tt={rec['three_tier']['IGD+'][-1]:.4f} "
              f"| C(tt,nd)={rec['C_tt_nd'][-1]:.2f} C(nd,tt)={rec['C_nd_tt'][-1]:.2f}")
    return rec


def _ms(xs):
    return stats.mean(xs), (stats.pstdev(xs) if len(xs) > 1 else 0.0)


def report(rec):
    print("\n================ 多種子統計(平均 ± 標準差)================")
    for metric, better in (("HV", "越大越好"), ("IGD+", "越小越好")):
        ma, sa = _ms(rec["nd_first"][metric])
        mb, sb = _ms(rec["three_tier"][metric])
        print(f"{metric}({better}): nd_first {ma:.4f}±{sa:.4f} | three_tier {mb:.4f}±{sb:.4f}")
        try:
            stat, p = wilcoxon(rec["nd_first"][metric], rec["three_tier"][metric])
            print(f"    Wilcoxon 配對檢定 p = {p:.4f}")
        except Exception as e:
            print(f"    Wilcoxon 無法計算({e})")
    na, sa = _ms(rec["nd_first"]["n"]); nb, sb = _ms(rec["three_tier"]["n"])
    print(f"非支配解數: nd_first {na:.0f}±{sa:.0f} | three_tier {nb:.0f}±{sb:.0f}")
    mc1, sc1 = _ms(rec["C_tt_nd"]); mc2, sc2 = _ms(rec["C_nd_tt"])
    print("\nC-metric(覆蓋比例,越大代表越優):")
    print(f"    C(three_tier, nd_first) = {mc1:.3f}±{sc1:.3f}")
    print(f"    C(nd_first, three_tier) = {mc2:.3f}±{sc2:.3f}")


def plot_box(rec, save):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    try:
        axes[0].boxplot([rec["nd_first"]["HV"], rec["three_tier"]["HV"]], tick_labels=["nd_first", "three_tier"])
    except TypeError:
        axes[0].boxplot([rec["nd_first"]["HV"], rec["three_tier"]["HV"]], labels=["nd_first", "three_tier"])
    axes[0].set_title("Hypervolume(越大越好)"); axes[0].grid(alpha=0.3)
    try:
        axes[1].boxplot([rec["nd_first"]["IGD+"], rec["three_tier"]["IGD+"]], tick_labels=["nd_first", "three_tier"])
    except TypeError:
        axes[1].boxplot([rec["nd_first"]["IGD+"], rec["three_tier"]["IGD+"]], labels=["nd_first", "three_tier"])
    axes[1].set_title("IGD+(越小越好)"); axes[1].grid(alpha=0.3)
    fig.suptitle("多種子指標分布")
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)


if __name__ == "__main__":
    import time
    SEEDS = [1, 2, 3, 4, 5]          # 正式可加到 10+
    POP, FES = 100, 3000             # 正式可用 100 / 10000(較久)
    print(f"多種子實驗:seeds={SEEDS}, pop={POP}, FEs={FES}")
    t = time.time()
    rec = run_multiseed(SEEDS, POP, FES)
    report(rec)
    plot_box(rec, "metrics_boxplot.png")
    print(f"\n耗時 {time.time()-t:.1f}s | 圖:metrics_boxplot.png")
