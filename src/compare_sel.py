# -*- coding: utf-8 -*-
"""比對 μ+λ 環境選擇策略:nd_first(非支配優先)vs three_tier(極端值優先三層)。"""
import warnings; warnings.filterwarnings("ignore")
import random, time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import MOGA_GPSIFF_patrol_clean as M


def run_cmp(mode, seed, pop_size=100, max_fes=10100, snap_every=2, crossover_rate=None):
    random.seed(seed); np.random.seed(seed)
    cr = M.CROSSOVER_RATE if crossover_rate is None else crossover_rate
    fe = 0
    population = M.init_population(pop_size)
    for ch in population:
        M.evaluate(ch); fe += 1
    archive, seen, history, gen = [], set(), [], 0
    snapshots = []                                     # (fes, [(F1,F2,F3),...]) 供 HV 收斂曲線
    while fe < max_fes:
        gen += 1
        fit, _ = M.gpsiff(population)
        mating = M.binary_tournament(population, fit)
        offspring = []
        while len(offspring) < pop_size:
            if random.random() < cr:
                p1, p2 = random.sample(mating, 2)
                c1, c2 = M.crossover(p1, p2)
                offspring.append(M.mutation(c1))
                if len(offspring) < pop_size:
                    offspring.append(M.mutation(c2))
            else:
                offspring.append(M.mutation(deepcopy(random.choice(mating))))
        offspring = offspring[:pop_size]
        ev = 0
        for ch in offspring:
            M.evaluate(ch); fe += 1; ev += 1
            if fe >= max_fes:
                break
        offspring = offspring[:ev]
        combined = population + offspring
        cfit, cq = M.gpsiff(combined)
        nd = [i for i in range(len(combined)) if cq[i] == 0]
        dom = [i for i in range(len(combined)) if cq[i] > 0]
        if mode == "nd_first":
            chosen = (sorted(nd, key=lambda i: -cfit[i]) +
                      sorted(dom, key=lambda i: -cfit[i]))[:pop_size]
        elif mode == "three_tier":
            ex = M.nd_extremes(combined, nd); exset = set(ex)
            nd_rest = sorted((i for i in nd if i not in exset), key=lambda i: -cfit[i])
            chosen = (ex + nd_rest + sorted(dom, key=lambda i: -cfit[i]))[:pop_size]
        population = [combined[i] for i in chosen]
        pf = M.pareto_front(population)
        for ch in pf:
            key = (round(ch["F1"], 6), round(ch["F2"], 6), round(ch["F3"], 6))
            if key not in seen:
                seen.add(key)
                archive.append({"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
        history.append({"fes": fe,
                        "bestF1": max(c["F1"] for c in population),
                        "minF2": min(c["F2"] for c in population),
                        "minF3": min(c["F3"] for c in population),
                        "pareto": len(pf)})
        if gen % snap_every == 0 or fe >= max_fes:
            snapshots.append((fe, [(r["F1"], r["F2"], r["F3"]) for r in archive]))
    return population, archive, history, snapshots


def spread(arc, f):
    vals = [r[f] for r in arc]
    return min(vals), max(vals), max(vals) - min(vals)


def _hv3d(front, ref):
    """精確 3D hypervolume(最小化、參考點 ref 須大於所有點)。O(m^2)。"""
    pts = [tuple(p) for p in front if all(p[d] < ref[d] for d in range(3))]
    nd = []                                            # 3D 非支配過濾(保險)
    for p in pts:
        if not any(q != p and q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]
                   and (q[0] < p[0] or q[1] < p[1] or q[2] < p[2]) for q in pts):
            nd.append(p)
    if not nd:
        return 0.0
    nd.sort(key=lambda p: p[0])                        # 依 obj0 升冪切片
    r1, r2 = ref[1], ref[2]
    stair = []                                         # (obj1,obj2) 階梯,obj1 升冪、obj2 降冪
    hv = 0.0

    def area():
        a = 0.0
        for k in range(len(stair)):
            x, y = stair[k]
            nx = stair[k + 1][0] if k + 1 < len(stair) else r1
            a += (nx - x) * (r2 - y)
        return a

    for i in range(len(nd)):
        x0, y0 = nd[i][1], nd[i][2]
        if not any(sx <= x0 and sy <= y0 for sx, sy in stair):
            stair = [(sx, sy) for sx, sy in stair if not (sx >= x0 and sy >= y0)]
            stair.append((x0, y0)); stair.sort()
        t_i = nd[i][0]
        t_next = nd[i + 1][0] if i + 1 < len(nd) else ref[0]
        hv += (t_next - t_i) * area()
    return hv


def hv_compare(arcA, arcB, margin=1.1):
    """以兩集合聯集做共同正規化([0,1],皆轉最小化、0=最佳),參考點 (margin,)*3。"""
    allp = arcA + arcB
    lo = {f: min(r[f] for r in allp) for f in ("F1", "F2", "F3")}
    hi = {f: max(r[f] for r in allp) for f in ("F1", "F2", "F3")}
    rng = {f: (hi[f] - lo[f]) or 1.0 for f in ("F1", "F2", "F3")}

    def norm(arc):
        out = []
        for r in arc:
            o0 = (hi["F1"] - r["F1"]) / rng["F1"]      # F1 最大化 → 轉最小化(0=最佳)
            o1 = (r["F2"] - lo["F2"]) / rng["F2"]      # F2 最小化(0=最佳)
            o2 = (r["F3"] - lo["F3"]) / rng["F3"]      # F3 最小化(0=最佳)
            out.append((o0, o1, o2))
        return out

    ref = (margin, margin, margin)
    return _hv3d(norm(arcA), ref), _hv3d(norm(arcB), ref)


def hv_curve(snapshots, lo, hi, margin=1.1):
    """各快照(fes, 解集合)在固定正規化下的 HV → [(fes, hv), ...]。"""
    rng = {f: (hi[f] - lo[f]) or 1.0 for f in ("F1", "F2", "F3")}
    out = []
    for fes, pts in snapshots:
        norm = [((hi["F1"] - F1) / rng["F1"],
                 (F2 - lo["F2"]) / rng["F2"],
                 (F3 - lo["F3"]) / rng["F3"]) for (F1, F2, F3) in pts]
        out.append((fes, _hv3d(norm, (margin, margin, margin))))
    return out


def plot_hv_convergence(curveA, la, curveB, lb, save):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([c[0] for c in curveA], [c[1] for c in curveA], c="C0", marker="o", ms=3, label=la)
    ax.plot([c[0] for c in curveB], [c[1] for c in curveB], c="C3", marker="s", ms=3, label=lb)
    ax.set_xlabel("函數評估次數 (FEs)")
    ax.set_ylabel("Hypervolume(正規化,越大越好)")
    ax.set_title("Hypervolume 收斂曲線")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)


def plot_compare_pareto(arcA, la, arcB, lb, save):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (fx, fy) in zip(axes, [("F1", "F2"), ("F1", "F3"), ("F2", "F3")]):
        ax.scatter([r[fx] for r in arcA], [r[fy] for r in arcA], s=12, c="C0", alpha=0.45, label=la)
        ax.scatter([r[fx] for r in arcB], [r[fy] for r in arcB], s=12, c="C3", alpha=0.45, label=lb)
        ax.set_xlabel(fx); ax.set_ylabel(fy); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Pareto 前緣比對(跨代非支配解)")
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)


def plot_compare_conv(hA, la, hB, lb, save):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    keys = [("bestF1", "最佳 F1(↑)"), ("minF2", "最小 F2(↓)"), ("minF3", "最小 F3(↓)")]
    for ax, (k, t) in zip(axes, keys):
        ax.plot([h["fes"] for h in hA], [h[k] for h in hA], c="C0", label=la)
        ax.plot([h["fes"] for h in hB], [h[k] for h in hB], c="C3", label=lb)
        ax.set_title(t); ax.set_xlabel("FEs"); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("收斂比對")
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)


if __name__ == "__main__":
    SEED = 42
    t = time.time()
    popA, arcA, hA, snapA = run_cmp("nd_first", SEED)
    print("nd_first 完成 %.1fs | archive %d" % (time.time() - t, len(arcA)))
    t = time.time()
    popB, arcB, hB, snapB = run_cmp("three_tier", SEED)
    print("three_tier 完成 %.1fs | archive %d" % (time.time() - t, len(arcB)))

    print("\n=== Pareto 前緣延展(min, max, 範圍寬度)===")
    for f in ("F1", "F2", "F3"):
        a = spread(arcA, f); b = spread(arcB, f)
        print(f"{f}: nd_first 寬度 {a[2]:.1f}  | three_tier 寬度 {b[2]:.1f}  "
              f"(三層/前一版 = {b[2]/a[2]:.2f}x)")

    hvA, hvB = hv_compare(arcA, arcB, margin=1.1)
    print("\n=== Hypervolume(共同正規化、參考點 (1.1,1.1,1.1),越大越好,理論上限 1.331)===")
    print(f"nd_first   HV = {hvA:.4f}")
    print(f"three_tier HV = {hvB:.4f}  (三層/前一版 = {hvB/hvA:.3f}x)")

    # HV 收斂曲線(共同正規化:兩版最終 archive 聯集)
    allp = arcA + arcB
    lo = {f: min(r[f] for r in allp) for f in ("F1", "F2", "F3")}
    hi = {f: max(r[f] for r in allp) for f in ("F1", "F2", "F3")}
    curveA = hv_curve(snapA, lo, hi)
    curveB = hv_curve(snapB, lo, hi)
    print("HV 收斂(最終點):nd_first %.4f | three_tier %.4f" % (curveA[-1][1], curveB[-1][1]))

    plot_compare_pareto(arcA, "nd_first(非支配優先)", arcB, "three_tier(極端值優先)", f"cmp_pareto_{M.DRONE_TIER}.png")
    plot_compare_conv(hA, "nd_first", hB, "three_tier", f"cmp_convergence_{M.DRONE_TIER}.png")
    plot_hv_convergence(curveA, "nd_first(非支配優先)", curveB, "three_tier(極端值優先)", f"cmp_hv_curve_{M.DRONE_TIER}.png")
    print("\n圖已存:cmp_pareto.png, cmp_convergence.png, cmp_hv_curve.png")
