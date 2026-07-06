# -*- coding: utf-8 -*-
"""四方法之 HV 收斂曲線(三海域,代表性種子)。斷點續跑(每海域存檔)。"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, pickle, random
from copy import deepcopy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC","Noto Sans CJK JP","Microsoft JhengHei","DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import MOGA_GPSIFF_patrol_clean as M
import moea_metrics as MM
import baselines as B
from compare_sel import run_cmp
OUT = f"conv_snaps_{M.DRONE_TIER}.pkl"

def es_snap(seed, pop, fes, snap_gen=2):
    random.seed(seed); np.random.seed(seed)
    P = M.init_population(pop); fe = 0
    for ch in P: M.evaluate(ch); fe += 1
    arc, snaps, gen = [], [], 0
    while fe < fes:
        gen += 1
        Q = [M.mutation(deepcopy(random.choice(P))) for _ in range(pop)]
        ev = 0
        for ch in Q:
            M.evaluate(ch); fe += 1; ev += 1
            if fe >= fes: break
        Q = Q[:ev]; C = P + Q; fit, q = M.gpsiff(C)
        nd = [i for i in range(len(C)) if q[i] == 0]; dom = [i for i in range(len(C)) if q[i] > 0]
        P = [C[i] for i in (sorted(nd, key=lambda i: -fit[i]) + sorted(dom, key=lambda i: -fit[i]))[:pop]]
        for ch in M.pareto_front(P):
            arc = B._nd_add(arc, {"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
        if gen % snap_gen == 0 or fe >= fes:
            snaps.append((fe, [(a["F1"], a["F2"], a["F3"]) for a in arc]))
    return snaps

def rand_snap(seed, pop, fes, snap_ev=200):
    random.seed(seed); np.random.seed(seed)
    arc, snaps, fe, nxt = [], [], 0, 200
    for _ in range(fes):
        ch = M.init_population(1)[0]; M.evaluate(ch); fe += 1
        arc = B._nd_add(arc, {"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
        if fe >= nxt:
            snaps.append((fe, [(a["F1"], a["F2"], a["F3"]) for a in arc])); nxt += snap_ev
    snaps.append((fe, [(a["F1"], a["F2"], a["F3"]) for a in arc]))
    return snaps

def run_conv(seed=1, pop=100, fes=3000, max_wall=235):
    B.load_cache()
    scns = B.make_scenarios()
    data = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {"meta": {"seed": seed, "pop": pop, "fes": fes}, "scn": {}}
    t0 = time.time()
    for sname, wmap in scns.items():
        if sname in data["scn"]: continue
        if time.time() - t0 > max_wall:
            print(f"[時間到] 已完成 {len(data['scn'])}/3,請續跑。"); return False
        M.weight_map = wmap
        t = time.time()
        _, _, _, snap_self = run_cmp("three_tier", seed, pop, fes, snap_every=2)
        snap_es = es_snap(seed, pop, fes)
        snap_rd = rand_snap(seed, pop, fes)
        gd = B.run_greedy()
        data["scn"][sname] = {"本方法": snap_self, "ES": snap_es, "隨機": snap_rd, "貪婪": gd}
        pickle.dump(data, open(OUT, "wb")); B.save_cache()
        print(f"  存檔 {sname} ({time.time()-t:.0f}s) 進度 {len(data['scn'])}/3")
    print("== 全部完成 ==" if len(data["scn"]) >= 3 else "partial")
    return len(data["scn"]) >= 3

def run_conv_multi(seeds=(1, 2, 3, 4, 5), pop=100, fes=3000, out=f"conv_snaps_multi_{M.DRONE_TIER}.pkl", max_wall=200):
    B.load_cache()
    scns = B.make_scenarios()
    data = pickle.load(open(out, "rb")) if os.path.exists(out) else {"meta": {"seeds": list(seeds), "pop": pop, "fes": fes}, "scn": {}, "greedy": {}}
    total = len(scns) * len(seeds); done = lambda: sum(len(v["本方法"]) for v in data["scn"].values())
    t0 = time.time()
    for sname, wmap in scns.items():
        M.weight_map = wmap
        data["scn"].setdefault(sname, {"本方法": {}, "ES": {}, "隨機": {}})
        if sname not in data["greedy"]:
            data["greedy"][sname] = B.run_greedy(); pickle.dump(data, open(out, "wb"))
        for sd in seeds:
            if sd in data["scn"][sname]["本方法"]: continue
            if time.time() - t0 > max_wall:
                print(f"[時間到] 進度 {done()}/{total},請續跑。"); return False
            t = time.time()
            data["scn"][sname]["本方法"][sd] = run_cmp("three_tier", sd, pop, fes, snap_every=2)[3]
            data["scn"][sname]["ES"][sd] = B.es_snap(sd, pop, fes)[1]
            data["scn"][sname]["隨機"][sd] = B.rand_snap(sd, pop, fes)[1]
            pickle.dump(data, open(out, "wb")); B.save_cache()
            print(f"  存檔 {sname} seed {sd} ({time.time()-t:.0f}s) 進度 {done()}/{total}")
    print("== 全部完成 ==" if done() >= total else f"進度 {done()}/{total}")
    return done() >= total


def plot_conv(save=f"hv_conv_scenarios_{M.DRONE_TIER}.png"):
    data = pickle.load(open(OUT, "rb")); scns = list(data["scn"].keys())
    fig, axes = plt.subplots(1, len(scns), figsize=(5.3 * len(scns), 4.7))
    if len(scns) == 1: axes = [axes]
    for ax, s in zip(axes, scns):
        rec = data["scn"][s]
        finals = []
        for m in ["本方法", "ES", "隨機"]:
            finals += [{"F1": a[0], "F2": a[1], "F3": a[2]} for a in rec[m][-1][1]]
        finals += rec["貪婪"]
        lo, hi = MM._bounds(finals)
        for m, c in [("本方法", "#1F4E79"), ("ES", "#2E9E5B"), ("隨機", "#E08A1E")]:
            xs = [fe for fe, _ in rec[m]]
            ys = [MM.hv([{"F1": a[0], "F2": a[1], "F3": a[2]} for a in pts], lo, hi) for _, pts in rec[m]]
            ax.plot(xs, ys, color=c, marker="o", ms=2.5, label=m)
        ax.axhline(MM.hv(rec["貪婪"], lo, hi), color="#999999", ls="--", label="貪婪(建構即得)")
        ax.set_title(s, fontsize=12); ax.set_xlabel("函數評估次數 (FEs)"); ax.set_ylabel("Hypervolume")
        ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
    fig.suptitle("Hypervolume 收斂曲線(四方法,三模擬海域;代表性種子)", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(save, dpi=130); plt.close(fig)
    from PIL import Image; print("saved", save, Image.open(save).size)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "plot": plot_conv()
    elif mode == "multi": run_conv_multi(max_wall=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else: run_conv(max_wall=int(sys.argv[2]) if len(sys.argv) > 2 else 235)
