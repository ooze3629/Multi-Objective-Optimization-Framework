# -*- coding: utf-8 -*-
"""
本方法-ss:穩態 (μ+1) 版的 HVg 骨架。

融合三條血緣,**不含 2-opt**(已封存):
  - 本方法:GPSIFF 二元競爭母代、兩段編碼、F1/F2/F3、繞行修補、FEs 計數。
  - HVg:存活時保護 F1/F2/F3 之六個目標極值(保前緣延展)。
  - SMS-EMOA:穩態 μ+1——每代生 1 子代,併入後移除「最差前緣中 HV 貢獻最小者」。

與 SMS-EMOA 的差別僅兩處:(i) 母代用 GPSIFF 二元競爭(非隨機);(ii) 淘汰時保護極值。
回傳:(population, archive, history, snapshots),與 compare_sel.run_cmp 同。
"""
import warnings; warnings.filterwarnings("ignore")
import random
from copy import deepcopy
import numpy as np
import MOGA_GPSIFF_patrol_clean as M
from sms_emoa import _fronts, _least_hv_contributor


def _gpsiff_pick(P, fit):
    """GPSIFF 二元競爭:抽 2 取適應值高者(回傳個體)。"""
    i, j = random.sample(range(len(P)), 2)
    return P[i] if fit[i] >= fit[j] else P[j]


def run_method_ss(seed, pop_size=100, max_fes=10100, snap_every=2, crossover_rate=None, crossover_fn=None, mutate_fn=None):
    random.seed(seed); np.random.seed(seed)
    cr = M.CROSSOVER_RATE if crossover_rate is None else crossover_rate
    cx = crossover_fn or M.crossover
    mut = mutate_fn or M.mutation
    fe = 0
    population = M.init_population(pop_size)
    for ch in population:
        M.evaluate(ch); fe += 1

    archive, seen, history = [], set(), []
    snapshots = []
    snap_fes = pop_size * snap_every
    next_snap = fe + snap_fes

    def record(force=False):
        nonlocal next_snap
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
        if force or fe >= next_snap:
            snapshots.append((fe, [(r["F1"], r["F2"], r["F3"]) for r in archive]))
            next_snap = fe + snap_fes

    while fe < max_fes:
        # (a) 母代:GPSIFF 二元競爭
        fit, _ = M.gpsiff(population)
        p1 = _gpsiff_pick(population, fit)
        # (b) 單一子代:交配取一 + 探索型突變(無 2-opt)
        if random.random() < cr:
            p2 = _gpsiff_pick(population, fit)
            c1, _ = cx(p1, p2)
            child = mut(c1)
        else:
            child = mut(deepcopy(p1))
        M.evaluate(child); fe += 1
        # (c) μ+1 → 移除最差前緣中 HV 貢獻最小者,但保護目標極值
        combined = population + [child]
        fronts = _fronts(combined)
        nd = fronts[0]
        R = fronts[-1]
        E = set(M.nd_extremes(combined, nd))          # ≤6 受保護極值
        cand = [i for i in R if i not in E]
        if not cand:                                  # 整層皆極值才退讓
            cand = list(R)
        drop = _least_hv_contributor(combined, cand)
        population = [combined[i] for i in range(len(combined)) if i != drop]
        record()

    record(force=True)
    return population, archive, history, snapshots


if __name__ == "__main__":
    import time, baselines as B
    M.set_environment("taiwan_real_ddn")
    try:
        M.load_route_cache()
    except Exception as e:
        print("快取載入略過:", e)
    M.weight_map = list(B.make_scenarios("taiwan_real_ddn").values())[0]
    t = time.time()
    pop, arc, hist, snaps = run_method_ss(seed=1, pop_size=20, max_fes=200, snap_every=1)
    cross = sum(M.crosses_no_go(v[a], v[a + 1])
                for c in pop for v in [c["routes"][0]] for a in range(len(v) - 1))
    print(f"本方法-ss 自測 {time.time()-t:.1f}s | archive {len(arc)} | iters {len(hist)} "
          f"| 末代禁航穿越(船0) {cross}")
