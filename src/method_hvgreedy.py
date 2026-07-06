# -*- coding: utf-8 -*-
"""
本方法-HVg:在「本方法(MOGA+GPSIFF,世代式 μ+λ)」上,**只把環境選擇**改為:
  (1) 先保留非支配集合中 F1/F2/F3 的六個極值(min/max,去重 ≤6)——沿用本方法之極值層;
  (2) 再以「前向貪婪 hypervolume」遞增:每次加入「使已選集合 HV 增加最多」者,
      直到滿足族群大小 μ(在固定正規化尺度上計算;平手以 GPSIFF 高者優先)。
  若非支配解不足 μ,餘額由被支配解依 GPSIFF 由高到低補足。

與 `本方法-HV`(後向:於溢出層逐一移除 HV 貢獻最小者)為同一目標的兩種貪婪方向;
本版明確保留本方法的極值保護(邊界延展),再以前向貪婪 HV 充實內部。

保留(與本方法相同):兩段編碼、F1/F2/F3、繞行修補、FEs、**GPSIFF 二元競爭母代**、世代式 μ+λ。
回傳:(population, archive, history, snapshots),與 compare_sel.run_cmp 同。
"""
import warnings; warnings.filterwarnings("ignore")
import random
from copy import deepcopy
import numpy as np
import MOGA_GPSIFF_patrol_clean as M
from compare_sel import _hv3d


def _fronts(pop):
    n = len(pop)
    S = [[] for _ in range(n)]; ndom = [0] * n; fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if M.dominates(pop[p], pop[q]):
                S[p].append(q)
            elif M.dominates(pop[q], pop[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1; fronts.append(nxt)
    fronts.pop()
    return fronts


def _norm_all(combined, margin=1.1):
    """以整個 combined 之各目標範圍正規化成最小化點(F1 轉最小化);固定尺度供貪婪用。"""
    F1 = [c["F1"] for c in combined]; F2 = [c["F2"] for c in combined]; F3 = [c["F3"] for c in combined]
    lo = (min(F1), min(F2), min(F3)); hi = (max(F1), max(F2), max(F3))
    rng = tuple((hi[d] - lo[d]) or 1.0 for d in range(3))
    return [((hi[0] - c["F1"]) / rng[0], (c["F2"] - lo[1]) / rng[1], (c["F3"] - lo[2]) / rng[2])
            for c in combined]


def _survive_extreme_greedy_hv(combined, mu):
    ref = (1.1, 1.1, 1.1)
    fronts = _fronts(combined)
    nd = fronts[0]
    dom_order = [i for f in fronts[1:] for i in f]
    cfit, _ = M.gpsiff(combined)
    pts = _norm_all(combined)

    def hv_of(idxset):
        return _hv3d([pts[i] for i in idxset], ref)

    # 非支配不足 μ:全收 + 被支配補足
    if len(nd) <= mu:
        S = list(nd)
        rest = sorted(dom_order, key=lambda i: -cfit[i])
        S += rest[:mu - len(S)]
        return S[:mu]

    # (1) 先放六個極值(非支配集合中各目標 min/max)
    ex = M.nd_extremes(combined, nd)
    S = list(dict.fromkeys(ex))
    inS = set(S)
    cur = hv_of(S)
    cand = [i for i in nd if i not in inS]
    # (2) 前向貪婪 HV,直到 μ
    while len(S) < mu and cand:
        best_i, best_gain = None, None
        for c in cand:
            g = hv_of(S + [c]) - cur
            if (best_gain is None or g > best_gain + 1e-12 or
                    (abs(g - best_gain) <= 1e-12 and cfit[c] > cfit[best_i])):
                best_gain, best_i = g, c
        S.append(best_i); inS.add(best_i); cur += max(best_gain, 0.0)
        cand.remove(best_i)
    # 若 nd 已全收仍不足(不會發生,因 len(nd)>mu),保險補被支配
    if len(S) < mu:
        rest = sorted((i for i in dom_order if i not in inS), key=lambda i: -cfit[i])
        S += rest[:mu - len(S)]
    return S[:mu]


def run_method_hvgreedy(seed, pop_size=100, max_fes=10100, snap_every=2, crossover_rate=None):
    random.seed(seed); np.random.seed(seed)
    cr = M.CROSSOVER_RATE if crossover_rate is None else crossover_rate
    fe = 0
    population = M.init_population(pop_size)
    for ch in population:
        M.evaluate(ch); fe += 1

    archive, seen, history, gen = [], set(), [], 0
    snapshots = []
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
        chosen = _survive_extreme_greedy_hv(combined, pop_size)
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


if __name__ == "__main__":
    import time
    M.set_environment("taiwan_real_ddn")
    try:
        M.load_route_cache()
    except Exception as e:
        print("快取載入略過:", e)
    sc = __import__("baselines").make_scenarios("taiwan_real_ddn")
    M.weight_map = list(sc.values())[0]
    t = time.time()
    pop, arc, hist, snaps = run_method_hvgreedy(seed=1, pop_size=20, max_fes=200, snap_every=1)
    cross = sum(M.crosses_no_go(v[a], v[a+1])
                for c in pop for v in [c["routes"][0]] for a in range(len(v)-1))
    print(f"本方法-HVg 自測 {time.time()-t:.1f}s | archive {len(arc)} | gens {len(hist)} "
          f"| 末代禁航穿越(船0) {cross}")
