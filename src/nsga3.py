# -*- coding: utf-8 -*-
"""
NSGA-III(Deb & Jain, 2014)作為對照演算法。

設計原則(維持與本方法之公平比較):
  - 編碼、目標函式(F1/F2/F3)、交配、突變、繞行修補、FEs 計數 **完全沿用**
    MOGA_GPSIFF_patrol_clean(M),唯一差異 = 環境選擇機制。
  - 本方法用「GPSIFF 三層 μ+λ」;NSGA-III 用「快速非支配排序 + 參考點利基保留」。
  - 交配選擇:依 Deb & Jain 原作,母代隨機選取(多樣性壓力來自參考點利基,
    不另用適應值錦標賽),其餘運算子與 FEs 預算與本方法一致。
  - 目標統一為最小化:g = (-F1, F2, F3);非支配判定直接重用 M.dominates
    (其已定義 F1 取大、F2/F3 取小),正規化/關聯則用 g 向量。

回傳格式與 compare_sel.run_cmp 完全相同:(population, archive, history, snapshots)
"""
import warnings; warnings.filterwarnings("ignore")
import random
from copy import deepcopy
from collections import defaultdict
import numpy as np
import MOGA_GPSIFF_patrol_clean as M


# ---------------------------------------------------------------- 參考點
def das_dennis(p, n_obj=3):
    """Das-Dennis 結構化參考點(單位單純形),共 C(p+n_obj-1, n_obj-1) 個。"""
    pts = []
    def rec(point, left, depth):
        if depth == n_obj - 1:
            point[depth] = left / p
            pts.append(point.copy()); return
        for i in range(left + 1):
            point[depth] = i / p
            rec(point, left - i, depth + 1)
    rec([0.0] * n_obj, p, 0)
    return np.array(pts, dtype=float)

def _choose_p(pop_size, n_obj=3):
    """挑選 p 使參考點數最接近且不少於 pop_size(三目標)。"""
    best_p, best = 2, None
    for p in range(2, 40):
        cnt = 1
        # C(p+n_obj-1, n_obj-1)
        from math import comb
        cnt = comb(p + n_obj - 1, n_obj - 1)
        if best is None or (cnt >= pop_size and (best < pop_size or cnt < best)) or \
           (best < pop_size and cnt > best):
            best, best_p = cnt, p
        if cnt >= pop_size:
            return p
    return best_p


# ---------------------------------------------------------------- NSGA-III 內核
def fast_nondominated_sort(pop):
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

def _objm(pop):
    """最小化目標矩陣 g = (-F1, F2, F3)。"""
    return np.array([[-c["F1"], c["F2"], c["F3"]] for c in pop], dtype=float)

def _normalize(G, Sl, ideal):
    """Deb & Jain 正規化:平移→ASF 求極端點→超平面截距→以截距縮放。"""
    m = G.shape[1]
    Gt = G - ideal
    Sl = np.asarray(Sl)
    extreme = np.empty(m, dtype=int)
    for j in range(m):
        w = np.full(m, 1e-6); w[j] = 1.0
        asf = (Gt[Sl] / w).max(axis=1)
        extreme[j] = Sl[int(np.argmin(asf))]
    try:
        Z = Gt[extreme]
        plane = np.linalg.solve(Z, np.ones(m))
        intercepts = 1.0 / plane
        if not np.all(np.isfinite(intercepts)) or np.any(intercepts <= 1e-6):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = Gt[Sl].max(axis=0)
    intercepts = np.where(intercepts <= 1e-6, 1e-6, intercepts)
    return Gt / intercepts

def _associate(Gn, refs_unit):
    """每點關聯到垂距最小之參考線(過原點、方向 = 參考點)。"""
    proj = Gn @ refs_unit.T                      # (N,R)
    norm2 = np.sum(Gn * Gn, axis=1, keepdims=True)
    d2 = np.maximum(norm2 - proj ** 2, 0.0)
    dist = np.sqrt(d2)
    assign = dist.argmin(axis=1)
    dmin = dist.min(axis=1)
    return assign, dmin

def nsga3_survival(combined, pop_size, refs_unit):
    fronts = fast_nondominated_sort(combined)
    St, last = [], []
    for fr in fronts:
        if len(St) + len(fr) <= pop_size:
            St.extend(fr)
            if len(St) == pop_size:
                return St
        else:
            last = fr; break
    Sl = St + last
    K = pop_size - len(St)
    G = _objm(combined)
    ideal = G[Sl].min(axis=0)
    Gn = _normalize(G, Sl, ideal)
    assign, dmin = _associate(Gn, refs_unit)
    rho = np.zeros(refs_unit.shape[0], dtype=int)
    for i in St:
        rho[assign[i]] += 1
    ref_members = defaultdict(list)
    for i in last:
        ref_members[assign[i]].append(i)
    chosen = list(St)
    while len(chosen) < pop_size:
        cand = [j for j in ref_members if ref_members[j]]
        # 利基數最小之參考點優先;平手取編號小者(決定性)
        jmin = min(cand, key=lambda j: (rho[j], j))
        members = ref_members[jmin]
        if rho[jmin] == 0:
            pick = min(members, key=lambda i: (dmin[i], i))
        else:
            pick = random.choice(members)
        chosen.append(pick)
        members.remove(pick)
        rho[jmin] += 1
    return chosen[:pop_size]


# ---------------------------------------------------------------- 驅動(與 run_cmp 同介面)
def run_nsga3(seed, pop_size=100, max_fes=10100, snap_every=2, crossover_rate=None, mutate_fn=None):
    random.seed(seed); np.random.seed(seed)
    cr = M.CROSSOVER_RATE if crossover_rate is None else crossover_rate
    mut = mutate_fn or M.mutation
    refs = das_dennis(_choose_p(pop_size))
    refs_unit = refs / np.linalg.norm(refs, axis=1, keepdims=True)

    fe = 0
    population = M.init_population(pop_size)
    for ch in population:
        M.evaluate(ch); fe += 1

    archive, seen, history, gen = [], set(), [], 0
    snapshots = []
    while fe < max_fes:
        gen += 1
        # 交配選擇:隨機母代(NSGA-III 原作;多樣性壓力來自參考點利基)
        offspring = []
        while len(offspring) < pop_size:
            if random.random() < cr:
                p1, p2 = random.sample(population, 2)
                c1, c2 = M.crossover(p1, p2)
                offspring.append(mut(c1))
                if len(offspring) < pop_size:
                    offspring.append(mut(c2))
            else:
                offspring.append(mut(deepcopy(random.choice(population))))
        offspring = offspring[:pop_size]
        ev = 0
        for ch in offspring:
            M.evaluate(ch); fe += 1; ev += 1
            if fe >= max_fes:
                break
        offspring = offspring[:ev]
        combined = population + offspring
        chosen = nsga3_survival(combined, pop_size, refs_unit)
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
    M.set_environment("taiwan_real_ddn"); 
    try:
        M.load_route_cache()
    except Exception as e:
        print("快取載入略過:", e)
    sc = __import__("baselines").make_scenarios("taiwan_real_ddn")
    M.weight_map = list(sc.values())[0]
    t = time.time()
    pop, arc, hist, snaps = run_nsga3(seed=1, pop_size=20, max_fes=200, snap_every=1)
    cross = sum(M.crosses_no_go(v[a], v[a+1])
                for c in pop for v in [c["routes"][0]] for a in range(len(v)-1))
    print(f"NSGA-III 自測 {time.time()-t:.1f}s | archive {len(arc)} | gens {len(hist)} "
          f"| 參考點 {len(das_dennis(_choose_p(20)))} | 末代禁航穿越(船0) {cross}")
