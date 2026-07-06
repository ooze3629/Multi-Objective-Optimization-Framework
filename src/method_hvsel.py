# -*- coding: utf-8 -*-
"""
本方法-HV:在「本方法(MOGA+GPSIFF,世代式 μ+λ)」上,**只把環境選擇**由三層
(極端值→其餘非支配→被支配,皆用 GPSIFF 排序)換成 **SMS-EMOA 的 HV 貢獻準則**。

保留(與本方法完全相同):
  - 兩段編碼、F1/F2/F3、繞行修補、FEs 嚴格計數;
  - **母代選擇 = GPSIFF 二元競爭**(p−q+c);
  - 世代式:每代產 μ 個子代(均勻交配 + 逐點突變),合併 P∪Q = 2μ。
唯一變更(環境選擇):
  - 對 2μ 做快速非支配排序;由佳到劣納入整層,至某層加入會超過 μ 時,
    於該「分割層」**逐一移除 HV 貢獻最小者**直到剩餘恰為所需名額(= SMS-EMOA
    淘汰準則由穩態 (μ+1) 推廣到世代式 μ+λ 的 truncation 版)。
  - HV 貢獻於「以該層目標範圍正規化([0,1]、F1 轉最小化)、參考點 (1.1)³」之
    內部尺度上逐步計算;最終 archive 仍由 experiment_eval 以全方法聯集公平正規化評分。

回傳格式與 compare_sel.run_cmp 相同:(population, archive, history, snapshots)
"""
import warnings; warnings.filterwarnings("ignore")
import random
from copy import deepcopy
import numpy as np
import MOGA_GPSIFF_patrol_clean as M
from compare_sel import _hv3d
from sms_emoa import _fronts, _norm_min


def _hv_truncate(pop, R, k_keep):
    """於分割層 R(索引)中保留 k_keep 個:逐一移除 HV 貢獻最小者(每步重算)。"""
    R = list(R)
    ref = (1.1, 1.1, 1.1)
    while len(R) > k_keep:
        pts = _norm_min(pop, R)
        base = _hv3d(pts, ref)
        worst_k, worst_c = 0, None
        for k in range(len(R)):
            contrib = base - _hv3d(pts[:k] + pts[k + 1:], ref)
            if worst_c is None or contrib < worst_c:
                worst_c, worst_k = contrib, k
        R.pop(worst_k)
    return R


def _survive_hv(combined, mu):
    fronts = _fronts(combined)
    St = []
    for fr in fronts:
        if len(St) + len(fr) <= mu:
            St.extend(fr)
            if len(St) == mu:
                return St
        else:
            return St + _hv_truncate(combined, fr, mu - len(St))
    return St


def run_method_hv(seed, pop_size=100, max_fes=10100, snap_every=2, crossover_rate=None):
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
        # 母代選擇:GPSIFF 二元競爭(與本方法相同)
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
        # 環境選擇:HV 貢獻 truncation(取代三層)
        combined = population + offspring
        chosen = _survive_hv(combined, pop_size)
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
    pop, arc, hist, snaps = run_method_hv(seed=1, pop_size=20, max_fes=200, snap_every=1)
    cross = sum(M.crosses_no_go(v[a], v[a+1])
                for c in pop for v in [c["routes"][0]] for a in range(len(v)-1))
    print(f"本方法-HV 自測 {time.time()-t:.1f}s | archive {len(arc)} | gens {len(hist)} "
          f"| 末代禁航穿越(船0) {cross}")
