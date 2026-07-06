# -*- coding: utf-8 -*-
"""
本方法-Mem:由「本方法-HVg」衍生的 memetic(模因)版本。

GA 部分與 `本方法-HVg` 完全相同:GPSIFF 二元競爭母代、均勻交配、探索型突變
(換點 / 換無人機數)、世代式 μ+λ、環境選擇 = 極值保留 + 前向貪婪 HV。
新增(memetic 之「局部搜尋」):每個子代在突變後,對**每艘船的訪點順序**做
**2-opt 局部搜尋**(Lamarckian:把改良後的順序寫回染色體)。

設計要點:
  - 2-opt 只重排既有訪點順序(不換點、不改無人機),屬於 exploitation;
    探索(換點/換無人機)仍由原突變負責 → 兩者互補,避免停滯。
  - 2-opt 目標 = **直線距離**(訪點順序改善之 cheap surrogate;區內候選點多半無禁航阻隔,直線≈繞行);
    真正 F2 仍由 evaluate() 以實際 mission edges(route_around 計入禁航繞行)計算。以 O(1) delta 加速。
  - **FEs 公平性**:2-opt 僅計算路徑長度、不評估 F1/F2/F3,故**不計入 FEs 預算**;
    僅最後對精煉後子代之整體 evaluate 計為 1 FE,與其他方法之預算對等。

回傳:(population, archive, history, snapshots),與 compare_sel.run_cmp 同。
"""
import warnings; warnings.filterwarnings("ignore")
import random
from copy import deepcopy
import numpy as np
import MOGA_GPSIFF_patrol_clean as M
from method_hvgreedy import _survive_extreme_greedy_hv
from local_search import mutate_2opt as _mutate_2opt   # 共用 2-opt(與 NSGA-III-Mem/SMS-EMOA-Mem 完全相同)


def run_method_memetic(seed, pop_size=100, max_fes=10100, snap_every=2, crossover_rate=None):
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
                offspring.append(_mutate_2opt(c1))
                if len(offspring) < pop_size:
                    offspring.append(_mutate_2opt(c2))
            else:
                offspring.append(_mutate_2opt(deepcopy(random.choice(mating))))
        offspring = offspring[:pop_size]
        ev = 0
        for ch in offspring:
            M.evaluate(ch); fe += 1; ev += 1                # 僅整體評估計 FE;2-opt 不計
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
    pop, arc, hist, snaps = run_method_memetic(seed=1, pop_size=20, max_fes=200, snap_every=1)
    cross = sum(M.crosses_no_go(v[a], v[a+1])
                for c in pop for v in [c["routes"][0]] for a in range(len(v)-1))
    print(f"本方法-Mem 自測 {time.time()-t:.1f}s | archive {len(arc)} | gens {len(hist)} "
          f"| 末代禁航穿越(船0) {cross}")
