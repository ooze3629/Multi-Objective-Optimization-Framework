# -*- coding: utf-8 -*-
"""
骨架改進運算子:基地區塊交配(BBX)+ 反轉突變(inversion)。

BBX(crossover_block):以「整個基地區塊」(該基地 SV+USV 各 N 點 + 兩船無人機數)為
  交換單位做均勻交配。候選點基地本地、各基地索引區間互斥 → 子代天生可行(免點層級修補)、
  可繼承性高(父代成形之基地簇整包遺傳)。

反轉突變(inversion):順序運算子——反轉某船訪點走訪序之一段。只重排順序(影響 F1/F2),
  不改變選點集合(不影響 F3)。屬隨機突變,非 2-opt(無目標資訊之局部搜尋),符合「2-opt 封存」界線。
  - mutate_inv:drone 重抽 + inversion(純;不換點)
  - mutate_inv_resel:再加低速率換點(維持選點多樣性、防早熟)
"""
import random
from copy import deepcopy
import MOGA_GPSIFF_patrol_clean as M

RESEL_RATE = 0.05   # -bx+ 之殘餘換點率(維持選點新鮮度)


def crossover_block(p1, p2):
    """基地區塊均勻交配:每基地以 0.5 機率整包交換(兩船點集 + 兩船無人機數)。"""
    c1 = {"drones": list(p1["drones"]), "assignment": [list(s) for s in p1["assignment"]]}
    c2 = {"drones": list(p2["drones"]), "assignment": [list(s) for s in p2["assignment"]]}
    for b in range(M.NUM_BASES):
        if random.random() < 0.5:                      # 整個基地區塊交換
            for v in (2 * b, 2 * b + 1):
                c1["drones"][v] = p2["drones"][v]; c1["assignment"][v] = list(p2["assignment"][v])
                c2["drones"][v] = p1["drones"][v]; c2["assignment"][v] = list(p1["assignment"][v])
    return c1, c2                                       # 不建路徑;由 mutate 重建(repair 為安全網,實際零修補)


def _mutate(chromo, with_reselect):
    m = deepcopy(chromo)
    used = set(p for ship in m["assignment"] for p in ship)
    # 無人機數:逐基因重抽
    for v in range(M.NUM_VESSELS):
        if random.random() < M.MUTATION_RATE:
            m["drones"][v] = random.choice(M.DRONE_DOMAIN)
    # 反轉突變:逐船以 MUTATION_RATE 反轉一段走訪序(只動順序,不動集合)
    for v in range(M.NUM_VESSELS):
        seq = m["assignment"][v]
        if len(seq) >= 2 and random.random() < M.MUTATION_RATE:
            i, j = sorted(random.sample(range(len(seq)), 2))
            seq[i:j + 1] = seq[i:j + 1][::-1]
    # 安全閥:低速率換點(維持選點多樣性)
    if with_reselect:
        for v in range(M.NUM_VESSELS):
            start, end = M.ship_candidate_range[v]
            for k in range(M.N_POINTS):
                if random.random() < RESEL_RATE:
                    pool = [i for i in range(start, end) if i not in used and M._legal_point(i)]
                    if pool:
                        nw = random.choice(pool)
                        used.discard(m["assignment"][v][k]); used.add(nw)
                        m["assignment"][v][k] = nw
    return M.build_routes(m)


def mutate_inv(chromo):
    """純反轉突變(drone 重抽 + inversion;不換點)。"""
    return _mutate(chromo, with_reselect=False)


def mutate_inv_resel(chromo):
    """反轉突變 + 低速率換點(防選點凍結)。"""
    return _mutate(chromo, with_reselect=True)
