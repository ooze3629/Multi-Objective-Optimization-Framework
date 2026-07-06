# -*- coding: utf-8 -*-
"""
共用局部搜尋運算子(memetic 用)。所有引擎(本方法-Mem / NSGA-III-Mem / SMS-EMOA-Mem)
共用同一個 `mutate_2opt`,確保「公平化 memetic 比較」中**局部搜尋完全相同、僅演化骨架不同**。

mutate_2opt = 探索型突變(換點 / 換無人機,M.mutation)+ 各船訪點順序之 2-opt 局部搜尋
              (Lamarckian:改良順序寫回染色體)。
2-opt 以**直線距離**重排(區內候選點多半無禁航阻隔,直線≈繞行;真正繞行 F2 於 evaluate 計算),
以 O(1) delta 加速;且 2-opt **只算距離、不評估 F1/F2/F3 → 不計入 FEs**。
(synthetic philippines 任務段為 Exit↔Patrol、F1/F2 排除 Base↔Exit,故該環境 2-opt 以 exit point 作 home。)
"""
import math
import MOGA_GPSIFF_patrol_clean as M


def two_opt_indices(home, idx_list, max_pass=3):
    """對單船訪點(索引序)做 2-opt:以直線距離最小化 home→…→home 巡訪序(delta 加速)。"""
    tour = list(idx_list)
    n = len(tour)
    if n < 3:
        return tour
    PP = M.Patrol_Point
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    pt = lambda idx: (float(PP[idx][0]), float(PP[idx][1]))
    improved, passes = True, 0
    while improved and passes < max_pass:
        improved = False; passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                A = home if i == 0 else pt(tour[i - 1])
                B = pt(tour[i])
                Cc = pt(tour[j])
                D = home if j == n - 1 else pt(tour[j + 1])
                if d(A, Cc) + d(B, D) + 1e-9 < d(A, B) + d(Cc, D):
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def home_anchor(v):
    """2-opt 巡訪序之 home 端點。synthetic philippines 任務段為 Exit↔Patrol
    (F1/F2 排除 Base↔Exit),故該環境以 exit point 作 home;其餘環境用 land base。"""
    bid = v // 2
    if getattr(M, "ENV_NAME", None) == "philippines" and bid < len(M.exit_coords):
        ex = M.exit_coords[bid]
        if ex is not None:
            return tuple(int(c) for c in ex)
    return tuple(int(c) for c in M.base_coords[bid])


def mutate_2opt(chromo):
    """探索型突變(換點/換無人機)+ 各船 2-opt 局部搜尋(Lamarckian),最後重建路徑。"""
    m = M.mutation(chromo)                          # 換點/換無人機 + 修補(已 build_routes)
    for v in range(M.NUM_VESSELS):
        m["assignment"][v] = two_opt_indices(home_anchor(v), m["assignment"][v])
    return M.build_routes(m)
