# -*- coding: utf-8 -*-
"""
建置「資料驅動節點」環境 taiwan_real_ddn(第一版:K=40/基地,drop-in)
========================================================================
底圖 / 禁航 / 基地 = 沿用 taiwan_real(finalmap_taiwan_real.npy、同一組 13 基地);
唯一差異 = 候選點(Patrol_Point)由「人工網格」改為「資料驅動 + 基地責任區」:

  (1) 責任區劃分:由 13 個基地對合法海域做多源 BFS(8 鄰、海路測地距離),
      每格指派給「走海路最近」的基地。各區互斥、聯集 = 全部合法海域
      (用海路而非直線,東/西岸不會被台灣島隔著誤配)。
  (2) 區內節點:在每個基地責任區內,對「三情境融合依據場」做加權 k-means(k=40):
      依據場 = normalize( 航運密度[AIS,S21] + 暗船查緝[SAR暗船,S22] + 海纜基礎設施[海纜v1] )
      (三軸各自正規化後相加再整體正規化;彼此相關低:AIS↔暗船0.42、AIS↔海纜~0.1),
      與三個 F1 情境一致使節點對三情境一視同仁、同時反映航運/暗船/海纜走廊,
      snap 到區內合法格、去重補足成 40 個。
  (3) 維持每基地 40 點、依基地順序排列 → ship_candidate_range 不需更動(drop-in)。

決定性:k-means 以固定亂數種子(seed=基地編號)→ 可重現。
產物:Patrol_Point_taiwan_real_ddn.npy,(520,2) int32,列為 (x,y)。
"""
import os
import numpy as np
from collections import deque
import MOGA_GPSIFF_patrol_clean as M

K = 40                       # 每基地節點數(= CANDIDATES_PER_BASE)
SEED_BASE = "taiwan_real"    # 底圖/禁航/基地來源
NB8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def geodesic_districts(legal, bases):
    """多源 BFS:每個合法海格指派給海路最近的基地。回傳 owner 陣列(-1=非合法)。"""
    H, W = legal.shape
    dist = np.full((H, W), 1 << 30, int)
    owner = np.full((H, W), -1, int)
    dq = deque()
    for b, (x, y) in enumerate(bases):
        if legal[y, x]:
            dist[y, x] = 0; owner[y, x] = b; dq.append((x, y))
    while dq:
        x, y = dq.popleft()
        for dx, dy in NB8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and legal[ny, nx] and dist[ny, nx] == (1 << 30):
                dist[ny, nx] = dist[y, x] + 1; owner[ny, nx] = owner[y, x]; dq.append((nx, ny))
    return owner


def wkmeans(pts, w, k, iters=40, seed=0):
    """輕量加權 k-means(k-means++ 加權初始化 + Lloyd)。pts:(n,2);w:(n,)。"""
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n <= k:
        return pts.astype(float)
    w = w + 1e-6                                   # 全零權重 → 退化為空間均布
    p = w / w.sum()
    idx = [int(rng.choice(n, p=p))]
    d2 = ((pts - pts[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        pp = w * d2; s = pp.sum(); pp = pp / s if s > 0 else np.ones(n) / n
        j = int(rng.choice(n, p=pp)); idx.append(j)
        d2 = np.minimum(d2, ((pts - pts[j]) ** 2).sum(1))
    C = pts[idx].astype(float)
    for _ in range(iters):
        D = ((pts[:, None, :] - C[None, :, :]) ** 2).sum(2)
        a = D.argmin(1); newC = C.copy()
        for c in range(k):
            m = a == c
            if m.any():
                ws = w[m]; newC[c] = (pts[m] * ws[:, None]).sum(0) / ws.sum()
        if np.allclose(newC, C):
            break
        C = newC
    return C


def snap_distinct(centroids, dpts, used):
    """把每個質心 snap 到區內『最近且尚未使用』的合法格,確保 K 個相異節點。"""
    out = []
    for cx, cy in centroids:
        order = np.argsort((dpts[:, 0] - cx) ** 2 + (dpts[:, 1] - cy) ** 2)
        for j in order:
            cell = (int(dpts[j, 0]), int(dpts[j, 1]))
            if cell not in used:
                used.add(cell); out.append(cell); break
    return out


def main():
    s = M.set_environment(SEED_BASE)
    ng = M.no_go_zone; legal = (ng == 0)
    bases = [(int(x), int(y)) for (x, y) in M.base_coords]
    DATA = os.path.join(M.SCRIPT_DIR, "..", "data")

    def _norm(a):
        a = np.where(ng == 0, a.astype(float), 0.0)
        return a / a.max() if a.max() > 0 else a

    # 節點依據場 = 三『正交』情境真實訊號融合(各自正規化到[0,1]後相加,再整體正規化):
    #   航運密度(AIS, S21) + 暗船查緝(SAR 暗船, S22) + 海纜基礎設施(海纜 v1),
    #   與三個 F1 情境一致,使節點對三情境一視同仁(三軸彼此相關低:AIS↔暗船0.42、AIS↔海纜~0.1)。
    pri = (_norm(np.load(os.path.join(DATA, "scenario_TWreal_ais_v1.npy")))
           + _norm(np.load(os.path.join(DATA, "scenario_TWreal_sar_dark_v1.npy")))
           + _norm(np.load(os.path.join(DATA, "scenario_TWreal_cable_geo_v1.npy"))))
    pri[ng != 0] = 0.0
    if pri.max() > 0:
        pri = pri / pri.max()
    assert pri.shape == ng.shape, "依據場與底圖尺寸不符"

    owner = geodesic_districts(legal, bases)
    assert (owner[legal] >= 0).all(), "有合法海格未被分區"

    nodes = []
    for b in range(len(bases)):
        cells = np.argwhere(owner == b)                # (y,x)
        dpts = cells[:, ::-1].astype(float)            # (x,y)
        w = pri[cells[:, 0], cells[:, 1]]
        C = wkmeans(dpts, w, K, seed=b)
        used = {(int(bases[b][0]), int(bases[b][1]))}   # 排除基地自身格:候選點不得等於出航點
        snapped = snap_distinct(C, dpts, used)
        # 若因區太小不足 K(理論上各區 >= 159 > 40,不會發生),補區內未用格
        if len(snapped) < K:
            for j in range(len(dpts)):
                cell = (int(dpts[j, 0]), int(dpts[j, 1]))
                if cell not in used:
                    used.add(cell); snapped.append(cell)
                if len(snapped) == K:
                    break
        assert len(snapped) == K, f"基地 {b} 僅得 {len(snapped)} 點"
        nodes.extend(snapped)

    PP = np.array(nodes, dtype=np.int32)               # (520,2),(x,y),依基地順序
    # --- 驗證 ---
    assert PP.shape == (len(bases) * K, 2), PP.shape
    assert (ng[PP[:, 1], PP[:, 0]] == 0).all(), "有節點落在非合法海域"
    for b in range(len(bases)):
        seg = PP[b * K:(b + 1) * K]
        assert len({tuple(p) for p in seg}) == K, f"基地 {b} 節點不相異"

    out = os.path.join(M.SCRIPT_DIR, "Patrol_Point_taiwan_real_ddn.npy")
    np.save(out, PP)
    print(f"環境來源={s['env']}  底圖={ng.shape}  合法海域={int(legal.sum())}")
    print(f"節點={PP.shape}  每基地={K}  全部合法海域={int((ng[PP[:,1],PP[:,0]]==0).sum())}/{len(PP)}")
    print("各基地責任區(格)/區內優先度Σ:")
    for b in range(len(bases)):
        cells = np.argwhere(owner == b)
        print(f"  基地{b:2d} {tuple(bases[b])!s:9s} 區={len(cells):4d}  依據場Σ={pri[cells[:,0],cells[:,1]].sum():6.1f}")
    print("已存檔:", out)


if __name__ == "__main__":
    main()
