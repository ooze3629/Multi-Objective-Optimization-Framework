# -*- coding: utf-8 -*-
"""建置 japan_real/philippines_real(均勻幾何 k-means)與 *_real_ddn(活動融合 k-means)候選點。
與 build_ddn_env 同方法:基地責任區(海路 BFS)+ 區內加權 k-means(k=40),決定性 seed=基地序。
  _real     權重 = 均勻(幾何均布,對應台灣 taiwan_real 人工網格的「不依活動」精神)
  _real_ddn 權重 = norm(AIS 航運 + 暗船 SAR + 海纜 v1)(對應 taiwan_real_ddn)
底圖 = finalmap_<c>_real.npy(no_go = 陸 + 真實禁航區;日/菲已烘入,見 build_real_maps RESTRICTED)。
輸出:Patrol_Point_<c>_real.npy / Patrol_Point_<c>_real_ddn.npy ,(N,2) int32,(x,y),依基地順序。
"""
import os, pickle, numpy as np
from build_ddn_env import geodesic_districts, wkmeans, snap_distinct
import MOGA_GPSIFF_patrol_clean as M

K = 40
HERE = M.SCRIPT_DIR
DATA = os.path.join(HERE, "..", "data")
TAG = {"japan": "JP", "philippines": "PH"}


def _norm(a, ng):
    a = np.where(ng == 0, a.astype(float), 0.0)
    return a / a.max() if a.max() > 0 else a


def gen(country, bases, weighted):
    ng = np.load(os.path.join(HERE, f"finalmap_{country}_real.npy"))
    sea = (ng == 0); legal = sea
    pri = None
    if weighted:
        t = TAG[country]
        pri = (_norm(np.load(os.path.join(DATA, f"scenario_{t}real_ais_v1.npy")), ng)
               + _norm(np.load(os.path.join(DATA, f"scenario_{t}real_sar_dark_v1.npy")), ng)
               + _norm(np.load(os.path.join(DATA, f"scenario_{t}real_cable_geo_v1.npy")), ng))
        pri[ng != 0] = 0.0
        if pri.max() > 0:
            pri = pri / pri.max()
    owner = geodesic_districts(legal, bases)
    # 後援:未被海路 BFS 觸及的孤立合法海域 → 就近(歐氏)指派,確保全部海格有歸屬
    un = np.argwhere((owner < 0) & legal)
    if len(un):
        bz = np.array(bases)
        for (y, x) in un:
            owner[y, x] = int(((bz[:, 0] - x) ** 2 + (bz[:, 1] - y) ** 2).argmin())
    nodes = []
    for b in range(len(bases)):
        cells = np.argwhere(owner == b)            # (y,x)
        if len(cells) == 0:
            cells = np.argwhere(sea)
        dpts = cells[:, ::-1].astype(float)        # (x,y)
        w = pri[cells[:, 0], cells[:, 1]] if weighted else np.ones(len(dpts))
        C = wkmeans(dpts, w, K, seed=b)
        used = {(int(bases[b][0]), int(bases[b][1]))}   # 排除基地自身格
        snapped = snap_distinct(C, dpts, used)
        if len(snapped) < K:                        # 區內補足
            for j in range(len(dpts)):
                c = (int(dpts[j, 0]), int(dpts[j, 1]))
                if c not in used:
                    used.add(c); snapped.append(c)
                if len(snapped) == K:
                    break
        if len(snapped) < K:                        # 全域後援(理論上用不到)
            for (y, x) in np.argwhere(sea):
                c = (int(x), int(y))
                if c not in used:
                    used.add(c); snapped.append(c)
                if len(snapped) == K:
                    break
        assert len(snapped) == K, f"{country} base {b}: only {len(snapped)}"
        nodes.extend(snapped)
    PP = np.array(nodes, dtype=np.int32)
    assert PP.shape == (len(bases) * K, 2)
    assert (ng[PP[:, 1], PP[:, 0]] == 0).all(), "節點落在 no_go"
    for b in range(len(bases)):
        seg = PP[b * K:(b + 1) * K]
        assert len({tuple(p) for p in seg}) == K, f"base {b} 節點不相異"
    return PP


def _snap_to_sea(gx, gy, sea):
    from collections import deque
    H, W = sea.shape
    gx = min(max(int(round(gx)), 0), W - 1); gy = min(max(int(round(gy)), 0), H - 1)
    if sea[gy, gx]:
        return gx, gy
    seen = {(gx, gy)}; q = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen:
                if sea[ny, nx]:
                    return nx, ny
                seen.add((nx, ny)); q.append((nx, ny))
    return gx, gy


# 各地區 100x100 格網地理基準(與 gfw_fetch_*.py / 海纜 / AIS·SAR binning 同一組)
BBOX = {"japan": (128.0, 30.0, 146.0, 46.0), "philippines": (116.0, 4.5, 127.0, 21.0)}


def real_bases(country):
    """由 base_coords_real.json 之真實經緯度,以 bbox 換算格座標並 snap 離陸。"""
    import json
    bc = json.load(open(os.path.join(HERE, "base_coords_real.json"), encoding="utf-8"))
    lon0, lat0, lon1, lat1 = BBOX[country]
    sea = (np.load(os.path.join(HERE, f"finalmap_{country}_real.npy")) == 0)
    out = []
    for e in bc[country]:
        gx = (e["lon"] - lon0) / (lon1 - lon0) * 100
        gy = (e["lat"] - lat0) / (lat1 - lat0) * 100
        out.append(_snap_to_sea(gx, gy, sea))
    return out


def main():
    for country in ["japan", "philippines"]:
        bases = real_bases(country)
        for weighted, suf in [(False, "real"), (True, "real_ddn")]:
            PP = gen(country, bases, weighted)
            out = os.path.join(HERE, f"Patrol_Point_{country}_{suf}.npy"); np.save(out, PP)
            print(f"[{country}_{suf}] {PP.shape} 基地{len(bases)} 點/基地={len(PP)//len(bases)} 全合法OK")


if __name__ == "__main__":
    main()
