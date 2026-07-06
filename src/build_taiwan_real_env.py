# -*- coding: utf-8 -*-
"""
建置 taiwan_real 真實環境:把真實禁航多邊形(射擊 S17 / 風場 S18·S19 / 彰化 S20·海纜 S15)
設為「硬 no_go_zone」。陸地與示意離島(澎湖/小琉球,真實地理)保留;底圖本無人造海上虛擬禁區。
落入新禁航的候選點/發射點,就近移到最近合法航行海域(保留每基地 40 點結構)。
同步重存所有 TWreal 情境(海纜重算、優先度/漁業/AIS/SAR 重新海域裁切)。

產出:
  finalmap_taiwan_real.npy            taiwan_real 專屬硬 no_go(陸+離島+真實多邊形)
  Patrol_Point_taiwan_real.npy        候選點(40 衝突點已就近移到合法海域)
  /tmp/twreal_bases.json              基地調整(供主程式 _BASES 覆寫)
  data/scenario_TWreal_*.npy          以新 no_go 重存/重裁
不動到台灣原環境(finalmap_correct.npy / Patrol_Point.npy 原封不動)。
"""
import os, json
import numpy as np
from collections import deque
import real_geo_sources as R
import tw_real_env as TR
import MOGA_GPSIFF_patrol_clean as M

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CODE, "..", "data")


def nearest_legal_sea(legal, x, y, occupied):
    """從 (x,y) 向外 BFS,找最近的合法海域格(legal=True)且未被佔用。"""
    H, W = legal.shape
    if legal[y, x] and (x, y) not in occupied:
        return (x, y)
    seen = {(x, y)}; dq = deque([(x, y)])
    while dq:
        cx, cy = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen:
                seen.add((nx, ny))
                if legal[ny, nx] and (nx, ny) not in occupied:
                    return (nx, ny)
                dq.append((nx, ny))
    return (x, y)


def main():
    base = np.load(os.path.join(CODE, "finalmap_correct.npy")); H, W = base.shape
    gr = R.Georeferencer()
    ov, layers, info = R.build_overlay_v1(base, gr, clip_sea=True)
    new = base.copy().astype(np.int64)
    new[ov] = 1                                  # 真實多邊形 → 硬禁航;陸地/離島(原本即 1)保留
    np.save(os.path.join(CODE, "finalmap_taiwan_real.npy"), new)
    legal = (new == 0)
    print(f"硬 no_go:陸+離島 {int((base==1).sum())} + 真實多邊形(海域)"
          f"{int((ov & (base==0)).sum())} → 共 {int((new==1).sum())} 禁航格;合法海域 {int(legal.sum())}")
    for k, m in layers.items():
        print(f"   {k:9s} {int((m & (base==0)).sum())} 海域格")

    # --- 候選點:衝突者就近移到合法海域 ---
    PP = np.load(os.path.join(CODE, "Patrol_Point.npy"))
    occupied = set((int(x), int(y)) for (x, y) in PP if legal[int(y), int(x)])
    PPr = PP.copy(); moved = 0
    for i, (x, y) in enumerate(PP):
        x, y = int(x), int(y)
        if not legal[y, x]:
            nx, ny = nearest_legal_sea(legal, x, y, occupied)
            PPr[i] = [nx, ny]; occupied.add((nx, ny)); moved += 1
    np.save(os.path.join(CODE, "Patrol_Point_taiwan_real.npy"), PPr)
    bad = sum(1 for (x, y) in PPr if not legal[int(y), int(x)])
    print(f"候選點:移動 {moved}/{len(PP)};移動後仍在禁航 {bad}(應為 0)")

    # --- 基地:衝突者就近移到合法海域(僅動到落在禁航者)---
    names = list(M._BASES["taiwan"].keys()); declared = list(M._BASES["taiwan"].values())
    newbases = {}; bmoved = []
    for nm, (x, y) in zip(names, declared):
        x, y = int(x), int(y)
        if not legal[y, x]:
            nx, ny = nearest_legal_sea(legal, x, y, occupied)
            newbases[nm] = [nx, ny]; occupied.add((nx, ny)); bmoved.append((nm, (x, y), (nx, ny)))
        else:
            newbases[nm] = [x, y]
    json.dump({"bases": newbases, "moved": bmoved}, open("/tmp/twreal_bases.json", "w"), ensure_ascii=False)
    print(f"基地:移動 {len(bmoved)} 個 → {bmoved}")

    # --- 重存/重裁所有 TWreal 情境(以新 no_go)---
    cable = TR.build_cable_weight(new)
    np.save(os.path.join(DATA, "scenario_TWreal_cable.npy"), cable)
    reclip = ["scenario_TWreal_priority_v1.npy", "scenario_TWreal_fishing_v1.npy",
              "scenario_TWreal_ais_v1.npy", "scenario_TWreal_sar_v1.npy",
              "scenario_TWreal_sar_dark_v1.npy"]
    done = []
    for fn in reclip:
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            w = np.load(p); w[new != 0] = 0.0; np.save(p, w); done.append(fn.split("_")[2])
    print(f"情境重存:海纜(重算)+ 重裁 {done}")
    # 同步更新疊層檔(供文件/檢視)
    np.save(os.path.join(DATA, "no_go_TWreal_overlay.npy"), ov.astype(np.uint8))


if __name__ == "__main__":
    main()
