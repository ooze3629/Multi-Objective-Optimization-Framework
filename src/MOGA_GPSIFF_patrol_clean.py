# -*- coding: utf-8 -*-
"""
多載具協同海域巡邏路徑規劃 — 多目標遺傳演算法(MOGA + GPSIFF)乾淨重寫版
======================================================================

Implementation of the MOGA + GPSIFF optimization workflow:

    初始化  →  迴圈{ GPSIFF 計分 → 二元競爭選擇 → 均勻交配 → 突變
                     → 修補 → 評估子代 → μ+λ 環境選擇 }  →  輸出 Pareto front

不含原始程式的額外機制(強制保留單一目標最佳、前段整體替換的兩階段選擇、
移民、2-opt、重啟),全程一律 μ+λ。

目標函式(經確認之定案版):
  F1  最大化  總覆蓋率:沿「整條航跡」(基地→各巡邏點→基地)的走廊覆蓋,
              有效感測半徑 r = rho0 + Delta(u)(圓盤足跡),以膨脹遮罩取聯集、
              依權重地圖加總;只計合法(非禁航)海域格。
  F2  最小化  距離成本 = C_V * sum_v ( u_v * L_v ),L_v 為含起航段與返航段
              之總航線長度(與 F1 的航段處理一致)。
  F3  最小化  協同運作距離(平方距離分解):每基地
              ||cP-cQ||^2 + var_P + var_Q,逐基地加總
              (cP, cQ 為 SV/USV 巡邏點中心點,var 為點對中心的平均平方距離)。

染色體:第一段 = 各載具無人機數 u_v;第二段 = 各載具巡邏點序列 r_{v,i}。
環境:13 基地、每基地 1 SV(偶數索引)+ 1 USV(奇數索引)、每基地 40 候選點、
      每船選 N_POINTS 點、返回「原基地」、穿越禁航之航段以「繞行頂點」繞過
      (垂直偏移試點 + BFS 後備,保證零穿越;巡邏點不變故 F3 不受影響)。
"""

import os
import csv
import math
import random
import datetime
import pickle
import hashlib
from copy import deepcopy
from collections import deque

import numpy as np
from scipy.ndimage import binary_dilation

import matplotlib
matplotlib.use("Agg")                         # 無視窗環境也能存圖
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401  (註冊 3D 投影)
# 中文字型:Windows 用 Microsoft JhengHei,Linux 退回 Noto CJK
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ======================================================================
# 參數
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SHIPS_PER_BASE = 2                       # 0=SV(有人), 1=USV(無人)
CANDIDATES_PER_BASE = 40
N_POINTS = 10                            # 每船巡邏點數(可一般化)

# === 無人機數值域:三層覆蓋校準(env 變數 DRONE_TIER 切換,預設 LOWER)===
# 由最小覆蓋(min set-cover)導出;船核 rho0=1、單機足跡 3x3、無人機不佔船艦格。
# 三層皆映射 r∈{2,3,4,5} → F1/DISK 不變,僅 F2=C_V*Σ(u*L) 之尺度隨層級改變。
# 推導與驗證見 v2_drone_calibration.py(verify_tiers / suggest_C_V)。
# 跨層比較:C_V 三層保持一致(勿逐層調);HV 參考點/正規化取三層解之聯集。
import os as _os
_DRONE_TIERS = {
    "LOWER":     {3: 2, 4: 3, 8: 4, 12: 5},   # 純覆蓋下界(>=1 重;F2 最省)
    "OPERATING": {6: 2, 8: 3, 14: 4, 24: 5},  # 營運:容許 1 台派遣仍全覆蓋(>=2 重)
    "SAFETY":    {8: 2, 14: 3, 22: 4, 36: 5}, # 安全上界:容許 2 台(>=3 重)
}
DRONE_TIER = _os.environ.get("DRONE_TIER", "LOWER").upper()
if DRONE_TIER not in _DRONE_TIERS:
    raise ValueError("未知 DRONE_TIER=%r;可用:%s" % (DRONE_TIER, list(_DRONE_TIERS)))
_RADIUS_OF = _DRONE_TIERS[DRONE_TIER]
DRONE_DOMAIN = sorted(_RADIUS_OF)        # 各船可搭載之無人機數(依所選層級)
BASE_R = 1                               # rho0:船本身基礎感測半徑
DELTA = {u: r - BASE_R for u, r in _RADIUS_OF.items()}  # r(u)=BASE_R+DELTA[u] → r∈{2,3,4,5}
C_V = 10                                 # 每單位無人機之距離成本率

POP_SIZE = 100
MAX_FES = 20000                          # 函數評估次數預算(嚴格計數)
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.1
SEED = None                              # 設為整數可重現

# ======================================================================
# 環境定義與切換(台灣 / 日本 / 菲律賓)
# ----------------------------------------------------------------------
# 本內層以「模組級全域」持有目前環境(底圖、候選點、基地、權重地圖等)。
# 外層 experiment.py 在跑實驗前呼叫 set_environment("taiwan"/"japan") 即可切換;
# baselines.make_scenarios(env) 再依環境產生對應的情境權重地圖。
# 預設環境為台灣,且台灣分支與原始版本逐行一致,可重現性不受影響。
# 座標慣例:點存 (x, y),取地圖值一律用 no_go_zone[y, x](列=y、行=x)。
# ======================================================================
CURRENT_ENV = "taiwan"
exit_point_coords = []

# 各環境之基地座標(同基地兩船共用)
_BASES = {
    "taiwan": {
        "基隆": (72, 86), "淡水": (63, 87), "新竹": (52, 76), "台中": (44, 65),
        "嘉義": (36, 46), "澎湖": (28, 50), "台南": (36, 34), "高雄": (39, 24),
        "恆春": (52, 10), "台東": (62, 30), "花蓮": (70, 56), "蘇澳": (75, 70),
        "澳底": (78, 81),
    },
    # 日本:海上保安廳 10 個管區本部
    "japan": {
        "小樽": (62, 81), "鹽竈": (68, 60), "橫濱": (62, 37), "名古屋": (56, 32),
        "神戶": (47, 30), "廣島": (41, 28), "北九州": (29, 36), "舞鶴": (47, 48),
        "新潟": (51, 51), "鹿兒島": (29, 15),
    },
    # 菲律賓：菲律賓海岸防衛隊 16 個區域基地
    "philippines": {
        "東北呂宋": (49, 74), "西北呂宋": (44, 71), "首都區/中呂宋": (46, 58), "南塔加洛": (52, 52),
        "巴拉望": (32, 35), "比科爾": (61, 52), "東維薩亞": (65, 43), "西維薩亞": (51, 42),
        "中維薩亞": (58, 43), "南維薩亞": (61, 35), "北民答那峨": (55, 24), "東北民答那峨": (66, 31),
        "東南民答那峨": (70, 20), "西南民答那峨": (45, 15), "南民答那峨": (66, 16), "邦薩摩洛自治區": (51, 18),
},
    # 真實環境(EEZ 真實海陸 + 真實海保基地;日/菲 no_go 已含真實禁航區,見 build_real_maps RESTRICTED)
    "japan_real": { "小樽": (72, 82), "鹽竈": (72, 51), "橫濱": (65, 34), "名古屋": (48, 30), "神戶": (40, 28), "廣島": (25, 26), "北九州": (15, 25), "舞鶴": (42, 34), "新潟": (61, 50), "鹿兒島": (15, 10) },
    "japan_real_ddn": { "小樽": (72, 82), "鹽竈": (72, 51), "橫濱": (65, 34), "名古屋": (48, 30), "神戶": (40, 28), "廣島": (25, 26), "北九州": (15, 25), "舞鶴": (42, 34), "新潟": (61, 50), "鹿兒島": (15, 10) },
    "philippines_real": { "東北呂宋": (51, 84), "西北呂宋": (38, 73), "首都區/中呂宋": (44, 61), "南塔加洛": (46, 55), "巴拉望": (25, 32), "比科爾": (71, 53), "東維薩亞": (81, 41), "西維薩亞": (61, 37), "中維薩亞": (73, 35), "南維薩亞": (62, 37), "北民答那峨": (78, 24), "東北民答那峨": (86, 32), "東南民答那峨": (88, 16), "西南民答那峨": (56, 15), "南民答那峨": (83, 8), "邦薩摩洛自治區": (74, 17) },
    "philippines_real_ddn": { "東北呂宋": (51, 84), "西北呂宋": (38, 73), "首都區/中呂宋": (44, 61), "南塔加洛": (46, 55), "巴拉望": (25, 32), "比科爾": (71, 53), "東維薩亞": (81, 41), "西維薩亞": (61, 37), "中維薩亞": (73, 35), "南維薩亞": (62, 37), "北民答那峨": (78, 24), "東北民答那峨": (86, 32), "東南民答那峨": (88, 16), "西南民答那峨": (56, 15), "南民答那峨": (83, 8), "邦薩摩洛自治區": (74, 17) },
}

# 各環境之底圖 / 候選點檔名(均與本程式同目錄)
_ENV_FILES = {
    "taiwan": ("finalmap_correct.npy", "Patrol_Point.npy"),
    "japan":  ("finalmap_japan_correct.npy", "Patrol_Point_japan.npy"),
    "philippines": ("finalmap_philippines_correct.npy", "Patrol_Point_philippines.npy"),
    # taiwan_real:真實禁航多邊形(射擊/風場/海纜)併入硬 no_go;陸地+示意離島保留。
    # 落入新禁航之候選點(40)與發射點(1,嘉義)已就近移到合法海域(見 build_taiwan_real_env.py)。
    "taiwan_real": ("finalmap_taiwan_real.npy", "Patrol_Point_taiwan_real.npy"),
    # taiwan_real_ddn:同 taiwan_real 之底圖/禁航/基地,唯候選點改為「資料驅動 + 基地責任區」
    # (測地最近劃分各基地責任區,區內依巡邏優先度做加權 k-means 取 40 點;見 build_ddn_env.py)。
    # 與 taiwan_real 形成乾淨 A/B:僅候選點生成方式不同,可比較「人工網格 vs 資料驅動節點」。
    "taiwan_real_ddn": ("finalmap_taiwan_real.npy", "Patrol_Point_taiwan_real_ddn.npy"),
    "japan_real":            ("finalmap_japan_real.npy", "Patrol_Point_japan_real.npy"),
    "japan_real_ddn":        ("finalmap_japan_real.npy", "Patrol_Point_japan_real_ddn.npy"),
    "philippines_real":      ("finalmap_philippines_real.npy", "Patrol_Point_philippines_real.npy"),
    "philippines_real_ddn":  ("finalmap_philippines_real.npy", "Patrol_Point_philippines_real_ddn.npy"),
}
_BASES["taiwan_real"] = dict(_BASES["taiwan"])   # 基地同台灣海巡轄區港...
_BASES["taiwan_real"]["嘉義"] = (35, 45)          # ...惟嘉義原點 (36,46) 落入真實禁航,就近移至合法海域
_BASES["taiwan_real_ddn"] = dict(_BASES["taiwan_real"])   # 基地與 taiwan_real 完全相同

ENV_NAME = None   # 目前環境名稱(由 set_environment 設定)
exit_coords = []

# 各環境是否將「出航點」吸附到貼陸地的海岸海域格:
#   台灣 = False(維持原始座標,§4.4 正式結果與 S1 情境綁定此組座標,不可變動)
#   日本 = True (全新環境,無鎖定結果,將基地由外海拉回海岸,取代手動 y±1~2)
_SNAP_DEFAULT = {
    "taiwan": False,
    "japan": True,
    "taiwan_real": False,
    "taiwan_real_ddn": False,
    "philippines": False,
}

# 菲律賓：基地對應出海口；None 表示基地可直接出海
_EXIT_POINTS = {
    "philippines": {
        "東北呂宋": (55, 74),
        "西北呂宋": (39, 71),
        "首都區/中呂宋": (41, 58),
        "南塔加洛": (48, 46),
        "巴拉望": (32, 40),
        "比科爾": (62, 52),
        "東維薩亞": (48, 46),
        "西維薩亞": (50, 42),
        "中維薩亞": (48, 46),
        "南維薩亞": (54, 30),
        "北民答那峨": (55, 20),
        "東北民答那峨": (54, 30),
        "東南民答那峨": (71, 14),
        "西南民答那峨": (44, 15),
        "南民答那峨": (66, 13),
        "邦薩摩洛自治區": (49, 18),
    }
}

# BFS 鄰居順序固定 → 吸附結果具確定性(可重現)
_NB4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _nearest_land(ngz, x, y):
    """由 (x,y) 找最近的陸地格(no_go==1),回傳 (x,y);供『港口貼海岸』顯示。"""
    from collections import deque
    H, W = ngz.shape
    seen = {(x, y)}; dq = deque([(x, y)])
    while dq:
        cx, cy = dq.popleft()
        if 0 <= cx < W and 0 <= cy < H and ngz[cy, cx] == 1:
            return (cx, cy)
        for dx, dy in _NB4:
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and n not in seen:
                seen.add(n); dq.append(n)
    return (x, y)


def _coastal_sea(ngz, x, y):
    """由 (x,y) 找最近、且與陸地相鄰的『海岸海域格』(no_go==0 且四鄰有陸地),
    供出航點使用:既是合法海域格(出航從海上起算),又緊貼海岸(不漂浮外海)。"""
    from collections import deque
    H, W = ngz.shape
    def is_coastal_sea(cx, cy):
        if not (0 <= cx < W and 0 <= cy < H) or ngz[cy, cx] != 0:
            return False
        return any(0 <= cx + dx < W and 0 <= cy + dy < H and ngz[cy + dy, cx + dx] == 1
                   for dx, dy in _NB4)
    seen = {(x, y)}; dq = deque([(x, y)])
    while dq:
        cx, cy = dq.popleft()
        if is_coastal_sea(cx, cy):
            return (cx, cy)
        for dx, dy in _NB4:
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and n not in seen:
                seen.add(n); dq.append(n)
    return (x, y)


def _build_weight_map(name, ngz):
    """各環境之『預設』權重地圖(供主程式單次最佳化與繪圖之底)。
    正式比較實驗會由 baselines.make_scenarios() 以 M.weight_map 覆蓋。"""
    H, W = ngz.shape
    w = np.where(ngz >= 0.9, 0.0, 0.2).astype(float)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    if name.startswith("taiwan"):
        # 海域基礎 0.2、方形熱區 0.8、方形抑制 0、圓形熱區距離遞減至 1.0(與原始版本一致)
        for x1, x2, y1, y2 in [(2, 5, 25, 35), (33, 36, 28, 38), (37, 40, 63, 73), (4, 7, 65, 75)]:
            w[y1:y2, x1:x2] = 0.8
        for x1, x2, y1, y2 in [(20, 25, 20, 30), (82, 87, 50, 65), (37, 39, 26, 30), (64, 68, 93, 95)]:
            w[y1:y2, x1:x2] = 0.0
        for (cx, cy) in [(35, 12), (82, 12), (20, 80), (15, 48), (90, 85)]:
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            m = d <= 15
            w[m] = np.maximum(w[m], 1 - d[m] / 15)
    else:  # japan:沿用日本程式之 5 熱區(半徑 15,僅限海域)
        for (cx, cy) in [(30, 15), (75, 20), (82, 58), (25, 58), (45, 78)]:
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            m = (d <= 15) & (ngz == 0)
            w[m] = np.maximum(w[m], 1 - d[m] / 15)
    return w


def set_environment(name="taiwan", snap_bases=None):
    """切換實驗環境。重新繫結所有與環境相關之模組級全域,並清空路徑快取
    (繞行結果與底圖綁定,跨環境不可共用)。回傳環境摘要 dict。

    base_ports  = 港口顯示座標(吸附到最近陸地格,讓基地『貼海岸』,僅供繪圖)。
    base_coords = 出航/返航座標(供路徑規劃、情境、貪婪使用)。
      snap_bases=True  → 吸附到貼陸地的海岸海域格(出航從海上起算,且緊貼海岸);
      snap_bases=False → 沿用宣告座標(維持既有可重現性)。
      省略時依環境預設(_SNAP_DEFAULT:台灣 False、日本 True)。"""
    global ENV_NAME, NUM_BASES, NUM_VESSELS, no_go_zone, Patrol_Point
    global MAP_H, MAP_W, base, base_coords, base_ports, weight_map, LEGAL, ship_candidate_range
    global _route_cache, exit_coords
    name = (name or "taiwan").lower()
    if name not in _ENV_FILES:
        raise ValueError(f"未知環境:{name}(可用:{list(_ENV_FILES)})")

    map_file, point_file = _ENV_FILES[name]
    no_go_zone = np.asarray(np.load(os.path.join(SCRIPT_DIR, map_file)))  # 0=海 1=陸/禁航
    Patrol_Point = np.load(os.path.join(SCRIPT_DIR, point_file))
    MAP_H, MAP_W = no_go_zone.shape

    declared = list(_BASES[name].values())                       # 宣告座標(冠廷原始輸入)
    names = list(_BASES[name].keys())
    snap = _SNAP_DEFAULT.get(name, False) if snap_bases is None else bool(snap_bases)
    base_ports = [_nearest_land(no_go_zone, x, y) for (x, y) in declared]   # 港口貼海岸(顯示)
    if snap:
        base_coords = [_coastal_sea(no_go_zone, x, y) for (x, y) in declared]  # 出航點吸附海岸
    else:
        base_coords = list(declared)                              # 維持原座標(可重現)
    base = dict(zip(names, base_coords))                          # base 與 base_coords 一致(路徑/貪婪用)
    if name == "philippines":
        exit_dict = _EXIT_POINTS["philippines"]
        exit_coords = [exit_dict[n] for n in names]
    else:
        exit_coords = [None for _ in range(len(base_coords))]
    NUM_BASES = len(base_coords)
    NUM_VESSELS = NUM_BASES * SHIPS_PER_BASE

    weight_map = _build_weight_map(name, no_go_zone)
    LEGAL = (no_go_zone == 0)
    ship_candidate_range = {i: (i // 2 * CANDIDATES_PER_BASE,
                                i // 2 * CANDIDATES_PER_BASE + CANDIDATES_PER_BASE)
                            for i in range(NUM_VESSELS)}
    _route_cache = {}   # 底圖改變 → 繞行快取必須清空
    ENV_NAME = name
    return {"env": name, "bases": NUM_BASES, "vessels": NUM_VESSELS,
            "map": no_go_zone.shape, "points": len(Patrol_Point), "snap_bases": snap}


# 預設環境:台灣(維持與原始版本完全一致)
set_environment("taiwan")

# 有效半徑與圓盤核(啟動時預建)
def radius_of(u):
    if u not in DELTA:
        raise ValueError("drone value %r 不在 DRONE_DOMAIN=%s(DRONE_TIER=%s)" % (u, DRONE_DOMAIN, DRONE_TIER))
    return BASE_R + DELTA[u]

def _disk_kernel(r):
    ky, kx = np.ogrid[-r:r + 1, -r:r + 1]
    return (kx * kx + ky * ky) <= r * r

DISK = {radius_of(u): _disk_kernel(radius_of(u)) for u in DRONE_DOMAIN}


def set_drone_tier(tier):
    """執行期切換無人機層級:同步更新 DRONE_TIER/DRONE_DOMAIN/DELTA/DISK。
    供 merge_and_refigure 等「以非預設層級重繪/finalize」之流程使用,確保 results meta
    的 drone_tier 與 drone_domain 一致(避免只改 DRONE_TIER 造成 provenance 不符)。"""
    global DRONE_TIER, DRONE_DOMAIN, DELTA, DISK
    tier = str(tier).upper()
    if tier not in _DRONE_TIERS:
        raise ValueError("未知 DRONE_TIER=%r;可用:%s" % (tier, list(_DRONE_TIERS)))
    rad = _DRONE_TIERS[tier]
    DRONE_TIER = tier
    DRONE_DOMAIN = sorted(rad)
    DELTA = {u: r - BASE_R for u, r in rad.items()}
    DISK = {radius_of(u): _disk_kernel(radius_of(u)) for u in DRONE_DOMAIN}
    return {"DRONE_TIER": DRONE_TIER, "DRONE_DOMAIN": list(DRONE_DOMAIN), "DELTA": dict(DELTA)}


# ======================================================================
# 幾何工具
# ======================================================================
def get_line(p1, p2):
    """Bresenham:回傳 p1→p2 直線上的整數格點。"""
    x0, y0 = int(p1[0]), int(p1[1])
    x1, y1 = int(p2[0]), int(p2[1])
    pts = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    x, y = x0, y0
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            pts.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            pts.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    pts.append((x, y))
    return pts

def crosses_no_go(p1, p2):
    """線段是否穿越禁航(陸地)格(Bresenham 逐格,短線段最快)。"""
    for (x, y) in get_line(p1, p2):
        if 0 <= x < MAP_W and 0 <= y < MAP_H and no_go_zone[y][x] == 1:
            return True
    return False


# ======================================================================
# 染色體:{'drones': [26], 'assignment': [26][N_POINTS], 'routes': [26] of 頂點序列}
# ======================================================================
def _legal_point(idx):
    x, y = Patrol_Point[idx]
    return 0 <= x < MAP_W and 0 <= y < MAP_H and no_go_zone[y][x] == 0

def _sample_ship(ship, used):
    """從該船候選區間中,抽 N_POINTS 個未使用且合法的點。"""
    start, end = ship_candidate_range[ship]
    cand = [i for i in range(start, end) if i not in used and _legal_point(i)]
    if len(cand) < N_POINTS:
        return None
    return random.sample(cand, N_POINTS)

def repair_assignment(assignment):
    """點層級修補:每船 N_POINTS 個合法、全域不重複、在區間內的點;不足則補足。"""
    used = set()
    new_assignment = []
    for i in range(NUM_VESSELS):
        allowed = set(range(*ship_candidate_range[i]))
        kept = []
        for pt in assignment[i]:
            if pt in allowed and _legal_point(pt) and pt not in used:
                kept.append(pt)
                used.add(pt)
        need = N_POINTS - len(kept)
        if need > 0:
            pool = [i2 for i2 in allowed if i2 not in used and _legal_point(i2)]
            random.shuffle(pool)
            add = pool[:need]
            kept.extend(add)
            used.update(add)
        new_assignment.append(kept)
    return new_assignment

# ----------------------------------------------------------------------
# 禁航段「繞行」修補:不更動巡邏點(F3 不受影響),只在穿越段插入繞行頂點。
#   1) 垂直偏移試點:在穿越段法向量兩側、由近到遠找一個繞行點 W,
#      使 prev→W、W→pt 皆不穿越禁航。
#   2) BFS 後備:單點不成則於合法海域格上做 BFS 找通路,再以視線簡化成少數頂點,
#      保證(只要存在合法走廊)能繞過。
# ----------------------------------------------------------------------
def _perpendicular_detour(p1, p2, max_off=12):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None
    nx, ny = -dy / L, dx / L                 # 單位法向量
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for off in range(1, max_off + 1):
        for s in (1, -1):
            wx = int(round(mx + s * off * nx))
            wy = int(round(my + s * off * ny))
            if 0 <= wx < MAP_W and 0 <= wy < MAP_H and no_go_zone[wy][wx] == 0:
                W = (wx, wy)
                if not crosses_no_go(p1, W) and not crosses_no_go(W, p2):
                    return W
    return None

def _bfs_path(start, goal):
    """合法海域格上的 8 連通 BFS;起點/終點允許通過。回傳格點路徑或 None。"""
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if start == goal:
        return [start]
    visited = {start}
    prev = {}
    dq = deque([start])
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while dq:
        cur = dq.popleft()
        if cur == goal:
            path = [cur]
            while path[-1] != start:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        cx, cy = cur
        for ddx, ddy in nbrs:
            nx2, ny2 = cx + ddx, cy + ddy
            if 0 <= nx2 < MAP_W and 0 <= ny2 < MAP_H and (nx2, ny2) not in visited:
                if (nx2, ny2) == goal or no_go_zone[ny2][nx2] == 0:
                    visited.add((nx2, ny2))
                    prev[(nx2, ny2)] = cur
                    dq.append((nx2, ny2))
    return None

def _simplify(path):
    """視線簡化:回傳內部繞行頂點(不含頭尾)。"""
    out = []
    anchor = 0
    n = len(path)
    while True:
        far = anchor + 1
        for j in range(anchor + 1, n):
            if not crosses_no_go(path[anchor], path[j]):
                far = j
        if far >= n - 1:
            break
        out.append(path[far])
        anchor = far
    return out

_route_cache = {}                            # (p1,p2) → 繞行頂點 list(地圖靜態,結果固定)

def _poly_ok(verts):
    """整條折線每段皆不穿越 no_go。"""
    return all(not crosses_no_go(verts[i], verts[i + 1]) for i in range(len(verts) - 1))

def route_around(p1, p2):
    """回傳 p1→p2 之間需插入的繞行頂點(座標 list);可直達則為空。已快取。
    若直線穿越 no_go 但找不到任何合法繞行(perpendicular / BFS 皆失敗或仍穿越),raise——
    不再靜默回傳 [](那等於把穿越段當直達,會污染 F1/F2 與可行性)。"""
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    key = (p1, p2)
    cached = _route_cache.get(key)
    if cached is not None:
        return cached
    if not crosses_no_go(p1, p2):
        res = []
    else:
        res = None
        W = _perpendicular_detour(p1, p2)
        if W is not None and _poly_ok([p1, W, p2]):
            res = [W]
        if res is None:
            path = _bfs_path(p1, p2)
            if path is not None and len(path) > 2:
                simp = _simplify(path)
                if _poly_ok([p1] + simp + [p2]):
                    res = simp
                elif _poly_ok(path):                 # 退回逐格 BFS 路徑(必不穿越)
                    res = path[1:-1]
        if res is None:
            raise RuntimeError(
                f"route_around 找不到合法繞行:{p1}->{p2}(海域可能不連通或端點不合法)")
    _route_cache[key] = res
    return res

def _map_sha(arr):
    """目前 no_go_zone 的指紋(供快取與底圖綁定檢核)。"""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def save_route_cache(path=None):
    """把目前環境已算好的繞行快取存檔(含底圖指紋)。需先 set_environment + 預熱。"""
    global _route_cache
    if path is None:
        path = os.path.join(SCRIPT_DIR, "..", "data", f"route_cache_{ENV_NAME}.pkl")
    meta = {"env": ENV_NAME, "map_shape": list(no_go_zone.shape),
            "map_sha": _map_sha(no_go_zone), "n_bases": NUM_BASES,
            "n_points": int(len(Patrol_Point)),
            "points_sha": _map_sha(Patrol_Point),
            "cache": _route_cache}
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    return path, len(_route_cache)

def load_route_cache(path=None, strict=True):
    """載入預先算好的繞行快取。必須先 set_environment 同一環境;會以底圖指紋核對,
    不符時(strict)拋錯,避免把 A 環境的快取套到 B 底圖上。"""
    global _route_cache
    if path is None:
        path = os.path.join(SCRIPT_DIR, "..", "data", f"route_cache_{ENV_NAME}.pkl")
    with open(path, "rb") as f:
        meta = pickle.load(f)
    cache = meta["cache"] if isinstance(meta, dict) and "cache" in meta else meta
    cur = _map_sha(no_go_zone)
    if isinstance(meta, dict):
        # env 不符:即使底圖相同(如 taiwan_real vs taiwan_real_ddn 共用底圖)也擋下
        if meta.get("env") and meta["env"] != ENV_NAME:
            msg = f"快取環境不符:檔案 env={meta.get('env')} != 目前 {ENV_NAME}。"
            if strict:
                raise ValueError(msg)
            print("[warn]", msg)
        # 候選點指紋不符:同底圖但不同候選點集(資料驅動 vs 網格)會被擋下
        pts_cur = _map_sha(Patrol_Point)
        if meta.get("points_sha") and meta["points_sha"] != pts_cur:
            msg = (f"快取候選點指紋不符:檔案 {meta.get('points_sha')} != 目前 {pts_cur}"
                   f"(env={meta.get('env')} vs {ENV_NAME});候選點集可能不同。")
            if strict:
                raise ValueError(msg)
            print("[warn]", msg)
    if isinstance(meta, dict) and meta.get("map_sha") and meta["map_sha"] != cur:
        msg = (f"快取底圖指紋不符:檔案 {meta.get('map_sha')} != 目前 {cur}"
               f"(env={meta.get('env')} vs {ENV_NAME});底圖可能已改變。")
        if strict:
            raise ValueError(msg)
        print("[warn]", msg)
    _route_cache = dict(cache)
    return len(_route_cache)


def build_routes(chromo):
    """
    建立各船航線頂點序列。

    一般環境：
        Base -> Patrol Points -> Base

    菲律賓環境：
        Base -> Exit Point -> Patrol Points -> Exit Point -> Base

    說明：
    1. 菲律賓多數基地位於陸上，因此保留真實基地座標。
    2. 若該基地有出海口，則路徑必須先經出海口再進入巡邏區。
    3. Base <-> Exit Point 視為出港/返港轉場段，直接加入路徑。
    4. Exit Point <-> Patrol Points 以及巡邏點之間仍透過 route_around() 繞開 no_go。
    """
    assignment = repair_assignment(chromo['assignment'])
    routes = []

    for v in range(NUM_VESSELS):
        base_id = v // 2
        home = base_coords[base_id]
        exit_pt = exit_coords[base_id] if ENV_NAME == "philippines" else None

        verts = [home]

        if exit_pt is not None:
            # Base -> Exit：出港轉場段，不呼叫 route_around，避免基地位於陸地造成問題
            verts.append(exit_pt)
            prev = exit_pt

            # Exit -> Patrol Points -> Exit
            targets = [tuple(Patrol_Point[idx]) for idx in assignment[v]] + [exit_pt]
        else:
            # 一般環境，或菲律賓中不需出海口之基地
            prev = home
            targets = [tuple(Patrol_Point[idx]) for idx in assignment[v]] + [home]

        for pt in targets:
            verts.extend(route_around(prev, pt))
            verts.append(pt)
            prev = pt

        if exit_pt is not None:
            # Exit -> Base：返港轉場段，直接加入
            verts.append(home)

        routes.append(verts)

    chromo['assignment'] = assignment
    chromo['routes'] = routes
    return chromo

def is_base_exit_edge(p1, p2, ship_idx):
    """
    判斷 p1 -> p2 是否為菲律賓環境的 Base <-> Exit 出港/返港轉場段。
    此段保留在 routes 內供畫圖，但不列入 F1 覆蓋與 F2 距離成本。
    """
    if ENV_NAME != "philippines":
        return False

    base_id = ship_idx // 2
    if base_id >= len(exit_coords):
        return False

    exit_pt = exit_coords[base_id]
    if exit_pt is None:
        return False

    home = base_coords[base_id]

    return (
        tuple(p1) == tuple(home) and tuple(p2) == tuple(exit_pt)
    ) or (
        tuple(p1) == tuple(exit_pt) and tuple(p2) == tuple(home)
    )


def mission_edges(chromo, ship_idx):
    """
    回傳真正巡邏任務航段。
    菲律賓會排除 Base <-> Exit 轉場段；
    台灣、日本維持原本所有航段皆計入。
    """
    verts = chromo['routes'][ship_idx]

    for a in range(len(verts) - 1):
        p1, p2 = verts[a], verts[a + 1]

        if is_base_exit_edge(p1, p2, ship_idx):
            continue

        yield p1, p2

def init_chromosome():
    drones = [random.choice(DRONE_DOMAIN) for _ in range(NUM_VESSELS)]
    assignment = []
    used = set()
    for v in range(NUM_VESSELS):
        ship = _sample_ship(v, used)
        if ship is None:
            return None
        assignment.append(ship)
        used.update(ship)
    return build_routes({'drones': drones, 'assignment': assignment})

def init_population(n):
    pop = []
    guard = 0
    while len(pop) < n and guard < n * 50:
        c = init_chromosome()
        if c is not None:
            pop.append(c)
        guard += 1
    return pop


# ======================================================================
# 目標函式
# ======================================================================
def calculate_F1(chromo):
    """
    走廊覆蓋(圓盤足跡、半徑 = rho0 + Delta(u)),以「依半徑分組、整體膨脹一次」計算:
      1) 將同半徑各船的航跡格點打到同一張 path mask;
      2) 對每個半徑的 path mask 用其圓盤結構元做一次 binary_dilation,交合法海域;
      3) 各半徑結果聯集後,依權重地圖加總。
    與逐格貼圖的結果完全相同(皆為航跡 ⊕ 圓盤 ∩ 合法的聯集),但快很多。
    """
    path_masks = {r: np.zeros((MAP_H, MAP_W), dtype=bool) for r in DISK}
    for v in range(NUM_VESSELS):
        r = radius_of(chromo['drones'][v])
        pm = path_masks[r]
        for p1, p2 in mission_edges(chromo, v):
            for (x, y) in get_line(p1, p2):
                if 0 <= x < MAP_W and 0 <= y < MAP_H:
                    pm[y, x] = True
    covered = np.zeros((MAP_H, MAP_W), dtype=bool)
    for r, pm in path_masks.items():
        if pm.any():
            covered |= binary_dilation(pm, structure=DISK[r]) & LEGAL
    return float(weight_map[covered].sum())

def calculate_F2(chromo):
    """
    距離成本 = C_V * sum_v (u_v * L_v)

    台灣/日本：
        L_v 計完整 Base -> Patrol -> Base。

    菲律賓：
        L_v 排除 Base <-> Exit 出港/返港轉場段，
        只計算 Exit -> Patrol Points -> Exit。
    """
    total = 0.0

    for v in range(NUM_VESSELS):
        L = 0.0

        for p1, p2 in mission_edges(chromo, v):
            L += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        total += chromo['drones'][v] * L

    return C_V * total

def calculate_F3(chromo):
    """協同運作距離(平方距離分解):每基地 ||cP-cQ||^2 + var_P + var_Q。"""
    assignment = chromo['assignment']
    total = 0.0
    for b in range(NUM_BASES):
        P = np.array([Patrol_Point[i] for i in assignment[2 * b]], dtype=float)
        Q = np.array([Patrol_Point[i] for i in assignment[2 * b + 1]], dtype=float)
        cP, cQ = P.mean(axis=0), Q.mean(axis=0)
        var_P = float(np.mean(np.sum((P - cP) ** 2, axis=1)))
        var_Q = float(np.mean(np.sum((Q - cQ) ** 2, axis=1)))
        total += float(np.sum((cP - cQ) ** 2)) + var_P + var_Q
    return total

def evaluate(chromo):
    chromo['F1'] = calculate_F1(chromo)
    chromo['F2'] = calculate_F2(chromo)
    chromo['F3'] = calculate_F3(chromo)
    return chromo


# ======================================================================
# GPSIFF / 選擇 / 交配 / 突變
# ======================================================================
def dominates(a, b):
    """a 支配 b?F1 取大、F2/F3 取小。"""
    not_worse = (a['F1'] >= b['F1']) and (a['F2'] <= b['F2']) and (a['F3'] <= b['F3'])
    strictly = (a['F1'] > b['F1']) or (a['F2'] < b['F2']) or (a['F3'] < b['F3'])
    return not_worse and strictly

def gpsiff(pop):
    """
    GPSIFF:fitness_i = p_i - q_i + c,c = 族群大小。
    一併回傳每個個體的 q_i(被支配數);q_i = 0 即為(該集合內的)非支配解。
    回傳 (fit_list, q_list)。
    """
    n = len(pop)
    fit, q_list = [], []
    for i in range(n):
        p = q = 0
        for j in range(n):
            if i == j:
                continue
            if dominates(pop[i], pop[j]):
                p += 1
            elif dominates(pop[j], pop[i]):
                q += 1
        fit.append(p - q + n)
        q_list.append(q)
    return fit, q_list

def binary_tournament(pop, fit):
    selected = []
    n = len(pop)
    for _ in range(n):
        i, j = random.sample(range(n), 2)
        selected.append(pop[i] if fit[i] >= fit[j] else pop[j])
    return selected

def crossover(parent1, parent2):
    """均勻交配:巡邏點段(逐船逐點 mask)與無人機數段(逐船 mask)。"""
    a1, a2 = [], []
    for v in range(NUM_VESSELS):
        p1, p2 = parent1['assignment'][v], parent2['assignment'][v]
        c1, c2 = [], []
        for k in range(N_POINTS):
            if random.random() < 0.5:
                c1.append(p1[k]); c2.append(p2[k])
            else:
                c1.append(p2[k]); c2.append(p1[k])
        a1.append(c1); a2.append(c2)
    d1, d2 = [], []
    for v in range(NUM_VESSELS):
        if random.random() < 0.5:
            d1.append(parent1['drones'][v]); d2.append(parent2['drones'][v])
        else:
            d1.append(parent2['drones'][v]); d2.append(parent1['drones'][v])
    # 交配後不建路徑:子代必接著突變並由 mutation 重建,避免重複 build_routes
    return {'drones': d1, 'assignment': a1}, {'drones': d2, 'assignment': a2}

def mutation(chromo):
    """逐點突變:每個巡邏點、每個無人機數基因各以 MUTATION_RATE 變異。"""
    m = deepcopy(chromo)
    used = set(p for ship in m['assignment'] for p in ship)
    # 無人機數:逐基因重抽
    for v in range(NUM_VESSELS):
        if random.random() < MUTATION_RATE:
            m['drones'][v] = random.choice(DRONE_DOMAIN)
    # 巡邏點:逐點以區間內未使用之合法點替換
    for v in range(NUM_VESSELS):
        start, end = ship_candidate_range[v]
        for k in range(N_POINTS):
            if random.random() < MUTATION_RATE:
                pool = [i for i in range(start, end) if i not in used and _legal_point(i)]
                if pool:
                    new_pt = random.choice(pool)
                    used.discard(m['assignment'][v][k])
                    used.add(new_pt)
                    m['assignment'][v][k] = new_pt
    return build_routes(m)


# ======================================================================
# Pareto front
# ======================================================================
def pareto_front(pop):
    front = []
    for i in range(len(pop)):
        if not any(j != i and dominates(pop[j], pop[i]) for j in range(len(pop))):
            front.append(pop[i])
    return front

def nd_extremes(pop, nd):
    """
    回傳非支配集合 nd 中,各目標極端值的個體索引:
    F1/F2/F3 各取 min 與 max(共 ≤6,去重後保留順序)。
    用於保住 Pareto 前緣在各目標軸上的延展範圍(邊界解)。
    """
    if not nd:
        return []
    idx = []
    for f in ("F1", "F2", "F3"):
        idx.append(min(nd, key=lambda i: pop[i][f]))   # 該目標最小者
        idx.append(max(nd, key=lambda i: pop[i][f]))   # 該目標最大者
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ======================================================================
# 主流程:純 MOGA + GPSIFF(全程 μ+λ)
# ======================================================================
def run_evolution(pop_size=POP_SIZE, max_fes=MAX_FES,
                  crossover_rate=CROSSOVER_RATE, verbose=True, snapshot_every=5):
    if SEED is not None:
        random.seed(SEED); np.random.seed(SEED)

    fe = 0
    population = init_population(pop_size)
    for ch in population:
        evaluate(ch); fe += 1

    archive = []                       # 跨代 Pareto 解(去重)
    seen = set()
    history = []                       # 每代收斂記錄
    snapshots = []                     # 每 snapshot_every 代之 (fes, [(F1,F2,F3)...]) 供事後畫 HV 收斂
    gen = 0
    while fe < max_fes:
        gen += 1
        # 1) GPSIFF 計分(取適應值;q 在環境選擇時才用)
        fit, _ = gpsiff(population)
        # 2) 二元競爭選擇
        mating = binary_tournament(population, fit)
        # 3) 交配 + 突變 → 子代
        offspring = []
        while len(offspring) < pop_size:
            if random.random() < crossover_rate:
                p1, p2 = random.sample(mating, 2)
                c1, c2 = crossover(p1, p2)
                offspring.append(mutation(c1))
                if len(offspring) < pop_size:
                    offspring.append(mutation(c2))
            else:
                offspring.append(mutation(deepcopy(random.choice(mating))))
        offspring = offspring[:pop_size]
        # 4) 評估子代
        ev = 0
        for ch in offspring:
            evaluate(ch); fe += 1; ev += 1
            if fe >= max_fes:
                break
        offspring = offspring[:ev]
        # 5) μ+λ 環境選擇(三層優先序):
        #    (1) 非支配解中各目標的極端值(每目標 min/max,共 ≤6,保住前緣延展)
        #    (2) 其餘非支配解(q=0),GPSIFF 高者優先
        #    (3) 被支配解,依 GPSIFF 由高到低填補
        combined = population + offspring
        cfit, cq = gpsiff(combined)
        nd = [i for i in range(len(combined)) if cq[i] == 0]     # 非支配(q=0)
        dom = [i for i in range(len(combined)) if cq[i] > 0]     # 被支配
        extremes = nd_extremes(combined, nd)                     # (1) 各目標極端值(去重後 ≤6)
        ex_set = set(extremes)
        nd_rest = sorted((i for i in nd if i not in ex_set), key=lambda i: -cfit[i])  # (2)
        dom.sort(key=lambda i: -cfit[i])                         # (3)
        ordered = extremes + nd_rest + dom
        population = [combined[i] for i in ordered[:pop_size]]
        # 記錄 Pareto front 與收斂指標
        pf = pareto_front(population)
        for ch in pf:
            key = (round(ch['F1'], 6), round(ch['F2'], 6), round(ch['F3'], 6))
            if key not in seen:
                seen.add(key)
                archive.append({'F1': ch['F1'], 'F2': ch['F2'], 'F3': ch['F3']})
        best_f1 = max(c['F1'] for c in population)
        min_f2 = min(c['F2'] for c in population)
        min_f3 = min(c['F3'] for c in population)
        history.append({'gen': gen, 'fes': fe, 'bestF1': best_f1,
                        'minF2': min_f2, 'minF3': min_f3, 'pareto': len(pf)})
        if gen % snapshot_every == 0 or fe >= max_fes:
            snapshots.append((fe, [(r['F1'], r['F2'], r['F3']) for r in archive]))
        if verbose and (gen % 10 == 0 or fe >= max_fes):
            print(f"世代 {gen:>4} | FEs {fe:>6} | bestF1 {best_f1:.2f} "
                  f"| minF2 {min_f2:.1f} | minF3 {min_f3:.2f} | Pareto {len(pf)}")

    return population, archive, history, snapshots


def save_pareto_csv(archive, path):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['F1', 'F2', 'F3'])
        for r in archive:
            w.writerow([r['F1'], r['F2'], r['F3']])


# ======================================================================
# 繪圖:路徑圖、收斂圖、Pareto 圖
# ======================================================================
def plot_paths(chromo, title, save_path):
    """路徑圖:權重地圖為底(熱區=高重要性),陸地灰底;各基地一色,SV 實線、USV 虛線。"""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(weight_map, cmap="YlOrRd", origin="lower", alpha=0.55, vmin=0, vmax=1)
    land = np.ma.masked_where(no_go_zone == 0, no_go_zone)
    ax.imshow(land, cmap="gray_r", origin="lower", vmin=0, vmax=1)
    colors = cm.get_cmap("tab20", NUM_BASES)
    for v in range(NUM_VESSELS):
        verts = chromo["routes"][v]
        xs = [p[0] for p in verts]; ys = [p[1] for p in verts]
        c = colors(v // 2)
        ls = "-" if v % 2 == 0 else "--"          # SV 實線 / USV 虛線
        ax.plot(xs, ys, color=c, lw=1.4, ls=ls, alpha=0.85)
        pts = [Patrol_Point[i] for i in chromo["assignment"][v]]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=c, s=12, zorder=3)
    ports = base_ports if "base_ports" in globals() and base_ports else list(base.values())
    for (name, (lx, ly)), (px, py) in zip(base.items(), ports):
        ax.plot([px, lx], [py, ly], c="red", lw=0.7, alpha=0.5, zorder=4)   # 港口↔出航點
        ax.scatter(lx, ly, c="red", s=16, marker="o", zorder=5,
                   edgecolors="k", linewidths=0.3)                          # 出航點(海域格)
        ax.scatter(px, py, c="red", s=80, marker="*", zorder=6,
                   edgecolors="k", linewidths=0.4)                          # 港口(貼海岸)
        ax.text(px + 1, py + 1, name, fontsize=8, color="black")
    ax.set_title(f"{title}\nF1={chromo['F1']:.1f}  F2={chromo['F2']:.0f}  F3={chromo['F3']:.1f}")
    ax.set_xlim(0, MAP_W); ax.set_ylim(0, MAP_H); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(save_path, dpi=130); plt.close(fig)

def plot_convergence(history, save_path):
    """收斂圖:三目標的代表值對 FEs 的變化。"""
    fes = [h["fes"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(fes, [h["bestF1"] for h in history], color="C0")
    axes[0].set_title("最佳 F1(覆蓋,越大越好)")
    axes[1].plot(fes, [h["minF2"] for h in history], color="C1")
    axes[1].set_title("最小 F2(成本,越小越好)")
    axes[2].plot(fes, [h["minF3"] for h in history], color="C2")
    axes[2].set_title("最小 F3(協同運作距離,越小越好)")
    for ax in axes:
        ax.set_xlabel("函數評估次數 (FEs)"); ax.grid(alpha=0.3)
    fig.suptitle("收斂曲線")
    fig.tight_layout(); fig.savefig(save_path, dpi=130); plt.close(fig)

def plot_pareto(archive, save_path):
    """Pareto 圖:3D 散佈 + 三個 2D 投影。"""
    if not archive:
        return
    F1 = np.array([r["F1"] for r in archive])
    F2 = np.array([r["F2"] for r in archive])
    F3 = np.array([r["F3"] for r in archive])
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(F1, F2, F3, c=F1, cmap="viridis", s=16)
    ax.set_xlabel("F1"); ax.set_ylabel("F2"); ax.set_zlabel("F3")
    ax.set_title("Pareto Front(3D)")
    a = fig.add_subplot(2, 2, 2); a.scatter(F1, F2, s=12, c="C0")
    a.set_xlabel("F1(覆蓋)↑"); a.set_ylabel("F2(成本)↓"); a.set_title("F1–F2"); a.grid(alpha=0.3)
    b = fig.add_subplot(2, 2, 3); b.scatter(F1, F3, s=12, c="C1")
    b.set_xlabel("F1(覆蓋)↑"); b.set_ylabel("F3(協同)↓"); b.set_title("F1–F3"); b.grid(alpha=0.3)
    d = fig.add_subplot(2, 2, 4); d.scatter(F2, F3, s=12, c="C2")
    d.set_xlabel("F2(成本)↓"); d.set_ylabel("F3(協同)↓"); d.set_title("F2–F3"); d.grid(alpha=0.3)
    fig.suptitle(f"跨代 Pareto Front(共 {len(archive)} 個非支配解)")
    fig.tight_layout(); fig.savefig(save_path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    t0 = datetime.datetime.now()
    pop, archive, history, snapshots = run_evolution()
    out_dir = os.path.join(SCRIPT_DIR, "results_" + DRONE_TIER + "_" + t0.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    save_pareto_csv(archive, os.path.join(out_dir, "Pareto_Front.csv"))
    # 每隔幾代的 Pareto 快照(供日後畫 HV 收斂,不需重算)
    with open(os.path.join(out_dir, "pareto_snapshots.pkl"), "wb") as f:
        pickle.dump({"snapshots": snapshots, "final_archive": archive,
                     "params": {"pop": POP_SIZE, "max_fes": MAX_FES, "seed": SEED}}, f)

    best_f1 = max(pop, key=lambda c: c["F1"])
    best_f2 = min(pop, key=lambda c: c["F2"])
    best_f3 = min(pop, key=lambda c: c["F3"])

    # 路徑圖、收斂圖、Pareto 圖
    plot_paths(best_f1, "F1 最佳(最大覆蓋)路徑", os.path.join(out_dir, "path_bestF1.png"))
    plot_paths(best_f2, "F2 最佳(最低成本)路徑", os.path.join(out_dir, "path_bestF2.png"))
    plot_paths(best_f3, "F3 最佳(最小協同距離)路徑", os.path.join(out_dir, "path_bestF3.png"))
    plot_convergence(history, os.path.join(out_dir, "convergence.png"))
    plot_pareto(archive, os.path.join(out_dir, "pareto_front.png"))

    print("\n=== 完成 ===")
    print(f"最佳 F1(覆蓋): {best_f1['F1']:.2f}")
    print(f"最低 F2(成本): {best_f2['F2']:.2f}")
    print(f"最低 F3(協同距離): {best_f3['F3']:.2f}")
    print(f"跨代 Pareto 解數: {len(archive)}")
    print(f"輸出資料夾: {out_dir}")
    print(f"耗時: {datetime.datetime.now() - t0}")
