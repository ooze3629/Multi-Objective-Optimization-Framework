# -*- coding: utf-8 -*-
"""
基準演算法比較:本方法(MOGA+GPSIFF, three_tier) vs (μ+λ)-ES、隨機搜尋、貪婪。
三種模擬海域(權重分布不同)。指標 HV / IGD⁺ 的正規化與參考前緣皆取「全演算法聯集」。
"""
import warnings; warnings.filterwarnings("ignore")
import random, time, pickle, os, sys
from copy import deepcopy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC","Noto Sans CJK JP","Microsoft JhengHei","DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import MOGA_GPSIFF_patrol_clean as M
import moea_metrics as MM
from compare_sel import run_cmp

# ---------------- 三種模擬海域(權重地圖)----------------
def _blank():
    w = np.where(M.no_go_zone == 0, 0.2, 0.0).astype(float)
    return w
def _bump(w, cx, cy, amp, sig):
    yy, xx = np.ogrid[0:M.MAP_H, 0:M.MAP_W]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    w[:] = np.maximum(w, amp * np.exp(-d2 / (2 * sig * sig)))
def _scenarios_taiwan():
    scn = {}
    # S1 近岸熱區:各基地附近高權重
    w = _blank()
    for (bx, by) in M.base_coords:
        _bump(w, bx, by, 1.0, 7)
    w[M.no_go_zone != 0] = 0.0; scn["S1 近岸熱區"] = w
    # S2 外海熱區:離岸/邊緣海域
    w = _blank()
    for (cx, cy, a) in [(12, 80, 1.0), (88, 78, 1.0), (12, 22, 0.9), (88, 26, 0.9), (50, 92, 0.85)]:
        _bump(w, cx, cy, a, 9)
    w[M.no_go_zone != 0] = 0.0; scn["S2 外海熱區"] = w
    # S3 分散群集:散布多點
    w = _blank()
    for (cx, cy, a) in [(20, 50, 0.9), (82, 52, 0.9), (50, 14, 0.85), (38, 86, 0.85), (72, 78, 0.8), (26, 30, 0.8)]:
        _bump(w, cx, cy, a, 7)
    w[M.no_go_zone != 0] = 0.0; scn["S3 分散群集"] = w
    return scn


def _scenarios_japan():
    """日本三情境,與台灣同設計理念(近岸 / 外海 / 分散),確保四項標準產物可同規格比較。
    J2 外海採用日本程式原本記載之 5 個重點巡邏海域(保留學生原始設定)。"""
    scn = {}
    # J1 近岸熱區:10 個管區本部附近高權重(與 S1 同配方)
    w = _blank()
    for (bx, by) in M.base_coords:
        _bump(w, bx, by, 1.0, 7)
    w[M.no_go_zone != 0] = 0.0; scn["J1 近岸熱區"] = w
    # J2 外海熱區:沿用日本程式原本 5 熱區(九州南、本州太平洋南、東北太平洋、日本海西、北海道/日本海北)
    w = _blank()
    for (cx, cy, a) in [(30, 15, 1.0), (75, 20, 1.0), (82, 58, 0.95), (25, 58, 0.9), (45, 78, 0.85)]:
        _bump(w, cx, cy, a, 9)
    w[M.no_go_zone != 0] = 0.0; scn["J2 外海熱區"] = w
    # J3 分散群集:列島四周散布多點
    w = _blank()
    for (cx, cy, a) in [(45, 90, 0.9), (20, 70, 0.9), (80, 46, 0.9), (78, 16, 0.85), (38, 50, 0.85), (28, 26, 0.8)]:
        _bump(w, cx, cy, a, 7)
    w[M.no_go_zone != 0] = 0.0; scn["J3 分散群集"] = w
    return scn

def _scenarios_philippines():
    """菲律賓三個模型化任務情境。
    P1：西菲律賓海敏感海域
    P2：巴拉望與南部海域巡防
    P3：東側外海巡防
    """
    H, W = M.no_go_zone.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    def hotspot(centers):
        w = np.where(M.no_go_zone == 0, 0.2, 0.0).astype(float)

        for cx, cy, radius in centers:
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            mask = (d <= radius) & (M.no_go_zone == 0)
            w[mask] = np.maximum(w[mask], 1 - d[mask] / radius)

        return w

    return {
        "P1 西菲律賓海敏感海域": hotspot([  #對應西側敏感海域、海上執法與主權巡護需求
            (35, 62, 15),
            (42, 54, 10),
        ]),
        "P2 巴拉望與南部海域": hotspot([  #對應巴拉望、民答那峨與南部航道巡防
            (27, 35, 14),
            (49, 18, 12),
            (66, 13, 12),
        ]),
        "P3 東側外海巡防": hotspot([  #對應菲律賓東側外海與開放太平洋方向巡防
            (76, 45, 16),
            (85, 25, 14),
        ]),
    }

def _scenarios_taiwan_real():
    """台灣真實環境:三個『正交』真實情境,各自獨立最佳化、分開呈現(多任務比較)。
    經相關性稽核,選彼此最不重疊的三軸(避免優先度≈漁業≈AIS 之冗餘):
      1) 航運密度(AIS, S21)      —— 一般海運交通 / SAR 需求 / 在場背景
      2) 暗船查緝(SAR 暗船, S22) —— 可疑 / 不發報目標(查緝取向;與 AIS 相關僅 0.42)
      3) 海纜基礎設施(海纜 v1)   —— 真實海纜幾何風險(與 AIS 相關僅 ~0.1;結構真實、權重 proxy)
    成分/衍生層(priority/fishing/shipping/sar-all/cable-v0)留存 data/ 供 ablation,預設不納入主迴圈
    (理由:與 AIS 高相關 0.86–0.96 之冗餘;見 data/TWreal_provenance.md)。
    為「真實港口 + 真實熱點」之半合成情境,須與人造 S1/S2/S3 分開呈現。"""
    D = os.path.join(M.SCRIPT_DIR, "..", "data")
    spec = [("TWreal 航運密度(AIS)", "scenario_TWreal_ais_v1.npy"),
            ("TWreal 暗船查緝(SAR)", "scenario_TWreal_sar_dark_v1.npy"),
            ("TWreal 海纜基礎設施", "scenario_TWreal_cable_geo_v1.npy")]
    scn = {}
    for name, fn in spec:
        w = np.load(os.path.join(D, fn))
        assert w.shape == M.no_go_zone.shape, f"{fn} 尺寸與底圖不符"
        scn[name] = w
    return scn


def _scenarios_real(tag):
    """真實環境三正交情境(航運密度 AIS / 暗船 SAR未匹配 / 海纜 v1)。tag: JP 或 PH。
    與 taiwan_real 同方法。本函式僅建三正交情境權重;no_go(含真實禁航區)在底圖 finalmap_*_real.npy。"""
    D = os.path.join(M.SCRIPT_DIR, "..", "data")
    spec = [(f"{tag}real 航運密度(AIS)", f"scenario_{tag}real_ais_v1.npy"),
            (f"{tag}real 暗船查緝(SAR未匹配)", f"scenario_{tag}real_sar_dark_v1.npy"),
            (f"{tag}real 海纜基礎設施", f"scenario_{tag}real_cable_geo_v1.npy")]
    scn = {}
    for name, fn in spec:
        w = np.load(os.path.join(D, fn))
        assert w.shape == M.no_go_zone.shape, f"{fn} 尺寸與底圖不符"
        scn[name] = w
    return scn


def _scenarios_japan_real():
    return _scenarios_real("JP")


def _scenarios_philippines_real():
    return _scenarios_real("PH")


def make_scenarios(env=None):
    """依環境產生情境權重地圖。env 省略時取 M.ENV_NAME(目前環境)。
    回傳 {情境名: weight_map}。weight_map 形狀與目前環境底圖一致。"""
    env = (env or getattr(M, "ENV_NAME", "taiwan")).lower()
    if env == "japan":
        return _scenarios_japan()
    if env in ("taiwan_real", "taiwan_real_ddn"):
        return _scenarios_taiwan_real()   # 同底圖/權重圖,僅候選點不同 → 共用三真實情境
    if env in ("japan_real", "japan_real_ddn"):
        return _scenarios_japan_real()
    if env in ("philippines_real", "philippines_real_ddn"):
        return _scenarios_philippines_real()
    if env == "philippines":
        return _scenarios_philippines()
    return _scenarios_taiwan()


def make_mixed_scenario(env=None, mode="equal_mass"):
    """把該環境三正交層(AIS / SAR暗船 / 海纜)合成單一『混合』情境。
      mode='equal_mass'(等質量):各層先除以自身海域總質量(每層 Σ=1,等質量)再相加,最後對最大值
        正規化到 [0,1]——海纜等低量值層不會被高量值的 AIS 吃掉(等質量比例在相加時已固定,/max 僅整體縮放)。
      mode='equal_max'(等峰值,對照):各層除以自身海域最大值後相加(峰值對齊,非質量對齊)。
    僅真實環境(具三正交層)支援。回傳 {情境名: weight_map}(形狀同底圖;非海域格權重 0)。"""
    env = (env or getattr(M, "ENV_NAME", "taiwan")).lower()
    base = make_scenarios(env)
    names = list(base.keys()); layers = list(base.values())
    if len(layers) != 3:
        raise ValueError(f"{env}: 混合情境需恰三個正交層,實得 {len(layers)}({names})")
    sea = (M.no_go_zone == 0)
    mixed = np.zeros(M.no_go_zone.shape, dtype=float)
    for w in layers:
        w = w.astype(float)
        if mode == "equal_mass":
            denom = float(w[sea].sum())
        elif mode == "equal_max":
            denom = float(w[sea].max())
        else:
            raise ValueError(f"未知 mode: {mode}(可用 equal_mass / equal_max)")
        mixed += (w / denom) if denom > 0 else w
    mx = float(mixed[sea].max())
    if mx > 0:
        mixed = mixed / mx                          # → [0,1](保留各層等質量比例,僅整體縮放,與既有情境同尺度)
    mixed[~sea] = 0.0                                 # 非海域格權重歸零(F1 僅計合法海域格)
    tag = names[0].split()[0]                         # TWreal / JPreal / PHreal
    phrase = "等質量混合(AIS+SAR+海纜)" if mode == "equal_mass" else "等峰值混合(AIS+SAR+海纜)"
    return {f"{tag} {phrase}": mixed}

# ---------------- 非支配典藏輔助 ----------------
def _nd_add(arc, c):
    if any(M.dominates(a, c) for a in arc):
        return arc
    arc = [a for a in arc if not M.dominates(c, a)]
    arc.append(c)
    return arc

# ---------------- 基準一:(μ+λ)-ES(突變繁殖、無交配)----------------
def run_es(seed, pop=100, fes=1200):
    random.seed(seed); np.random.seed(seed)
    P = M.init_population(pop); fe = 0
    for ch in P: M.evaluate(ch); fe += 1
    arc, seen = [], set()
    def add(chs):
        nonlocal arc
        for ch in chs:
            k = (round(ch["F1"], 6), round(ch["F2"], 6), round(ch["F3"], 6))
            if k in seen: continue
            seen.add(k); arc = _nd_add(arc, {"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
    while fe < fes:
        Q = [M.mutation(deepcopy(random.choice(P))) for _ in range(pop)]
        ev = 0
        for ch in Q:
            M.evaluate(ch); fe += 1; ev += 1
            if fe >= fes: break
        Q = Q[:ev]
        C = P + Q; fit, q = M.gpsiff(C)
        nd = [i for i in range(len(C)) if q[i] == 0]; dom = [i for i in range(len(C)) if q[i] > 0]
        chosen = (sorted(nd, key=lambda i: -fit[i]) + sorted(dom, key=lambda i: -fit[i]))[:pop]
        P = [C[i] for i in chosen]
        add(M.pareto_front(P))
    return arc

# ---------------- 基準二:隨機搜尋 ----------------
def run_random(seed, pop=100, fes=1200):
    # 隨機基準:pop 參數被忽略;fes = 獨立隨機取樣次數(逐一 evaluate),非族群大小/選擇行為。
    random.seed(seed); np.random.seed(seed)
    arc = []
    for _ in range(fes):
        ch = M.init_population(1)[0]; M.evaluate(ch)
        arc = _nd_add(arc, {"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
    return arc

# ---------------- 基準三:貪婪建構 ----------------
def _local_weight(pt, r):
    x, y = pt
    x0, x1 = max(0, x - r), min(M.MAP_W, x + r + 1)
    y0, y1 = max(0, y - r), min(M.MAP_H, y + r + 1)
    return float(M.weight_map[y0:y1, x0:x1].sum())

def run_greedy():
    def build(mode, dv):
        used, assignment = set(), []
        for v in range(M.NUM_VESSELS):
            lo, hi = M.ship_candidate_range[v]
            cands = [i for i in range(lo, hi) if M._legal_point(i) and i not in used]
            if mode == "dist":
                bx, by = M.base_coords[v // 2]
                cands.sort(key=lambda i: (M.Patrol_Point[i][0] - bx) ** 2 + (M.Patrol_Point[i][1] - by) ** 2)
            else:
                r = M.radius_of(dv)
                cands.sort(key=lambda i: -_local_weight(M.Patrol_Point[i], r))
            pick = cands[:M.N_POINTS]; used.update(pick); assignment.append(pick)
        ch = M.build_routes({"drones": [dv] * M.NUM_VESSELS, "assignment": assignment})
        M.evaluate(ch); return ch
    arc = []
    _vals = sorted(M.DRONE_DOMAIN)   # v2:依當前層級取值(dist 用低成本/小半徑,cover 用大半徑)
    for mode, dv in [("dist", _vals[0]), ("dist", _vals[1]), ("cover", _vals[-1]), ("cover", _vals[-2])]:
        ch = build(mode, dv)
        arc = _nd_add(arc, {"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
    return arc


# ---- 記錄快照之變體(供 HV 收斂曲線;不需事後重算)----
def es_snap(seed, pop=100, fes=3000, snap_gen=2):
    random.seed(seed); np.random.seed(seed)
    P = M.init_population(pop); fe = 0
    for ch in P: M.evaluate(ch); fe += 1
    arc, snaps, gen = [], [], 0
    while fe < fes:
        gen += 1
        Q = [M.mutation(deepcopy(random.choice(P))) for _ in range(pop)]
        ev = 0
        for ch in Q:
            M.evaluate(ch); fe += 1; ev += 1
            if fe >= fes: break
        Q = Q[:ev]; C = P + Q; fit, q = M.gpsiff(C)
        nd = [i for i in range(len(C)) if q[i] == 0]; dom = [i for i in range(len(C)) if q[i] > 0]
        P = [C[i] for i in (sorted(nd, key=lambda i: -fit[i]) + sorted(dom, key=lambda i: -fit[i]))[:pop]]
        for ch in M.pareto_front(P):
            arc = _nd_add(arc, {"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
        if gen % snap_gen == 0 or fe >= fes:
            snaps.append((fe, [(a["F1"], a["F2"], a["F3"]) for a in arc]))
    return arc, snaps

def rand_snap(seed, pop=100, fes=3000, snap_ev=200):
    # 隨機基準(含快照):pop 參數被忽略;fes = 獨立隨機取樣次數,非族群大小/選擇行為。
    random.seed(seed); np.random.seed(seed)
    arc, snaps, fe, nxt = [], [], 0, 200
    for _ in range(fes):
        ch = M.init_population(1)[0]; M.evaluate(ch); fe += 1
        arc = _nd_add(arc, {"F1": ch["F1"], "F2": ch["F2"], "F3": ch["F3"]})
        if fe >= nxt:
            snaps.append((fe, [(a["F1"], a["F2"], a["F3"]) for a in arc])); nxt += snap_ev
    snaps.append((fe, [(a["F1"], a["F2"], a["F3"]) for a in arc]))
    return arc, snaps

# ---------------- 驅動 ----------------
ALGOS = ["本方法", "NSGA-III", "SMS-EMOA", "ES", "隨機", "貪婪"]


def _ext_solve(method, seed, pop, fes):
    """取 experiment._solve 之 archive(NSGA-III / SMS-EMOA;懶載入避免與 experiment 循環匯入)。"""
    from experiment import _solve, Config
    r = _solve(method, seed, Config(pop=pop, fes=fes, snapshot_every=2), record_snap=True)
    return r["archive"], r["snaps"]

def run_all(seeds, pop=100, fes=1200):
    if fes < pop:
        raise SystemExit(f"[參數錯誤] fes({fes}) < pop({pop}):ES/MOEA 基準初始族群即用掉 pop 次評估會超支 FEs;請設 fes ≥ pop。")
    scns = make_scenarios()
    res = {s: {a: {"HV": [], "IGD+": []} for a in ALGOS} for s in scns}
    cm = {s: {a: [] for a in ["NSGA-III", "SMS-EMOA", "ES", "隨機", "貪婪"]} for s in scns}
    for sname, wmap in scns.items():
        M.weight_map = wmap
        greedy = run_greedy()
        for sd in seeds:
            arcs = {}
            arcs["本方法"] = run_cmp("three_tier", sd, pop, fes)[1]
            arcs["NSGA-III"] = _ext_solve("NSGA-III", sd, pop, fes)[0]
            arcs["SMS-EMOA"] = _ext_solve("SMS-EMOA", sd, pop, fes)[0]
            arcs["ES"] = run_es(sd, pop, fes)
            arcs["隨機"] = run_random(sd, pop, fes)
            arcs["貪婪"] = greedy
            allarc = [a for al in ALGOS for a in arcs[al]]
            lo, hi = MM._bounds(allarc); ref = MM.nondominated(allarc)
            for al in ALGOS:
                res[sname][al]["HV"].append(MM.hv(arcs[al], lo, hi))
                res[sname][al]["IGD+"].append(MM.igd_plus(arcs[al], ref, lo, hi))
            for al in ["NSGA-III", "SMS-EMOA", "ES", "隨機", "貪婪"]:
                cm[sname][al].append((MM.c_metric(arcs["本方法"], arcs[al]), MM.c_metric(arcs[al], arcs["本方法"])))
            print(f"  [{sname}] seed {sd}: HV 本方法={res[sname]['本方法']['HV'][-1]:.3f} "
                  f"ES={res[sname]['ES']['HV'][-1]:.3f} 隨機={res[sname]['隨機']['HV'][-1]:.3f} 貪婪={res[sname]['貪婪']['HV'][-1]:.3f}")
    return res, cm, list(scns.keys())

def mstd(xs):
    import statistics as st
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)

def report(res, cm, scn_names):
    for s in scn_names:
        print(f"\n=== {s} ===")
        for metric in ("HV", "IGD+"):
            line = f"{metric}: "
            for a in ALGOS:
                m, sd = mstd(res[s][a][metric]); line += f"{a} {m:.3f}±{sd:.3f}  "
            print(line)
        for a in ["NSGA-III", "SMS-EMOA", "ES", "隨機", "貪婪"]:
            cab = mstd([x[0] for x in cm[s][a]])[0]; cba = mstd([x[1] for x in cm[s][a]])[0]
            print(f"  C(本方法,{a})={cab:.2f}  C({a},本方法)={cba:.2f}")

def plot_bars(res, scn_names, save, algos=None, title_suffix=""):
    algos = list(algos) if algos is not None else list(ALGOS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    n = len(algos); x = np.arange(len(scn_names)); w = 0.8 / n
    colors = ["#1F4E79", "#C0504D", "#7030A0", "#2E9E5B", "#E08A1E", "#999999"][:n]
    for j, metric in enumerate(("HV", "IGD+")):
        for i, a in enumerate(algos):
            means = [mstd(res[s][a][metric])[0] for s in scn_names]
            errs = [mstd(res[s][a][metric])[1] for s in scn_names]
            axes[j].bar(x + (i - (n - 1) / 2.0) * w, means, w, yerr=errs, capsize=3, label=a, color=colors[i])
        axes[j].set_xticks(x); axes[j].set_xticklabels(scn_names, fontsize=10)
        axes[j].set_title(("Hypervolume(越大越好)" if metric == "HV" else "IGD+(越小越好)"))
        axes[j].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=9, ncol=2)
    fig.suptitle("本方法 vs 基準演算法(三模擬海域)" + (f"  {title_suffix}" if title_suffix else ""), fontweight="bold")
    fig.tight_layout(); fig.savefig(save, dpi=130); plt.close(fig)

# ============================================================
# 路徑快取持久化 + 預熱(航線只用到同基地候選點間線段,數量有限)
# ============================================================
CACHE_PATH = "route_cache.pkl"

def _cache_path(path=None):
    """各環境之路徑快取分檔:台灣維持 route_cache.pkl(相容既有),其餘環境加環境後綴。
    繞行結果與底圖綁定,跨環境不可共用。"""
    if path is not None:
        return path
    env = getattr(M, "ENV_NAME", "taiwan")
    return CACHE_PATH if env == "taiwan" else f"route_cache_{env}.pkl"

def load_cache(path=None):
    """繞行快取載入(新舊相容)。
    path=None:走 M.load_route_cache → data/route_cache_<env>.pkl(meta 格式 + 底圖指紋核對);
    給定 path:相容讀取(meta 之 'cache' 或 raw dict)。
    註:新流程的 canonical 快取在 data/,由 build_route_caches.py 預建。"""
    if path is None:
        try:
            return M.load_route_cache(strict=True)
        except FileNotFoundError:
            return len(M._route_cache)
    obj = pickle.load(open(path, "rb"))
    cache = obj["cache"] if isinstance(obj, dict) and "cache" in obj else obj
    M._route_cache.update(cache)
    return len(M._route_cache)

def save_cache(path=None):
    """繞行快取存檔。path=None → M.save_route_cache(存 data/,meta 格式);給定 path → raw 存該路徑。"""
    if path is None:
        return M.save_route_cache()
    pickle.dump(M._route_cache, open(path, "wb"))

def prewarm(path=None, max_wall=240):
    """[legacy] 漸進預熱繞行快取。新流程請改用 build_route_caches.py
    (寫 data/、含完整性檢查與 synthetic philippines 之 exit anchor)。
    path=None → 走 data/ canonical(M.load/save_route_cache);給定 path → raw 檔。"""
    if path is None:
        try:
            load_cache()        # data/ canonical(缺檔則 _route_cache 維持現狀)
        except Exception:
            pass
    elif os.path.exists(path):
        load_cache(path)
    t0 = time.time()
    for b in range(M.NUM_BASES):
        if M.ENV_NAME == "philippines" and M.exit_coords[b] is not None:
            home = tuple(int(x) for x in M.exit_coords[b])     # 出海口錨點(同 build_route_caches)
        else:
            home = tuple(int(x) for x in M.base_coords[b])
        lo, hi = b * M.CANDIDATES_PER_BASE, (b + 1) * M.CANDIDATES_PER_BASE
        pts = [home] + [tuple(M.Patrol_Point[i]) for i in range(lo, hi) if M._legal_point(i)]
        for p in pts:
            for q in pts:
                if p != q:
                    M.route_around(p, q)
        save_cache(path)
        print(f"  base {b} 預熱完成,cache={len(M._route_cache)},{time.time()-t0:.0f}s")
        if time.time() - t0 > max_wall and b < M.NUM_BASES - 1:
            print(f"[時間到] 預熱到 base {b};再次執行 prewarm 續跑。")
            return False
    print(f"== 預熱完成,cache={len(M._route_cache)} ==")
    return True


# ============================================================
# 正式規模:斷點續跑(每完成一個 (海域,種子) 即存檔,可多次執行接續)
# ============================================================
def run_formal(seeds, pop=100, fes=8000, out="formal_results.pkl", max_wall=240):
    if fes < pop:
        raise SystemExit(f"[參數錯誤] fes({fes}) < pop({pop}):ES/MOEA 基準初始族群即用掉 pop 次評估會超支 FEs;請設 fes ≥ pop。")
    load_cache()
    scns = make_scenarios()
    if os.path.exists(out):
        data = pickle.load(open(out, "rb"))
        _m = data.get("meta", {})
        # provenance 一致性:同一 pkl 內所有結果必須同設定;任一不符即拒(避免污染正式 baseline)
        _expect = {"pop": pop, "fes": fes, "seeds": list(seeds), "algos": list(ALGOS),
                   "drone_tier": getattr(M, "DRONE_TIER", None),
                   "drone_domain": list(getattr(M, "DRONE_DOMAIN", []) or []),
                   "delta": dict(getattr(M, "DELTA", {}) or {}),
                   "env": getattr(M, "ENV_NAME", None)}
        for _k, _live in _expect.items():
            _old = _m.get(_k)
            if _old is not None and _old != _live:
                raise SystemExit(f"[{_k} 不符] 既有 {out} 的 {_k}={_old} ≠ 本次 {_live};請勿把不同設定結果併入同一檔(請刪除舊檔或改檔名)。")
    else:
        data = {"meta": {"pop": pop, "fes": fes, "seeds": list(seeds)}, "greedy": {}, "runs": {}, "snaps": {}, "best": {}}
    data["meta"]["algos"] = list(ALGOS)
    data["meta"].update({"drone_tier": getattr(M, "DRONE_TIER", None),
                         "drone_domain": list(getattr(M, "DRONE_DOMAIN", []) or []),
                         "delta": dict(getattr(M, "DELTA", {}) or {}),
                         "env": getattr(M, "ENV_NAME", None)})
    data.setdefault("snaps", {}); data.setdefault("best", {})
    total = len(scns) * len(seeds)
    done = lambda: sum(len(v) for v in data["runs"].values())
    t0 = time.time()
    for sname, wmap in scns.items():
        M.weight_map = wmap
        data["runs"].setdefault(sname, {})
        if sname not in data["greedy"]:
            data["greedy"][sname] = run_greedy()
            pickle.dump(data, open(out, "wb"))
        for sd in seeds:
            if sd in data["runs"][sname]:
                continue
            if time.time() - t0 > max_wall:
                print(f"[時間到 {max_wall}s] 本批先停;已完成 {done()}/{total},請再次執行續跑。")
                return False
            t = time.time()
            rec = {}; snr = {}
            _p, _a, _x, _sn = run_cmp("three_tier", sd, pop, fes); rec["本方法"] = _a; snr["本方法"] = _sn
            if "本方法" not in data.setdefault("best", {}).setdefault(sname, {}):
                data["best"][sname]["本方法"] = {"F1": max(_p, key=lambda c: c["F1"]),
                                               "F2": min(_p, key=lambda c: c["F2"]),
                                               "F3": min(_p, key=lambda c: c["F3"])}
            rec["NSGA-III"], snr["NSGA-III"] = _ext_solve("NSGA-III", sd, pop, fes)
            rec["SMS-EMOA"], snr["SMS-EMOA"] = _ext_solve("SMS-EMOA", sd, pop, fes)
            rec["ES"], snr["ES"] = es_snap(sd, pop, fes)
            rec["隨機"], snr["隨機"] = rand_snap(sd, pop, fes)
            data["runs"][sname][sd] = rec
            data.setdefault("snaps", {}).setdefault(sname, {})[sd] = snr
            pickle.dump(data, open(out, "wb"))
            save_cache()
            print(f"  存檔 [{sname}] seed {sd}  ({time.time()-t:.0f}s)  進度 {done()}/{total}")
    finished = done() >= total
    print("== 全部完成 ==" if finished else f"== 本批結束,進度 {done()}/{total} ==")
    return finished


def aggregate(out="formal_results.pkl", fig="baseline_formal.png"):
    from scipy.stats import wilcoxon, friedmanchisquare
    data = pickle.load(open(out, "rb"))
    seeds = data["meta"]["seeds"]; scns = list(data["runs"].keys())
    pop = data["meta"]["pop"]; fes = data["meta"]["fes"]
    _samp = next(iter(next(iter(data["runs"].values())).values()))
    algos = [a for a in ALGOS if (a in _samp or a == "貪婪")]   # 依資料實際方法集(相容舊 4 方 pkl)
    _hm = data.get("meta", {})
    print(f"== aggregate provenance ==\n  env={_hm.get('env')} tier={_hm.get('drone_tier')} "
          f"domain={_hm.get('drone_domain')} delta={_hm.get('delta')}\n  "
          f"pop={_hm.get('pop')} fes={_hm.get('fes')} seeds={_hm.get('seeds')} algos={_hm.get('algos')}")
    comparators = [a for a in algos if a != "本方法"]
    metrics = {s: {a: {"HV": [], "IGD+": [], "N": []} for a in algos} for s in scns}
    cmet = {s: {a: [] for a in comparators} for s in scns}
    for s in scns:
        g = data["greedy"][s]
        for sd in seeds:
            if sd not in data["runs"][s]:
                continue
            rec = dict(data["runs"][s][sd]); rec["貪婪"] = g
            rec = {al: MM.nondominated(rec[al]) for al in algos}   # 清理為真正非支配前緣(公平比較)
            allarc = [a for al in algos for a in rec[al]]
            lo, hi = MM._bounds(allarc); ref = MM.nondominated(allarc)
            for al in algos:
                metrics[s][al]["HV"].append(MM.hv(rec[al], lo, hi))
                metrics[s][al]["IGD+"].append(MM.igd_plus(rec[al], ref, lo, hi))
                metrics[s][al]["N"].append(len(rec[al]))
            for al in comparators:
                cmet[s][al].append((MM.c_metric(rec["本方法"], rec[al]), MM.c_metric(rec[al], rec["本方法"])))

    print(f"\n========== 正式規模彙整(pop={pop}, FEs={fes}, seeds={len(seeds)}) ==========")
    for s in scns:
        print(f"\n=== {s} ===")
        for metric in ("HV", "IGD+"):
            line = f"{metric}: "
            for a in algos:
                m, sd = mstd(metrics[s][a][metric]); line += f"{a} {m:.3f}±{sd:.3f}  "
            print(line)
            try:
                st, p = friedmanchisquare(*[metrics[s][a][metric] for a in algos])
                print(f"    Friedman χ²={st:.2f}, p={p:.4g}")
            except Exception as e:
                print(f"    Friedman 無法計算({e})")
            for a in comparators:
                try:
                    st, p = wilcoxon(metrics[s]["本方法"][metric], metrics[s][a][metric])
                    print(f"    Wilcoxon 本方法 vs {a}: p={p:.4g}")
                except Exception as e:
                    print(f"    Wilcoxon 本方法 vs {a}: 無法計算({e})")
        for a in comparators:
            cab = mstd([x[0] for x in cmet[s][a]])[0]; cba = mstd([x[1] for x in cmet[s][a]])[0]
            print(f"  C(本方法,{a})={cab:.2f}  C({a},本方法)={cba:.2f}")
    # ---- 標準表格與圖(與真實環境 formal 對齊:tier/env-tagged CSV + C-metric 盒鬚 + 長條)----
    import csv as _csv
    _meta = data.get("meta", {})
    T = _meta.get("drone_tier") or getattr(M, "DRONE_TIER", "")
    EV = _meta.get("env") or getattr(M, "ENV_NAME", "")
    _lt = getattr(M, "DRONE_TIER", None); _le = getattr(M, "ENV_NAME", None)
    if _meta.get("drone_tier") and _lt and _meta["drone_tier"] != _lt:
        print(f"[warn] pkl drone_tier={_meta['drone_tier']} ≠ 目前 {_lt};標籤以 pkl meta 為準。")
    if _meta.get("env") and _le and _meta["env"] != _le:
        print(f"[warn] pkl env={_meta['env']} ≠ 目前 {_le};標籤以 pkl meta 為準。")
    _b = fig[:-4] if fig.endswith(".png") else fig
    csv_path = (_b.replace("baseline_formal", "baseline_compare") if "baseline_formal" in _b else _b + "_compare") + ".csv"
    cbox = (_b.replace("baseline_formal", "baseline_cmetric") if "baseline_formal" in _b else _b + "_cmetric") + ".png"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as _f:
        w = _csv.writer(_f)
        w.writerow(["Tier", "Env", "Scenario", "Method", "HV_mean", "HV_sd",
                    "IGD+_mean", "IGD+_sd", "N_mean", "C(本方法->M)", "C(M->本方法)"])
        for s in scns:
            for a in algos:
                hm, hs = mstd(metrics[s][a]["HV"]); im, is_ = mstd(metrics[s][a]["IGD+"]); nm = mstd(metrics[s][a]["N"])[0]
                if a in comparators:
                    cab = mstd([x[0] for x in cmet[s][a]])[0]; cba = mstd([x[1] for x in cmet[s][a]])[0]
                    cab, cba = f"{cab:.3f}", f"{cba:.3f}"
                else:
                    cab = cba = ""
                w.writerow([T, EV, s, a, f"{hm:.4f}", f"{hs:.4f}", f"{im:.4f}", f"{is_:.4f}", f"{nm:.1f}", cab, cba])
    # C-metric 盒鬚(本方法 vs 各法;左:本方法覆蓋對手、右:對手覆蓋本方法)
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, ttl in enumerate(("C(本方法, ·):本方法覆蓋對手", "C(·, 本方法):對手覆蓋本方法")):
        bd = [[x[k] for s in scns for x in cmet[s][a]] for a in comparators]
        ax2[k].boxplot(bd, showmeans=True)
        ax2[k].set_xticks(range(1, len(comparators) + 1)); ax2[k].set_xticklabels(comparators, rotation=20)
        ax2[k].set_ylim(-0.02, 1.02); ax2[k].set_title(ttl); ax2[k].grid(alpha=0.3, axis="y")
    fig2.suptitle(f"C-metric 盒鬚(本方法 vs 各法)" + (f"  [{T} tier]" if T else ""), fontweight="bold")
    fig2.tight_layout(); fig2.savefig(cbox, dpi=130); plt.close(fig2)
    plot_bars(metrics, scns, fig, algos=algos, title_suffix=f"({EV}, {T} tier)")
    print(f"\n四指標表:{csv_path}\nC-metric 盒鬚:{cbox}\nHV/IGD+ 長條:{fig}")
    # HV 收斂曲線(逐代快照;重用 experiment_eval,與 formal/mixed 同款)
    snaps_all = data.get("snaps", {})
    if snaps_all:
        try:
            import experiment_eval as EE
            scn_d = {}
            for s in scns:
                scn_d[s] = {}
                for sd in seeds:
                    for m, sn in snaps_all.get(s, {}).get(sd, {}).items():
                        if m in algos and sn:
                            scn_d[s].setdefault(m, {})[sd] = sn
            if any(scn_d[s] for s in scns):
                multi = {"scn": scn_d, "greedy": {s: data["greedy"][s] for s in scns},
                         "meta": {"fes": fes, "drone_tier": T}}
                hvp = (_b.replace("baseline_formal", "baseline_hv_convergence") if "baseline_formal" in _b else _b + "_hvconv") + ".png"
                EE.hv_convergence_multiseed(multi, algos, hvp)
                print(f"HV 收斂:{hvp}")
        except Exception as e:
            print(f"[warn] HV 收斂略過:{e}")
    # 最佳路徑拼圖(本方法 F1/F2/F3 最佳解;重用 experiment_eval,與 formal/mixed 同款)
    if data.get("best"):
        try:
            import experiment_eval as EE
            scns_w = make_scenarios()
            s0 = next((s for s in scns if data["best"].get(s, {}).get("本方法")), None)
            if s0:
                M.weight_map = scns_w[s0]
                best = data["best"][s0]["本方法"]
                ttl = (f"{EE.method_en('本方法')}: best F1 / F2 / F3 patrol routes ({EE.scen_en(s0)})"
                       + (f"  [{T} tier]" if T else ""))
                bp = (_b.replace("baseline_formal", "baseline_best_paths") if "baseline_formal" in _b else _b + "_bestpaths") + ".png"
                EE.best_path_montage(best, ttl, bp)
                print(f"最佳路徑:{bp}")
        except Exception as e:
            print(f"[warn] 最佳路徑略過:{e}")
    return metrics, cmet


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        prog="baselines.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="人工(合成)環境 baseline runner:本方法 / NSGA-III / SMS-EMOA / ES / 隨機 / 貪婪。",
        epilog="注意:本檔為『歷史 / 合成環境』baseline 下界軌;v2 三層『真實 / DDN / tier』正式 formal\n"
               "請改用 run_formal_resumable.py 或 run_baselines_alltiers.sh。\n\n"
               "範例:\n"
               "  python baselines.py formal taiwan 700          # 合成 taiwan,本批 700s 續跑\n"
               "  python baselines.py formal philippines 700 --seeds 10 --pop 100 --fes 3000\n"
               "  python baselines.py agg japan                  # 彙整(四項標準產物)\n"
               "  python baselines.py demo                       # 小型示範\n"
               "  python baselines.py prewarm taiwan 300         # 路由快取預熱")
    ap.add_argument("mode", choices=["prewarm", "formal", "agg", "aggregate", "demo"], help="執行模式")
    ap.add_argument("env", nargs="?", default="taiwan", choices=["taiwan", "japan", "philippines"],
                    help="合成環境(預設 taiwan;不支援真實/DDN 環境)")
    ap.add_argument("wall", nargs="?", type=int, default=240,
                    help="本批時間預算(秒;formal/prewarm 續跑用,預設 240)")
    ap.add_argument("--seeds", type=int, default=10, help="formal 種子數(預設 10)")
    ap.add_argument("--pop", type=int, default=100, help="族群大小(預設 100;隨機基準忽略)")
    ap.add_argument("--fes", type=int, default=3000, help="每法 FEs / 隨機取樣次數(預設 3000)")
    a = ap.parse_args()
    mode = "agg" if a.mode == "aggregate" else a.mode
    env, wall = a.env, a.wall
    M.set_environment(env)
    _T = M.DRONE_TIER
    out = f"formal_results_{_T}.pkl" if env == "taiwan" else f"formal_results_{env}_{_T}.pkl"
    fig = f"baseline_formal_{_T}.png" if env == "taiwan" else f"baseline_formal_{env}_{_T}.png"
    print(f"== 環境={env}:{M.NUM_BASES} 基地 / {M.NUM_VESSELS} 載具 / tier={_T} / 快取={_cache_path()} ==")
    if mode == "prewarm":
        prewarm(max_wall=wall)
    elif mode == "formal":
        run_formal(list(range(1, a.seeds + 1)), pop=a.pop, fes=a.fes, out=out, max_wall=wall)
    elif mode == "agg":
        aggregate(out=out, fig=fig)
    else:
        SEEDS = [1, 2, 3]; POP, FES = a.pop, min(a.fes, 1000)
        print(f"基準比較(demo):seeds={SEEDS}, pop={POP}, FEs={FES}")
        res, cm, scn_names = run_all(SEEDS, POP, FES)
        report(res, cm, scn_names)
        plot_bars(res, scn_names, fig.replace("formal", "bars"))
