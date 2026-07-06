# -*- coding: utf-8 -*-
"""
縮小規模的精確 Pareto 前緣(依層級 + 情境參數化):窮舉(三目標真值)+ MILP(PuLP/CBC)雙目標復現驗證。

目的:為 HV / IGD⁺ 提供「精確前緣」錨點。窮舉給出縮小實例的三目標真前緣(HV 真值);
      MILP 對可線性化的雙目標核心(F1 覆蓋 vs F2 成本)做 ε-constraint,精確復現窮舉之
      (F1,F2) 投影 → 證明 MILP 構式精確、為可放大之精確法。F3(平方協同)為二次 → MIQP。

層級參數化(v2.6+):DRONE[u]=M.radius_of(u)(覆蓋半徑;三層皆 {2,3,4,5})、UVAL[u]=u
      (F2 成本係數=無人機數,與主模型 F2=C_V·Σ u_v·L_v 一致)、CV=M.C_V。三層覆蓋幾何相同、
      差異進入 F2 成本 → 三層精確前緣分別為 LOWER/OPERATING/SAFETY 之 HV/IGD⁺ 錨點。

情境參數化(v2.8+):縮小實例亦比照主模型的「正交三層 + 等質量混合」情境設計:
      - 單層情境 ais / sar / cable:分別對應 AIS 航運密度(平滑場)、SAR 暗船(零散尖點)、
        海纜基礎設施(線狀走廊);
      - 混合情境 mixed:三層各除自身(合法海域)總質量(equal_mass)或各除峰值(equal_max)後相加,
        對應主模型 run_mixed_2opt_ab 的等質量混合權重情境。
      覆蓋表 COVER 僅依幾何(候選點/半徑/合法格),與情境無關;情境只改 F1 之權重圖 W_MAP。

需額外套件(僅 --milp 交叉驗證需要):pip install pulp(內含 CBC)。
用法:
      DRONE_TIER=SAFETY python milp_exact_front.py                       # 無位置參數 → 用環境變數層級(此例 SAFETY)× mixed
      python milp_exact_front.py SAFETY                                  # SAFETY × mixed
      python milp_exact_front.py all                                     # 三層 × mixed(亦可列出 LOWER OPERATING SAFETY)
      python milp_exact_front.py SAFETY --scenario ais                  # 指定單層情境
      python milp_exact_front.py all --scenario all                     # 三層 × {ais,sar,cable,mixed}
      python milp_exact_front.py SAFETY --scenario mixed --mix-mode equal_max
      python milp_exact_front.py SAFETY --scenario mixed --milp         # MILP(CBC)交叉驗證(ε 抽樣,可 --milp-eps N 調);未裝 pulp 會略過並提示
"""
import warnings; warnings.filterwarnings("ignore")
import itertools, math
import numpy as np
import MOGA_GPSIFF_patrol_clean as M

SCENARIOS = ("ais", "sar", "cable", "mixed")
MILP_EPS = 16   # --milp 之 ε 抽樣上限(完整 ε 在放大格點上過慢;抽樣足以驗證構式精確)

# 以下模組級全域由 configure() 依當前層級 + 情境建立
TIER = None; SCN = None; MIX_MODE = "equal_mass"
LEVELS = []; DRONE = {}; UVAL = {}; CV = 1.0
H = W = 0; BASE = (0, 0); CANDS = []; C = 0; N = 2; NOGO = set()
W_MAP = None; LEGAL = None; COVER = {}


def disk_cells(center, r):
    cx, cy = center; out = []
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r and LEGAL[y, x]:
                out.append((x, y))
    return out


def _gauss_layer(centers, sig):
    L = np.zeros((H, W))
    for (cx, cy, amp) in centers:
        for y in range(H):
            for x in range(W):
                L[y, x] += amp * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))
    return L


def _line_layer(p0, p1, halfwidth):
    """沿線段 p0→p1 之走廊權重(對應海纜基礎設施)。"""
    (x0, y0), (x1, y1) = p0, p1
    L = np.zeros((H, W)); dx, dy = x1 - x0, y1 - y0; seg2 = dx * dx + dy * dy or 1.0
    for y in range(H):
        for x in range(W):
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg2))
            px, py = x0 + t * dx, y0 + t * dy
            dist2 = (x - px) ** 2 + (y - py) ** 2
            L[y, x] = math.exp(-dist2 / (2 * halfwidth ** 2))
    return L


def _build_layers():
    """三正交層(座標依格點比例縮放):AIS 平滑場 / SAR 零散尖點 / 海纜線狀走廊。"""
    fx = lambda a: int(round(a * (W - 1)))
    ais = _gauss_layer([(fx(0.30), fx(0.62), 0.9), (fx(0.66), fx(0.55), 0.8)], sig=max(2.5, 0.16 * W))
    sar = _gauss_layer([(fx(0.20), fx(0.80), 1.0), (fx(0.82), fx(0.80), 1.0),
                        (fx(0.52), fx(0.26), 1.0), (fx(0.86), fx(0.40), 0.9)], sig=max(1.2, 0.045 * W))
    cable = _line_layer((fx(0.10), fx(0.22)), (fx(0.90), fx(0.72)), halfwidth=max(1.0, 0.035 * W))
    return {"ais": np.clip(ais, 0, None), "sar": np.clip(sar, 0, None), "cable": np.clip(cable, 0, None)}


def _scenario_map(layers, scenario, mix_mode):
    """產生情境權重圖;單層直接取該層,mixed 依 equal_mass/equal_max 合併三層。皆正規化至峰值=1。"""
    legal = LEGAL
    if scenario in layers:
        Wm = layers[scenario].copy()
    elif scenario == "mixed":
        Wm = np.zeros((H, W))
        for L in layers.values():
            denom = (L.max() if mix_mode == "equal_max" else float(L[legal].sum())) or 1.0
            Wm += L / denom
    else:
        raise ValueError("未知情境 %r;可用:%s" % (scenario, SCENARIOS))
    Wm[~legal] = 0.0
    mx = Wm.max() or 1.0
    return Wm / mx


def configure(tier=None, scenario="mixed", mix_mode="equal_mass"):
    """依當前(或指定)層級 + 情境重建縮小實例的所有全域。"""
    global TIER, SCN, MIX_MODE, LEVELS, DRONE, UVAL, CV
    global H, W, BASE, CANDS, C, N, NOGO, W_MAP, LEGAL, COVER
    if tier:
        M.set_drone_tier(tier)
    TIER = M.DRONE_TIER; SCN = scenario; MIX_MODE = mix_mode
    LEVELS = sorted(M.DRONE_DOMAIN)
    DRONE = {u: M.radius_of(u) for u in LEVELS}      # 級別鍵=無人機數 u → 覆蓋半徑(三層皆 {2,3,4,5})
    UVAL = {u: float(u) for u in LEVELS}             # F2 成本係數 = 無人機數 u
    CV = float(getattr(M, "C_V", 1.0))
    rmax = max(DRONE.values())
    H = W = max(20, rmax * 5)                         # 格點隨最大半徑放大,避免覆蓋飽和
    N = 2
    BASE = (W // 2, 0)
    fr = [(0.18, 0.35), (0.35, 0.72), (0.60, 0.85), (0.82, 0.60), (0.75, 0.25), (0.25, 0.78)]
    CANDS = [(int(round(a * (W - 1))), int(round(b * (H - 1)))) for (a, b) in fr]
    C = len(CANDS)
    NOGO = {(W // 2, H // 2), (W // 2 + 1, H // 2), (W // 2, H // 2 + 1)}
    LEGAL = np.ones((H, W), dtype=bool)
    for (x, y) in NOGO:
        if 0 <= y < H and 0 <= x < W:
            LEGAL[y, x] = False
    W_MAP = _scenario_map(_build_layers(), scenario, mix_mode)
    COVER = {}                                        # 僅依幾何,與情境無關
    for idx, pt in enumerate(CANDS):
        for lv, r in DRONE.items():
            COVER[("n", idx, lv)] = set(disk_cells(pt, r))
    for lv, r in DRONE.items():
        COVER[("base", lv)] = set(disk_cells(BASE, r))
    return TIER, SCN


def tour_len(sel):
    best = math.inf
    for perm in itertools.permutations(sel):
        seq = [BASE] + [CANDS[i] for i in perm] + [BASE]
        L = sum(math.hypot(seq[a + 1][0] - seq[a][0], seq[a + 1][1] - seq[a][1])
                for a in range(len(seq) - 1))
        best = min(best, L)
    return best


def objectives(selP, lvP, selQ, lvQ):
    covered = set(COVER[("base", lvP)]) | set(COVER[("base", lvQ)])
    for i in selP:
        covered |= COVER[("n", i, lvP)]
    for i in selQ:
        covered |= COVER[("n", i, lvQ)]
    F1 = float(sum(W_MAP[y, x] for (x, y) in covered))
    F2 = CV * (UVAL[lvP] * tour_len(selP) + UVAL[lvQ] * tour_len(selQ))
    P = np.array([CANDS[i] for i in selP], float); Q = np.array([CANDS[i] for i in selQ], float)
    cP, cQ = P.mean(0), Q.mean(0)
    varP = float(np.mean(np.sum((P - cP) ** 2, axis=1)))
    varQ = float(np.mean(np.sum((Q - cQ) ** 2, axis=1)))
    F3 = float(np.sum((cP - cQ) ** 2)) + varP + varQ
    return F1, F2, F3


def dominates(a, b):
    return (a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]) and \
           (a[0] > b[0] or a[1] < b[1] or a[2] < b[2])


def nondominated(pts):
    out = [p for p in pts if not any(q != p and dominates(q, p) for q in pts)]
    return sorted(set(out))


def enumerate_front():
    all_obj = []
    for selP in itertools.combinations(range(C), N):
        restP = [i for i in range(C) if i not in selP]
        for selQ in itertools.combinations(restP, N):
            for lvP in DRONE:
                for lvQ in DRONE:
                    all_obj.append(objectives(selP, lvP, selQ, lvQ))
    return nondominated(all_obj), len(all_obj)


def solve_milp_biobj_front():
    """精確雙目標(F1 max / F2 min)前緣:對 F2 做 ε-constraint;與窮舉之 (F1,F2) 投影比對。"""
    import pulp
    cov = set()
    for v in COVER.values():
        cov |= v
    cells = sorted(cov)                              # 僅可覆蓋格(非覆蓋格 y≡0,精確不變、大幅縮模)

    def build(eps2):
        m = pulp.LpProblem("patrol_biobj", pulp.LpMaximize)
        x = {(v, i): pulp.LpVariable(f"x_{v}_{i}", cat="Binary") for v in (0, 1) for i in range(C)}
        d = {(v, l): pulp.LpVariable(f"d_{v}_{l}", cat="Binary") for v in (0, 1) for l in DRONE}
        y = {a: pulp.LpVariable(f"y_{a[0]}_{a[1]}", cat="Binary") for a in cells}
        for v in (0, 1):
            m += pulp.lpSum(x[(v, i)] for i in range(C)) == N
            m += pulp.lpSum(d[(v, l)] for l in DRONE) == 1
        for i in range(C):
            m += x[(0, i)] + x[(1, i)] <= 1
        w = {}
        for v in (0, 1):
            for i in range(C):
                for l in DRONE:
                    wv = pulp.LpVariable(f"w_{v}_{i}_{l}", cat="Binary")
                    m += wv <= x[(v, i)]; m += wv <= d[(v, l)]; m += wv >= x[(v, i)] + d[(v, l)] - 1
                    w[(v, i, l)] = wv
        for a in cells:
            covers = []
            for v in (0, 1):
                for l in DRONE:
                    if a in COVER[("base", l)]:
                        covers.append(d[(v, l)])
                    for i in range(C):
                        if a in COVER[("n", i, l)]:
                            covers.append(w[(v, i, l)])
            m += (y[a] <= pulp.lpSum(covers)) if covers else (y[a] == 0)
        F1 = pulp.lpSum(W_MAP[a[1], a[0]] * y[a] for a in cells)
        F2_terms = []
        for v in (0, 1):
            for (i, j) in itertools.combinations(range(C), 2):
                pij = pulp.LpVariable(f"p_{v}_{i}_{j}", cat="Binary")
                m += pij <= x[(v, i)]; m += pij <= x[(v, j)]; m += pij >= x[(v, i)] + x[(v, j)] - 1
                Lij = tour_len((i, j))
                for l in DRONE:
                    t = pulp.LpVariable(f"t_{v}_{i}_{j}_{l}", cat="Binary")
                    m += t <= pij; m += t <= d[(v, l)]; m += t >= pij + d[(v, l)] - 1
                    F2_terms.append(CV * UVAL[l] * Lij * t)
        F2 = pulp.lpSum(F2_terms)
        m += F1
        m += (F2 <= eps2 + 1e-6)
        return m, F1, F2

    truth, _ = enumerate_front()
    f2_grid = sorted(set(round(p[1], 6) for p in truth))
    if len(f2_grid) > MILP_EPS:                      # 抽樣 ε(均勻取點)
        ix = np.linspace(0, len(f2_grid) - 1, MILP_EPS).round().astype(int)
        f2_grid = [f2_grid[i] for i in sorted(set(int(k) for k in ix))]
    got = []
    for e2 in f2_grid:
        m, F1, F2 = build(e2)
        m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=15))
        if pulp.LpStatus[m.status] == "Optimal":
            got.append((round(pulp.value(F1), 6), round(pulp.value(F2), 6)))
    nd = [p for p in got if not any(q != p and q[0] >= p[0] and q[1] <= p[1]
                                    and (q[0] > p[0] or q[1] < p[1]) for q in got)]
    return sorted(set(nd))


def hv_of_front(front, margin=1.1):
    from compare_sel import _hv3d
    f1 = [p[0] for p in front]; f2 = [p[1] for p in front]; f3 = [p[2] for p in front]
    lo = (min(f1), min(f2), min(f3)); hi = (max(f1), max(f2), max(f3))
    rng_ = [(hi[d] - lo[d]) or 1.0 for d in range(3)]
    pts = [((hi[0] - p[0]) / rng_[0], (p[1] - lo[1]) / rng_[1], (p[2] - lo[2]) / rng_[2]) for p in front]
    return _hv3d(pts, (margin, margin, margin)), lo, hi


# 匯入時先以當前環境變數層級 + 預設 mixed 情境建立一份實例
configure()


def run_case(tier, scenario, mix_mode="equal_mass", do_milp=False, outdir="milp_out"):
    """跑單一(層級 × 情境):窮舉精確前緣(+選配 MILP),存 tier×scn 標記之 CSV/PNG,回傳摘要。"""
    import time, csv, os
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure(tier, scenario, mix_mode)
    tag = f"{TIER}_{SCN}" + (f"-{mix_mode}" if SCN == "mixed" else "")
    t = time.time(); truth, total = enumerate_front()
    hv, lo, hi = hv_of_front(truth)
    print(f"[{TIER} · {SCN}{'/'+mix_mode if SCN=='mixed' else ''}] "
          f"DRONE_DOMAIN={LEVELS} CV={CV:g}  窮舉 {total} 解 → 前緣 {len(truth)} 點"
          f"({time.time()-t:.2f}s)  HV={hv:.4f}  F1∈[{lo[0]:.2f},{hi[0]:.2f}] F2∈[{lo[1]:.1f},{hi[1]:.1f}]")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"exact_front_{tag}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f); wr.writerow(["Tier", "Scenario", "F1", "F2", "F3"])
        for p in truth:
            wr.writerow([TIER, SCN] + [f"{v:.6f}" for v in p])
    milp = None
    if do_milp:
        import importlib.util
        if importlib.util.find_spec("pulp") is None:
            print("    [SKIP] --milp 需要 pulp;請執行 pip install -r requirements-dev.txt(或 pip install pulp)")
        else:
            tm = time.time(); milp = solve_milp_biobj_front()
            proj = [(p[0], p[1]) for p in truth]
            proj_nd = sorted(set(p for p in proj if not any(
                q != p and q[0] >= p[0] and q[1] <= p[1] and (q[0] > p[0] or q[1] < p[1]) for q in proj)))
            R = lambda S: set((round(a, 3), round(b, 3)) for (a, b) in S)
            sub = R(milp) <= R(proj_nd)
            print(f"    MILP(CBC) 抽樣 {len(milp)} 點({time.time()-tm:.1f}s)  "
                  f"全部落在窮舉投影前緣? {sub}  (窮舉投影 {len(proj_nd)} 點)")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    F1 = [p[0] for p in truth]; F2 = [p[1] for p in truth]; F3 = [p[2] for p in truth]
    ax[0].scatter(F1, F2, s=18, facecolors="none", edgecolors="k", label="窮舉精確前緣")
    if milp:
        ax[0].scatter([p[0] for p in milp], [p[1] for p in milp], s=60, marker="x", c="k", label="MILP(CBC)")
        ax[0].legend(fontsize=8)
    ax[0].set_xlabel("F1 覆蓋 (↑)"); ax[0].set_ylabel("F2 成本 (↓)"); ax[0].set_title("F1–F2")
    ax[1].scatter(F1, F3, s=18, facecolors="none", edgecolors="k"); ax[1].set_xlabel("F1 (↑)"); ax[1].set_ylabel("F3 (↓)"); ax[1].set_title("F1–F3")
    ax[2].scatter(F2, F3, s=18, facecolors="none", edgecolors="k"); ax[2].set_xlabel("F2 (↓)"); ax[2].set_ylabel("F3 (↓)"); ax[2].set_title("F2–F3")
    fig.suptitle(f"縮小實例精確 Pareto 前緣  [{TIER} tier · {SCN}{'/'+mix_mode if SCN=='mixed' else ''}]({len(truth)} 點)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"exact_front_{tag}.png"), dpi=130, bbox_inches="tight")
    return {"tier": TIER, "scenario": SCN, "n_feasible": total, "n_front": len(truth), "hv": hv, "lo": lo, "hi": hi}


def _build_argparser():
    import argparse
    ap = argparse.ArgumentParser(
        prog="milp_exact_front.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="縮小實例之精確 Pareto 前緣錨點(依層級 × 情境參數化)",
        epilog="範例:\n"
               "  DRONE_TIER=SAFETY python milp_exact_front.py            # 無位置參數 → 環境變數層級 × mixed\n"
               "  python milp_exact_front.py SAFETY                       # SAFETY × mixed\n"
               "  python milp_exact_front.py all                          # 三層 × mixed\n"
               "  python milp_exact_front.py all --scenario all           # 三層 × {ais,sar,cable,mixed}\n"
               "  python milp_exact_front.py SAFETY --scenario mixed --mix-mode equal_max\n"
               "  python milp_exact_front.py SAFETY --milp                # MILP(CBC)交叉驗證(需 pulp;每案約 60–90s)\n"
               "  python milp_exact_front.py SAFETY --milp --milp-eps 12  # 調 ε 抽樣上限(越小越快)")
    ap.add_argument("tiers", nargs="*", default=[],
                    help="層級 LOWER/OPERATING/SAFETY 或 all;省略則用環境變數 DRONE_TIER")
    ap.add_argument("--scenario", default="mixed", help="ais|sar|cable|mixed|all(預設 mixed)")
    ap.add_argument("--mix-mode", dest="mix_mode", choices=["equal_mass", "equal_max"],
                    default="equal_mass", help="mixed 情境之合併方式")
    ap.add_argument("--milp", action="store_true", help="另做 MILP(CBC)交叉驗證(需 pulp)")
    ap.add_argument("--milp-eps", dest="milp_eps", type=int, default=MILP_EPS,
                    help="MILP ε 抽樣上限")
    return ap


def main(argv=None):
    global MILP_EPS
    ap = _build_argparser()
    a = ap.parse_args(argv)
    MILP_EPS = a.milp_eps
    toks = [t.upper() for t in a.tiers]
    if any(t == "ALL" for t in toks):
        want_tiers = ["LOWER", "OPERATING", "SAFETY"]
    elif toks:
        bad = [t for t in toks if t not in M._DRONE_TIERS]
        if bad:
            ap.error("未知層級 %s;可用 %s 或 all" % (bad, list(M._DRONE_TIERS)))
        want_tiers = toks
    else:
        want_tiers = [M.DRONE_TIER]
    scn = a.scenario.lower()
    if scn == "all":
        want_scn = list(SCENARIOS)
    elif scn in SCENARIOS:
        want_scn = [scn]
    else:
        ap.error("未知情境 %r;可用 %s 或 all" % (scn, list(SCENARIOS)))
    if a.milp:
        ncase = len(want_tiers) * len(want_scn)
        print(f"[注意] --milp 每個(層級×情境)約需 ~60–90s(ε 抽樣 ≤{MILP_EPS} + CBC,timeLimit 15s/解);"
              f"本次 {ncase} 案,預估約 {ncase}–{ncase*2} 分鐘。窮舉前緣本身為秒級,MILP 僅作雙目標投影交叉驗證。")
    print(f"=== 縮小實例精確前緣錨點:層級 {want_tiers} × 情境 {want_scn}"
          f"{'(混合用 '+a.mix_mode+')' if 'mixed' in want_scn else ''}{'(含 MILP)' if a.milp else ''} ===")
    rows = [run_case(t, s, a.mix_mode, a.milp) for t in want_tiers for s in want_scn]
    print("\n=== 摘要(同覆蓋幾何;F2 隨層級放大、F1 隨情境改變)===")
    print(f"{'Tier':<10}{'情境':<8}{'可行解':>7}{'前緣點':>7}{'HV':>9}{'F1範圍':>18}{'F2範圍':>20}")
    for r in rows:
        print(f"{r['tier']:<10}{r['scenario']:<8}{r['n_feasible']:>7}{r['n_front']:>7}{r['hv']:>9.4f}"
              f"{('['+format(r['lo'][0],'.1f')+','+format(r['hi'][0],'.1f')+']'):>18}"
              f"{('['+format(r['lo'][1],'.0f')+','+format(r['hi'][1],'.0f')+']'):>20}")
    print("精確前緣輸出:milp_out/exact_front_<TIER>_<SCN>[-mixmode].csv / .png")
    return rows


if __name__ == "__main__":
    main()
