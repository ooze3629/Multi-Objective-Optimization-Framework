# -*- coding: utf-8 -*-
"""
標準評估模組:每個比對實驗皆產生四項標準產物(每張圖同時輸出「彩色」與「黑白友善 _bw」版)
  (1) 四指標比對表  HV↑ / IGD⁺↓ / 前緣延展(extent)↑ / 非支配解數
  (2) Hypervolume 收斂曲線(各方法)
  (3) C-metric 盒鬚圖(本方法對各基準之覆蓋率,多種子分布)
  (4) F1 / F2 / F3 最佳解之路徑規劃圖

圖面標籤一律英文(情境名、方法名、軸標、標題);情境/方法之 results 字典「鍵」維持原樣
(僅顯示時翻譯)。黑白版以「灰階 + 線型 + 標記 + 填充紋路」四重區分,適合單色列印。

設計原則(公平性):各方法之解先過濾為「真正非支配前緣」;HV / IGD⁺ 的正規化
與 IGD⁺ 參考前緣皆取「全方法解集合之聯集」;HV 參考點 (1.1,1.1,1.1)。

輸入資料格式(見 EVAL_PROTOCOL.md):
  results = {
    "meta": {"pop":.., "fes":.., "seeds":[...], "proposed":"本方法"},
    "greedy": {scn: [archive dicts]},
    "runs":  {scn: {seed: {method: [archive dicts]}}},
    "snaps": {scn: {method: [(fes,[(F1,F2,F3)..])..]}},
    "best":  {scn: {method: {"F1":chromo,"F2":chromo,"F3":chromo}}},
  }
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, pickle, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

import MOGA_GPSIFF_patrol_clean as M
import moea_metrics as MM
import baselines as B
from compare_sel import run_cmp

OBJS = ("F1", "F2", "F3")

# ============ 配色/樣式(彩色 + 黑白友善)============
# 每方法四重區分:顏色/灰階 + 線型 + 標記 + 填充紋路。索引依 methods 之順序。
_COLORS  = ["#1F4E79", "#C0392B", "#2E9E5B", "#E08A1E", "#7D3C98", "#555555"]
_GRAYS   = ["#000000", "#7A7A7A", "#B5B5B5", "#4D4D4D", "#9A9A9A", "#2B2B2B"]
_BARFILL = ["#FFFFFF", "#BFBFBF", "#7F7F7F", "#3F3F3F", "#E5E5E5", "#1F1F1F"]  # 黑白長條漸層
_LS      = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
_MK      = ["o", "s", "^", "D", "v", "P"]
_HATCH   = ["", "//", "..", "xx", "\\\\", "++"]

def _idx(methods, m):
    try: return list(methods).index(m)
    except ValueError: return len(list(methods))

def _style(methods, m, bw):
    i = _idx(methods, m) % len(_COLORS)
    return {"line": (_GRAYS if bw else _COLORS)[i],
            "fill": (_BARFILL[i] if bw else _COLORS[i]),
            "ls": _LS[i % len(_LS)], "marker": _MK[i % len(_MK)], "hatch": _HATCH[i % len(_HATCH)]}

def _bw_path(save):
    root, ext = os.path.splitext(save)
    return root + "_bw" + ext

# ---- 中文情境名/方法名 → 英文(僅顯示;不動 results 鍵)----
_SCEN_REPL = [
    ("TWreal", "TW"), ("JPreal", "JP"), ("PHreal", "PH"),
    ("航運密度(AIS)", "Shipping Density (AIS)"),
    ("暗船查緝(SAR未匹配)", "Dark-Vessel (SAR, unmatched)"),
    ("暗船查緝(SAR)", "Dark-Vessel (SAR)"),
    ("海纜基礎設施", "Submarine Cable"),
    ("等質量混合(AIS+SAR+海纜)", "Equal-Mass Mix (AIS+SAR+Cable)"),
    ("等峰值混合(AIS+SAR+海纜)", "Equal-Peak Mix (AIS+SAR+Cable)"),
    ("近岸熱區", "Nearshore Hotspot"),
    ("外海熱區", "Offshore Hotspot"),
    ("近岸航道", "Nearshore Lane"),
    ("巴拉望與南部海域", "Palawan & Southern Waters"),
    ("東側外海巡防", "Eastern Offshore Patrol"),
    ("外海熱區巡防", "Offshore Patrol"),
]
def scen_en(s):
    out = str(s)
    for a, b in _SCEN_REPL:
        out = out.replace(a, b)
    return out.strip()

_METHOD_EN = {"本方法-ss": "Proposed-MOGA", "本方法-ss-2opt": "Proposed-MOGA (2-opt)", "本方法": "Proposed",
              "ES": "ES", "隨機": "Random", "貪婪": "Greedy"}
def method_en(m): return _METHOD_EN.get(m, m)

def _ms(x): return (st.mean(x), st.pstdev(x) if len(x) > 1 else 0.0)
def _extent(arc, lo, hi):
    if not arc: return 0.0
    return sum((max(a[o] for a in arc) - min(a[o] for a in arc)) / ((hi[o] - lo[o]) or 1.0) for o in OBJS) / 3.0

def _tier_tag(results):
    """由 results meta 取無人機層級,接到圖/表標題(舊結果無此欄則回空字串)。"""
    t = (results.get("meta") or {}).get("drone_tier") or getattr(M, "DRONE_TIER", None)
    return f"  [{t} tier]" if t else ""


def _tier_str(results):
    return (results.get("meta") or {}).get("drone_tier") or getattr(M, "DRONE_TIER", "") or ""


def _csv_with_tier(rows, results):
    """寫 CSV 時於首欄加入 Tier(PNG 表標題已含層級,故僅作用於 CSV)。"""
    t = _tier_str(results)
    return [["Tier"] + rows[0]] + [[t] + r for r in rows[1:]]


def _per_seed_clean(runs, greedy, scn, seed, methods):
    rec = dict(runs[scn][seed])
    if greedy and "貪婪" in methods and "貪婪" not in rec:
        rec["貪婪"] = greedy[scn]
    rec = {m: MM.nondominated(rec[m]) for m in methods if m in rec}
    allarc = [a for m in rec for a in rec[m]]
    lo, hi = MM._bounds(allarc); ref = MM.nondominated(allarc)
    return rec, lo, hi, ref

# ---------- (1) 四指標比對表 ----------
def four_metric_table(results, methods, save_csv=None):
    runs, greedy = results["runs"], results.get("greedy", {})
    seeds = results["meta"]["seeds"]; scns = list(runs.keys())
    agg = {s: {m: {"HV": [], "IGD+": [], "extent": [], "n": []} for m in methods} for s in scns}
    for s in scns:
        for sd in seeds:
            if sd not in runs[s]: continue
            rec, lo, hi, ref = _per_seed_clean(runs, greedy, s, sd, methods)
            for m in rec:
                agg[s][m]["HV"].append(MM.hv(rec[m], lo, hi))
                agg[s][m]["IGD+"].append(MM.igd_plus(rec[m], ref, lo, hi))
                agg[s][m]["extent"].append(_extent(rec[m], lo, hi))
                agg[s][m]["n"].append(len(rec[m]))
    rows = [["Scenario", "Method", "HV(up)", "IGD+(down)", "Extent(up)", "N"]]
    print("=== 四指標比對表(平均±標準差) ===")
    for s in scns:
        print(f"[{s}]")
        for m in methods:
            if not agg[s][m]["HV"]: continue
            hv, ig, ex, n = _ms(agg[s][m]["HV"]), _ms(agg[s][m]["IGD+"]), _ms(agg[s][m]["extent"]), _ms(agg[s][m]["n"])
            print(f"  {m:4s} HV {hv[0]:.3f}±{hv[1]:.3f}  IGD+ {ig[0]:.3f}±{ig[1]:.3f}  延展 {ex[0]:.3f}  解數 {n[0]:.0f}")
            rows.append([scen_en(s), method_en(m), f"{hv[0]:.3f}±{hv[1]:.3f}",
                         f"{ig[0]:.3f}±{ig[1]:.3f}", f"{ex[0]:.3f}±{ex[1]:.3f}", f"{n[0]:.0f}±{n[1]:.0f}"])
    if save_csv:
        import csv
        with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(_csv_with_tier(rows, results))
        print("表格存檔:", save_csv)
    return rows

# ---------- (2) HV 收斂曲線 ----------
def _emit(save, build):
    """以 color / bw 兩種模式各畫一次,輸出 save 與 _bw 版。"""
    for bw in (False, True):
        fig = build(bw)
        out = save if not bw else _bw_path(save)
        fig.savefig(out, dpi=130); plt.close(fig)
        print(("彩色" if not bw else "黑白") + "圖存檔:", out)

def hv_convergence(results, methods, save):
    snaps = results.get("snaps"); runs = results["runs"]; greedy = results.get("greedy", {})
    scns = list(runs.keys()); rep = results["meta"]["seeds"][0]
    def build(bw):
        fig, axes = plt.subplots(1, len(scns), figsize=(5.3 * len(scns), 4.7))
        if len(scns) == 1: axes = [axes]
        for ax, s in zip(axes, scns):
            rec, lo, hi, _ = _per_seed_clean(runs, greedy, s, rep, methods)
            for m in [x for x in methods if x != "貪婪"]:
                sn = snaps[s][m] if snaps and s in snaps and m in snaps[s] else None
                if sn is None: continue
                xs = [fe for fe, _ in sn]
                ys = [MM.hv([{"F1": a[0], "F2": a[1], "F3": a[2]} for a in pts], lo, hi) for _, pts in sn]
                stl = _style(methods, m, bw)
                ax.plot(xs, ys, color=stl["line"], ls=stl["ls"], marker=stl["marker"], ms=3.5,
                        markevery=3, lw=1.6, label=method_en(m))
            if "貪婪" in rec:
                stl = _style(methods, "貪婪", bw)
                ax.axhline(MM.hv(rec["貪婪"], lo, hi), color=stl["line"], ls=stl["ls"], lw=1.6, label="Greedy (constructive)")
            ax.set_title(scen_en(s), fontsize=12); ax.set_xlabel("Function Evaluations (FEs)"); ax.set_ylabel("Hypervolume")
            ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
        fig.suptitle("Hypervolume Convergence (per method)" + _tier_tag(results), fontweight="bold", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); return fig
    _emit(save, build)

def hv_convergence_multiseed(multi, methods, save):
    scn_d = multi["scn"]; greedy = multi.get("greedy", {})
    fes = multi["meta"]["fes"]; scns = list(scn_d.keys())
    grid = np.linspace(fes / 30.0, fes, 30)
    def build(bw):
        fig, axes = plt.subplots(1, len(scns), figsize=(5.3 * len(scns), 4.7))
        if len(scns) == 1: axes = [axes]
        for ax, s in zip(axes, scns):
            finals = []
            for m in scn_d[s]:
                for sd, snaps in scn_d[s][m].items():
                    finals += [{"F1": a[0], "F2": a[1], "F3": a[2]} for a in snaps[-1][1]]
            if s in greedy: finals += greedy[s]
            lo, hi = MM._bounds(finals)
            for m in [x for x in methods if x != "貪婪" and x in scn_d[s]]:
                curves = []
                for sd, snaps in scn_d[s][m].items():
                    xs = [fe for fe, _ in snaps]
                    ys = [MM.hv([{"F1": a[0], "F2": a[1], "F3": a[2]} for a in pts], lo, hi) for _, pts in snaps]
                    curves.append(np.interp(grid, xs, ys))
                arr = np.array(curves); mean = arr.mean(0); sd_ = arr.std(0)
                stl = _style(methods, m, bw)
                ax.plot(grid, mean, color=stl["line"], ls=stl["ls"], marker=stl["marker"], ms=3.5,
                        markevery=3, lw=1.8, label=f"{method_en(m)} (n={arr.shape[0]})")
                ax.fill_between(grid, mean - sd_, mean + sd_, color=stl["line"], alpha=0.13)
            if s in greedy:
                stl = _style(methods, "貪婪", bw)
                ax.axhline(MM.hv(greedy[s], lo, hi), color=stl["line"], ls=stl["ls"], lw=1.6, label="Greedy (constructive)")
            ax.set_title(scen_en(s), fontsize=12); ax.set_xlabel("Function Evaluations (FEs)"); ax.set_ylabel("Hypervolume")
            ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
        fig.suptitle("Hypervolume Convergence: mean ± SD over seeds" + _tier_tag(multi), fontweight="bold", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); return fig
    _emit(save, build)

# ---------- (3) C-metric 盒鬚圖 ----------
def cmetric_boxplot(results, methods, save, proposed="本方法"):
    runs, greedy = results["runs"], results.get("greedy", {})
    seeds = results["meta"]["seeds"]; scns = list(runs.keys())
    others = [m for m in methods if m != proposed]
    box = {s: {m: [] for m in others} for s in scns}
    for s in scns:
        for sd in seeds:
            if sd not in runs[s]: continue
            rec, *_ = _per_seed_clean(runs, greedy, s, sd, methods)
            for m in others:
                if m in rec: box[s][m].append(MM.c_metric(rec[proposed], rec[m]))
    def build(bw):
        fig, axes = plt.subplots(1, len(scns), figsize=(4.7 * len(scns), 4.6))
        if len(scns) == 1: axes = [axes]
        for ax, s in zip(axes, scns):
            data = [box[s][m] for m in others]
            _labs = [f"vs {method_en(m)}" for m in others]
            _data = [d if len(d) else [0.0] for d in data]   # 空樣本以 [0.0] 佔位,避免 boxplot 對空陣列崩潰
            try:
                bp = ax.boxplot(_data, tick_labels=_labs, patch_artist=True,   # mpl>=3.9
                                medianprops=dict(color=("black" if not bw else "red")))
            except TypeError:
                bp = ax.boxplot(_data, labels=_labs, patch_artist=True,        # mpl<3.9 回退
                                medianprops=dict(color=("black" if not bw else "red")))
            for patch, m in zip(bp["boxes"], others):
                stl = _style(methods, m, bw)
                patch.set_facecolor(stl["fill"]); patch.set_alpha(0.55 if not bw else 1.0)
                patch.set_hatch(stl["hatch"]); patch.set_edgecolor("black")
            ax.set_title(scen_en(s), fontsize=12); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3, axis="y")
            ax.set_ylabel(f"C({method_en(proposed)}, .) — higher = more dominated")
        fig.suptitle(f"Coverage C-metric of {method_en(proposed)} vs. baselines (over seeds)" + _tier_tag(results),
                     fontweight="bold", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); return fig
    _emit(save, build)

# ---------- (3b) HV / IGD⁺ 長條圖 ----------
def metric_bars(results, methods, save):
    runs, greedy = results["runs"], results.get("greedy", {})
    seeds = results["meta"]["seeds"]; scns = list(runs.keys())
    agg = {s: {m: {"HV": [], "IGD+": []} for m in methods} for s in scns}
    for s in scns:
        for sd in seeds:
            if sd not in runs[s]: continue
            rec, lo, hi, ref = _per_seed_clean(runs, greedy, s, sd, methods)
            for m in rec:
                agg[s][m]["HV"].append(MM.hv(rec[m], lo, hi))
                agg[s][m]["IGD+"].append(MM.igd_plus(rec[m], ref, lo, hi))
    x = np.arange(len(scns)); w = 0.8 / max(1, len(methods))
    labels_en = [scen_en(s) for s in scns]
    def build(bw):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for j, metric in enumerate(("HV", "IGD+")):
            for i, m in enumerate(methods):
                means = [np.mean(agg[s][m][metric]) if agg[s][m][metric] else 0 for s in scns]
                errs = [np.std(agg[s][m][metric]) if agg[s][m][metric] else 0 for s in scns]
                stl = _style(methods, m, bw)
                axes[j].bar(x + (i - (len(methods) - 1) / 2) * w, means, w, yerr=errs, capsize=3,
                            label=method_en(m), color=stl["fill"], alpha=(0.85 if not bw else 1.0),
                            hatch=stl["hatch"], edgecolor="black", linewidth=0.6)
            axes[j].set_xticks(x); axes[j].set_xticklabels(labels_en, fontsize=9.5, rotation=8)
            axes[j].set_title("Hypervolume (higher is better)" if metric == "HV" else "IGD+ (lower is better)")
            axes[j].grid(alpha=0.3, axis="y")
        axes[0].legend(fontsize=9, ncol=2)
        fig.suptitle("Proposed-MOGA vs. Baseline Algorithms" + _tier_tag(results), fontweight="bold")
        fig.tight_layout(); return fig
    _emit(save, build)

# ---------- (4) F1/F2/F3 最佳解路徑圖 ----------
def _restricted_mask(env):
    """該環境『海上禁航區』獨立遮罩(bool,與 no_go 同形);合成環境/無禁航區回 None。
    來源:taiwan_real* → no_go_TWreal_overlay.npy(射擊/離岸風場/彰化);
          japan_real* / philippines_real* → build_real_maps.RESTRICTED 多邊形(MoD 危險區 / Tubbataha 核心)。"""
    try:
        H, W = M.no_go_zone.shape
        ng = M.no_go_zone.astype(bool)
        D = os.path.join(M.SCRIPT_DIR, "..", "data")
        if env.startswith("taiwan_real"):
            f = os.path.join(D, "no_go_TWreal_overlay.npy")
            if os.path.exists(f):
                m = np.load(f).astype(bool) & ng
                return m if m.any() else None
            return None
        if env.startswith("japan_real") or env.startswith("philippines_real"):
            import build_real_maps as BRM
            from matplotlib.path import Path as _MPath
            country = "japan" if env.startswith("japan_real") else "philippines"
            lon0, lat0, lon1, lat1 = BRM.BBOX[country]
            xs = lon0 + (np.arange(W) + 0.5) / W * (lon1 - lon0)
            ys = lat0 + (np.arange(H) + 0.5) / H * (lat1 - lat0)
            LON, LAT = np.meshgrid(xs, ys); pts = np.stack([LON.ravel(), LAT.ravel()], 1)
            mask = np.zeros(H * W, bool)
            for poly in BRM.RESTRICTED.get(country, []):
                mask |= _MPath(np.asarray(poly)).contains_points(pts)
            m = mask.reshape(H, W) & ng
            return m if m.any() else None
    except Exception:
        return None
    return None


def _restricted_style(bw):
    """海上禁航區呈現樣式:(fill 色, fill alpha, hatch, 邊界色)。陸地一律黑;禁航區與陸地以不同方式區分。"""
    if bw:
        return "#CFCFCF", 1.0, "////", "black"      # 黑白:淺灰底 + 黑斜線 + 黑邊界(對比實心黑陸地)
    return "#7D3C98", 0.50, "////", "#4A235A"        # 彩色:紫底半透明 + 斜線 + 深紫邊界(避開 YlOrRd 權重與航線色)


def _draw_path(ax, ch, bw, restricted=None):
    import matplotlib as _mpl
    from matplotlib.colors import ListedColormap, PowerNorm
    ng = M.no_go_zone.astype(bool)
    land = ng & ~restricted if restricted is not None else ng
    # 權重底圖:√ 色階(PowerNorm γ=0.5)讓低密度(如日本沿岸薄帶)也看得見
    im0 = ax.imshow(M.weight_map, cmap=("Greys" if bw else "YlOrRd"), origin="lower",
                    alpha=(0.45 if bw else 0.55), norm=PowerNorm(0.5, vmin=0, vmax=1))
    # 陸地:一律黑
    ax.imshow(np.ma.masked_where(~land, land.astype(float)), cmap=ListedColormap(["black"]),
              origin="lower", vmin=0, vmax=1)
    # 海上禁航區:獨立呈現(底色 + 斜線紋路 + 邊界)
    if restricted is not None and restricted.any():
        fc, fa, hatch, ec = _restricted_style(bw)
        ax.imshow(np.ma.masked_where(~restricted, restricted.astype(float)),
                  cmap=ListedColormap([fc]), origin="lower", vmin=0, vmax=1, alpha=fa)
        R = restricted.astype(float)
        _ohc, _ohlw = _mpl.rcParams["hatch.color"], _mpl.rcParams["hatch.linewidth"]
        _mpl.rcParams["hatch.color"] = ec; _mpl.rcParams["hatch.linewidth"] = 0.7
        ax.contourf(R, levels=[0.5, 1.5], colors="none", hatches=[hatch], origin="lower")
        ax.contour(R, levels=[0.5], colors=ec, linewidths=0.9, origin="lower")
        _mpl.rcParams["hatch.color"] = _ohc; _mpl.rcParams["hatch.linewidth"] = _ohlw
    colors = cm.get_cmap("tab20", M.NUM_BASES)
    for v in range(M.NUM_VESSELS):
        verts = ch["routes"][v]
        col = "#222222" if bw else colors(v // 2)
        ax.plot([p[0] for p in verts], [p[1] for p in verts], color=col,
                lw=(0.9 if bw else 1.2), ls=("-" if v % 2 == 0 else "--"), alpha=(0.7 if bw else 0.85))
    ports = getattr(M, "base_ports", None) or list(M.base.values())
    pcol = "black" if bw else "red"
    for (px, py), (lx, ly) in zip(ports, M.base_coords):
        ax.plot([px, lx], [py, ly], c=pcol, lw=0.6, alpha=0.5, zorder=4)
        ax.scatter(lx, ly, c=pcol, s=12, marker="o", zorder=5, edgecolors="k", linewidths=0.3)
        ax.scatter(px, py, c=pcol, s=60, marker="*", zorder=6, edgecolors="k", linewidths=0.4)
    ax.set_xlim(0, M.MAP_W); ax.set_ylim(0, M.MAP_H); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return im0


def best_path_montage(best, label, save):
    from matplotlib.patches import Patch
    specs = [("F1", "Best F1 (max coverage)"), ("F2", "Best F2 (min cost)"), ("F3", "Best F3 (min cooperative dist.)")]
    restricted = _restricted_mask(getattr(M, "ENV_NAME", ""))
    def build(bw):
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
        im0 = None
        for ax, (k, lab) in zip(axes, specs):
            ch = best[k]; im0 = _draw_path(ax, ch, bw, restricted=restricted)
            ax.set_title("%s\nF1=%.0f  F2=%.0f  F3=%.0f" % (lab, ch["F1"], ch["F2"], ch["F3"]), fontsize=11)
        # 圖例:陸地 vs 海上禁航區
        handles = [Patch(facecolor="black", edgecolor="black", label="Land")]
        if restricted is not None and restricted.any():
            fc, fa, hatch, ec = _restricted_style(bw)
            handles.append(Patch(facecolor=fc, alpha=fa, hatch=hatch, edgecolor=ec,
                                 label="Maritime restricted zone (no-go)"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10, frameon=False,
                   bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(label, fontsize=14, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0.06, 0.92, 0.88])
        # 權重底圖 colorbar(右側,√ 色階):標清楚那層橘紅/灰是情境覆蓋權重
        if im0 is not None:
            cax = fig.add_axes([0.94, 0.28, 0.012, 0.44])
            cb = fig.colorbar(im0, cax=cax)
            cb.set_label("Scenario weight — coverage demand\n(normalized 0–1, √ colour scale)", fontsize=9)
            cb.ax.tick_params(labelsize=7)
        return fig
    for bw in (False, True):
        fig = build(bw)
        out = save if not bw else _bw_path(save)
        fig.savefig(out, dpi=110); plt.close(fig)
        print(("彩色" if not bw else "黑白") + "路徑圖存檔:", out)

def best_from_population(pop):
    return {"F1": max(pop, key=lambda c: c["F1"]), "F2": min(pop, key=lambda c: c["F2"]), "F3": min(pop, key=lambda c: c["F3"])}


# ---------- (6) 各演算法 union 後每目標最佳值表(含 Greedy)----------
def _pool_union(results, scn, m, seeds):
    """某情境、某方法跨所有 runs 的解聯集(原始 pool,不取非支配)。Greedy 取其單一前緣。"""
    if m == "貪婪":
        return list(results.get("greedy", {}).get(scn, []))
    runs = results["runs"]; sols = []
    for sd in seeds:
        sols += runs.get(scn, {}).get(sd, {}).get(m, [])
    return sols


def _render_table_png(rows, save, title=""):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    col_chars = [max(len(str(r[j])) for r in rows) for j in range(ncol)]   # 各欄最長內容字數
    total = max(1, sum(col_chars))
    colw = [c / total for c in col_chars]                                    # 欄寬依字數比例分配(和=1,撐滿軸)
    fig_w = max(10.0, 0.17 * total)                                          # 畫布寬隨總字數放大
    fig_h = max(2.2, 0.46 * (len(body) + 1) + 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="center", colWidths=colw)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for j in range(ncol):
        c = tbl[0, j]; c.set_text_props(fontweight="bold", color="white"); c.set_facecolor("#1F4E79")
    for i in range(1, len(body) + 1):
        for j in range(ncol):
            tbl[i, j].set_facecolor("#FFFFFF" if i % 2 else "#EEF2F7")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    fig.tight_layout(); fig.savefig(save, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("表格圖存檔:", save)


def best_obj_union_table(results, methods, save_csv=None, save_png=None):
    """各演算法跨所有 runs 之解聯集,列出每目標最佳值(F1 max↑, F2 min↓, F3 min↓)+ 聯集解數。
    注意:三個最佳值通常來自不同解(各目標各自取極值),非單一解同時達成三者。"""
    seeds = results["meta"]["seeds"]; scns = list(results["runs"].keys())
    rows = [["Scenario", "Method", "Best F1 (max ↑)", "Best F2 (min ↓)", "Best F3 (min ↓)", "N (union)"]]
    print("=== 各演算法 union 後每目標最佳值(F1↑ / F2↓ / F3↓)===")
    for s in scns:
        print(f"[{s}]")
        for m in methods:
            pool = _pool_union(results, s, m, seeds)
            if not pool:
                continue
            f1 = max(a["F1"] for a in pool); f2 = min(a["F2"] for a in pool); f3 = min(a["F3"] for a in pool)
            print(f"  {m:6s} F1max {f1:.0f}  F2min {f2:.0f}  F3min {f3:.0f}  (n={len(pool)})")
            rows.append([scen_en(s), method_en(m), f"{f1:.1f}", f"{f2:.1f}", f"{f3:.1f}", str(len(pool))])
    if save_csv:
        import csv
        with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(_csv_with_tier(rows, results))
        print("表格存檔:", save_csv)
    if save_png:
        _render_table_png(rows, save_png, title="Per-objective best over union of all runs (F1↑, F2↓, F3↓)" + _tier_tag(results))
    return rows


def _argbest_union(results, scn, m, seeds):
    """union pool 中 F1 最大、F2 最小、F3 最小的三個解(回傳 (s_f1, s_f2, s_f3) 或 None)。"""
    pool = _pool_union(results, scn, m, seeds)
    if not pool:
        return None
    return (max(pool, key=lambda a: a["F1"]),
            min(pool, key=lambda a: a["F2"]),
            min(pool, key=lambda a: a["F3"]))


def best_solutions_union_table(results, methods, save_csv=None, save_png=None):
    """各演算法跨所有 runs 聯集後,列出『F1 最佳解 / F2 最佳解 / F3 最佳解』各自完整的 (F1,F2,F3)。
    可直接查表看權衡:例如 F1 最佳的那個解,其 F2/F3 往往較差。每列一個 (情境×演算法)。"""
    seeds = results["meta"]["seeds"]; scns = list(results["runs"].keys())
    header = ["Scenario", "Method",
              "bestF1: F1", "bestF1: F2", "bestF1: F3",
              "bestF2: F1", "bestF2: F2", "bestF2: F3",
              "bestF3: F1", "bestF3: F2", "bestF3: F3"]
    rows = [header]
    print("=== 各演算法 union 後:F1/F2/F3 最佳解各自的完整三目標 ===")
    for s in scns:
        for m in methods:
            tri = _argbest_union(results, s, m, seeds)
            if tri is None:
                continue
            s1, s2, s3 = tri
            rows.append([scen_en(s), method_en(m),
                         f"{s1['F1']:.1f}", f"{s1['F2']:.0f}", f"{s1['F3']:.0f}",
                         f"{s2['F1']:.1f}", f"{s2['F2']:.0f}", f"{s2['F3']:.0f}",
                         f"{s3['F1']:.1f}", f"{s3['F2']:.0f}", f"{s3['F3']:.0f}"])
            print(f"  {scen_en(s)} | {method_en(m)}: "
                  f"bestF1=({s1['F1']:.0f},{s1['F2']:.0f},{s1['F3']:.0f}) "
                  f"bestF2=({s2['F1']:.0f},{s2['F2']:.0f},{s2['F3']:.0f}) "
                  f"bestF3=({s3['F1']:.0f},{s3['F2']:.0f},{s3['F3']:.0f})")
    if save_csv:
        import csv
        with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(_csv_with_tier(rows, results))
        print("表格存檔:", save_csv)
    if save_png:
        _render_table_png(rows, save_png,
                          title="Best-per-objective solutions over union — full (F1 up, F2 down, F3 down) of each" + _tier_tag(results))
    return rows


def greedy_solutions_table(results, save_csv=None, save_png=None):
    """列出 Greedy 每情境的所有解之 (F1↑, F2↓, F3↓)。便於看出單一解無法三目標兼優
    (低 F2/F3 的解往往 F1 覆蓋很低),與 EA 前緣相比整體居於劣勢。"""
    greedy = results.get("greedy", {})
    scns = [s for s in results["runs"].keys() if s in greedy and greedy[s]]
    if not scns:
        return []
    rows = [["Scenario", "Greedy sol.", "F1 (max ↑)", "F2 (min ↓)", "F3 (min ↓)"]]
    print("=== Greedy 所有解之三目標(F1↑ / F2↓ / F3↓)===")
    for s in scns:
        for i, a in enumerate(greedy[s], 1):
            rows.append([scen_en(s), f"#{i}", f"{a['F1']:.1f}", f"{a['F2']:.1f}", f"{a['F3']:.1f}"])
            print(f"  {scen_en(s)} #{i}  F1={a['F1']:.0f}  F2={a['F2']:.0f}  F3={a['F3']:.0f}")
    if save_csv:
        import csv
        with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(_csv_with_tier(rows, results))
        print("表格存檔:", save_csv)
    if save_png:
        _render_table_png(rows, save_png, title="Greedy: all solutions' objectives (F1↑, F2↓, F3↓)" + _tier_tag(results))
    return rows


# ---------- (5) 3D Pareto-front 對比圖(F1↑, F2↓, F3↓)----------
def _method_front(results, scn, m, seeds, greedy, methods, mode="union"):
    """回傳 (front, seed_label)。
      mode='union'    :該方法跨所有種子的解聯集 → 真非支配前緣(seed_label=None)。
      mode='median_run':該方法各種子前緣中 HV 取中位數的那一次 run 之前緣(seed_label=該種子)。
    Greedy 為確定性(每情境一份),兩模式皆回傳其單一前緣。"""
    runs = results["runs"]
    if m == "貪婪":
        g = MM.nondominated(greedy[scn]) if (greedy and scn in greedy) else []
        return g, None
    if mode == "median_run":
        scored = []
        for sd in seeds:
            if m not in runs.get(scn, {}).get(sd, {}):
                continue
            rec, lo, hi, _ = _per_seed_clean(runs, greedy, scn, sd, methods)   # 與指標一致的逐種子界
            if m in rec:
                scored.append((MM.hv(rec[m], lo, hi), sd, rec[m]))
        if not scored:
            return [], None
        scored.sort(key=lambda t: t[0])
        _, sd_med, front_med = scored[(len(scored) - 1) // 2]                  # 下中位數(偶數取較低者)
        return front_med, sd_med
    sols = []                                                                  # union
    for sd in seeds:
        cell = runs.get(scn, {}).get(sd, {})
        if m in cell:
            sols += cell[m]
    return (MM.nondominated(sols) if sols else []), None


def pareto3d_compare(results, methods, save, mode="union"):
    """各方法於 (F1↑,F2↓,F3↓) 三目標空間並陳前緣;每情境一個 3D 子圖。
      mode='union'     :跨種子綜合最佳已知前緣(樂觀包絡;不反映變異)。
      mode='median_run':各方法 HV 中位數的單一 run 之前緣(代表性單次結果)。
    彩色版以顏色區分;黑白版以灰階 + 不同 marker 區分(b&w friendly)。"""
    runs = results["runs"]; greedy = results.get("greedy", {})
    seeds = results["meta"]["seeds"]; scns = list(runs.keys())
    is_med = (mode == "median_run")

    def build(bw):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401(註冊 3d projection)
        n = max(1, len(scns))
        fig = plt.figure(figsize=(6.3 * n, 5.6))
        for j, s in enumerate(scns):
            ax = fig.add_subplot(1, n, j + 1, projection="3d")
            any_pt = False
            for m in methods:
                front, sd_med = _method_front(results, s, m, seeds, greedy, methods, mode)
                if not front:
                    continue
                any_pt = True
                st = _style(methods, m, bw)
                col = st["line"] if bw else st["fill"]
                xs = [a["F1"] for a in front]; ys = [a["F2"] for a in front]; zs = [a["F3"] for a in front]
                if is_med and sd_med is not None:
                    lab = f"{method_en(m)} (seed {sd_med}, n={len(front)})"
                elif is_med:                                   # Greedy:確定性,標 det.
                    lab = f"{method_en(m)} (det., n={len(front)})"
                else:
                    lab = f"{method_en(m)} (n={len(front)})"
                ax.scatter(xs, ys, zs, c=col, marker=st["marker"], s=36, alpha=0.9,
                           depthshade=False, edgecolors="k", linewidths=0.35, label=lab)
            ax.set_xlabel("F1 coverage ↑", fontsize=8, labelpad=2)
            ax.set_ylabel("F2 distance ↓", fontsize=8, labelpad=2)
            ax.set_zlabel("F3 coop. dist. ↓", fontsize=8, labelpad=2)
            ax.tick_params(labelsize=6)
            ax.set_title(scen_en(s), fontsize=10)
            ax.view_init(elev=22, azim=-52)
            if any_pt:
                ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
        sub = (f"median-HV single run per method" if is_med
               else f"non-dominated front pooled over {len(seeds)} seeds")
        fig.suptitle(f"3D Pareto-front comparison — {sub}" + _tier_tag(results), fontweight="bold", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig
    _emit(save, build)

# ---------- 一鍵:四項全做 ----------
def make_all(results, methods, outdir="eval_out", proposed="本方法", path_runner=None):
    """[legacy/手動入口] 正式管線請用 experiment.plot_all(自動帶層級檔名/標題)。
    此處檔名亦加 _<TIER> 後綴、標題帶 [TIER tier],以免與他層產物混淆。"""
    os.makedirs(outdir, exist_ok=True)
    _tier = _tier_str(results)
    def _t(name):
        if not _tier:
            return name
        root, ext = os.path.splitext(name); return f"{root}_{_tier}{ext}"
    four_metric_table(results, methods, save_csv=os.path.join(outdir, _t("compare_table.csv")))
    metric_bars(results, methods, os.path.join(outdir, _t("metric_bars.png")))
    cmetric_boxplot(results, methods, os.path.join(outdir, _t("cmetric_box.png")), proposed)
    pareto3d_compare(results, methods, os.path.join(outdir, _t("pareto3d_union.png")), mode="union")
    pareto3d_compare(results, methods, os.path.join(outdir, _t("pareto3d_median.png")), mode="median_run")
    if results.get("snaps"):
        hv_convergence(results, methods, os.path.join(outdir, _t("hv_convergence.png")))
    else:
        print("(無 snaps;略過 HV 收斂曲線)")
    best = None
    if results.get("best"):
        s0 = list(results["best"].keys())[0]; best = results["best"][s0].get(proposed)
    if best is None and path_runner is not None:
        best = best_from_population(path_runner())
    if best is not None:
        best_path_montage(best, f"{method_en(proposed)}: Patrol routes of best F1 / F2 / F3 solutions" + _tier_tag(results),
                          os.path.join(outdir, _t("best_paths.png")))
    else:
        print("(無 best 路徑資料且未提供 path_runner;略過路徑圖)")

def from_existing(formal_pkl="formal_results.pkl", conv_pkl="conv_snaps.pkl"):
    d = pickle.load(open(formal_pkl, "rb"))
    results = {"meta": d["meta"], "greedy": d.get("greedy", {}), "runs": d["runs"]}
    if conv_pkl and os.path.exists(conv_pkl):
        cd = pickle.load(open(conv_pkl, "rb"))
        results["snaps"] = {s: {m: cd["scn"][s][m] for m in ["本方法", "ES", "隨機"] if m in cd["scn"][s]} for s in cd["scn"]}
    return results

if __name__ == "__main__":
    B.load_cache()
    res = from_existing()
    methods = B.ALGOS
    def runner():
        res2 = B.make_scenarios(); M.weight_map = res2["S2 外海熱區"]
        return run_cmp("three_tier", 1, 100, 1500)[0]
    make_all(res, methods, outdir="eval_out", path_runner=runner)
    print("\n四項標準產物已輸出至 eval_out/")
