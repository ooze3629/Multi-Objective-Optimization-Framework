# -*- coding: utf-8 -*-
"""
aggregate_tiers.py —— 三層覆蓋校準(LOWER/OPERATING/SAFETY)之跨層彙整評估
Public portfolio version of the multi-objective patrol-planning project.

用途
----
三層各自 finalize 完成後(各有 results.pkl),本程式「最後一起算」:
對每個(情境 × 種子),以「**三層 × 全方法**之解的聯集」建立共同的
正規化界線(lo/hi)與 IGD⁺ 參考前緣(ref),再於此**共同正規化空間**計算每個
(層級, 方法)的四項指標。如此三層的 HV/IGD⁺ 才彼此可比(符合 EVAL_PROTOCOL §十)。

公平性規則完全沿用既有 moea_metrics / experiment_eval:
  * 各方法先取真實非支配前緣(MM.nondominated);
  * lo/hi 與 ref 取聯集(此處聯集跨三層);
  * HV 參考點 (1.1,1.1,1.1)(MM.hv 預設 margin=1.1);
  * F1 取大、F2/F3 取小(MM 內建)。

前提
----
三層需先各自跑完並 finalize:
  DRONE_TIER=LOWER     python run_formal_resumable.py <env> --finalize
  DRONE_TIER=OPERATING python run_formal_resumable.py <env> --finalize
  DRONE_TIER=SAFETY    python run_formal_resumable.py <env> --finalize
產生 experiment_out_pe_<env>_<TIER>_formal/results.pkl(本程式自動尋找)。

用法
----
  python aggregate_tiers.py <env>
  python aggregate_tiers.py taiwan_real_ddn --tiers LOWER OPERATING SAFETY --out cross_tier_<env>
  python aggregate_tiers.py taiwan_real_ddn --no-fig         # 不畫圖,只出表/CSV
  python aggregate_tiers.py taiwan_real_ddn --kind mixed     # 混合情境軌道(本方法-ss vs -2opt)
"""
import os, sys, pickle, argparse, statistics as st
import moea_metrics as MM
import experiment_eval as EE

OBJS = ("F1", "F2", "F3")
DEFAULT_TIERS = ["LOWER", "OPERATING", "SAFETY"]
FORMAL_METHODS = ["本方法-ss", "SMS-EMOA", "NSGA-III"]
MIXED_METHODS = ["本方法-ss", "本方法-ss-2opt"]


def _results_path(env, tier, kind="formal"):
    if kind == "mixed":
        return os.path.join(f"experiment_out_mixed2opt_{env}_{tier}", "results.pkl")
    return os.path.join(f"experiment_out_pe_{env}_{tier}_formal", "results.pkl")


def _ms(xs):
    xs = [x for x in xs if x == x]  # 去除 nan
    if not xs:
        return (float("nan"), 0.0)
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


def load_tiers(env, tiers, kind="formal"):
    data = {}
    missing = []
    for t in tiers:
        p = _results_path(env, t, kind)
        if not os.path.exists(p):
            missing.append((t, p)); continue
        with open(p, "rb") as fh:
            data[t] = pickle.load(fh)
    if missing:
        flag = "--finalize" if kind == "formal" else "--finalize(run_mixed_2opt_ab.py)"
        msg = f"缺少以下層級之 results.pkl(請先各層 {flag}):\n" + \
              "\n".join(f"  [{t}] {p}" for t, p in missing)
        raise FileNotFoundError(msg)
    return data


def aggregate(env, tiers, methods, kind="formal"):
    data = load_tiers(env, tiers, kind)
    runs0 = data[tiers[0]]["runs"]
    scns = list(runs0.keys())

    # (層級,方法) → 各 (情境,種子) 的四指標
    cells = {(t, m): {"HV": [], "IGD+": [], "extent": [], "card": []}
             for t in tiers for m in methods}
    # 供跨層前緣圖:每層彙整全情境/種子之非支配解(原始單位)
    pooled_front = {t: [] for t in tiers}

    for s in scns:
        seedsets = [set(data[t]["runs"].get(s, {}).keys()) for t in tiers]
        seeds = sorted(set.intersection(*seedsets)) if all(seedsets) else []
        for sd in seeds:
            rec = {}                      # (層級,方法) → 非支配前緣
            allarc = []                   # 三層 × 全方法之聯集
            for t in tiers:
                r = data[t]["runs"][s][sd]
                for m in methods:
                    if m in r:
                        nd = MM.nondominated(r[m])
                        rec[(t, m)] = nd
                        allarc += nd
                        pooled_front[t] += nd
            if not allarc:
                continue
            lo, hi = MM._bounds(allarc)           # ← 跨三層之共同正規化界線
            ref = MM.nondominated(allarc)         # ← 跨三層之共同參考前緣
            for (t, m), nd in rec.items():
                cells[(t, m)]["HV"].append(MM.hv(nd, lo, hi))
                cells[(t, m)]["IGD+"].append(MM.igd_plus(nd, ref, lo, hi))
                cells[(t, m)]["extent"].append(EE._extent(nd, lo, hi))
                cells[(t, m)]["card"].append(len(nd))
    return cells, scns, pooled_front, tiers, methods


def emit_table(cells, tiers, methods, save_csv):
    rows = []
    header = ["層級", "方法", "HV(mean±std)", "IGD+(mean±std)", "extent", "非支配數"]
    rows.append(header)
    for t in tiers:
        for m in methods:
            c = cells[(t, m)]
            hv_m, hv_s = _ms(c["HV"]); ig_m, ig_s = _ms(c["IGD+"])
            ex_m, ex_s = _ms(c["extent"]); cd_m, cd_s = _ms(c["card"])
            rows.append([t, m,
                         f"{hv_m:.4f}±{hv_s:.4f}",
                         f"{ig_m:.4f}±{ig_s:.4f}",
                         f"{ex_m:.3f}±{ex_s:.3f}",
                         f"{cd_m:.1f}±{cd_s:.1f}"])
    # 列印
    w = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    for ri, r in enumerate(rows):
        line = "  ".join(str(r[i]).ljust(w[i]) for i in range(len(header)))
        print(line)
        if ri == 0:
            print("  ".join("-" * w[i] for i in range(len(header))))
    if save_csv:
        import csv
        with open(save_csv, "w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerows(rows)
        print(f"\n[CSV] {save_csv}")


def plot_fronts(pooled_front, tiers, save):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"[warn] 略過畫圖:{e}"); return
    colors = {"LOWER": "#2e7d32", "OPERATING": "#1565c0", "SAFETY": "#c62828"}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for t in tiers:
        nd = MM.nondominated(pooled_front[t])
        if not nd:
            continue
        xs = [p["F1"] for p in nd]; ys = [p["F2"] for p in nd]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.scatter(xs, ys, s=18, color=colors.get(t, None), alpha=0.6, label=f"{t}")
    ax.set_xlabel("F1 加權覆蓋(越大越好)")
    ax.set_ylabel("F2 距離成本(越小越好)")
    ax.set_title("三層覆蓋校準之前緣對照(F1–F2 投影,原始單位)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=150, bbox_inches="tight")
    print(f"[圖] {save}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("env", nargs="?", default="taiwan_real_ddn")
    ap.add_argument("--kind", choices=["formal", "mixed"], default="formal",
                    help="formal=三方對照(預設);mixed=等質量混合情境下 本方法-ss vs -2opt")
    ap.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="預設依 --kind 自動選(formal 三方 / mixed 兩方)")
    ap.add_argument("--out", default=None, help="輸出前綴(預設 cross_tier_<kind>_<env>)")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args(argv)
    methods = a.methods or (MIXED_METHODS if a.kind == "mixed" else FORMAL_METHODS)
    out = a.out or f"cross_tier_{a.kind}_{a.env}"

    cells, scns, pooled, tiers, methods = aggregate(a.env, a.tiers, methods, a.kind)
    print(f"軌道={a.kind}  環境={a.env}  情境={len(scns)}  層級={tiers}  方法={methods}")
    print("(指標於『三層×全方法解聯集』之共同正規化空間計算;HV 參考點 1.1)\n")
    emit_table(cells, tiers, methods, save_csv=f"{out}_metrics.csv")
    if not a.no_fig:
        plot_fronts(pooled, tiers, save=f"{out}_fronts.png")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
