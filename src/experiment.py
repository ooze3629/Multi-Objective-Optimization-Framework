# -*- coding: utf-8 -*-
"""
外層實驗框架(雙層架構之外層)
  - Config:集中定義所有參數(預設 10 種子)
  - run_experiment:多種子 × 多海域 × 多方法,呼叫內層演算法(MOGA_GPSIFF / baselines)
  - analyze:統計檢定(Friedman / Wilcoxon)
  - plot_all:四項標準產物 + 長條圖(全部黑白列印友善:顏色 + 線型 + 標記 + 紋路)
內層 = MOGA_GPSIFF_patrol_clean(本方法)、baselines(ES/隨機/貪婪)。
用法:
  python experiment.py smoke    # 煙霧測試(小規模,驗證整條流程)
  python experiment.py          # 正式(pop100 / fes3000 / 10 種子)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, pickle
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import friedmanchisquare, wilcoxon

import MOGA_GPSIFF_patrol_clean as M
import baselines as B
import moea_metrics as MM
import experiment_eval as EE
from compare_sel import run_cmp
from nsga3 import run_nsga3
from sms_emoa import run_sms_emoa
from method_hvsel import run_method_hv
from method_hvgreedy import run_method_hvgreedy
from method_memetic import run_method_memetic
from method_ss import run_method_ss
from operators_bbx import crossover_block, mutate_inv, mutate_inv_resel
from local_search import mutate_2opt


# ============ 參數(全部集中於此)============
@dataclass
class Config:
    pop: int = 100
    fes: int = 3000
    crossover_rate: float = 0.6
    n_seeds: int = 10
    snapshot_every: int = 2
    methods: Tuple[str, ...] = ("本方法", "ES", "隨機", "貪婪")
    proposed: str = "本方法"
    record_snaps: bool = True          # 記錄 Pareto 快照(供多種子 HV 收斂曲線)
    env: str = "taiwan"                # 實驗環境:"taiwan" 或 "japan"
    out_dir: str = "experiment_out"
    seeds: Optional[Tuple[int, ...]] = None

    def resolved_seeds(self):
        return list(self.seeds) if self.seeds else list(range(1, self.n_seeds + 1))


# ============ 內層派發:單一方法 / 單一種子 / 當前海域 ============
def _solve(method, seed, cfg, record_snap):
    if method == "本方法":
        pop, arc, _, snaps = run_cmp("three_tier", seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "NSGA-III":
        pop, arc, _, snaps = run_nsga3(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-HV":
        pop, arc, _, snaps = run_method_hv(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "SMS-EMOA":
        pop, arc, _, snaps = run_sms_emoa(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-HVg":
        pop, arc, _, snaps = run_method_hvgreedy(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-Mem":
        pop, arc, _, snaps = run_method_memetic(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "NSGA-III-Mem":
        pop, arc, _, snaps = run_nsga3(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate, mutate_fn=mutate_2opt)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "SMS-EMOA-Mem":
        pop, arc, _, snaps = run_sms_emoa(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate, mutate_fn=mutate_2opt)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-ss":
        pop, arc, _, snaps = run_method_ss(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-ss-2opt":
        pop, arc, _, snaps = run_method_ss(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate, mutate_fn=mutate_2opt)
        best = {"F1": max(pop, key=lambda c: c["F1"]),
                "F2": min(pop, key=lambda c: c["F2"]),
                "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-ss-bx":
        pop, arc, _, snaps = run_method_ss(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate, crossover_fn=crossover_block, mutate_fn=mutate_inv)
        best = {"F1": max(pop, key=lambda c: c["F1"]), "F2": min(pop, key=lambda c: c["F2"]), "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-ss-bx+":
        pop, arc, _, snaps = run_method_ss(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate, crossover_fn=crossover_block, mutate_fn=mutate_inv_resel)
        best = {"F1": max(pop, key=lambda c: c["F1"]), "F2": min(pop, key=lambda c: c["F2"]), "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "本方法-ss-bbx":
        pop, arc, _, snaps = run_method_ss(seed, cfg.pop, cfg.fes, cfg.snapshot_every, cfg.crossover_rate, crossover_fn=crossover_block)
        best = {"F1": max(pop, key=lambda c: c["F1"]), "F2": min(pop, key=lambda c: c["F2"]), "F3": min(pop, key=lambda c: c["F3"])}
        return {"archive": arc, "snaps": snaps if record_snap else None, "best": best}
    if method == "ES":
        arc, snaps = B.es_snap(seed, cfg.pop, cfg.fes) if record_snap else (B.run_es(seed, cfg.pop, cfg.fes), None)
        return {"archive": arc, "snaps": snaps, "best": None}
    if method == "隨機":
        arc, snaps = B.rand_snap(seed, cfg.pop, cfg.fes) if record_snap else (B.run_random(seed, cfg.pop, cfg.fes), None)
        return {"archive": arc, "snaps": snaps, "best": None}
    if method == "貪婪":
        return {"archive": B.run_greedy(), "snaps": None, "best": None}
    raise ValueError("未知方法:" + method)


# ============ 編排:多種子 × 多海域 × 多方法 ============
def run_experiment(cfg, scenarios=None):
    M.set_environment(cfg.env)                          # 先切換環境(底圖/基地/候選點/快取)
    try:                                                # 在 set_environment 之後載入,避免被清空
        _nc = M.load_route_cache(strict=True)           # 讀 data/route_cache_<env>.pkl(meta 格式 + 底圖指紋核對)
        print(f"  [cache] 已載入 {cfg.env} 繞行快取 {_nc} 鍵(data/)")
    except FileNotFoundError:
        print(f"  [cache] 無 {cfg.env} 預建快取 → 即時繞行(可先跑 build_route_caches.py {cfg.env})")
    scns = scenarios if scenarios is not None else B.make_scenarios(cfg.env)
    seeds = cfg.resolved_seeds()
    os.makedirs(cfg.out_dir, exist_ok=True)
    data = {"meta": {"pop": cfg.pop, "fes": cfg.fes, "crossover_rate": cfg.crossover_rate,
                     "seeds": seeds, "proposed": cfg.proposed, "methods": list(cfg.methods),
                     "env": cfg.env, "drone_tier": M.DRONE_TIER, "drone_domain": list(M.DRONE_DOMAIN), "delta": dict(M.DELTA)},
            "runs": {}, "greedy": {}, "snaps": {}, "best": {}}
    t0 = time.time()
    print(f"  環境={cfg.env}:{M.NUM_BASES} 基地 / {M.NUM_VESSELS} 載具 / 底圖 {M.no_go_zone.shape} / "
          f"情境 {list(scns.keys())}")
    for sname, wmap in scns.items():
        M.weight_map = wmap
        data["runs"][sname] = {}; data["snaps"][sname] = {}
        if "貪婪" in cfg.methods:
            data["greedy"][sname] = B.run_greedy()
        for sd in seeds:
            data["runs"][sname][sd] = {}
            for m in cfg.methods:
                if m == "貪婪":
                    data["runs"][sname][sd][m] = data["greedy"][sname]; continue
                rec = _solve(m, sd, cfg, cfg.record_snaps)
                data["runs"][sname][sd][m] = rec["archive"]
                if rec["snaps"] is not None:
                    data["snaps"][sname].setdefault(m, {})[sd] = rec["snaps"]
                if rec["best"] is not None and m == cfg.proposed and sname not in data["best"]:
                    data["best"].setdefault(sname, {})[m] = rec["best"]
            print(f"  [{sname}] seed {sd} 完成({time.time()-t0:.0f}s)")
        # 註:繞行快取以 build_route_caches.py 預建於 data/(meta 格式);此處不再寫 cwd raw 副本
    pickle.dump(data, open(os.path.join(cfg.out_dir, "results.pkl"), "wb"))
    print(f"原始結果存檔:{os.path.join(cfg.out_dir, 'results.pkl')}")
    return data


# ============ 統計分析 ============
def analyze(data, cfg, save="stats.txt"):
    runs, greedy = data["runs"], data.get("greedy", {})
    seeds = data["meta"]["seeds"]; scns = list(runs.keys()); methods = list(cfg.methods)
    lines = [f"統計分析(pop={data['meta']['pop']}, fes={data['meta']['fes']}, seeds={len(seeds)})",
             f"DRONE_TIER={M.DRONE_TIER}  DRONE_DOMAIN={M.DRONE_DOMAIN}  DELTA={M.DELTA}"]
    for s in scns:
        per = {m: {"HV": [], "IGD+": []} for m in methods}
        for sd in seeds:
            rec, lo, hi, ref = EE._per_seed_clean(runs, greedy, s, sd, methods)
            for m in rec:
                per[m]["HV"].append(MM.hv(rec[m], lo, hi))
                per[m]["IGD+"].append(MM.igd_plus(rec[m], ref, lo, hi))
        lines.append(f"\n=== {s} ===")
        for metric in ("HV", "IGD+"):
            groups = [per[m][metric] for m in methods if per[m][metric]]
            if len(groups) >= 3 and len(set(len(g) for g in groups)) == 1 and len(groups[0]) >= 2:
                try:
                    chi, p = friedmanchisquare(*groups)
                    lines.append(f"  Friedman {metric}: chi2={chi:.2f}, p={p:.4g}")
                except Exception as e:
                    lines.append(f"  Friedman {metric}: {e}")
            for m in methods:
                if m == cfg.proposed or not per[m][metric]:
                    continue
                try:
                    _, p = wilcoxon(per[cfg.proposed][metric], per[m][metric])
                    lines.append(f"  Wilcoxon {metric} {cfg.proposed} vs {m}: p={p:.4g}")
                except Exception as e:
                    lines.append(f"  Wilcoxon {metric} {cfg.proposed} vs {m}: {e}")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(cfg.out_dir, save), "w", encoding="utf-8").write(txt + "\n")
    return txt


# ============ 全部繪圖(黑白友善)============
def plot_all(data, cfg):
    methods = list(cfg.methods); od = cfg.out_dir
    _tier = (data.get("meta") or {}).get("drone_tier") or getattr(M, "DRONE_TIER", None)
    def _t(name):
        if not _tier:
            return name
        r, e = os.path.splitext(name); return f"{r}_{_tier}{e}"
    # 每張產物各自 try/except:單張失敗只記警告,不拖垮其餘標準產物(避免一張崩圖吃掉後續所有圖)
    _fail = []
    def _safe(label, fn):
        try:
            fn()
        except Exception as e:
            import traceback
            _fail.append(label)
            print(f"[warn] 產物「{label}」繪製失敗,已跳過(不影響其餘產物):{type(e).__name__}: {e}")
            traceback.print_exc()
    all_methods = methods + (["貪婪"] if data.get("greedy") else [])
    _safe("compare_table", lambda: EE.four_metric_table(data, methods, save_csv=os.path.join(od, _t("compare_table.csv"))))
    _safe("best_obj_union", lambda: EE.best_obj_union_table(data, all_methods, save_csv=os.path.join(od, _t("best_obj_union.csv")),
                            save_png=os.path.join(od, _t("fig_best_obj_union.png"))))
    _safe("best_solutions_union", lambda: EE.best_solutions_union_table(data, all_methods, save_csv=os.path.join(od, _t("best_solutions_union.csv")),
                                  save_png=os.path.join(od, _t("fig_best_solutions_union.png"))))
    if data.get("greedy"):
        _safe("greedy_solutions", lambda: EE.greedy_solutions_table(data, save_csv=os.path.join(od, _t("greedy_solutions.csv")),
                                  save_png=os.path.join(od, _t("fig_greedy_solutions.png"))))
    _safe("metric_bars", lambda: EE.metric_bars(data, methods, os.path.join(od, _t("fig_metric_bars.png"))))
    _safe("cmetric_box", lambda: EE.cmetric_boxplot(data, methods, os.path.join(od, _t("fig_cmetric_box.png")), cfg.proposed))
    _p3d = {"both": [("union", "fig_pareto3d_union.png"), ("median_run", "fig_pareto3d_median.png")],
            "union": [("union", "fig_pareto3d_union.png")],
            "median": [("median_run", "fig_pareto3d_median.png")],
            "none": []}.get(os.environ.get("PARETO3D", "both").lower(),
                             [("union", "fig_pareto3d_union.png"), ("median_run", "fig_pareto3d_median.png")])
    for _md, _fn in _p3d:
        _safe(f"pareto3d:{_md}", (lambda md, fn: (lambda: EE.pareto3d_compare(data, methods, os.path.join(od, _t(fn)), mode=md)))(_md, _fn))
    snaps = data.get("snaps", {})
    has_multi = any(len(snaps.get(s, {}).get(m, {})) >= 2 for s in snaps for m in snaps.get(s, {}))
    if has_multi:
        multi = {"meta": data["meta"], "scn": snaps, "greedy": data.get("greedy", {})}
        _safe("hv_convergence", lambda: EE.hv_convergence_multiseed(multi, methods, os.path.join(od, _t("fig_hv_convergence.png"))))
    elif any(snaps.get(s) for s in snaps):
        single = {"meta": data["meta"], "runs": data["runs"], "greedy": data.get("greedy", {}),
                  "snaps": {s: {m: list(snaps[s][m].values())[0] for m in snaps[s]} for s in snaps}}
        _safe("hv_convergence", lambda: EE.hv_convergence(single, methods, os.path.join(od, _t("fig_hv_convergence.png"))))
    if data.get("best"):
        s0 = list(data["best"].keys())[0]; best = data["best"][s0].get(cfg.proposed)
        if best:
            _ttl = (f"{EE.method_en(cfg.proposed)}: Patrol routes of best F1 / F2 / F3 solutions "
                    f"({EE.scen_en(s0)})" + (f"  [{_tier} tier]" if _tier else ""))
            _safe("best_paths", lambda: EE.best_path_montage(best, _ttl, os.path.join(od, _t("fig_best_paths.png"))))
    if _fail:
        print(f"[plot_all] 完成,但 {len(_fail)} 項產物失敗:{_fail}(其餘已正常輸出;可修正後重跑 --refigure 補齊)")
    else:
        print("[plot_all] 全部標準產物輸出完成")


def main(cfg):
    t = time.time()
    print(f"== 實驗開始 == 環境={cfg.env} pop={cfg.pop} fes={cfg.fes} seeds={cfg.n_seeds} methods={cfg.methods}")
    data = run_experiment(cfg)
    analyze(data, cfg)
    plot_all(data, cfg)
    print(f"\n== 完成 == 輸出於 {cfg.out_dir}/(總耗時 {time.time()-t:.0f}s)")


if __name__ == "__main__":
    # 用法:python experiment.py [smoke] [<env>]   (關鍵字順序不限)
    # <env> 可為下列九個環境字串之一;省略則預設 taiwan。
    VALID_ENVS = [
        "taiwan", "japan", "philippines",
        "taiwan_real", "taiwan_real_ddn",
        "japan_real", "japan_real_ddn",
        "philippines_real", "philippines_real_ddn",
    ]
    args = [a.lower() for a in sys.argv[1:]]
    env_tokens = [a for a in args if a in VALID_ENVS]
    unknown = [a for a in args if a not in ("smoke", "micro") and a not in VALID_ENVS]
    if unknown:
        raise SystemExit(
            f"未知參數:{unknown}。環境須為下列之一:{VALID_ENVS};另可加 smoke 或 micro。")
    if len(env_tokens) > 1:
        raise SystemExit(
            f"experiment.py 一次只支援一個環境(收到 {env_tokens});"
            f"多環境依序請用:python run_ddn_nsga3_ab.py formal {' '.join(env_tokens)}")
    env = env_tokens[0] if env_tokens else "taiwan"
    smoke = "smoke" in args
    micro = "micro" in args
    if micro:
        M.set_environment(env)
        scns = B.make_scenarios(env); scns = {k: scns[k] for k in list(scns)[:2]}
        cfg = Config(pop=10, fes=50, n_seeds=1, snapshot_every=1,
                     methods=("本方法-ss", "NSGA-III"), proposed="本方法-ss",
                     env=env, out_dir=f"experiment_out_micro_{env}_{M.DRONE_TIER}")
        print(f"== micro 健檢(環境={env};pop10/fes50/1seed/2方法/2情境/不繪圖)==")
        t = time.time()
        d = run_experiment(cfg, scenarios=scns)
        analyze(cfg=cfg, data=d)
        print(f"micro 完成({time.time()-t:.0f}s),輸出於 {cfg.out_dir}/(僅 runs/stats,無圖)")
    elif smoke:
        cfg = Config(pop=20, fes=200, n_seeds=2, snapshot_every=1,
                     env=env, out_dir=f"experiment_out_smoke_{env}_{M.DRONE_TIER}")
        # 煙霧:僅取兩個情境加速
        M.set_environment(env)
        scns = B.make_scenarios(env)
        scns = {k: scns[k] for k in list(scns)[:2]}
        print(f"== 煙霧測試(環境={env})==")
        t = time.time()
        d = run_experiment(cfg, scenarios=scns)
        analyze(cfg=cfg, data=d)
        plot_all(d, cfg)
        print(f"\n煙霧測試完成({time.time()-t:.0f}s),輸出於 {cfg.out_dir}/")
    else:
        main(Config(env=env, out_dir=(f"experiment_out_{M.DRONE_TIER}" if env=="taiwan" else f"experiment_out_{env}_{M.DRONE_TIER}")))
