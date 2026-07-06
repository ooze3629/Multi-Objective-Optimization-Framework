# -*- coding: utf-8 -*-
"""組裝多台電腦的實驗結果(formal 三方 / mixed 2-opt)並重新生圖 + 出跨環境總表。

每台用 run_formal_resumable.py 或 run_mixed_2opt_ab.py 跑(「按環境分機」或「同環境 --seed-list 分種子」),
產出 experiment_out_pe_<env>_formal/ 或 experiment_out_mixed2opt_<env>/(內含 ckpt/ 逐格 .pkl,跨機可攜;
或已 finalize 的 results.pkl)。本工具自動辨識兩種實驗,逐環境彙整 → ckpt 齊則 finalize(含 Greedy 基準),
否則用現成 results.pkl 重繪 → combined_summary.csv 跨環境/實驗總表。

前提:各台須用同一程式版本 + 同一份 data/(各台 verify.py cache 並比對 MANIFEST.sha256);
同名 ckpt 格內容不一致會發出衝突警訊(通常代表版本/資料不一致)。

用法(於 code/ 下執行,路徑可絕對):
  python merge_and_refigure.py --sources DIR_A DIR_B DIR_C [--target ./merged] [--envs e1 e2 ...]
  python merge_and_refigure.py --sources A B --refigure-only        # 不彙整,只就地從各 results.pkl 重繪
每個 --sources 目錄須含一或多個 experiment_out_pe_<env>_formal/ 或 experiment_out_mixed2opt_<env>/ 子目錄。
Public portfolio version of the multi-objective patrol-planning project.
"""
import os, glob, shutil, argparse, hashlib, csv, io, pickle
import run_formal_resumable as RFR
import run_mixed_2opt_ab as RMX
import MOGA_GPSIFF_patrol_clean as M
KNOWN_TIERS = ("LOWER", "OPERATING", "SAFETY")

# kind, 前綴, 後綴, finalize_fn, refigure_fn
KINDS = [
    ("formal", "experiment_out_pe_", "_formal", RFR.finalize_env, RFR.refigure_env),
    ("mixed",  "experiment_out_mixed2opt_", "", RMX.finalize_env, RMX.refigure_env),
]
_FIN = {k: fin for k, _, _, fin, _ in KINDS}
_REF = {k: ref for k, _, _, _, ref in KINDS}
_PRE = {k: pre for k, pre, _, _, _ in KINDS}
_SUF = {k: suf for k, _, suf, _, _ in KINDS}


def _discover(root):
    """root 下所有實驗輸出目錄 → {(kind, env, tier): dir}。tier 取自夾名尾端 _<TIER>(無則 None)。"""
    res = {}
    for d in sorted(glob.glob(os.path.join(root, "experiment_out_*"))):
        if not os.path.isdir(d):
            continue
        b = os.path.basename(d)
        for kind, pre, suf, _, _ in KINDS:
            if b.startswith(pre) and (b.endswith(suf) if suf else True):
                mid = b[len(pre):len(b) - len(suf)] if suf else b[len(pre):]
                tier = None
                for t in KNOWN_TIERS:
                    if mid.endswith("_" + t):
                        tier = t; mid = mid[:-(len(t) + 1)]; break
                res[(kind, mid, tier)] = d
                break
    return res


def _pool_ckpt(src_dirs, target_ck, expect_tier=None):
    os.makedirs(target_ck, exist_ok=True)
    seen = {}; n = 0; conflicts = []; tier_mismatch = []
    for sd in src_dirs:
        for f in sorted(glob.glob(os.path.join(sd, "ckpt", "*.pkl"))):
            base = os.path.basename(f); raw = open(f, "rb").read()
            if expect_tier:                       # tier 一致性檢查(防跨機把他層 ckpt 放錯夾)
                try:
                    ct = pickle.loads(raw).get("drone_tier")
                except Exception:
                    ct = None
                if ct and ct != expect_tier:
                    tier_mismatch.append((base, ct)); continue
            sha = hashlib.sha256(raw).hexdigest()
            if base in seen:
                if seen[base] != sha:
                    conflicts.append(base)
                continue
            open(os.path.join(target_ck, base), "wb").write(raw)
            seen[base] = sha; n += 1
    return n, conflicts, tier_mismatch


def _read_table(path, env, kind, tier):
    if not os.path.exists(path):
        return [], None
    with io.open(path, "r", encoding="utf-8-sig", newline="") as fh:
        r = csv.reader(fh); header = next(r, None)
        rows = [row for row in r if row]
    if header and header[0] == "Tier":   # CSV 已自帶 Tier 欄 → 去重(改用資料夾 tier)
        header = header[1:]; rows = [row[1:] for row in rows]
    rows = [[kind, env, tier] + row for row in rows]
    return rows, (["Experiment", "Env", "Tier"] + header if header else None)


def _first_with_results(dirs):
    return next((d for d in dirs if os.path.exists(os.path.join(d, "results.pkl"))), None)


def _finalize(kind, env, out, a):
    fn = _FIN[kind]
    return fn(env, out, a.pop, a.fes, a.seeds, a.mode) if kind == "mixed" else fn(env, out, a.pop, a.fes, a.seeds)


def _refigure(kind, env, out, a):
    fn = _REF[kind]
    return fn(env, out, a.pop, a.fes, a.seeds, a.mode) if kind == "mixed" else fn(env, out, a.pop, a.fes, a.seeds)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="各台輸出根目錄")
    ap.add_argument("--target", default=".", help="彙整輸出根目錄(預設當前)")
    ap.add_argument("--envs", nargs="*", default=None, help="只處理這些環境(預設全部)")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--fes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mode", choices=["equal_mass", "equal_max"], default="equal_mass", help="mixed 實驗的混合模式")
    ap.add_argument("--refigure-only", action="store_true", help="不彙整 ckpt,只就地從各 results.pkl 重繪")
    ap.add_argument("--pareto3d", choices=["both", "union", "median", "none"], default="both",
                    help="3D Pareto-front 圖:both=綜合前緣+中位數run 兩張(預設)")
    a = ap.parse_args(argv)
    os.environ["PARETO3D"] = a.pareto3d

    src_map = {}                                  # (kind, env, tier) -> [dir,...]
    for root in a.sources:
        for key, d in _discover(root).items():
            src_map.setdefault(key, []).append(d)
    keys = sorted(src_map, key=lambda k: (k[0], k[1], k[2] or ""))
    if a.envs:
        keys = [k for k in keys if k[1] in a.envs]
    if not keys:
        print("未在任何來源發現 experiment_out_pe_*_formal/ 或 experiment_out_mixed2opt_*/。"); return False

    summary_rows = []; summary_header = None
    for (kind, env, tier) in keys:
        dirs = src_map[(kind, env, tier)]
        if tier:                                  # 同步切換 live 層級(DRONE_TIER/DOMAIN/DELTA/DISK)→ meta 一致
            M.set_drone_tier(tier)
        tlabel = tier or "-"; _tsuf = f"_{tier}" if tier else ""
        print(f"\n===== [{kind}] {env}(tier={tlabel},{len(dirs)} 個來源)=====")

        if a.refigure_only:
            tgt = _first_with_results(dirs)
            if not tgt:
                print(f"[{kind}/{env}/{tlabel}] 無 results.pkl,略過。"); continue
            ok, msg = _refigure(kind, env, tgt, a); print(msg)
            tbl = os.path.join(tgt, f"compare_table{_tsuf}.csv")
        else:
            out = os.path.join(a.target, _PRE[kind] + env + _tsuf + _SUF[kind])
            n, conflicts, mism = _pool_ckpt(dirs, os.path.join(out, "ckpt"), expect_tier=tier)
            note = (f";⚠ 同格內容衝突 {len(conflicts)}:{conflicts[:3]}(版本/資料不一致?)" if conflicts else "")
            if mism:
                note += f";⚠ 層級不符已剔除 {len(mism)}:{mism[:3]}(ckpt drone_tier≠{tier})"
            print(f"[{kind}/{env}/{tlabel}] 彙整 ckpt 格 {n} 個{note}")
            ok = False; msg = ""
            if n > 0:
                ok, msg = _finalize(kind, env, out, a); print(msg)
            if not ok:
                src_rp = _first_with_results(dirs)
                if src_rp:
                    os.makedirs(out, exist_ok=True)
                    if os.path.abspath(src_rp) != os.path.abspath(out):
                        shutil.copy(os.path.join(src_rp, "results.pkl"), os.path.join(out, "results.pkl"))
                    ok, msg = _refigure(kind, env, out, a)
                    print(("(ckpt 未齊,改用現成 results.pkl 重繪)" if n > 0 else "(無 ckpt,用現成 results.pkl 重繪)"), msg)
                elif n == 0:
                    print(f"[{kind}/{env}/{tlabel}] 既無 ckpt 亦無 results.pkl,略過。"); continue
            tbl = os.path.join(out, f"compare_table{_tsuf}.csv")

        rows, hdr = _read_table(tbl, env, kind, tlabel)
        if hdr and summary_header is None:
            summary_header = hdr
        summary_rows.extend(rows)

    if summary_rows and summary_header:
        sp = os.path.join(a.target, "combined_summary.csv")
        with io.open(sp, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh); w.writerow(summary_header); w.writerows(summary_rows)
        n_cell = len(set((r[0], r[1], r[2]) for r in summary_rows))
        print(f"\n跨環境/實驗/層級總表 → {sp}({len(summary_rows)} 列,{n_cell} 個 實驗×環境×層級)")
    print("\n完成。")
    return True


if __name__ == "__main__":
    main()
