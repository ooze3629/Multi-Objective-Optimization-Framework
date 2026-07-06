# -*- coding: utf-8 -*-
"""
分環境預先建置並存檔繞行快取(route cache)
================================================
為每個實驗環境(九個字串:taiwan / japan / philippines / taiwan_real / taiwan_real_ddn /
japan_real / japan_real_ddn / philippines_real / philippines_real_ddn)預先算好 build_routes 會用到的
所有繞行頂點,存成 data/route_cache_<env>.pkl(含底圖指紋)。比對實驗開跑前載入,
即可省去重算、且確保各環境結果可重現。

涵蓋範圍(完整性):各環境皆為標準 Base→Patrol→Base(出海口路段僅合成 philippines 啟用,
其餘含 philippines_real(_ddn) 不使用出海口),故每艘船只巡訪「自己基地的候選點」並返航;
每個基地 b 需要的點對 = {home_b} ∪ candidates_b 之所有有序配對,對所有基地聯集即為全集。
本程式以此預熱,並做自我檢核:預熱後再大量產生隨機染色體,快取「零新增」才算完整。

用法:
    python build_route_caches.py                 # 九個環境全做(philippines 家族較久)
    python build_route_caches.py taiwan_real     # 只做指定環境
    python build_route_caches.py japan_real japan_real_ddn philippines_real philippines_real_ddn  # 補真實環境
載入(實驗端):
    M.set_environment("taiwan_real"); M.load_route_cache()    # 由 data/route_cache_<env>.pkl 載入(指紋核對)
"""
import sys, os, random
import MOGA_GPSIFF_patrol_clean as M

ENVS = ["taiwan", "japan", "philippines",
        "taiwan_real", "taiwan_real_ddn",
        "japan_real", "japan_real_ddn",
        "philippines_real", "philippines_real_ddn"]


def _anchor(env, b):
    """繞行錨點:synthetic philippines 為 Base→Exit→Patrol→Exit→Base,
    route_around 的 mission pair 以 exit point 為錨(非陸上 base);其餘環境用 base_coords。"""
    if env == "philippines" and M.exit_coords[b] is not None:
        return tuple(int(x) for x in M.exit_coords[b])
    return tuple(int(x) for x in M.base_coords[b])


def warm(env):
    s = M.set_environment(env)
    CPB = M.CANDIDATES_PER_BASE
    M._route_cache = {}            # 確保從空白開始
    for b in range(M.NUM_BASES):
        home = _anchor(env, b)
        cands = [tuple(int(x) for x in M.Patrol_Point[i]) for i in range(b * CPB, (b + 1) * CPB)]
        S = [home] + cands
        for p in S:
            for q in S:
                if p != q:
                    M.route_around(p, q)
    warmed = len(M._route_cache)
    detours = sum(1 for v in M._route_cache.values() if v)        # 需繞行(非直達)
    return s, warmed, detours


def completeness_check(env, n=400):
    """預熱後大量建構染色體,確認 build_routes 不再新增任何「非自環」快取鍵(=預熱完整)。
    自環鍵 (p,p)(空指派載具之退化 exit→exit)零成本、恆為 [],不計入完整性。"""
    before = len([k for k in M._route_cache if k[0] != k[1]])
    random.seed(20260606)
    built = 0
    for _ in range(n):
        c = M.init_chromosome()
        if c is None:
            continue
        built += 1
        # 再做一次 build_routes(等同實際評估路徑)
        M.build_routes(c)
    after = len([k for k in M._route_cache if k[0] != k[1]])
    return before, after, built


def main():
    targets = [a.lower() for a in sys.argv[1:]] or ENVS
    failed = []
    for env in targets:
        s, warmed, detours = warm(env)
        warm_snapshot = dict(M._route_cache)          # warm 純淨結果(不含 completeness 殘留)
        before, after, built = completeness_check(env)
        complete = (after == before)
        print(f"[{env}] bases={s['bases']} pts={s['points']} map={s['map']} "
              f"snap={s['snap_bases']}")
        print(f"    快取鍵={warmed}(需繞行 {detours},直達 {warmed - detours})")
        print(f"    完整性(非自環):預熱後 {before} → 建 {built} 條染色體後 {after} → "
              f"{'完整(零新增)' if complete else '不完整,新增 '+str(after-before)}")
        if not complete:
            print(f"    [FAIL] {env} 預熱不完整,未存檔。請檢查 warm() 之 mission pair 涵蓋(如出海口錨點)。")
            failed.append(env)
            continue
        M._route_cache = warm_snapshot                # 還原為 warm 純淨快照再存
        path, n = M.save_route_cache()
        rel = os.path.relpath(path, os.path.join(M.SCRIPT_DIR, ".."))
        print(f"    存檔:{rel}({n} 鍵)")
    if failed:
        print(f"\n[ERROR] 下列環境預熱不完整,未產生快取:{failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
