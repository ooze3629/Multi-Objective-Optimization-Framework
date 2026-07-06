# -*- coding: utf-8 -*-
"""
建置「非漁船航運」真實層(乾淨版,取代正規化相減)
====================================================
原理:GFW `presence` 涵蓋全部 AIS 船;`fishing-effort` 是其中的漁船子集。
      非漁船航運(時數) = 總船舶存在(時數) − 視在漁撈(時數),**原始時數相減、單位一致**,
      再走與 S21/S23 完全相同的 finalize(log1p→smooth→海域clip→正規化[0,1]),確保三層對齊可比。

Prerequisite: prepare the GFW input CSV files externally:兩個原始 CSV,**同期間/同區域/同解析度**:
  1) 總船舶存在  public-global-presence:v4.0    例:gfw_fetch_presence.py --start 2025-06-01 --end 2026-06-01 --res HIGH
     欄位含「Vessel Presence Hours」(可 group-by flag,本程式會自動跨 flag 累加到每格)。
  2) 視在漁撈    public-global-fishing-effort:v4.0  同 S23 參數(date 2025-06-01..2026-06-01,HIGH 0.01°,group-by flagAndGearType)
     欄位含「Apparent Fishing Hours」。

用法:
  python build_nonfishing_layer.py <presence.csv> <fishing.csv> [out_stem]
  預設 out_stem = scenario_TWreal_shipping_v1  → 存 data/scenario_TWreal_shipping_v1.npy
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

import MOGA_GPSIFF_patrol_clean as M
import real_geo_sources as R
from gfw_to_hotspot import accumulate_csv_stream, finalize, DATA, FIG


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法: python build_nonfishing_layer.py <presence.csv> <fishing.csv> [out_stem]")
    pres_csv, fish_csv = sys.argv[1], sys.argv[2]
    out_stem = sys.argv[3] if len(sys.argv) > 3 else "scenario_TWreal_shipping_v1"
    for p in (pres_csv, fish_csv):
        if not os.path.exists(p):
            raise SystemExit(f"找不到檔案:{p}")

    M.set_environment("taiwan_real")
    no_go = M.no_go_zone; H, W = no_go.shape
    gr = R.Georeferencer()

    # 原始時數格網(同一套 Georeferencer;跨 flag/geartype 自動累加到每格)
    pres_acc, pin, pout = accumulate_csv_stream(pres_csv, gr, H, W)
    fish_acc, fin, fout = accumulate_csv_stream(fish_csv, gr, H, W)
    print(f"presence 落圖內 {pin} 界外 {pout}  總時數Σ={pres_acc.sum():.0f}")
    print(f"fishing  落圖內 {fin} 界外 {fout}  總時數Σ={fish_acc.sum():.0f}")

    # 非漁船航運 = 總存在 − 視在漁撈(原始時數;負值歸零=漁撈估計局部超過總量的誤差)
    neg = float(np.clip(fish_acc - pres_acc, 0, None).sum())
    nonfish_hours = np.clip(pres_acc - fish_acc, 0, None)
    print(f"漁撈>存在 之溢出時數(歸零處理)={neg:.0f}  非漁航運總時數Σ={nonfish_hours.sum():.0f}")

    # 與 S21/S23 相同後處理(0.01° 用 sigma=0)
    w = finalize(nonfish_hours, 0.0)
    out_npy = os.path.join(DATA, out_stem + ".npy")
    np.save(out_npy, w)
    sea = (no_go == 0)
    print(f"非漁航運層:非零海域格 {int((w[sea] > 0).sum())} / {int(sea.sum())}  "
          f"mean={w[sea].mean():.3f} max={w.max():.2f}")

    # 與其他真實場相關(快速獨立性檢查)
    try:
        ais = np.load(os.path.join(DATA, "scenario_TWreal_ais_v1.npy"))
        fish = np.load(os.path.join(DATA, "scenario_TWreal_fishing_v1.npy"))
        dark = np.load(os.path.join(DATA, "scenario_TWreal_sar_dark_v1.npy"))
        c = lambda a, b: round(float(np.corrcoef(a[sea], b[sea])[0, 1]), 2)
        print(f"相關: 航運↔AIS={c(w,ais)} 航運↔漁業={c(w,fish)} 航運↔暗船={c(w,dark)}")
    except Exception as e:
        print("(相關檢查略過:", e, ")")

    # 圖
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(np.where(no_go == 1, 0.55, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=1)
    im = ax.imshow(np.where(sea, w, np.nan), origin="lower", cmap="hot_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized intensity")
    ax.set_title("TWreal 非漁船航運層 v1(presence − fishing,原始時數)", fontsize=11)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    fig_out = os.path.join(FIG, "twreal_shipping_hotspot.png")
    fig.tight_layout(); fig.savefig(fig_out, dpi=120)
    print("saved:", out_npy, "and", fig_out)


if __name__ == "__main__":
    main()
