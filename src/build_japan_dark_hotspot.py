# -*- coding: utf-8 -*-
"""japan_real / japan_real_ddn 之 F1「暗船查緝(SAR 未匹配)」層 —— 烘入大和堆真實查緝熱點。

語意:GFW SAR『未匹配 AIS』暗船 presence 在本島周邊有覆蓋,但**日本海大和堆(Yamato Bank)**
      —— 日本海上保安廳年報明列之**外國漁船非法捕魚查緝重點**(較禁航區更貼合 F1「暗船查緝」軸)
      —— 在 GFW 暗船匯出格網中為 0(該海域 SAR 覆蓋稀疏),致此一現實上最關鍵的暗船查緝場
      在資料層缺席。本步以官方年報為據,於大和堆位置疊加一真實加權熱點,補回此查緝語意。

版本慣例(沿用海纜 v0→v1):
  scenario_JPreal_sar_dark_v0_gfwonly.npy  = 純 GFW 暗船 presence(本步首次執行時自動快照保存)
  scenario_JPreal_sar_dark_v1.npy          = v0_gfwonly ⊕ 大和堆加權熱點(本步輸出;baselines 載入此檔)

冪等:每次皆由 v0_gfwonly 重建,**不會**重複疊加;清掉本熱點即刪 _v1 後把 v0_gfwonly 複製回 _v1。

出處:
  - 日本海上保安廳《Annual Report 2026》(kaiho.mlit.go.jp):大和堆為外國漁船非法捕魚取締重點海域。
  - 位置:約 39°30′N / 134°30′E(日本海西部);本島 bbox(lon128–146/lat30–46)內。
  - 與能登 5009(MoD 禁區,no_go)空間鄰近但語意不同:5009 為「禁航」、大和堆為「查緝熱點」,二者分屬不同層。
Public portfolio version of the multi-objective patrol-planning project.
"""
import os, numpy as np
import MOGA_GPSIFF_patrol_clean as M

HERE = M.SCRIPT_DIR
DATA = os.path.join(HERE, "..", "data")
FIG  = os.path.join(HERE, "..", "figures")
BBOX = (128.0, 30.0, 146.0, 46.0)   # 與 build_real_maps / build_real_envs / GFW binning 同一組
H = W = 100

# 大和堆查緝熱點(自帶出處):中心經緯度 + 高斯足跡(以格為單位)+ 峰值權重。
# 峰值 1.0 → 與既有層最大值同級(年報定位為「查緝重點」,取與最強 GFW 熱點同級而非壓制之)。
YAMATO = {
    "name": "Yamato_Bank_大和堆",
    "lat": 39.5, "lon": 134.5,     # JCG Annual Report 2026:外國漁船非法捕魚查緝重點
    "sigma_cells": 2.2,            # ≈ ±2σ 直徑約 1.4°(~120–150 km),貼合堆+查緝場尺度
    "cap_cells": 5.0,              # 硬截半徑(超過即 0),保持局部化、不外溢沿岸
    "peak": 1.0,                   # 與層最大值同級
    "source": "JCG Annual Report 2026",
}


def _to_grid(lat, lon):
    lon0, lat0, lon1, lat1 = BBOX
    gx = (lon - lon0) / (lon1 - lon0) * W
    gy = (lat - lat0) / (lat1 - lat0) * H
    return gx, gy


def build():
    base_fp = os.path.join(DATA, "scenario_JPreal_sar_dark_v0_gfwonly.npy")
    v1_fp   = os.path.join(DATA, "scenario_JPreal_sar_dark_v1.npy")
    ng = np.load(os.path.join(HERE, "finalmap_japan_real.npy"))
    sea = (ng == 0)

    # 1) 取/建純 GFW base(冪等關鍵:永遠由 base 重建,不疊加在已加過的 _v1 上)
    if not os.path.exists(base_fp):
        cur = np.load(v1_fp).astype(np.float32)   # 首次執行:現有 _v1 即純 GFW,快照為 v0
        np.save(base_fp, cur)
        print(f"[snapshot] 純 GFW 暗船層存為 {os.path.basename(base_fp)}")
    base = np.load(base_fp).astype(np.float32)
    assert base.shape == (H, W)

    # 2) 大和堆高斯熱點(clip 到海域)
    cx, cy = _to_grid(YAMATO["lat"], YAMATO["lon"])
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    g = YAMATO["peak"] * np.exp(-r2 / (2.0 * YAMATO["sigma_cells"] ** 2))
    g[r2 > YAMATO["cap_cells"] ** 2] = 0.0
    g[~sea] = 0.0                                  # 不落陸/禁航

    # 3) 疊加:取較大值。base 與 g 皆已 ∈[0,1] 且 g 僅落海域,故 max 後天然 ∈[0,1],
    #    **無需重新正規化** → 除大和堆海域格外,其餘格(含 GFW base 原有之少數陸上惰性權重)
    #    與 base 逐位元組不變,變更純粹可稽核為「僅烘入大和堆」。
    #    (F1 覆蓋 ∩ 合法海域、ddn 融合 _norm 皆已歸零 no_go,陸上權重對結果為惰性。)
    out = np.maximum(base, g).astype(np.float32)
    assert out.max() <= 1.0 + 1e-6, "疊加後超出 [0,1],請檢查 peak/正規化"
    np.save(v1_fp, out)

    added = int(((out > 0) & (base <= 0)).sum())
    changed = int((out != base).sum())
    cxi, cyi = int(round(cx)), int(round(cy))
    print(f"[japan_real SAR-dark] 大和堆中心格 (x={cxi}, y={cyi})  峰值權重 {out[cyi, cxi]:.3f}")
    print(f"  純 GFW 非零 {int((base>0).sum())} → 烘入後非零 {int((out>0).sum())}  "
          f"(新增暗船查緝海域格 {added};**僅** {changed} 格與 base 不同 → 純大和堆變更)  max {out.max():.3f}")
    return out, (cxi, cyi)


def figure(out, center):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ng = np.load(os.path.join(HERE, "finalmap_japan_real.npy"))
    disp = np.ma.masked_where(ng != 0, out)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.where(ng != 0, 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=1)
    im = ax.imshow(disp, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax.scatter([center[0]], [center[1]], s=80, facecolors="none",
               edgecolors="cyan", linewidths=1.8, label="Yamato Bank 大和堆")
    ax.set_title("japan_real F1 暗船查緝 (SAR) + Yamato Bank hotspot")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight [0,1]")
    out_fp = os.path.join(FIG, "twreal_jp_sar_dark_yamato.png")
    fig.tight_layout(); fig.savefig(out_fp, dpi=120); plt.close(fig)
    print(f"[fig] {out_fp}")


if __name__ == "__main__":
    out, center = build()
    try:
        figure(out, center)
    except Exception as e:
        print("[fig] 跳過繪圖:", e)
