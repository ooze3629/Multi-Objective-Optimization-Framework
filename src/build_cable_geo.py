# -*- coding: utf-8 -*-
"""
海纜風險場 v1(真實幾何)重建器 — 多地區版(taiwan / japan / philippines)
==========================================================================
由 TeleGeography(submarinecablemap.com,CC BY-SA;經 GitHub 鏡像取得)之
「該地區子集海纜/登陸 GeoJSON」轉到 100×100 格,建構海纜風險場:

  風險 = max( 海域底值 0.10,
              鄰近最近纜線緩衝(≤3.5 格,線性遞減至 0.55),
              匯聚帶(該格 3.5 格內不同纜線數 / 最大數,峰值 1.0),
              登陸站高斯(振幅 0.6、σ 3) )
  → 海域 clip、正規化 [0,1]。

★ 多地區改動
  - **taiwan**:沿用 real_geo_sources.Georeferencer 仿射校正 + taiwan_real 海域遮罩,
    輸出 scenario_TWreal_cable_geo_v1.npy(與原版完全相同,確保可重現)。
  - **japan / philippines**:沒有仿射校正,改用「bbox 等距投影」把 lon/lat 線性映到
    100×100 格(gx=(lon-lon0)/(lon1-lon0)*W, gy=(lat-lat0)/(lat1-lat0)*H);
    **這組 bbox 必須與 gfw_fetch_*.py / AIS·SAR binning 同一組**,三層才會對齊。
    海域遮罩需以 --map 指向該地區的真實底圖(由 EEZ 幾何先建好的 finalmap_*real.npy)。

來源檔(預設,可用 --cable/--landing 覆寫):
    sources/mirror/S15_data/cable_geo_<region>.json
    sources/mirror/S15_data/landing_<region>.json
輸出:
    data/scenario_<TAG>_cable_geo_v1.npy   (TAG: taiwan→TWreal, japan→JPreal, philippines→PHreal)

用法:
    python build_cable_geo.py                                  # taiwan(原行為)
    python build_cable_geo.py --region japan --map ../code/finalmap_japan_real.npy
    python build_cable_geo.py --region philippines --map ../code/finalmap_philippines_real.npy \
        --cable some_cable.json --landing some_landing.json
    python build_cable_geo.py --region japan --bbox 128 30 146 46 --map ...   # 自訂框

注意:海纜場為「結構真實(走廊幾何/登陸站)、座標近似、權重模型推估」之風險 proxy,
非量測資料(與 AIS/SAR 之 GFW 量測不同層級;見 sources/SOURCES.md S15-geo)。
"""
import os, json, argparse
import numpy as np
import MOGA_GPSIFF_patrol_clean as M

R_BUF = 3.5
SRC = os.path.join(M.SCRIPT_DIR, "..", "sources", "mirror", "S15_data")

# ── 地區地理基準(lon_min, lat_min, lon_max, lat_max)── 與 gfw_fetch_*.py 同一組 ──
REGIONS = {
    "taiwan":      (119.0, 21.5, 122.6, 25.6),
    "japan":       (128.0, 30.0, 146.0, 46.0),
    "philippines": (116.0,  4.5, 127.0, 21.0),
}
TAG = {"taiwan": "TWreal", "japan": "JPreal", "philippines": "PHreal"}


class BBoxGeoref:
    """bbox 等距投影:lon/lat → 連續格座標(與 AIS/SAR binning 一致)。"""
    def __init__(self, bbox, W, H):
        self.lon0, self.lat0, self.lon1, self.lat1 = bbox
        self.W, self.H = W, H

    def to_grid(self, lon, lat):
        gx = (lon - self.lon0) / (self.lon1 - self.lon0) * self.W
        gy = (lat - self.lat0) / (self.lat1 - self.lat0) * self.H
        return gx, gy


def _densify_to_grid(gr, segs, H, W, step_deg=0.03):
    pts = []
    for ln in segs:
        for (x0, y0), (x1, y1) in zip(ln[:-1], ln[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step_deg))
            for t in np.linspace(0, 1, n):
                gx, gy = gr.to_grid(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
                if 0 <= gx < W and 0 <= gy < H:
                    pts.append((gx, gy))
    return np.array(pts) if pts else np.empty((0, 2))


def load_sea_mask(region, map_path):
    """回傳 (no_go, sea_bool)。taiwan 用 taiwan_real;其餘需 --map 指向真實底圖。"""
    if map_path:
        ng = np.load(map_path)
    elif region == "taiwan":
        M.set_environment("taiwan_real")
        ng = M.no_go_zone
    else:
        raise SystemExit(f"--region {region} 需要 --map 指向該地區真實底圖(finalmap_*real.npy)")
    return ng, (ng == 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="taiwan", choices=sorted(REGIONS))
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON0", "LAT0", "LON1", "LAT1"),
                    help="自訂經緯度框,覆寫 --region(僅 japan/philippines 等距投影用)")
    ap.add_argument("--map", dest="map_path", default=None,
                    help="該地區真實底圖 finalmap_*.npy(japan/philippines 必填)")
    ap.add_argument("--cable", default=None, help="海纜線 GeoJSON(預設 cable_geo_<region>.json)")
    ap.add_argument("--landing", default=None, help="登陸站 GeoJSON(預設 landing_<region>.json)")
    ap.add_argument("--out", default=None, help="輸出 .npy 路徑(預設 data/scenario_<TAG>_cable_geo_v1.npy);verify 用 temp 以免變動受管檔")
    a = ap.parse_args(argv)

    region = a.region
    bbox = tuple(a.bbox) if a.bbox else REGIONS[region]
    ng, sea = load_sea_mask(region, a.map_path)
    H, W = ng.shape

    # 座標→格:台灣用仿射校正(原行為);日本/菲律賓用 bbox 等距投影
    if region == "taiwan" and not a.bbox:
        import real_geo_sources as R
        gr = R.Georeferencer()
    else:
        gr = BBoxGeoref(bbox, W, H)

    cable_path = a.cable or os.path.join(SRC, f"cable_geo_{region}.json")
    landing_path = a.landing or os.path.join(SRC, f"landing_{region}.json")
    cab = json.load(open(cable_path, encoding="utf-8"))["features"]
    lds = json.load(open(landing_path, encoding="utf-8"))["features"]

    yy, xx = np.mgrid[0:H, 0:W]
    grid = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    corridor = np.zeros(H * W); ncab = np.zeros(H * W); ncables = 0
    for f in cab:
        geom = f["geometry"]
        # 支援 LineString 與 MultiLineString
        segs = geom["coordinates"]
        if geom.get("type") == "LineString":
            segs = [segs]
        pts = _densify_to_grid(gr, segs, H, W)
        if len(pts) == 0:
            continue
        ncables += 1
        best = np.full(H * W, np.inf)
        for i in range(0, len(pts), 64):
            ch = pts[i:i + 64]
            d = np.sqrt(((grid[:, None, :] - ch[None, :, :]) ** 2).sum(2)).min(1)
            best = np.minimum(best, d)
        corridor = np.maximum(corridor, np.clip(1 - best / R_BUF, 0, 1))
        ncab += (best <= R_BUF).astype(float)
    corridor = corridor.reshape(H, W)
    conv = (ncab / ncab.max() if ncab.max() > 0 else ncab).reshape(H, W)

    w = np.where(sea, 0.10, 0.0)
    w = np.maximum(w, 0.55 * corridor)
    w = np.maximum(w, conv)
    for f in lds:
        lo, la = f["geometry"]["coordinates"]
        gx, gy = gr.to_grid(lo, la)
        if 0 <= gx < W and 0 <= gy < H:
            w = np.maximum(w, 0.6 * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * 9.0)))
    w[~sea] = 0.0
    if w.max() > 0:
        w = w / w.max()
    w = w.astype(np.float32)

    out = a.out or os.path.join(M.SCRIPT_DIR, "..", "data", f"scenario_{TAG[region]}_cable_geo_v1.npy")
    np.save(out, w)
    print(f"[{region}] 海纜 v1:纜線 {ncables} 條、登陸 {len(lds)} 個 → 非零海域格 "
          f"{int((w[sea]>0).sum())}/{int(sea.sum())} mean={w[sea].mean():.3f} max={w.max():.2f}  存:{out}")


if __name__ == "__main__":
    main()
