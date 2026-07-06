# -*- coding: utf-8 -*-
"""
GFW 船舶存在(AIS vessel presence)格網 → TWreal 真實熱點(100×100)
======================================================================
輸入:Global Fishing Watch 4Wings「AIS vessel presence」匯出檔(CSV 或 GeoJSON),
      需含每格中心之緯經度與一個數值欄(每格停留時數 hours / count)。
      常見欄名自動辨識:lat/latitude/Lat、lon/lng/longitude/Lon、
                        hours/value/vessel_hours/apparent_fishing_hours/count。
處理:用 real_geo_sources.Georeferencer 把 (lon,lat) 仿射映到 100×100 格,
      把落在同一格的數值加總,海域 clip、對數壓縮後正規化到 [0,1],得真實熱點。
產出:data/scenario_TWreal_ais_v1.npy、figures/twreal_ais_hotspot.png
之後可在 baselines._scenarios_taiwan_real 以此檔取代人造高斯(§9.3 第 2 步)。

用法:
  python gfw_to_hotspot.py /path/to/gfw_vessel_presence.csv
  python gfw_to_hotspot.py /path/to/gfw_export.geojson
"""
import os, sys, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import real_geo_sources as R
import MOGA_GPSIFF_patrol_clean as M

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "..", "data")
FIG = os.path.join(SCRIPT_DIR, "..", "figures")

LAT_KEYS = ["lat", "latitude", "cell_lat", "y"]
LON_KEYS = ["lon", "lng", "long", "longitude", "cell_lon", "x"]
VAL_KEYS = ["vessel presence hours", "vessel_presence_hours", "detections",
            "apparent fishing hours", "apparent_fishing_hours", "fishing hours",
            "hours", "value", "vessel_hours",
            "fishing_hours", "presence", "count", "hours_sum"]


def _pick(keys, available):
    al = {k.lower(): k for k in available}
    for k in keys:
        if k in al:
            return al[k]
    return None


def load_points(path):
    """回傳 (lon, lat, val) 三個 1D array。支援 CSV / GeoJSON。"""
    if path.lower().endswith((".geojson", ".json")):
        obj = json.load(open(path, encoding="utf-8"))
        feats = obj.get("features", obj if isinstance(obj, list) else [])
        lons, lats, vals = [], [], []
        for ft in feats:
            geom = ft.get("geometry", {}); props = ft.get("properties", ft)
            vk = _pick(VAL_KEYS, props.keys())
            v = float(props.get(vk, 1.0)) if vk else 1.0
            if geom.get("type") == "Point":
                lon, lat = geom["coordinates"][:2]
            else:  # polygon/cell: 取質心
                cs = np.array(geom.get("coordinates", []), dtype=object)
                pts = np.array(geom["coordinates"][0]) if geom.get("coordinates") else None
                if pts is None:
                    continue
                lon, lat = pts[:, 0].mean(), pts[:, 1].mean()
            lons.append(lon); lats.append(lat); vals.append(v)
        return np.array(lons), np.array(lats), np.array(vals)
    else:
        import csv
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        cols = rows[0].keys()
        lk, ok, vk = _pick(LAT_KEYS, cols), _pick(LON_KEYS, cols), _pick(VAL_KEYS, cols)
        if not (lk and ok):
            raise SystemExit(f"找不到緯經度欄;可用欄位={list(cols)}")
        lons = np.array([float(r[ok]) for r in rows])
        lats = np.array([float(r[lk]) for r in rows])
        vals = np.array([float(r[vk]) if vk and r[vk] not in ("", None) else 1.0 for r in rows])
        return lons, lats, vals


def accumulate_csv_stream(path, gr, H, W, chunk=200000):
    """串流(分塊)讀大型 CSV,直接累積進格點,避免一次載入記憶體(178MB/2M 列也安全)。"""
    import csv
    f = open(path, encoding="utf-8-sig", newline="")
    rd = csv.reader(f)
    header = next(rd)
    cols = {c.lower(): i for i, c in enumerate(header)}
    li = next((cols[k] for k in LAT_KEYS if k in cols), None)
    oi = next((cols[k] for k in LON_KEYS if k in cols), None)
    vi = next((cols[k] for k in VAL_KEYS if k in cols), None)
    if li is None or oi is None:
        raise SystemExit(f"找不到緯經度欄;可用欄位={header}")
    acc = np.zeros((H, W), float); nin = nout = 0
    buf_lat = []; buf_lon = []; buf_val = []

    def flush():
        nonlocal nin, nout
        if not buf_lat:
            return
        lat = np.array(buf_lat); lon = np.array(buf_lon); val = np.array(buf_val)
        g = gr.to_grid(lon, lat)
        gx = np.round(g[:, 0]).astype(int); gy = np.round(g[:, 1]).astype(int)
        m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        np.add.at(acc, (gy[m], gx[m]), val[m])
        nin += int(m.sum()); nout += int((~m).sum())
        buf_lat.clear(); buf_lon.clear(); buf_val.clear()

    for row in rd:
        try:
            buf_lat.append(float(row[li])); buf_lon.append(float(row[oi]))
            buf_val.append(float(row[vi]) if vi is not None and row[vi] not in ("", None) else 1.0)
        except (ValueError, IndexError):
            continue
        if len(buf_lat) >= chunk:
            flush()
    flush(); f.close()
    return acc, nin, nout


def finalize(acc, smooth_sigma):
    no_go = M.no_go_zone
    if smooth_sigma and smooth_sigma > 0:        # 僅低解析(0.1°)需平滑填補網格空隙;0.01° 設 0
        from scipy.ndimage import gaussian_filter
        acc = gaussian_filter(acc, sigma=smooth_sigma)
    acc = acc.copy(); acc[no_go != 0] = 0.0
    w = np.log1p(acc)                            # 對數壓縮(時數/偵測為重尾分布)
    if w.max() > 0:
        w = w / w.max()
    return np.clip(w, 0, 1)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python gfw_to_hotspot.py <gfw_export.csv|.geojson> [out_stem] [title] [smooth_sigma]")
    path = sys.argv[1]
    out_stem = sys.argv[2] if len(sys.argv) > 2 else "scenario_TWreal_ais_v1"
    title = sys.argv[3] if len(sys.argv) > 3 else "TWreal real hotspot from GFW AIS vessel presence"
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8   # 0.1°→0.8;0.01°→建議 0
    no_go = M.no_go_zone; H, W = no_go.shape
    gr = R.Georeferencer()
    if path.lower().endswith((".geojson", ".json")):
        lons, lats, vals = load_points(path)
        g = gr.to_grid(lons, lats)
        gx = np.round(g[:, 0]).astype(int); gy = np.round(g[:, 1]).astype(int)
        acc = np.zeros((H, W), float)
        m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        np.add.at(acc, (gy[m], gx[m]), vals[m]); nin = int(m.sum()); nout = int((~m).sum())
    else:
        acc, nin, nout = accumulate_csv_stream(path, gr, H, W)   # 串流,記憶體安全
    w = finalize(acc, sigma)
    np.save(os.path.join(DATA, out_stem + ".npy"), w)
    print(f"落在底圖內 {nin}、界外 {nout};熱點非零格 {int((w>0).sum())};smooth_sigma={sigma}")

    fig_name = "twreal_" + out_stem.replace("scenario_TWreal_", "").replace("_v1", "") + "_hotspot.png"
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(np.where(no_go == 1, 1, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=1.5)
    im = ax.imshow(np.where(no_go == 0, w, np.nan), origin="lower", cmap="hot_r", vmin=0, vmax=1, alpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized intensity")
    for (x, y) in M.base_ports:
        ax.plot(x, y, "*", ms=9, mfc="gold", mec="k", mew=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    out = os.path.join(FIG, fig_name)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("saved:", os.path.join(DATA, out_stem + ".npy"), "and", out)


if __name__ == "__main__":
    main()
