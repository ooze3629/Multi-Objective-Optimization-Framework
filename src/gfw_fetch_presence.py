# -*- coding: utf-8 -*-
"""
======================================================================
抓指定海域之 AIS 船舶存在(每格停留時數),輸出成 lat,lon,hours 的 CSV,
直接餵給 code/gfw_to_hotspot.py 產生真實熱點。

★ 本版改為多地區:用 --region 選 taiwan / japan / philippines,或用 --bbox 自訂。
  REGIONS 內的 bbox 就是「該地區 100×100 底圖的地理基準」——之後 gfw_to_hotspot
  必須用『同一組 bbox』把 lat/lon 對映到格網,三層(AIS/暗船/海纜)與基地才會對齊。

安全:**token 不要寫進程式,也不要貼給任何人**。請設環境變數:
    macOS/Linux:  export GFW_TOKEN="你的token"
    Windows PS :  $env:GFW_TOKEN="你的token"
然後:
    python gfw_fetch_presence.py --region taiwan          # 近 12 個月、LOW(0.1°)
    python gfw_fetch_presence.py --region japan --res HIGH  # 0.01° 較細(較大較慢)
    python gfw_fetch_presence.py --region philippines --start 2025-01-01 --end 2025-12-31
    python gfw_fetch_presence.py --bbox 116 4.5 127 21      # 自訂 lon_min lat_min lon_max lat_max
    python gfw_fetch_presence.py --region japan --dry-run    # 只印請求,不呼叫
產出:
    gfw_presence_<region>.csv  (lat,lon,hours) ← 上傳這個給我
    gfw_presence_<region>_raw.json  (原始回應,備查)

依 GFW API v3 文件(4Wings report):
  POST https://gateway.api.globalfishingwatch.org/v3/4wings/report
  dataset = public-global-presence:latest;hours = 每格船舶存在時數;
  spatial-aggregation=false + temporal-resolution=ENTIRE → 每格一列 {lat,lon,hours}。
  date-range 上限 366 天;同一帳號同時間僅允許一份報告(否則 429)。
Reference: Global Fishing Watch; use only under the applicable non-commercial/academic terms.
"""
import os, sys, json, time, argparse, datetime, urllib.parse, urllib.request, urllib.error

BASE = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-presence:latest"

# ── 地區地理基準(lon_min, lat_min, lon_max, lat_max)──────────────────────────
# 注意:這組 bbox 是該地區 100×100 底圖的地理基準,gfw_to_hotspot / 海纜子集 / 暗船事件
# 都要沿用同一組,否則三層與基地對不齊。台灣為原既有設定。
REGIONS = {
    "taiwan":      (119.0, 21.5, 122.6, 25.6),   # 海峽 + 東部(原設定)
    "japan":       (128.0, 30.0, 146.0, 46.0),   # 本州/四國/九州/北海道周邊(不含沖繩)
    "philippines": (116.0,  4.5, 127.0, 21.0),   # 全群島 + 西菲律賓海 + 東側太平洋
}


def aoi_polygon(bbox):
    lon0, lat0, lon1, lat1 = bbox
    return {"type": "Polygon", "coordinates": [[
        [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]
    ]]}


def build_request(aoi, start, end, res, fmt="JSON"):
    params = [
        ("spatial-resolution", res),          # LOW=0.1°, HIGH=0.01°
        ("spatial-aggregation", "false"),     # 要每格,不要整體加總
        ("temporal-resolution", "ENTIRE"),    # 整段時間每格一個值
        ("datasets[0]", DATASET),
        ("date-range", f"{start},{end}"),
        ("format", fmt),
    ]
    url = BASE + "?" + urllib.parse.urlencode(params, safe="[]:,")
    body = json.dumps({"geojson": aoi}).encode("utf-8")
    return url, body


# 瀏覽器型 UA:GFW gateway 在 Cloudflare 後,預設 Python-urllib UA 會被
# Browser Integrity Check 擋(HTTP 403 / error code 1010)。帶上一般瀏覽器 UA 即可放行。
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def post(url, body, token):
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _UA,
    })
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def poll_last_report(token, tries=20, wait=15):
    """524/長時間報告:用 last-report 端點輪詢取回。"""
    url = "https://gateway.api.globalfishingwatch.org/v3/4wings/last-report"
    for _ in range(tries):
        time.sleep(wait)
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError:
            continue
        if isinstance(data, dict) and data.get("status") == "running":
            print("  ...報告產製中,續候"); continue
        return data
    raise SystemExit("last-report 逾時")


def flatten(resp):
    """從 entries[0][dataset_version] 取 {lat,lon,hours} 列。"""
    rows = []
    for entry in resp.get("entries", []):
        for _ver, recs in entry.items():
            if isinstance(recs, list):
                for rec in recs:
                    if "lat" in rec and "lon" in rec:
                        rows.append((rec["lat"], rec["lon"], rec.get("hours", rec.get("vesselIDs", 1))))
    return rows


def resolve_region(a):
    """回傳 (tag, bbox)。--bbox 優先於 --region。"""
    if a.bbox:
        return "custom", tuple(a.bbox)
    return a.region, REGIONS[a.region]


def main():
    ap = argparse.ArgumentParser()
    end_default = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
    start_default = (datetime.date.today() - datetime.timedelta(days=6 + 365)).isoformat()
    ap.add_argument("--region", default="taiwan", choices=sorted(REGIONS),
                    help="地區預設 bbox(taiwan/japan/philippines)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON0", "LAT0", "LON1", "LAT1"),
                    help="自訂經緯度框,覆寫 --region")
    ap.add_argument("--start", default=start_default)
    ap.add_argument("--end", default=end_default)
    ap.add_argument("--res", default="LOW", choices=["LOW", "HIGH"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tag, bbox = resolve_region(a)
    aoi = aoi_polygon(bbox)
    out_csv = f"gfw_presence_{tag}.csv"
    out_raw = f"gfw_presence_{tag}_raw.json"

    url, body = build_request(aoi, a.start, a.end, a.res)
    if a.dry_run:
        print(f"地區={tag}  bbox(lon0,lat0,lon1,lat1)={bbox}")
        print("POST", url)
        print("BODY", body.decode())
        print("HEADER Authorization: Bearer <GFW_TOKEN>")
        print("輸出將為:", out_csv)
        return

    token = os.environ.get("GFW_TOKEN")
    if not token:
        raise SystemExit("請先設定環境變數 GFW_TOKEN(勿寫進程式/勿外傳)")
    print(f"地區={tag} bbox={bbox}  抓取 {a.start}..{a.end}  解析度={a.res} …(同帳號同時只能一份報告)")
    try:
        resp = post(url, body, token)
    except urllib.error.HTTPError as e:
        if e.code == 524:
            print("  524 逾時,改用 last-report 輪詢取回…")
            resp = poll_last_report(token)
        elif e.code == 429:
            raise SystemExit("429:你已有報告在執行,稍後再試(同帳號同時僅一份)")
        else:
            raise SystemExit(f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:500]}")

    json.dump(resp, open(out_raw, "w", encoding="utf-8"), ensure_ascii=False)
    rows = flatten(resp)
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["lat", "lon", "hours"]); w.writerows(rows)
    print(f"完成:{len(rows)} 格 → {out_csv}(上傳這個給我)")
    if not rows:
        print(f"⚠ 無資料列;檢查 token、日期範圍(<=366 天)、或改 --res,並看 {out_raw}")


if __name__ == "__main__":
    main()
