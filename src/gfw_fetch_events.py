# -*- coding: utf-8 -*-
"""
GFW 行為事件擷取器(可疑行為/對峙風險、暗船查緝熱點用) — v2,抗逾時
====================================================================
抓指定海域的 Encounter / Loitering / GAP(AIS-off)事件,每筆帶經緯度,
輸出 lat,lon,type 的 CSV,再餵給 gfw_to_hotspot.py 做熱點(每格事件數)。

★ 本版改為多地區:用 --region 選 taiwan / japan / philippines,或用 --bbox 自訂。
  REGIONS 內的 bbox 必須與 gfw_fetch_presence.py 同一組(同地區同框),三層才對齊。

v2 修正逾時:① client timeout 拉長到 300s;② 時間切成「逐月」查(每次請求輕很多);
③ 逾時/暫時性錯誤自動重試;④ 單頁筆數降到 500。

安全:token 設環境變數 GFW_TOKEN,勿寫進程式/勿外傳。
相依:pip install gfw-api-python-client

用法:
    export GFW_TOKEN="你的token"                      # Windows PowerShell: $env:GFW_TOKEN="..."
    python gfw_fetch_events.py --region taiwan        # 近 12 個月、三類事件
    python gfw_fetch_events.py --region japan --types ENCOUNTER
    python gfw_fetch_events.py --region philippines --start 2025-06-01 --end 2026-06-01
    python gfw_fetch_events.py --bbox 116 4.5 127 21  # 自訂 lon_min lat_min lon_max lat_max
    python gfw_fetch_events.py --region japan --dry-run
產出:gfw_events_<region>.csv (lat,lon,type)
"""
import os, sys, csv, asyncio, argparse, datetime

# ── 地區地理基準(lon_min, lat_min, lon_max, lat_max)── 與 gfw_fetch_presence 同 ──
REGIONS = {
    "taiwan":      (119.0, 21.5, 122.6, 25.6),
    "japan":       (128.0, 30.0, 146.0, 46.0),
    "philippines": (116.0,  4.5, 127.0, 21.0),
}
DATASET = {
    "ENCOUNTER": "public-global-encounters-events:latest",
    "LOITERING": "public-global-loitering-events:latest",
    "GAP":       "public-global-gaps-events:latest",   # AIS-off
}
PAGE = 500
RETRY = 3


def aoi_polygon(bbox):
    lon0, lat0, lon1, lat1 = bbox
    return {"type": "Polygon", "coordinates": [[
        [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]
    ]]}


def month_windows(start, end):
    """把 [start,end] 切成逐月 (s,e) 字串窗格。"""
    s = datetime.date.fromisoformat(start); e = datetime.date.fromisoformat(end)
    out = []
    cur = s
    while cur < e:
        if cur.month == 12:
            nxt = datetime.date(cur.year + 1, 1, 1)
        else:
            nxt = datetime.date(cur.year, cur.month + 1, 1)
        out.append((cur.isoformat(), min(nxt, e).isoformat()))
        cur = nxt
    return out


def _latlon(item):
    pos = getattr(item, "position", None) or (item.get("position") if isinstance(item, dict) else None)
    if pos is None:
        return None, None
    lat = getattr(pos, "lat", None); lon = getattr(pos, "lon", None)
    if lat is None and isinstance(pos, dict):
        lat, lon = pos.get("lat"), pos.get("lon")
    return lat, lon


async def _req(client, ds, s, e, offset, aoi):
    last = None
    for k in range(RETRY):
        try:
            return await client.events.get_all_events(
                datasets=[ds], start_date=s, end_date=e,
                geometry=aoi, limit=PAGE, offset=offset,
            )
        except Exception as ex:                      # 逾時/暫時錯誤 → 退避重試
            last = ex
            await asyncio.sleep(5 * (k + 1))
    raise last


async def fetch_type(client, etype, start, end, seen, aoi):
    rows = []
    for (ms, me) in month_windows(start, end):
        offset = 0
        while True:
            res = await _req(client, DATASET[etype], ms, me, offset, aoi)
            items = res.data()
            if not isinstance(items, list):
                items = [items] if items else []
            if not items:
                break
            for it in items:
                eid = getattr(it, "id", None) or (it.get("id") if isinstance(it, dict) else None)
                key = (eid or "").split(".")[0]
                if key and key in seen:
                    continue
                seen.add(key)
                lat, lon = _latlon(it)
                if lat is not None and lon is not None:
                    rows.append((lat, lon, etype))
            if len(items) < PAGE:
                break
            offset += PAGE
        print(f"  {etype:9s} {ms}..{me}: 累積 {len(rows)}")
    print(f"  {etype:9s} 合計 {len(rows)} 筆")
    return rows


async def run(types, start, end, aoi, out_csv):
    import gfwapiclient as gfw
    token = os.environ.get("GFW_TOKEN")
    if not token:
        raise SystemExit("請先設定環境變數 GFW_TOKEN(勿寫進程式/勿外傳)")
    client = gfw.Client(access_token=token, timeout=300.0, connect_timeout=30.0)
    allrows = []; seen = set()
    for et in types:
        allrows += await fetch_type(client, et, start, end, seen, aoi)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["lat", "lon", "type"]); w.writerows(allrows)
    print(f"完成:{len(allrows)} 筆事件 → {out_csv}(上傳給我)")


def resolve_region(a):
    if a.bbox:
        return "custom", tuple(a.bbox)
    return a.region, REGIONS[a.region]


def main():
    ap = argparse.ArgumentParser()
    end_d = (datetime.date.today() - datetime.timedelta(days=4)).isoformat()
    start_d = (datetime.date.today() - datetime.timedelta(days=4 + 365)).isoformat()
    ap.add_argument("--region", default="taiwan", choices=sorted(REGIONS),
                    help="地區預設 bbox(taiwan/japan/philippines)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON0", "LAT0", "LON1", "LAT1"),
                    help="自訂經緯度框,覆寫 --region")
    ap.add_argument("--start", default=start_d)
    ap.add_argument("--end", default=end_d)
    ap.add_argument("--types", nargs="+", default=["ENCOUNTER", "LOITERING", "GAP"],
                    choices=["ENCOUNTER", "LOITERING", "GAP"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tag, bbox = resolve_region(a)
    aoi = aoi_polygon(bbox)
    out_csv = f"gfw_events_{tag}.csv"

    if a.dry_run:
        print(f"DRY-RUN  地區={tag}  bbox(lon0,lat0,lon1,lat1)={bbox}")
        for et in a.types:
            print(f"  {et} -> {DATASET[et]}")
        print("  date-range:", a.start, "..", a.end, "(逐月切窗)")
        print("  月窗數:", len(month_windows(a.start, a.end)))
        print("  timeout=300s, page=500, retry=3")
        print("  輸出將為:", out_csv)
        return
    asyncio.run(run(a.types, a.start, a.end, aoi, out_csv))


if __name__ == "__main__":
    main()
