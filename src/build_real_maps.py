# -*- coding: utf-8 -*-
"""建置 japan_real / philippines_real 真實海陸底圖(可重跑)。
no_go = 真實陸地(Natural Earth 50m,bbox 內柵格化;含外國海岸,避免穿陸)
        + 真實禁航區多邊形(已烘入,見 RESTRICTED;v1.4 起生效)
輸出 finalmap_<country>_real.npy(0=可航行海域,1=陸/禁航)。

現況真實禁航區(已納入 RESTRICTED,v1.4):
  日本(_JP_EX):JCG NtM No.1/2026 之 11 個 in-bbox MoD 演習・危險區。
  菲律賓(_PH_TUBBATAHA_CORE):Tubbataha 核心區(RA 10067 §4/§19)。
  台灣(taiwan_real):經官方文件審查結論無可烘入項,no_go 維持「僅真實陸地」。
若日後要新增限制區:於 RESTRICTED[country] 增 [(lon,lat) 多邊形],重跑本腳本 → 重建底圖,
  再依序重跑 build_cable_geo / AIS·SAR binning / build_real_envs(候選點)即可。

來源:sources/mirror/ne_50m_land.geojson(Natural Earth,公有領域)。
bbox 與 gfw_fetch_*.py / build_cable_geo.py / build_real_envs.py 一致。
"""
import os, json, numpy as np
from matplotlib.path import Path
import MOGA_GPSIFF_patrol_clean as M

HERE = M.SCRIPT_DIR
LAND = os.path.join(HERE, "..", "sources", "mirror", "ne_50m_land.geojson")
BBOX = {"japan": (128.0, 30.0, 146.0, 46.0), "philippines": (116.0, 4.5, 127.0, 21.0)}
# ★真實禁航區多邊形(lon,lat)。
# 日本:MoD 火砲/轟炸演習・危險區,座標取自 JCG 航船布告 No.1/2026(2026 年 2 月演習排程),
#   經 bbox(lat 30–46 / lon 128–146)篩選後之 11 個 in-bbox 區(沖繩 5017–19、小笠原 5014 等外海者已排除)。
#   屬臨時/週期性指定危險區,本模型以「常設應避航 no_go」抽象之;清空 _JP_EX 即還原為僅陸地 no_go。
def _dms(s):
    h = s[-1]; p = s[:-1].split("-")
    v = float(p[0]) + float(p[1]) / 60 + (float(p[2]) / 3600 if len(p) > 2 else 0.0)
    return -v if h in "SW" else v
_JP_EX = {  # name: [(lat,lon) DMS]，MoD via JCG NtM No.1/2026
    "5005_Erimo_W":       [("42-04-09N","142-16-46E"),("41-44-09N","142-57-46E"),("41-27-10N","142-42-46E"),("41-45-39N","142-05-17E"),("41-59-09N","142-03-47E")],
    "5006_Erimo_SW":      [("41-43-09N","142-59-46E"),("41-20-10N","142-59-46E"),("41-20-10N","142-07-47E"),("41-45-39N","142-05-17E"),("41-27-10N","142-42-46E"),("41-44-09N","142-57-46E")],
    "5007_Erimo_SSW":     [("41-40-45N","143-26-26E"),("41-33-10N","143-29-46E"),("41-10-10N","143-19-46E"),("41-10-10N","142-09-47E"),("41-20-10N","142-07-47E"),("41-20-10N","142-59-46E"),("41-38-14N","142-59-46E")],
    "5008_TsunoShima_NW": [("34-08-52N","130-29-01E"),("34-16-57N","130-12-37E"),("34-51-11N","130-35-06E"),("34-43-31N","130-52-01E")],
    "5009_Noto_NW":       [("37-14-11N","136-09-49E"),("36-33-11N","134-44-50E"),("37-40-10N","133-24-50E"),("38-33-10N","134-01-50E"),("39-27-10N","136-09-49E")],
    "5010_Hachinohe_ENE": [("41-10-10N","142-09-47E"),("41-10-10N","143-19-46E"),("40-53-10N","143-13-46E"),("40-44-10N","142-59-46E"),("40-50-10N","142-59-46E"),("40-50-10N","142-10-47E")],
    "5011_Hachinohe_E":   [("40-24-10N","142-13-47E"),("40-50-10N","142-10-47E"),("40-50-10N","142-59-46E"),("40-44-10N","142-59-46E"),("40-24-10N","142-32-47E")],
    "5012_KashimaNada":   [("36-00-12N","141-04-48E"),("36-40-11N","141-04-48E"),("36-40-11N","141-20-48E"),("36-00-12N","141-20-48E")],
    "5013_Inubo_NE":      [("36-05-00N","141-20-48E"),("36-38-36N","141-20-48E"),("36-40-43N","142-10-46E"),("36-09-59N","141-59-52E"),("36-05-00N","141-46-04E")],
    "5015_Goto_S":        [("31-47-12N","128-45-52E"),("32-20-12N","128-45-52E"),("32-20-12N","129-09-52E"),("31-47-12N","129-09-52E")],
    "5016_Toi_ENE":       [("31-30-43N","132-09-21E"),("32-00-13N","132-34-51E"),("32-09-13N","132-59-51E"),("31-48-13N","132-59-51E"),("32-02-13N","133-29-51E"),("31-42-13N","133-29-51E"),("31-04-13N","132-07-51E"),("31-25-13N","132-07-51E")],
}
# 菲律賓:Tubbataha 群礁自然公園核心區(RA 10067 §4,PRS92;§19 明定 off-limits to navigation;
#   並經 IMO 劃為 PSSA/應避航區 ATBA)。10 浬緩衝(§5)屬特別管制、非航行禁止,僅記錄於 provenance,不烘入。
_PH_TUBBATAHA_CORE = [  # (lat,lon) DMS,RA 10067 §4(約 97,030 ha)
    ("9-04-52N","119-46-10E"),("9-06-05N","119-48-22E"),("8-58-09N","120-03-12E"),
    ("8-53-29N","120-03-30E"),("8-41-33N","119-50-41E"),("8-43-09N","119-45-46E"),
]
RESTRICTED = {
    "japan": [[(_dms(lo), _dms(la)) for la, lo in v] for v in _JP_EX.values()],
    "philippines": [[(_dms(lo), _dms(la)) for la, lo in _PH_TUBBATAHA_CORE]],
}


def _polys(feat):
    g = feat["geometry"]
    return [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]


def build(country, H=100, W=100):
    lon0, lat0, lon1, lat1 = BBOX[country]
    xs = lon0 + (np.arange(W) + 0.5) / W * (lon1 - lon0)
    ys = lat0 + (np.arange(H) + 0.5) / H * (lat1 - lat0)
    LON, LAT = np.meshgrid(xs, ys)
    pts = np.stack([LON.ravel(), LAT.ravel()], 1)
    land = json.load(open(LAND, encoding="utf-8"))["features"]
    mask = np.zeros(H * W, bool)
    for f in land:
        for poly in _polys(f):
            o = np.asarray(poly[0])
            if (o[:, 0].max() < lon0 - 0.2 or o[:, 0].min() > lon1 + 0.2 or
                    o[:, 1].max() < lat0 - 0.2 or o[:, 1].min() > lat1 + 0.2):
                continue
            mask |= Path(o).contains_points(pts)
    for poly in RESTRICTED.get(country, []):       # 禁航區(已烘入;清空 RESTRICTED 即還原為僅陸地)
        mask |= Path(np.asarray(poly)).contains_points(pts)
    ng = mask.reshape(H, W).astype(np.int16)
    out = os.path.join(HERE, f"finalmap_{country}_real.npy")
    np.save(out, ng)
    print(f"[{country}_real] 陸/禁 {int(ng.sum())}/{H*W}  海 {int((ng==0).sum())}  "
          f"禁航多邊形 {len(RESTRICTED.get(country, []))} 個  → {out}")


if __name__ == "__main__":
    for c in ["japan", "philippines"]:
        build(c)
