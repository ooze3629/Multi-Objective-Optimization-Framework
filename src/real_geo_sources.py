# -*- coding: utf-8 -*-
"""
真實禁航/限制區地理來源(WGS84)與經緯度→格點近似校正
======================================================================
This module encodes public-source「交通部航港局航船布告」「空軍實彈射擊報告單」「離岸風場
施工作業計畫」「彰化風場航行空間船舶交通服務指南」中之 **實際 WGS84 座標多邊形**
編碼為可程式化幾何,並提供:

  1. DMS / 十進位分(decimal-minute)→ 十進位度 解析。
  2. 圓(中心+半徑 NM)→ 多邊形近似。
  3. 由 MOGA_GPSIFF_patrol_clean._BASES['taiwan'] 之 13 個基地港,
     最小平方擬合 (lon,lat)→(x,y) 仿射(affine)轉換 — 因台灣底圖為 100×100
     示意座標(非經緯度換算),此為「近似地理校正」,殘差另行報告。
  4. 將真實多邊形轉到格點、再點陣化(rasterize)成布林遮罩。

真實性宣告(務必照 EVAL_PROTOCOL §9.2 載明):
  - 幾何「座標」為官方公告之 WGS84 實際值(真實);
  - 但底圖為示意座標,故疊到 100×100 之「定位」為仿射近似(±約 1~2 格 ≈ 5~10 km),
    非精確地理疊合;此屬 §9.4「座標經緯度校正」之過渡版本(real v1)。
  - 軍事射擊/限制區僅取公開航船布告之公告座標,屬公開近似資料。

來源(逐項見 sources/SOURCES.md S13、S17–S20;鏡像於 sources/mirror/):
  - S17 空軍 115 年 7 月實彈射擊報告單(RCR-6/7/9/11/12/17/38/42),航船布告 318。
  - S18 渢妙(FEM1)離岸風場水下基礎保護工程安全區,航船布告 304。
  - S19 大肚一號離岸風場地球物理勘測作業區,航船布告 324。
  - S20 彰化風場航行空間船舶交通服務指南(TSS 分道通航 + 海纜/管線 + 禁錨),航港局 MPB。
"""
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# 座標解析
# ----------------------------------------------------------------------
def dms(d, m=0, s=0.0):
    """度分秒 → 十進位度。"""
    return d + m / 60.0 + s / 3600.0


def dm(d, mm):
    """度 + 十進位分(航船 VTS 指南格式,如 24°09'.96 = 24 度 09.96 分)→ 十進位度。"""
    return d + mm / 60.0


NM_DEG = 1.0 / 60.0   # 1 海浬 ≈ 1/60 度緯度


def circle_polygon(lon_c, lat_c, radius_nm, n=48):
    """以(lon,lat)為心、radius_nm 海浬為半徑之圓 → 多邊形(經度依緯度作 cos 修正)。"""
    r_lat = radius_nm * NM_DEG
    r_lon = r_lat / max(0.2, np.cos(np.radians(lat_c)))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[lon_c + r_lon * np.cos(t), lat_c + r_lat * np.sin(t)]


# ----------------------------------------------------------------------
# 幾何資料(全部 WGS84,(lon,lat) 十進位度)
# ----------------------------------------------------------------------
# ---- 空軍 7 月實彈射擊危險區(S17;航船布告 318 + 報告單附件)----
#   時間:115/7 月 1-3、6-10、13-17、20-24、27-31 日(日間時段,各區略異)。
#   圓形者(center+radius)/多邊形者(verts)。半徑/扇形之細節於 note 說明,
#   小型外海圓區(2.6~5NM)在 ~5km/格 之解析度下近似為小圓盤。
FIRING_ZONES = [
    {"id": "RCR-6",  "kind": "circle", "center": (dms(120,35,29), dms(22,26,54)),
     "radius_nm": 5.0, "src": "S17", "note": "中心半徑 5NM。"},
    {"id": "RCR-7",  "kind": "circle", "center": (dms(119,20,33), dms(23,18,18)),
     "radius_nm": 2.6, "src": "S17",
     "note": "核心 2.6NM(0-15,000ft)另含 6.5NM 西半扇形(除去 360°順→180°);此處取 2.6NM 核心近似。"},
    {"id": "RCR-9",  "kind": "poly", "src": "S17", "note": "四點多邊形,最小半徑約 31NM。",
     "verts": [(dms(120,48), dms(25,48)), (dms(121,20), dms(25,48)),
               (dms(120,23), dms(24,52)), (dms(119,55), dms(24,52))]},
    {"id": "RCR-11", "kind": "poly", "src": "S17", "note": "A/B/C/D 四點(臺灣海峽中西,台中/彰化外海以西)。",
     "verts": [(dms(119,50), dms(24,48)), (dms(120,20), dms(24,48)),
               (dms(119,27), dms(23,55)), (dms(119,8),  dms(24,10))]},
    {"id": "RCR-12", "kind": "poly", "src": "S17", "note": "四點(澎湖南/海峽西南)。",
     "verts": [(dms(118,55), dms(22,34)), (dms(119,39), dms(22,59)),
               (dms(119,37), dms(22,16)), (dms(118,55), dms(22,21))]},
    {"id": "RCR-17", "kind": "poly", "src": "S17", "note": "A/B/C/D 四點(台灣東方太平洋)。",
     "verts": [(dms(122,5), dms(23,43)), (dms(122,38), dms(23,43)),
               (dms(122,38), dms(23,20)), (dms(122,5),  dms(23,0))]},
    {"id": "RCR-38", "kind": "circle", "center": (dms(120,7,59), dms(23,29,54)),
     "radius_nm": 5.0, "src": "S17",
     "note": "中心 5NM,除去 120°10'59\"E 以東;此處取 5NM 圓近似(嘉義/雲林外海)。"},
    {"id": "RCR-42", "kind": "poly", "src": "S17", "note": "A/B/C/D 四點(台東/蘭嶼東南外海)。",
     "verts": [(dms(121,40,29), dms(22,14,53)), (dms(122,0,29), dms(22,26,53)),
               (dms(122,0,29), dms(21,32,53)), (dms(121,40,29), dms(21,41,53))]},
]

# ---- 離岸風場施工/勘測作業區(礙航;S18 渢妙、S19 大肚一號)----
FEM1_SAFETY = [   # 渢妙 FEM1 水下基礎保護工程安全區(航船布告 304),2026-03-15..12-31
    (dms(120,9,33.8),  dms(24,24,46.9)), (dms(120,9,7.5),  dms(24,24,55.7)),
    (dms(120,4,45.9),  dms(24,24,56.7)), (dms(120,0,24.4), dms(24,24,57.4)),
    (dms(120,0,19.9),  dms(24,24,40.1)), (dms(120,4,5.3),  dms(24,22,57.8)),
    (dms(120,7,50.9),  dms(24,21,16.0)), (dms(120,8,32.0), dms(24,21,13.5)),
    (dms(120,8,40.7),  dms(24,21,43.6)), (dms(120,9,2.9),  dms(24,23,0.2)),
    (dms(120,9,20.6),  dms(24,24,1.3)),
]
DADU1_SURVEY = [  # 大肚一號離岸風場地球物理勘測作業區(航船布告 324),2026-06-15..2027-12-31
    (dms(120,7,50.9),  dms(24,21,34.1)), (dms(120,8,32.3),  dms(24,21,15.4)),
    (dms(120,8,38.0),  dms(24,21,34.9)), (dms(120,14,35.9), dms(24,23,49.5)),
    (dms(120,17,53.8), dms(24,24,49.6)), (dms(120,23,53.5), dms(24,24,48.5)),
    (dms(120,33,25.0), dms(24,25,22.5)), (dms(120,34,15.7), dms(24,25,10.7)),
    (dms(120,35,26.7), dms(24,24,30.2)), (dms(120,35,51.3), dms(24,24,14.2)),
    (dms(120,35,22.3), dms(24,23,36.5)), (dms(120,35,13.2), dms(24,23,29.2)),
    (dms(120,35,8.5),  dms(24,23,29.7)), (dms(120,35,15.4), dms(24,23,23.7)),
    (dms(120,35,11.0), dms(24,23,13.1)), (dms(120,34,58.8), dms(24,23,16.9)),
    (dms(120,33,50.6), dms(24,24,8.0)),  (dms(120,33,11.5), dms(24,24,17.4)),
    (dms(120,30,8.9),  dms(24,24,5.8)),  (dms(120,28,37.5), dms(24,23,44.5)),
    (dms(120,23,56.5), dms(24,23,27.3)), (dms(120,23,56.4), dms(24,23,27.3)),
    (dms(120,17,19.7), dms(24,23,28.6)), (dms(120,16,5.4),  dms(24,23,2.5)),
    (dms(120,15,18.1), dms(24,22,37.5)), (dms(120,10,45.0), dms(24,21,10.0)),
    (dms(120,8,50.0),  dms(24,20,29.5)), (dms(120,8,29.8),  dms(24,20,26.7)),
    (dms(119,58,9.1),  dms(24,21,2.7)),  (dms(119,57,33.9), dms(24,21,37.3)),
    (dms(119,54,23.1), dms(24,21,48.2)), (dms(119,54,7.2),  dms(24,22,0.1)),
    (dms(119,54,12.2), dms(24,22,16.4)), (dms(119,57,17.3), dms(24,24,50.4)),
    (dms(119,59,36.9), dms(24,24,50.6)), (dms(120,6,46.5),  dms(24,21,37.7)),
]
WIND_ZONES = [
    {"id": "FEM1渢妙_水下基礎保護工程安全區", "src": "S18", "verts": FEM1_SAFETY,
     "valid": "2026-03-15..2026-12-31"},
    {"id": "大肚一號_地球物理勘測作業區",     "src": "S19", "verts": DADU1_SURVEY,
     "valid": "2026-06-15..2027-12-31"},
]

# ---- 彰化風場航行空間(S20;分道通航 TSS,decimal-minute 格式)----
#   15 個邊界點(decimal minute):點號 → (lon,lat)
CH_PTS = {
    1:  (dm(120, 12.42), dm(24, 9.96)),  2:  (dm(120, 9.24),  dm(24, 12.42)),
    3:  (dm(120, 8.40),  dm(24, 13.08)), 4:  (dm(120, 6.72),  dm(24, 14.34)),
    5:  (dm(120, 9.00),  dm(24, 8.52)),  6:  (dm(120, 0.36),  dm(23, 57.42)),
    7:  (dm(119, 59.22), dm(23, 52.98)), 8:  (dm(119, 55.62), dm(23, 54.78)),
    9:  (dm(119, 54.66), dm(23, 55.32)), 10: (dm(119, 52.74), dm(23, 56.28)),
    11: (dm(120, 5.47),  dm(24, 15.33)), 12: (dm(120, 11.29), dm(24, 7.26)),
    13: (dm(120, 2.88),  dm(23, 56.40)), 14: (dm(120, 1.85),  dm(23, 52.35)),
    15: (dm(119, 51.31), dm(23, 57.03)),
}
# 各分區(以點號序;見指南附圖三)。整體風場航行空間 = 外輪廓(11→12→14→15→11)。
CHANGHUA_ZONES = [
    {"id": "彰化_分隔區",   "type": "separation", "pts": [2, 8, 9, 3], "src": "S20"},
    {"id": "彰化_北向巷道", "type": "lane",       "pts": [1, 5, 6, 7, 8, 2], "src": "S20"},
    {"id": "彰化_南向巷道", "type": "lane",       "pts": [3, 9, 10, 4], "src": "S20"},
    {"id": "彰化_東側緩衝區", "type": "buffer",   "pts": [12, 13, 14, 7, 6, 5, 1], "src": "S20",
     "note": "東側緩衝區沿線有天然氣管線;指南載明海纜橫越航道。"},
    {"id": "彰化_西側緩衝區", "type": "buffer",   "pts": [4, 10, 15, 11], "src": "S20"},
]
CHANGHUA_OUTLINE = [11, 4, 15, 10, 9, 7, 14, 13, 12, 1]  # 風場航行空間概略外輪廓(禁錨/限制作業)


def changhua_zone_verts(zone):
    return [CH_PTS[i] for i in zone["pts"]]


def changhua_outline_verts():
    return [CH_PTS[i] for i in CHANGHUA_OUTLINE]


# ----------------------------------------------------------------------
# 經緯度 → 格點:由 13 基地港最小平方擬合仿射轉換
# ----------------------------------------------------------------------
# 13 基地港之近似真實 WGS84(lon,lat);與 _BASES['taiwan'] 之格點一一對應。
_PORT_REAL = {
    "基隆": (121.74, 25.13), "淡水": (121.41, 25.18), "新竹": (120.92, 24.85),
    "台中": (120.50, 24.29), "嘉義": (120.13, 23.38), "澎湖": (119.56, 23.57),
    "台南": (120.16, 23.00), "高雄": (120.28, 22.61), "恆春": (120.74, 21.95),
    "台東": (121.19, 22.79), "花蓮": (121.62, 23.98), "蘇澳": (121.86, 24.59),
    "澳底": (121.92, 25.02),
}


def fit_affine(grid_xy, lonlat):
    """最小平方擬合 (lon,lat,1)→x 與 →y。回傳 (cx, cy) 各 3 係數。"""
    A = np.c_[lonlat[:, 0], lonlat[:, 1], np.ones(len(lonlat))]
    cx, *_ = np.linalg.lstsq(A, grid_xy[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(A, grid_xy[:, 1], rcond=None)
    return cx, cy


def _load_base_grid():
    import MOGA_GPSIFF_patrol_clean as M
    bases = M._BASES["taiwan"]
    names = [n for n in bases if n in _PORT_REAL]
    G = np.array([bases[n] for n in names], float)
    L = np.array([_PORT_REAL[n] for n in names], float)
    return names, G, L


class Georeferencer:
    """(lon,lat)→(x,y) 仿射近似;預設用全部 13 港擬合,可選 west-coast 子集。"""
    def __init__(self, anchors=None):
        names, G, L = _load_base_grid()
        if anchors:
            idx = [names.index(n) for n in anchors if n in names]
            G, L, names = G[idx], L[idx], [names[i] for i in idx]
        self.names, self.G, self.L = names, G, L
        self.cx, self.cy = fit_affine(G, L)

    def to_grid(self, lon, lat):
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        x = self.cx[0] * lon + self.cx[1] * lat + self.cx[2]
        y = self.cy[0] * lon + self.cy[1] * lat + self.cy[2]
        return np.stack([x, y], axis=-1)

    def residuals(self):
        P = self.to_grid(self.L[:, 0], self.L[:, 1])
        d = np.hypot(P[:, 0] - self.G[:, 0], P[:, 1] - self.G[:, 1])
        return dict(zip(self.names, d)), float(d.mean()), float(d.max())

    def cell_deg(self):
        return (1.0 / np.hypot(self.cx[0], self.cy[0]),
                1.0 / np.hypot(self.cx[1], self.cy[1]))


# ----------------------------------------------------------------------
# 點陣化:多邊形(格點)→ 布林遮罩
# ----------------------------------------------------------------------
def rasterize_polygon(verts_grid, H, W):
    from matplotlib.path import Path
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.c_[xx.ravel(), yy.ravel()]
    mask = Path(np.asarray(verts_grid)).contains_points(pts, radius=0.5)
    return mask.reshape(H, W)


def geometry_to_grid_mask(geo, georef, H, W):
    """單一幾何(circle / poly / verts)→ 格點布林遮罩。"""
    if geo.get("kind") == "circle":
        lon_c, lat_c = geo["center"]
        poly_ll = circle_polygon(lon_c, lat_c, geo["radius_nm"])
    elif "verts" in geo:
        poly_ll = np.asarray(geo["verts"], float)
    else:
        poly_ll = np.asarray(geo["poly"], float)
    g = georef.to_grid(poly_ll[:, 0], poly_ll[:, 1])
    return rasterize_polygon(g, H, W)


# ----------------------------------------------------------------------
# 由真實多邊形建構 TWreal real-v1 之禁航疊層與熱點層
# ----------------------------------------------------------------------
def build_overlay_v1(no_go, georef=None, clip_sea=True):
    """真實禁航/限制疊層(real v1):射擊區 ∪ 離岸風場礙航區 ∪ 彰化航行空間 footprint。
    回傳 (overlay_bool, layers, info)。
      layers: {'firing':mask, 'owf':mask, 'changhua':mask}
      info  : [(id, type, cells_before_clip, on_land_before_clip), ...]
    clip_sea=True 時將疊層限於海域格(no_go==0),修正沿岸仿射誤差(禁航僅對海域有意義)。
    """
    if georef is None:
        georef = Georeferencer()
    H, W = no_go.shape
    sea = (no_go == 0)
    firing = np.zeros((H, W), bool)
    owf = np.zeros((H, W), bool)
    changhua = np.zeros((H, W), bool)
    info = []
    for z in FIRING_ZONES:
        m = geometry_to_grid_mask(z, georef, H, W)
        info.append((z["id"], "firing", int(m.sum()), int((m & ~sea).sum())))
        firing |= m
    for z in WIND_ZONES:
        m = geometry_to_grid_mask({"verts": z["verts"]}, georef, H, W)
        info.append((z["id"], "owf", int(m.sum()), int((m & ~sea).sum())))
        owf |= m
    # 彰化:以整體外輪廓 footprint 表示(細分區寬度 < 格距,無法逐巷道表現)
    m = rasterize_polygon(georef.to_grid(*np.array(changhua_outline_verts()).T), H, W)
    info.append(("彰化風場航行空間_footprint", "changhua", int(m.sum()), int((m & ~sea).sum())))
    changhua |= m
    if clip_sea:
        firing &= sea; owf &= sea; changhua &= sea
    overlay = firing | owf | changhua
    return overlay, {"firing": firing, "owf": owf, "changhua": changhua}, info


def build_real_hotspot_v1(no_go, base_weight=None, georef=None):
    """真實熱點層 v1:在 v0 海纜風險權重上,加入彰化海纜/管線走廊與離岸風場基礎之巡邏重要性。
    base_weight 省略時由 tw_real_env.build_cable_weight 取得 v0 權重。
    """
    if georef is None:
        georef = Georeferencer()
    H, W = no_go.shape
    sea = (no_go == 0)
    if base_weight is None:
        import tw_real_env as TR
        base_weight = TR.build_cable_weight(no_go)
    w = base_weight.copy()
    # 彰化航行空間(海纜橫越 + 東側管線)→ 高巡邏重要性
    ch = rasterize_polygon(georef.to_grid(*np.array(changhua_outline_verts()).T), H, W) & sea
    w[ch] = np.maximum(w[ch], 0.85)
    # 離岸風場施工/勘測礙航區(基礎/纜線維護周邊)→ 中高重要性
    for z in WIND_ZONES:
        m = geometry_to_grid_mask({"verts": z["verts"]}, georef, H, W) & sea
        w[m] = np.maximum(w[m], 0.70)
    w[no_go != 0] = 0.0
    return np.clip(w, 0, 1)


if __name__ == "__main__":
    gr = Georeferencer()                       # 全 13 港
    res, mean, mx = gr.residuals()
    cl, cw = gr.cell_deg()
    print("== 仿射校正(13 港)殘差(格) ==")
    for n, d in res.items():
        print(f"  {n}: {d:.2f}")
    print(f"  mean={mean:.2f} max={mx:.2f}  每格約 {cl:.3f}° lon × {cw:.3f}° lat")
    print(f"火砲射擊區 {len(FIRING_ZONES)}、離岸風場/勘測 {len(WIND_ZONES)}、"
          f"彰化分區 {len(CHANGHUA_ZONES)}(+外輪廓)")
