# -*- coding: utf-8 -*-
"""
建構 TWreal real-v1:由官方航船布告/射擊報告/離岸風場/彰化 VTS 之真實 WGS84 多邊形,
經仿射近似校正疊到台灣示意底圖,產生:
  - data/no_go_TWreal_real_v1.npy        (布林禁航/限制疊層;已 clip 至海域)
  - data/no_go_TWreal_real_v1_layers.npz (firing/owf/changhua 分層)
  - data/scenario_TWreal_cable_v1.npy    (真實熱點 v1:v0 海纜風險 + 彰化/風場)
  - data/TWreal_v1_provenance.md         (來源 S17–S20 + 仿射殘差 + 真實性/解析度宣告)
  - figures/twreal_real_v1.png           (交付檢視圖:禁航疊層 + 熱點)
不覆蓋 v0 檔(no_go_TWreal_overlay.npy / scenario_TWreal_cable.npy 保留);
Whether to merge into no_go_zone is controlled by project configuration(會影響候選點合法性與 §4.4 taiwan_real 可重現性)。
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import real_geo_sources as R
import MOGA_GPSIFF_patrol_clean as M
import tw_real_env as TR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "..", "data")
FIG = os.path.join(SCRIPT_DIR, "..", "figures")


if __name__ == "__main__":
    no_go = M.no_go_zone; H, W = no_go.shape
    gr = R.Georeferencer()
    res, mean_r, max_r = gr.residuals()
    cl, cw = gr.cell_deg()

    overlay, layers, info = R.build_overlay_v1(no_go, gr, clip_sea=True)
    hot_v0 = TR.build_cable_weight(no_go)
    hot_v1 = R.build_real_hotspot_v1(no_go, base_weight=hot_v0, georef=gr)

    np.save(os.path.join(DATA, "no_go_TWreal_real_v1.npy"), overlay)
    np.savez(os.path.join(DATA, "no_go_TWreal_real_v1_layers.npz"), **layers)
    np.save(os.path.join(DATA, "scenario_TWreal_cable_v1.npy"), hot_v1)

    # ---- 可行性檢查:若併入 no_go,有多少候選點/基地出航點會失效 ----
    pts = M.Patrol_Point
    pt_block = sum(1 for (x, y) in pts if overlay[int(y), int(x)])
    base_block = sum(1 for (x, y) in M.base_coords if overlay[int(y), int(x)])

    print("== 仿射殘差 == mean=%.2f max=%.2f 格;每格約 %.3f°lon × %.3f°lat" % (mean_r, max_r, cl, cw))
    print("== real-v1 禁航/限制疊層(已 clip 海域)==")
    for zid, typ, n_all, n_land in info:
        print(f"  {zid:30s} [{typ:8s}] 格={n_all:4d}  其中落陸(clip前)={n_land:3d}")
    for k, m in layers.items():
        print(f"  -> 層 {k:9s}: {int(m.sum()):4d} 海域格")
    print(f"  疊層合計(海域):{int(overlay.sum())} 格 / 海域 {int((no_go==0).sum())} 格"
          f" = {100*overlay.sum()/(no_go==0).sum():.1f}%")
    print("== 併入 no_go 之影響(供決策)==")
    print(f"  受影響候選巡邏點:{pt_block}/{len(pts)}({100*pt_block/len(pts):.1f}%)")
    print(f"  受影響基地出航點:{base_block}/{len(M.base_coords)}")

    # ---- 交付檢視圖 ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    def base_layer(ax):
        ax.imshow(np.where(no_go == 1, 1, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=1.5)
        for (x, y) in M.base_ports:
            ax.plot(x, y, "*", ms=9, mfc="gold", mec="k", mew=0.6)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")

    # (a) 禁航/限制疊層 + 分類輪廓
    ax = axes[0]; base_layer(ax)
    ax.contourf(layers["firing"].astype(float), levels=[0.5, 1.5], colors=["#d62728"], alpha=0.45, origin="lower")
    ax.contourf(layers["owf"].astype(float), levels=[0.5, 1.5], colors=["#ff7f0e"], alpha=0.6, origin="lower")
    ax.contourf(layers["changhua"].astype(float), levels=[0.5, 1.5], colors=["#2ca02c"], alpha=0.6, origin="lower")
    for z in R.FIRING_ZONES:
        m = R.geometry_to_grid_mask(z, gr, H, W)
        ys, xs = np.where(m)
        if len(xs): ax.text(xs.mean(), ys.mean(), z["id"], fontsize=7, color="#7a0000",
                            ha="center", va="center", fontweight="bold")
    ax.set_title("(a) real-v1 restricted overlay (clipped to sea)\n"
                 "red=firing RCR  orange=OWF works  green=Changhua channel", fontsize=10)

    # (b) 真實熱點 v1
    ax = axes[1]; base_layer(ax)
    im = ax.imshow(np.where(no_go == 0, hot_v1, np.nan), origin="lower", cmap="hot_r", vmin=0, vmax=1, alpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="patrol importance")
    ax.set_title("(b) real-v1 hotspot = cable risk (v0) + Changhua/OWF", fontsize=10)

    fig.suptitle("TWreal real-v1: official notices (S17-S20) -> stylized base map via affine georef "
                 f"(mean resid {mean_r:.1f} cells ~5-10 km)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "twreal_real_v1.png")
    fig.savefig(out, dpi=120)
    print("saved figure:", out)

    # ---- provenance ----
    prov = os.path.join(DATA, "TWreal_v1_provenance.md")
    with open(prov, "w", encoding="utf-8") as f:
        f.write("# 台灣真實環境 real-v1(TWreal_real_v1)— 來源與真實性/解析度宣告\n\n")
        f.write("## 狀態\n")
        f.write("真實港口 + 真實海纜熱點(v0)+ **由官方公告 WGS84 多邊形建構之真實禁航/限制疊層(本版新增)**。\n")
        f.write("座標屬官方實際值;疊到 100×100 示意底圖經仿射近似校正(過渡版,§9.4)。\n\n")
        f.write("## 仿射經緯度→格點校正(13 海巡轄區港最小平方)\n")
        f.write(f"- 殘差:mean={mean_r:.2f}、max={max_r:.2f} 格;每格約 {cl:.3f}° lon × {cw:.3f}° lat(≈ 5–6 km × 4–5 km)。\n")
        f.write("- 逐港殘差(格):" + "、".join(f"{n}:{d:.1f}" for n, d in res.items()) + "\n")
        f.write("- 西岸關注區(台中/彰化/澎湖)殘差 < 1 格;沿岸小型射擊圓區(RCR-6/38)落陸部分已 clip 至海域。\n\n")
        f.write("## 禁航/限制疊層(來源 → 幾何)\n")
        for zid, typ, n_all, n_land in info:
            f.write(f"- {zid}（{typ}）：{n_all} 格(clip 前落陸 {n_land}）\n")
        f.write("\n來源:S17(空軍 7 月實彈射擊 RCR-6/7/9/11/12/17/38/42)、S18(渢妙 FEM1 礙航)、"
                "S19(大肚一號勘測礙航)、S20(彰化 VTS:TSS/海纜管線/禁錨)。座標見 code/real_geo_sources.py。\n\n")
        f.write("## 真實性宣告(§9.2)\n")
        f.write("- 港口/基地:真實位置層級。\n")
        f.write("- 熱點:海纜風險(S02/S04/S05/S14/S15 等,v0)+ 彰化海纜/管線走廊與離岸風場(S20/S18/S19,v1);真實結構、座標近似。\n")
        f.write("- 禁航/限制:**真實**官方公告 WGS84 多邊形(S17–S20),經仿射近似定位;取代前版【示意】佔位。\n\n")
        f.write("## 解析度限制(務必載明)\n")
        f.write("- 100×100 示意底圖每格 ≈ 5 km;彰化 TSS 之分隔區(1 浬)、巷道(2 浬)等細結構小於格距,\n")
        f.write("  僅能以整體 footprint 表示,無法逐巷道分辨;射擊/礙航小區亦近似為數格。\n")
        f.write("- 欲精確表現分區結構,須建『高解析、真實經緯度』之西岸區域底圖(§9.3 完整版,後續工作)。\n\n")
        f.write("## 併入 no_go_zone 之影響(尚未併入)\n")
        f.write(f"- 若併入,受影響候選巡邏點 {pt_block}/{len(pts)}、基地出航點 {base_block}/{len(M.base_coords)}。\n")
        f.write("- 併入會改變候選點合法性、航線與路徑快取,並使 taiwan_real 結果與 v0 不一致;\n")
        f.write("  須重建 route_cache_taiwan_real、更新 verify.py 對應檢查後方可作為正式環境。\n")
    print("provenance:", prov)
