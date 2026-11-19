"""Evidence-bound research figures. This module never fits or scores a model.

Run from repository root with the project interpreter. Missing independent score
tables produce explicit skips, never numerical placeholders. --only takes a
comma-separated list of semantic names, with or without the upgrade_ prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import platform
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
RECORD = ROOT / "analysis/paper_upgrade_records/20261118"
OUT, DATA, QA = PAPER / "figures", PAPER / "data/figure_data", PAPER / "qa"
FONTS = json.loads((QA / "font_check.json").read_text())
BLUE, ORANGE, GRAY = "#456D90", "#B97A48", "#777777"
INK, LIGHT, GRID = "#303C44", "#F1F4F6", "#DFE3E6"
COLORS = {"uncorrected": GRAY, "ewma_reference": BLUE, "legacy_pair": ORANGE}
LABELS = {"uncorrected": "基础预测", "ewma_reference": "独立滚动校正", "legacy_pair": "锚点差异校正"}
STYLES = {"uncorrected": (":", "x"), "ewma_reference": ("--", "o"), "legacy_pair": ("-", "s")}
CAVEAT = "测试16期来自已探索队列，验证14期被重复使用；元数据为回溯快照。区间描述固定封存预测的探索性重采样。"
MANIFEST, ACTIVE_INPUTS = [], []
OPTIONS = None


class MissingInput(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path)
    if not path.exists():
        raise MissingInput(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
    record = {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "sha256": sha(path)}
    if record not in ACTIVE_INPUTS:
        ACTIVE_INPUTS.append(record)
    return path


def read_csv(path, key=None, *, snapshot=True, **kwargs):
    path = Path(path)
    local = DATA / (key or path.name)
    selected = path if path.exists() else local
    bind(selected)
    # Only finalized scorer bundles are consumed. CSV existence alone is not a
    # completion signal, particularly for interrupted chronological replays.
    if selected.is_relative_to(RECORD / "results/statistics"):
        manifest_path=selected.parent/"manifest.json"
        manifest=json.loads(bind(manifest_path).read_text())
        expected=manifest.get("artifacts",{}).get(selected.name)
        if expected is None or sha(selected)!=expected:
            raise ValueError("Independent-score artifact absent or changed: "+str(selected))
    if snapshot and selected != local:
        shutil.copyfile(selected, local)
        bind(local)
    return pd.read_csv(selected, float_precision="round_trip", **kwargs)


def read_json(path, key=None):
    path = Path(path)
    local = DATA / (key or path.name)
    selected = path if path.exists() else local
    bind(selected)
    if selected.is_relative_to(RECORD / "results/statistics"):
        manifest=json.loads(bind(selected.parent/"manifest.json").read_text())
        if manifest.get("artifacts",{}).get(selected.name)!=sha(selected):
            raise ValueError("Independent-score JSON absent or changed: "+str(selected))
    if selected != local:
        shutil.copyfile(selected, local)
        bind(local)
    return json.loads(selected.read_text())


def derived(frame, name):
    path = DATA / name
    frame.to_csv(path, index=False)
    bind(path)
    return frame


def style():
    for key in ["latin", "cjk"]:
        fm.fontManager.addfont(FONTS[key]["path"])
    plt.rcParams.update({
        "font.family": FONTS["font_family_order"], "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "axes.linewidth": .65,
        "lines.linewidth": 1.25, "lines.markersize": 3.5,
        "mathtext.fontset": "cm", "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.formatter.use_mathtext": False,
    })


def save(fig, name, label, purpose, caption, *, kind="real", notes=None):
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    formats = {}
    for ext in ["pdf", "svg", "png"]:
        dest = OUT / f"upgrade_{name}.{ext}"
        fig.savefig(dest, dpi=400, bbox_inches="tight", pad_inches=.055)
        formats[ext] = {"path": str(dest.relative_to(PAPER)), "sha256": sha(dest)}
    with Image.open(OUT / f"upgrade_{name}.png") as im:
        im.convert("L").save(QA / f"upgrade_figure_{name}_gray.png")
        px = list(im.size)
    MANIFEST.append({"name": f"upgrade_{name}", "label": label, "status": "generated",
        "scientific_purpose": purpose, "caption": caption, "evidence_type": kind,
        "inputs": list(ACTIVE_INPUTS), "formats": formats, "png_dpi": 400,
        "png_pixels": px, "nominal_size_cm": (fig.get_size_inches() * 2.54).tolist(),
        "minimum_fontsize_pt": min(sizes), "font_family_order": FONTS["font_family_order"],
        "font_paths": {k: FONTS[k]["path"] for k in ["latin", "cjk"]},
        "math_font": "Computer Modern (MathText)", "font_fallback": False,
        "svg_editability": "Native vector paths and live ordinary text; mathematical glyphs may be vector paths.",
        "limitations": CAVEAT if kind == "real" else None, "notes": notes or [],
        "visual_inspection": "pending"})
    plt.close(fig)


def clean(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=.45, zorder=0)
    ax.tick_params(width=.6, length=3)


def panel(ax, text, *, y=1.03):
    ax.text(0, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


def diagram(height=8.3):
    fig, ax = plt.subplots(figsize=(16 / 2.54, height / 2.54))
    ax.set(xlim=(0, 160), ylim=(0, height * 10)); ax.axis("off")
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.02)
    return fig, ax


def prism(ax, x, y, w, h, text, *, color=BLUE, depth=2.3, fontsize=9, gid=None):
    # Small isometric side faces encode a module, never a numerical magnitude.
    pale = "#E6EDF2" if color == BLUE else ("#F4EBE3" if color == ORANGE else "#EDF0F2")
    side = "#C8D6E0" if color == BLUE else ("#E2C9B6" if color == ORANGE else "#D7DDE1")
    items = [Polygon([(x,y+h),(x+depth,y+h+depth),(x+w+depth,y+h+depth),(x+w,y+h)], facecolor=pale, edgecolor=color, lw=.65),
             Polygon([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)], facecolor=side, edgecolor=color, lw=.65),
             Rectangle((x,y),w,h,facecolor="white",edgecolor=color,lw=.8)]
    for i, item in enumerate(items):
        item.set_zorder(3); item.set_gid(f"{gid or text.splitlines()[0]}-face-{i}"); ax.add_patch(item)
    label = ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,color=INK,linespacing=1.5,zorder=4)
    label.set_gid(f"{gid or text.splitlines()[0]}-label")
    return (x,y,w,h)


def arrow(ax, points, *, color=INK, dashed=False, gid="connector"):
    count=getattr(ax,"_upgrade_arrow_count",0)+1
    ax._upgrade_arrow_count=count
    gid=f"{gid}-{count}"
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]),color=color,lw=.85,ls="--" if dashed else "-",zorder=1,gid=gid)
    art = FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=8,
                          linewidth=.85,color=color,linestyle="--" if dashed else "-",zorder=1)
    art.set_gid(gid+"-arrow"); ax.add_patch(art)


def scene_framework():
    bind(ROOT / "AGENTS.md"); bind(ROOT / "analysis/v3_next_calibration.py")
    fig, ax = diagram(8.0)
    prism(ax,4,49,43,20,"本期可见前缀\n同活动、多排名、观测截止",gid="visible-prefixes",fontsize=8.6)
    prism(ax,58,49,43,20,"活动与卡牌信息\n国服发布时间过滤",gid="metadata",fontsize=8.6)
    prism(ax,112,49,43,20,"期初已完成历史\n已封存预测的误差",gid="released-history",fontsize=8.6)
    prism(ax,4,10,43,21,"基础终分预测\n两线 Context400／其余共享树",gid="base-forecast",fontsize=8.1)
    prism(ax,58,10,43,21,"历史偏差校正\n固定期初状态与更新规则",color=ORANGE,gid="bias-correction",fontsize=8.6)
    prism(ax,112,10,43,21,"封存本期预测\n活动结束后独立评分",gid="forecast-seal",fontsize=8.6)
    arrow(ax,[(25.5,49),(25.5,33.3)])
    arrow(ax,[(68,49),(45,33.3)])
    arrow(ax,[(122,49),(99,33.3)])
    arrow(ax,[(47,20.5),(58,20.5)])
    arrow(ax,[(101,20.5),(112,20.5)])
    ax.text(4,76,"音乐游戏活动的排名门槛预测",fontsize=9.5,color=INK)
    ax.text(79,39,"当前活动终分在预测时保持隔离",fontsize=8.4,ha="center",color=GRAY)
    ax.text(4,1,"观测单位：活动×排名×预测时刻；评分独立单位：活动",fontsize=8.3,color=GRAY)
    save(fig,"scene_framework","fig:upgrade-scene","Explain the entertainment task and the three information sources.",
         "音乐游戏活动的预测场景与信息来源。基础模型接收本期截至观测时刻的前缀及可用活动信息；校正器接收期初已结束活动的封存预测误差。本期终分在预测阶段隔离。浅等距侧面表示处理模块。",kind="mechanism")


def information_timeline():
    facts=read_json(PAPER/"data/facts.json","existing_facts.json")
    bind(ROOT/"AGENTS.md")
    fig,(ax,bx)=plt.subplots(2,1,figsize=(16/2.54,8.2/2.54),gridspec_kw={"height_ratios":[1,1.65]})
    for start,n,label,color,hatch in [(0,32,"基础热身 32","#E2E6E9",""),(32,35,"因果残差热身 35","#B9CCD9","//"),(67,14,"验证 14","#E3D4C8",""),(81,16,"已探索测试 16","#D8B89F","..")]:
        ax.add_patch(Rectangle((start,0),n,1,facecolor=color,edgecolor="white",hatch=hatch,lw=.8))
        ax.text(start+n/2,.5,label,ha="center",va="center",fontsize=8.1)
    ax.set(xlim=(0,97),ylim=(-.05,1.2),yticks=[])
    ax.set_xticks([0,32,67,81,97],labels=["0","32","67","81","97"])
    ax.set_xlabel("按国服实际开始时间排序的活动累计数")
    ax.spines[["left","right","top"]].set_visible(False);panel(ax,"(a) 固定97期归属")
    bx.set(xlim=(0,100),ylim=(0,40));bx.axis("off")
    bx.plot([3,97],[20,20],color=INK,lw=.9)
    pts=[(8,"期初历史冻结"),(35,"观测前缀并预测"),(64,"预测已封存"),(88,"活动结束")]
    for x,text in pts:
        bx.plot(x,20,"o",color=ORANGE if x==64 else BLUE,ms=4)
        bx.text(x,27,text,ha="center",fontsize=8.4)
    bx.text(8,13,"本期共享期初历史",ha="center",fontsize=8.2)
    bx.text(64,13,"全部案例先封存",ha="center",fontsize=8.2)
    bx.text(88,5,"终分到达后释放反馈",ha="center",fontsize=8.2)
    bx.annotate("",xy=(97,20),xytext=(88,20),arrowprops={"arrowstyle":"-|>","color":INK,"lw":.9})
    bx.text(50,36,"(b) 活动内的信息顺序（示意时间轴）",ha="center",fontsize=9)
    bx.text(3,-1,"下一期历史守卫：历史活动结束时间 < 下一期实际开始时间",fontsize=8.5,color=GRAY)
    fig.subplots_adjust(left=.055,right=.98,top=.90,bottom=.075,hspace=.65)
    save(fig,"information_timeline","fig:upgrade-timeline","Separate warmup, heldout partitions and forecast-before-feedback timing.",
         "固定97期活动的时间归属与预测反馈顺序。前67期分为32期基础热身和35期可用因果残差热身。当前活动全部预测共享期初历史；其终分在预测封存后释放，供后续严格满足时间守卫的活动使用。验证被重复使用，测试队列已有探索暴露。",kind="protocol")


def anchor_difference():
    bind(ROOT/"analysis/v3_next_calibration.py");bind(RECORD/"reports/B_method_design.md")
    fig,ax=diagram(9.6)
    prism(ax,3,66,36,16,"T2000 历史误差\n该线全部合法支持",gid="anchor-history",fontsize=8.5)
    prism(ax,3,34,36,16,"T1000 历史误差\n按活动、阶段配对",gid="target-history",fontsize=8.5)
    prism(ax,58,66,48,16,"慢锚点 A\n更新率0.2，收缩常数10",gid="slow-anchor",fontsize=8.3)
    prism(ax,58,34,48,16,"快差异 D\n更新率0.6，收缩常数5",color=ORANGE,gid="fast-difference",fontsize=8.3)
    prism(ax,121,50,33,25,"T2000：扣除 A\nT1000：扣除 A+D",gid="recombination",fontsize=8.5)
    arrow(ax,[(39,74),(58,74)])
    arrow(ax,[(39,42),(58,42)])
    arrow(ax,[(44,74),(48,74),(48,53),(58,50)])
    ax.text(46,58,"同活动差值",fontsize=7.9,rotation=90,ha="right",va="center",color=GRAY)
    arrow(ax,[(106,74),(121,68)])
    arrow(ax,[(106,42),(121,54)],color=ORANGE)
    ax.text(3,91,"历史对数误差：正值表示基础预测偏高",fontsize=9)
    ax.text(3,23,"总偏差截断后，使用基础预测乘以 exp(−偏差)；最后施加当前分数与增长上界。",fontsize=8.2)
    ax.text(3,15,"T1000 无配对支持时：回退自身独立暖校正。两种支持数分别计算。",fontsize=8.2,color=GRAY)
    ax.plot([3,157],[10,10],color=GRID,lw=.7)
    ax.text(3,3.7,"同面板、同线性滤波、同收缩的负控：",fontsize=8.3)
    ax.text(93,3.7,r"$sF(e_2)+sF(e_1-e_2)=sF(e_1)$",fontsize=10,va="baseline")
    save(fig,"anchor_difference","fig:upgrade-anchor","Expose the actual asymmetric filters, support and algebraic control.",
         "锚点差异校正的可编辑结构。T2000锚点利用该线全部合法历史，T1000差异使用同活动同阶段配对历史；α与收缩常数均不同。相同支持、线性滤波及收缩时，坐标分解与逐线滤波满足底部代数恒等式。该负控区分坐标变换与实际滤波差异。",kind="mechanism")


def sampling_diagnosis():
    x=read_json(ROOT/"analysis/data_quality/20261115_sampling_density/summary.json","archived_sampling_density.json")
    p=x["periods"];a=p["before_activity_226"];b=p["activity_226_onward_in_training"]
    fig,axes=plt.subplots(1,3,figsize=(16/2.54,6.2/2.54))
    keys=[("observations_within_activity","活动内观测数／条"),("median_gap_minutes","观测间隔／min"),("observations_per_day","每天观测数／条")]
    for ax,(key,ylabel),letter in zip(axes,keys,"abc"):
        values=[a[key],b[key]]
        ax.bar([0,1],values,width=.55,color=["#CCD2D7",BLUE],edgecolor=INK,lw=.6,hatch=["//",""])
        for j,v in enumerate(values):ax.text(j,v+max(values)*.035,f"{v:.1f}",ha="center",fontsize=8.5)
        ax.set_xticks([0,1],labels=["边界前\n186期","边界后\n12期"])
        ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.23);clean(ax);panel(ax,f"({letter})")
    fig.text(.5,.02,"归档训练曲线局部诊断；政策边界：2024-01-29 13:00（UTC+8）",ha="center",fontsize=8.3,color=GRAY)
    fig.subplots_adjust(left=.085,right=.98,top=.90,bottom=.25,wspace=.49)
    save(fig,"sampling_diagnosis","fig:upgrade-density","Document the archived sampling diagnosis without reopening early raw curves.",
         "历史归档的局部采样密度诊断。边界前186期306条合格训练曲线，边界后12期24条；图示活动内观测数中位数、逐曲线间隔中位数再取中位数、每天观测数。归档覆盖至2024年5月，用户指定的政策日期与技术采集切换时刻分别解释。数据从既有summary读取。",kind="archived_diagnosis")


def activity_forecasts():
    source_path=OPTIONS.stats_root/"test_case_losses.csv"
    if source_path.exists():
        f=read_csv(source_path,"test_case_losses.csv",snapshot=False,
                   usecols=["case_id","event","rank","start_ms","observed_at","current","method","predicted","actual"])
        order=f[["event","start_ms"]].drop_duplicates().sort_values(["start_ms","event"])
        if len(order)!=16:raise ValueError("Curve example requires all16 frozen test events")
        chosen=order.event.iloc[[0,7]].tolist()
        f=f[f.event.isin(chosen)&f["rank"].isin([1000,2000])&f.method.isin(COLORS)].copy()
        derived(f,"activity_forecast_examples.csv")
        selection={"rule":"actual-start ordered test positions1and8; fixed before plotting; not selected by error",
                   "full_event_order":order.event.astype(int).tolist(),"selected_event_ids":chosen,
                   "source_sha256":sha(source_path)}
        (DATA/"activity_forecast_selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2)+"\n")
        bind(DATA/"activity_forecast_selection.json")
    else:
        f=read_csv(DATA/"activity_forecast_examples.csv")
        chosen=read_json(DATA/"activity_forecast_selection.json")["selected_event_ids"]
    fig,axes=plt.subplots(2,2,figsize=(16/2.54,10.0/2.54),sharex=True)
    for i,event in enumerate(chosen):
        for j,rank in enumerate([1000,2000]):
            ax=axes[i,j];part=f[(f.event==event)&(f["rank"]==rank)]
            for method in COLORS:
                sub=part[part.method==method].sort_values("observed_at");t=(sub.observed_at-sub.start_ms)/86400000
                ls,mk=STYLES[method];ax.plot(t,sub.predicted/1e6,color=COLORS[method],ls=ls,marker=mk,ms=3,label=LABELS[method])
            one=part[part.method=="uncorrected"].sort_values("observed_at")
            t=(one.observed_at-one.start_ms)/86400000
            ax.plot(t,one.current/1e6,color=INK,lw=1,marker=".",label="冻结时点当前分数")
            actual=one.actual.unique()
            if len(actual)!=1:raise ValueError("One final score per event and rank")
            ax.axhline(actual[0]/1e6,color=INK,ls="-.",lw=.8,label="真实终分")
            panel(ax,f"({'abcd'[i*2+j]}) 活动{event} · T{rank}")
            clean(ax);ax.set_ylabel("分数／百万EP")
            if i==1:ax.set_xlabel("距实际开始的时间／d")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,1.005),ncol=3,fontsize=7.9)
    fig.subplots_adjust(left=.10,right=.98,top=.83,bottom=.12,hspace=.40,wspace=.32)
    save(fig,"activity_forecasts","fig:upgrade-curves","Show predeclared real activity examples and sealed forecasts at frozen origins.",
         "预定的两个真实活动示例：按实际开始时间排序的第1和第8测试活动，各展示T1000与T2000。当前分数和预测仅在冻结时点绘制，连接线辅助阅读；完整采样轨迹未在此复原。水平点划线为活动结束后独立评分所用终分。测试来自既有探索队列。")


def matched_correlations():
    f=read_csv(OPTIONS.corr_root/"matched_correlation_pairs.csv","matched_correlation_pairs.csv")
    panels=[("log_final","终分水平"),("remaining_growth_p50","50%进度剩余增长"),("raw_error_phase1","基础外推误差"),("independent_error_phase1","独立校正后误差")]
    ranks=sorted(set(f.rank_a)|set(f.rank_b));n=f.n.unique()
    if len(n)!=1:raise ValueError("Correlation panels must share the same complete event panel")
    fig,axes=plt.subplots(2,2,figsize=(16/2.54,13.8/2.54))
    cmap=LinearSegmentedColormap.from_list("muted_diverging",[BLUE,"#FAFAFA",ORANGE]);cmap.set_bad("#E1E4E7")
    for ax,(key,text),letter in zip(axes.flat,panels,"abcd"):
        m=np.eye(len(ranks));idx={r:i for i,r in enumerate(ranks)}
        for row in f[f.panel==key].itertuples():m[idx[row.rank_a],idx[row.rank_b]]=m[idx[row.rank_b],idx[row.rank_a]]=row.spearman
        edges=np.arange(len(ranks)+1)-.5
        im=ax.pcolormesh(edges,edges,np.ma.masked_invalid(m),vmin=-1,vmax=1,cmap=cmap,
                         shading="flat",rasterized=False,antialiased=False)
        ax.set(xlim=(-.5,len(ranks)-.5),ylim=(len(ranks)-.5,-.5),aspect="equal")
        ii,jj=np.where(m<0)
        ax.scatter(jj,ii,marker="x",s=7,linewidths=.35,color=INK)
        ticks=list(range(len(ranks)));ax.set_xticks(ticks,labels=[str(r) for r in ranks],rotation=90,fontsize=7.0)
        ax.set_yticks(ticks,labels=[str(r) for r in ranks],fontsize=7.0);ax.tick_params(length=2,pad=2)
        panel(ax,f"({letter}) {text}，n={int(n[0])}")
    fig.subplots_adjust(left=.08,right=.89,top=.95,bottom=.10,hspace=.40,wspace=.30)
    cax=fig.add_axes([.925,.26,.018,.49]);cb=fig.colorbar(im,cax=cax,label="Spearman",ticks=[-1,-.5,0,.5,1])
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face")
    fig.text(.48,.015,"×：负相关单元格；四个面板使用完全相同的活动及排名支持",ha="center",fontsize=8,color=GRAY)
    save(fig,"matched_correlations","fig:upgrade-correlation","Compare target and error correlations on exactly matched training support.",
         "同一批训练外推活动、同一共同排名面板的相关结构。终分、50%进度剩余增长、基础误差与独立校正误差均采用共同支持活动；图中n为活动数。T1500因训练缺失留在面板之外。T1000/T2000与其他线的基础模型路线不同；这些相关系数用于描述预测目标与残差结构。")


def eventwise_improvement():
    f=read_csv(OPTIONS.stats_root/"test_event_losses.csv","test_event_losses.csv")
    wide=f.pivot(index=["start_ms","event"],columns="method",values="mape_pct").sort_index()
    if len(wide)!=16:raise ValueError("Expected16 test events")
    delta=wide.legacy_pair-wide.ewma_reference
    used=pd.DataFrame({"event":wide.index.get_level_values("event"),"difference_points":delta.to_numpy()})
    derived(used,"eventwise_pair_improvement.csv")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,6.6/2.54),gridspec_kw={"width_ratios":[1.75,1]})
    x=np.arange(16);ax.bar(x,delta,color=[ORANGE if d<0 else "#CBD3D9" for d in delta],edgecolor=INK,lw=.45)
    ax.axhline(0,color=INK,lw=.7);ax.set_xticks(x,labels=[str(e) for e in used.event],rotation=60)
    ax.set_xlabel("活动ID（按实际开始排序）");ax.set_ylabel("双线−独立校正／百分点");clean(ax);panel(ax,"(a) 每活动配对差")
    a=np.sort(np.abs(delta.to_numpy()))[::-1];cum=np.cumsum(a)/a.sum() if a.sum() else np.zeros(16)
    bx.plot(np.arange(1,17),cum*100,color=BLUE,marker="o",ms=3)
    bx.plot([0,16],[0,100],color=GRAY,lw=.7,ls=":")
    bx.set(xlim=(0,16),ylim=(0,105),xticks=[1,3,8,16]);bx.set_xlabel("按绝对差排序的活动数")
    bx.set_ylabel("累计绝对差贡献／%");clean(bx);panel(bx,"(b) 差值集中程度")
    ax.text(.02,.97,f"改善 {(delta<0).sum()}/16期",transform=ax.transAxes,va="top",fontsize=8.4)
    fig.subplots_adjust(left=.095,right=.98,top=.90,bottom=.26,wspace=.40)
    save(fig,"eventwise_improvement","fig:upgrade-eventwise","Show all paired activities and the concentration of their loss differences.",
         "双线校正相对独立滚动校正的逐活动MAPE差，以及按绝对差排序后的累计贡献。负差表示双线较低；全部16期均保留。集中程度使用绝对差，改善期数单独标出。活动序列按国服实际开始时间排序。")


def sensitivity():
    frames={s:read_csv(OPTIONS.stats_root/f"{s}_metrics.csv",f"replay_{s}_metrics.csv").set_index("method") for s in ["validation","test"]}
    sweeps=[("anchor_alpha",[.1,.2,.3],.2,"锚点更新率"),("difference_alpha",[.3,.6,.9],.6,"差异更新率"),("anchor_shrinkage",[0,5,10,20],10,"锚点收缩常数"),("difference_shrinkage",[0,2.5,5,10],5,"差异收缩常数"),("minimum_pairs",[1,3,5,10],1,"最低配对活动数")]
    fig,axes=plt.subplots(2,3,figsize=(16/2.54,9.7/2.54))
    for ax,(key,values,base,xlabel),letter in zip(axes.flat,sweeps,"abcde"):
        for split,color,ls,mk in [("validation",BLUE,"--","o"),("test",ORANGE,"-","s")]:
            names=["legacy_pair" if v==base else f"pair_{key}_{v:g}".replace(".","p") for v in values]
            y=frames[split].loc[names,"event_macro_mape_pct"].to_numpy()
            ax.plot(values,y,color=color,ls=ls,marker=mk,ms=3,label="重复使用验证" if split=="validation" else "已探索测试")
        ax.axvline(base,color=GRAY,lw=.7,ls=":");ax.set_xticks(values);ax.set_xlabel(xlabel);ax.set_ylabel("活动等权 MAPE／%")
        clean(ax);panel(ax,f"({letter})")
    axes[1,2].axis("off");h,l=axes[0,0].get_legend_handles_labels();axes[1,2].legend(h,l,loc="upper center",bbox_to_anchor=(.5,.95),fontsize=8.2)
    axes[1,2].text(.5,.23,"每次只改变一个参数\n点线连接已运行设置\n竖虚线为固定原配方",ha="center",va="center",transform=axes[1,2].transAxes,fontsize=8.4,linespacing=1.6,color=GRAY)
    fig.subplots_adjust(left=.085,right=.98,top=.94,bottom=.13,hspace=.55,wspace=.50)
    save(fig,"sensitivity","fig:upgrade-sensitivity","Display the full preregistered one-factor sensitivity grid without interpolated combinations.",
         "预登记的单因素敏感性。每次仅改变锚点更新率、差异更新率、收缩常数或最低配对支持中的一项，其他设置固定。竖虚线为原配方，线段只连接实际运行点。验证和已探索测试均完整呈现，本图未按测试结果选参数。")


def structural_ablations():
    f=read_csv(OPTIONS.stats_root/"test_metrics.csv","replay_test_metrics.csv").set_index("method")
    summary=read_json(OPTIONS.stats_root/"summary.json","replay_summary.json")
    effects={r["method"]:r for r in summary["splits"]["test"]["paired_effects"] if r["weighting"]=="activity_case_weighting" and r["reference"]=="ewma_reference"}
    names=["ewma_reference","legacy_pair","pair_anchor_only","pair_difference_only","pair_swapped_anchor","same_filter_pair","same_panel_independent","window5","kalman_level","kalman_trend","dlm_fullcov"]
    labels=["独立暖EWMA","锚点＋差异","仅锚点分量","仅差异分量","交换锚点","同滤波分解","同面板独立","最近5期窗口","标量Kalman水平","标量Kalman趋势","完整协方差DLM"]
    v=f.loc[names,"event_macro_mape_pct"]-f.loc["ewma_reference","event_macro_mape_pct"]
    fig,ax=plt.subplots(figsize=(16/2.54,8.4/2.54));y=np.arange(len(names))
    for i,(name,value) in enumerate(zip(names,v)):
        bootstrap=effects[name].get("bootstrap",{})
        if bootstrap.get("5",{}).get("status")=="available":
            lo,hi=bootstrap["5"]["ci95"];ax.plot([lo,hi],[i+.08,i+.08],color=GRAY,lw=.8)
        if bootstrap.get("1",{}).get("status")=="available":
            lo,hi=bootstrap["1"]["ci95"];ax.plot([lo,hi],[i-.04,i-.04],color=BLUE,lw=1.7)
        ax.plot(value,i,"s" if name=="legacy_pair" else "o",color=ORANGE if name=="legacy_pair" else BLUE,ms=4)
    ax.axvline(0,color=GRAY,lw=.7,ls="--")
    ax.set_yticks(y,labels=labels);ax.invert_yaxis();ax.set_xlabel("相对独立暖EWMA的活动等权 MAPE 差／百分点")
    clean(ax,"x");fig.subplots_adjust(left=.28,right=.98,top=.97,bottom=.14)
    save(fig,"structural_ablations","fig:upgrade-ablation","Compare prespecified structural controls and classical filters on identical cases.",
         "固定基础预测下的结构消融与传统滤波对照。相同滤波分解与同面板独立控制具有相同历史支持和更新设置；原锚点差异法具有不同更新率、收缩与支持。点为16期测试的相对MAPE差，粗蓝线与细灰线分别为活动配对和5活动循环块的95%探索性重采样区间。仅差异分量在该测试的点估计较低，完整法与其配对差区间跨零。完整54方法清单见补充结果。")


def training_size():
    f=read_csv(OPTIONS.learning_root/"learning_metrics.csv","learning_metrics.csv")
    # Independent scorer owns final-score ensembles and activity equal metrics.
    required={"family","method","training_events","seed","event_macro_mape_pct"}
    if not required<=set(f):raise MissingInput("learning_metrics.csv schema: "+",".join(sorted(required)))
    f["family"]=f.family.replace({"catboost":"tree","nn":"neural"})
    if "structure" not in f:
        raise MissingInput("learning_metrics.csv requires structure metadata")
    structure_labels={"shared":"共享输出","rank_heads":"分线输出头","independent":"完全独立模型"}
    fig,axes=plt.subplots(1,2,figsize=(16/2.54,6.4/2.54))
    for ax,family,letter in zip(axes,["tree","neural"],"ab"):
        part=f[f.family==family]
        if part.empty:raise MissingInput("learning curve family "+family)
        for k,(method,sub) in enumerate(part.groupby("structure",sort=False)):
            color=[BLUE,ORANGE,GRAY][k%3];marker=["o","s","^"][k%3]
            seeds=sub[~sub.seed.astype(str).isin(["ensemble","mean","None","nan"])]
            if family=="neural":
                for _,s in seeds.groupby("seed"):
                    s=s.sort_values("training_events");ax.plot(s.training_events,s.event_macro_mape_pct,color=color,alpha=.3,lw=.7,ls=":")
            summary=sub[sub.seed.astype(str).isin(["ensemble","mean","42.0","42"])] if family=="tree" else sub[sub.seed.astype(str).isin(["ensemble","mean"])]
            summary=summary.sort_values("training_events")
            if summary.empty:raise MissingInput("independently scored ensemble for "+method)
            ax.plot(summary.training_events,summary.event_macro_mape_pct,color=color,marker=marker,ls=["-","--","-."][k%3],label=structure_labels.get(method,method),ms=4)
        ax.set_xticks([16,32,48,67]);ax.set_xlabel("实际用于拟合的训练活动数");ax.set_ylabel("验证活动等权 MAPE／%")
        clean(ax);panel(ax,f"({letter}) "+("CatBoost200" if family=="tree" else "16/16神经网络"));ax.legend(fontsize=7.6)
    fig.subplots_adjust(left=.095,right=.98,top=.90,bottom=.20,wspace=.36)
    save(fig,"training_size","fig:upgrade-learning","Show actual new fits at four training sizes on a fixed validation panel.",
         "真实基础模型训练规模实验。按实际开始时间取最早16、32、48、67期训练活动，在相同两线488个验证前缀上独立评分；树模型和神经网络使用相同本线输入、目标与有界解码。神经细虚线为42/123/2026逐种子，主线为三种子最终分数均值集成的独立成绩。共享输出、分线输出头及完全独立网络的参数量与CPU预算分别报告。该协议与19线混合基础模型分开解释。")


def rank_stage():
    f=read_csv(OPTIONS.stats_root/"test_stage_rank.csv","test_stage_rank.csv")
    stagecol="stage_id" if "stage_id" in f else "stage"
    stages=["progress_10pct","progress_25pct","progress_50pct","progress_75pct","progress_90pct","progress_95pct"]
    ranks=sorted(f["rank"].unique());methods=["ewma_reference","legacy_pair"]
    part=f[f[stagecol].isin(stages)&f.method.isin(methods)]
    pivot=part.pivot(index="rank",columns=["method",stagecol],values="event_macro_mape_pct")
    mat=np.full((len(ranks),len(stages)),np.nan);support=np.zeros_like(mat)
    for i,r in enumerate(ranks):
        for j,s in enumerate(stages):
            a=part[(part["rank"]==r)&(part[stagecol]==s)]
            if set(a.method)==set(methods):
                av=a.set_index("method");mat[i,j]=av.loc["legacy_pair","event_macro_mape_pct"]-av.loc["ewma_reference","event_macro_mape_pct"]
                support[i,j]=av.loc["ewma_reference","events"]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,10/2.54),gridspec_kw={"width_ratios":[1,1]})
    cmap=LinearSegmentedColormap.from_list("difference",[BLUE,"white",ORANGE]);cmap.set_bad("#E5E5E5")
    limit=max(float(np.nanmax(np.abs(mat))),.01)
    xedges=np.arange(len(stages)+1)-.5;yedges=np.arange(len(ranks)+1)-.5
    im=ax.pcolormesh(xedges,yedges,np.ma.masked_invalid(mat),vmin=-limit,vmax=limit,cmap=cmap,
                     shading="flat",rasterized=False,antialiased=False)
    sm=bx.pcolormesh(xedges,yedges,np.ma.masked_where(support==0,support),vmin=0,vmax=16,cmap="Greys",
                     shading="flat",rasterized=False,antialiased=False)
    for a in [ax,bx]:
        a.set(xlim=(-.5,len(stages)-.5),ylim=(len(ranks)-.5,-.5),aspect="auto")
        a.set_xticks(range(6),labels=["10","25","50","75","90","95"]);a.set_xlabel("活动进度／%")
        a.set_yticks(range(len(ranks)),labels=["T"+str(r) for r in ranks],fontsize=7.4)
    for i in range(len(ranks)):
        for j in range(6):
            if support[i,j]:
                bx.text(j,i,f"{support[i,j]:g}",ha="center",va="center",fontsize=7.5,color="white" if support[i,j]>8 else INK)
                if abs(mat[i,j])>1e-8:
                    ax.text(j,i,f"{mat[i,j]:+.2f}",ha="center",va="center",fontsize=7.0,color="white" if abs(mat[i,j])>.65*limit else INK)
    panel(ax,"(a) 双线−独立校正／百分点");panel(bx,"(b) 实际支持活动数")
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.03);sb=fig.colorbar(sm,ax=bx,fraction=.04,pad=.03,ticks=[0,8,16])
    cb.solids.set_rasterized(False);sb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face");sb.solids.set_edgecolor("face")
    fig.subplots_adjust(left=.10,right=.98,top=.93,bottom=.12,wspace=.43)
    save(fig,"rank_stage","fig:upgrade-rank-stage","Show heterogeneous stage/rank effects alongside exact event support.",
         "排名与活动进度的匹配误差差和支持数。左侧为双线校正减独立校正，负值为较低MAPE；右侧为相同单元格的活动数。缺失时点保留为空。总体指标按唯一案例计数，时间网格分面只使用成员关系。T1500在训练与验证中缺失，测试期按固定回退更新。")


def neural_comparison():
    f=read_csv(PAPER/"data/main_models.csv","existing_main_models.csv").set_index("method")
    names=["previous_neural","shared_single","rank_heads","related_static","related_phase"]
    labels=["多线门控网络","共享编码与输出","每线专属输出头","静态相关线分支","阶段图与注意力门控"]
    fig,ax=plt.subplots(figsize=(16/2.54,6.4/2.54))
    for i,name in enumerate(names):
        row=f.loc[name]
        ax.plot([row.block5_low,row.block5_high],[i+.09,i+.09],color=GRAY,lw=.9)
        ax.plot([row.iid_low,row.iid_high],[i-.04,i-.04],color=BLUE,lw=2)
        ax.plot(row.delta_vs_reference_points,i-.04,"o",color=BLUE,ms=4)
    ax.axvline(0,color=INK,lw=.7,ls="--");ax.set_yticks(range(len(names)),labels=labels);ax.invert_yaxis()
    ax.set_xlabel("相对独立滚动校正的活动等权 MAPE 差／百分点")
    ax.plot([],[],color=BLUE,lw=2,label="活动配对95%区间");ax.plot([],[],color=GRAY,lw=.9,label="5活动循环块95%区间")
    ax.legend(loc="upper center",bbox_to_anchor=(.45,1.24),ncol=2,fontsize=8);clean(ax,"x")
    fig.subplots_adjust(left=.30,right=.98,top=.78,bottom=.20)
    save(fig,"neural_comparison","fig:upgrade-neural","Retain existing matched neural negative results and their resampling uncertainty.",
         "既有神经结构在相同5220测试案例上的匹配比较。点为相对独立滚动校正的活动等权MAPE差，粗蓝线与细灰线分别为活动配对和5活动循环块的探索性95%重采样区间。训练预算与输入范围见正文；机制解释覆盖这些具体实现，长期方法搜索的不确定性另列。")


def synthetic_sharing():
    f=read_csv(RECORD/"synthetic/paired_comparisons.csv","synthetic_paired_comparisons.csv")
    audit=read_json(RECORD/"synthetic/negative_control_audit.json","synthetic_negative_control_audit.json")
    methods=[("fully_shared_ewma","完全共享"),("anchor_gap","锚点差异")]
    selected=f[(f.metric=="state_mse")&(f.history_length==32)&f.method.isin([x[0] for x in methods])].copy()
    if len(selected)!=32:raise ValueError("Expected16 cells for each of two synthetic methods")
    # Four row conditions x four drift conditions; all16 registered cells appear.
    rows=sorted(set(zip(selected.observation_variance,selected.mismatch_loading)))
    cols=sorted(set(zip(selected.q_common,selected.q_private)))
    matrices=[];errors=[]
    for method,_ in methods:
        mat=np.zeros((4,4));err=np.zeros((4,4))
        for r in selected[selected.method==method].itertuples():
            i=rows.index((r.observation_variance,r.mismatch_loading));j=cols.index((r.q_common,r.q_private))
            mat[i,j]=r.mean*1e3;err[i,j]=1.96*r.mc_standard_error*1e3
        matrices.append(mat);errors.append(err)
    lim=max(float(np.max(np.abs(m))) for m in matrices)
    fig,axes=plt.subplots(1,2,figsize=(16/2.54,8.5/2.54))
    cmap=LinearSegmentedColormap.from_list("synth",[BLUE,"#FFFFFF",ORANGE])
    for ax,mat,err,(_,label),letter in zip(axes,matrices,errors,methods,"ab"):
        edges=np.arange(5)-.5
        im=ax.pcolormesh(edges,edges,mat,cmap=cmap,vmin=-lim,vmax=lim,
                         shading="flat",rasterized=False,antialiased=False)
        ax.set(xlim=(-.5,3.5),ylim=(3.5,-.5),aspect="equal")
        ax.set_xticks(range(4),labels=[f"{c*1e4:g}/{d*1e4:g}" for c,d in cols],fontsize=8)
        ax.set_yticks(range(4),labels=[f"{r*1e4:g}/{m:g}" for r,m in rows],fontsize=8)
        ax.set_xlabel("共同／私有漂移方差\n（单位0.0001）");ax.set_ylabel("噪声方差／结构失配\n（方差单位0.0001）")
        panel(ax,f"({letter}) {label}：合成机制")
        for i in range(4):
            for j in range(4):
                sig=abs(mat[i,j])>err[i,j]
                ax.text(j,i,f"{mat[i,j]:.2f}"+("*" if sig else ""),ha="center",va="center",fontsize=7.8,color="white" if abs(mat[i,j])>.65*lim else INK)
    fig.subplots_adjust(left=.09,right=.88,top=.90,bottom=.30,wspace=.40)
    cax=fig.add_axes([.925,.30,.018,.48]);cb=fig.colorbar(im,cax=cax,label="相对独立EWMA的状态MSE差 ×10³")
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face")
    fig.text(.5,.015,"历史长度32；每格300条独立序列；*：配对Monte Carlo 95%区间位于零的一侧",ha="center",fontsize=8,color=GRAY)
    save(fig,"synthetic_sharing","fig:upgrade-synthetic","Map all registered synthetic mechanisms to sharing gain or loss.",
         "合成机制实验的16格共享效果图，历史长度32。两个方法均相对独立EWMA比较已截断偏差状态MSE，负差表示状态风险较低。每格300条独立模拟序列，星号表示配对Monte Carlo均值差的95%区间位于零的一侧；其余历史长度、方法及逐种子结果完整保留。同滤波代数负控最大偏差差为机器精度量级。该图的生成机制与真实活动评分分别解释。",kind="synthetic",notes=[f"same-filter maximum difference {audit['maximum_absolute_raw_bias_difference']:.4g}"])


def tail_interval():
    f=read_csv(OPTIONS.stats_root/"test_metrics.csv","replay_test_metrics.csv").set_index("method")
    g=read_csv(OPTIONS.stats_root/"test_interval_reliability.csv","test_interval_reliability.csv")
    extra=RECORD/"results/statistics/interval_joint_and_learning_contrasts_v1"
    joint=read_csv(extra/"interval_joint_summary.csv","interval_joint_summary.csv")
    coverage_ci=read_csv(extra/"interval_coverage_intervals.csv","interval_coverage_intervals.csv")
    fig,axes=plt.subplots(1,3,figsize=(16/2.54,7.1/2.54))
    ax,bx,cx=axes
    keys=["event_macro_mape_pct","p90_ape_pct","rmse_ep"]
    for name in COLORS:
        row=f.loc[name];ls,mk=STYLES[name]
        ratio=[100*row[k]/f.loc["uncorrected",k] for k in keys]
        ax.plot(range(3),ratio,color=COLORS[name],ls=ls,marker=mk,ms=4,label=LABELS[name])
    ax.set_xticks(range(3),labels=["MAPE","P90","RMSE"])
    ax.set_ylabel("相对基础预测／%（基础=100）");panel(ax,"(a) 尾部取舍");clean(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.24),fontsize=7.3)
    g=g[g.method=="ewma_reference"].sort_values("nominal_coverage")
    x=g.nominal_coverage*100
    joint=joint[(joint.split=="test")&(joint.method=="ewma_reference")].sort_values("nominal_coverage")
    for key,target,color,marker,ls,label in [
        ("event_equal_group_joint_coverage","group_joint_coverage",ORANGE,"s","-","排名×阶段联合目标"),
        ("event_equal_pointwise_coverage","pointwise_coverage",BLUE,"o","--","单前缀覆盖")]:
        values=joint[key].to_numpy()*100
        ci=coverage_ci[(coverage_ci.split=="test")&(coverage_ci.target==target)&(coverage_ci.block==1)].sort_values("nominal_coverage")
        if len(ci)!=len(joint):raise ValueError("Coverage CI target/level mismatch")
        bx.errorbar(x,values,yerr=np.vstack([values-ci.low95.to_numpy()*100,ci.high95.to_numpy()*100-values]),color=color,marker=marker,ls=ls,capsize=2,elinewidth=.7,label=label,ms=3.5)
    bx.plot([75,100],[75,100],color=GRAY,ls=":",lw=.8)
    bx.set(xlim=(75,100),ylim=(60,101));bx.set_xlabel("名义覆盖／%");bx.set_ylabel("活动等权经验覆盖／%");clean(bx);panel(bx,"(b) 历史经验区间")
    bx.legend(fontsize=7.1,loc="lower right")
    widths=g.supported_mean_width_ep/1e6
    cx.bar(x,widths,width=3,color=BLUE,edgecolor=INK,lw=.6)
    for xx,width,row in zip(x,widths,g.itertuples()):
        cx.text(xx,width+.04*widths.max(),f"{int(row.supported_cases)}/{int(row.cases)}",ha="center",fontsize=7.2)
    cx.set(xlim=(75,95),ylim=(0,widths.max()*1.25),xticks=x);cx.set_xlabel("名义覆盖／%");cx.set_ylabel("有限区间平均宽度／百万EP");clean(cx);panel(cx,"(c) 宽度与可用支持")
    fig.subplots_adjust(left=.085,right=.98,top=.88,bottom=.30,wspace=.63)
    save(fig,"tail_interval","fig:upgrade-interval","Retain point/tail tradeoffs and distinguish empirical coverage from availability.",
         "点预测尾部风险与历史经验区间可靠性。左侧将基础预测归一为100，比较活动等权MAPE、案例APE的P90和RMSE；中间以活动等权的排名×阶段全部前缀联合覆盖作为目标匹配的主曲线，单前缀覆盖为较弱目标。误差棒为16期活动配对重采样95%区间。右侧展示有限区间的案例平均宽度，柱上为有限支持案例数/总案例数。区间基于历史分组最大误差经验次序统计量，时间分布下的前瞻覆盖待新活动检验。")


def prospective_loop():
    bind(ROOT/"analysis/paper_upgrade_prospective.py")
    status={"status":"waiting_for_eligible_forecasts","events":0}
    if OPTIONS.prospective_root and (OPTIONS.prospective_root/"status.json").exists():
        status=read_json(OPTIONS.prospective_root/"status.json","prospective_status.json")
    fig,ax=diagram(8.8)
    prism(ax,3,54,43,20,"配置与来源封存\n固定主方法及次级方法",gid="registered-config",fontsize=8.6)
    prism(ax,59,54,43,20,"合格新活动\n实际开始晚于配置冻结",gid="eligible-event",fontsize=8.6)
    prism(ax,115,54,40,20,"逐时点预测封存\n时间戳、案例与版本",gid="issued-seal",fontsize=8.4)
    prism(ax,115,12,40,20,"终分到达与核实\n保留真实到达日志",color=ORANGE,gid="outcome-arrival",fontsize=8.4)
    prism(ax,59,12,43,20,"独立前瞻评分\n活动等权与完整支持",gid="prospective-score",fontsize=8.6)
    prism(ax,3,12,43,20,"下一期合法反馈\n结束与到达时间守卫",gid="released-next-history",fontsize=8.6)
    arrow(ax,[(46,64),(59,64)]);arrow(ax,[(102,64),(115,64)]);arrow(ax,[(135,54),(135,34.3)],color=ORANGE)
    arrow(ax,[(115,22),(104.3,22)]);arrow(ax,[(59,22),(48.3,22)])
    arrow(ax,[(24.5,32),(24.5,45),(80.5,45),(80.5,54)],dashed=True)
    ax.text(80,39,"反馈用于后续活动，当前预测保留原封存记录",ha="center",fontsize=8.0,color=GRAY)
    ax.text(3,83,"前瞻影子接口与部署记录",fontsize=9.5)
    wait=status.get("status","").startswith("waiting")
    ax.text(80,2,"当前状态：等待合格新活动及核实终分" if wait else "前瞻状态见封存日志与独立评分",ha="center",fontsize=9,color=ORANGE)
    save(fig,"prospective_loop","fig:upgrade-prospective","Separate a real prospective recording interface from retrospective precision claims.",
         "前瞻影子记录的闭环与当前等待状态。仅纳入实际开始晚于配置冻结的新活动；本轮冻结时已开始的活动排除于完整前瞻队列。配置、逐时点预测和终分到达分别记录，下一期反馈需同时满足时间守卫。当前前瞻精度等待合格新活动与核实结果；生产版本的工程状态与回溯测试精度分别报告。",kind="mechanism",notes=["Engineering interface, not an automatic production activation.",status])


FUNCTIONS = {f.__name__:f for f in [scene_framework,information_timeline,activity_forecasts,anchor_difference,sampling_diagnosis,matched_correlations,eventwise_improvement,sensitivity,structural_ablations,training_size,rank_stage,neural_comparison,synthetic_sharing,tail_interval,prospective_loop]}


def qa_exports():
    """Render the actual PDFs with Poppler and retain measurable export checks."""
    records,thumbs=[],[]
    font=ImageFont.truetype(FONTS["latin"]["path"],24)
    ns={"s":"http://www.w3.org/2000/svg"}
    for file in sorted(OUT.glob("upgrade_*.pdf")):
        prefix=QA/("upgrade_figure_pdf_"+file.stem.removeprefix("upgrade_"))
        subprocess.run(["pdftoppm","-scale-to","1600","-png","-singlefile",str(file),str(prefix)],check=True,capture_output=True)
        fonts=subprocess.run(["pdffonts",str(file)],check=True,capture_output=True,text=True).stdout
        svg=ET.parse(file.with_suffix(".svg")).getroot()
        ids=[e.get("id") for e in svg.iter() if e.get("id")]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate SVG IDs: "+str(file))
        with Image.open(file.with_suffix(".png")) as im:
            pixels=im.size;dpi=im.info.get("dpi")
        records.append({"name":file.stem,"pdffonts":fonts,"png_pixels":pixels,"png_dpi":dpi,
                        "svg_text_objects":len(svg.findall(".//s:text",ns)),
                        "svg_paths":len(svg.findall(".//s:path",ns)),
                        "svg_image_objects":len(svg.findall(".//s:image",ns)),
                        "svg_ids_unique":True,"pdf_render":str(prefix.relative_to(PAPER))+".png"})
        im=Image.open(prefix.with_suffix(".png")).convert("RGB");im.thumbnail((760,740))
        tile=Image.new("RGB",(800,800),"white");tile.paste(im,((800-im.width)//2,46))
        ImageDraw.Draw(tile).text((15,8),file.stem,font=font,fill="black");thumbs.append(tile)
    canvas=Image.new("RGB",(2400,800*((len(thumbs)+2)//3)),"white")
    for i,tile in enumerate(thumbs):canvas.paste(tile,((i%3)*800,(i//3)*800))
    canvas.save(QA/"upgrade_figure_contact_color.png")
    ImageOps.grayscale(canvas).save(QA/"upgrade_figure_contact_gray.png")
    (QA/"upgrade_figure_export_checks.json").write_text(json.dumps({"figures":records},ensure_ascii=False,indent=2)+"\n")


def main():
    global OPTIONS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only",default=",")
    p.add_argument("--stats-root",type=Path,default=RECORD/"results/statistics/fixed_replay_v2")
    p.add_argument("--corr-root",type=Path,default=RECORD/"results/statistics/matched_train_v3")
    p.add_argument("--learning-root",type=Path,default=RECORD/"results/statistics/learning_curve_v2")
    p.add_argument("--prospective-root",type=Path)
    p.add_argument("--qa",action="store_true",help="Render actual PDFs and make color/grayscale review sheets; requires Poppler.")
    OPTIONS=p.parse_args()
    for path in [OUT,DATA,QA]:path.mkdir(parents=True,exist_ok=True)
    style()
    requested=[x.removeprefix("upgrade_") for x in OPTIONS.only.split(",") if x]
    names=requested or list(FUNCTIONS)
    unknown=set(names)-set(FUNCTIONS)
    if unknown:raise ValueError("Unknown figure names: "+str(unknown))
    manifest_path=QA/"upgrade_figure_manifest.json"
    old=json.loads(manifest_path.read_text()).get("figures",[]) if manifest_path.exists() else []
    for name in names:
        ACTIVE_INPUTS.clear()
        try:
            FUNCTIONS[name]()
            print("GENERATED upgrade_"+name,flush=True)
        except MissingInput as e:
            plt.close("all")
            MANIFEST.append({"name":"upgrade_"+name,"status":"skipped_missing_input","missing_input":str(e),"inputs":list(ACTIVE_INPUTS)})
            print("SKIPPED upgrade_"+name+": "+str(e),flush=True)
    fresh={r["name"]:r for r in MANIFEST};combined={r["name"]:r for r in old};combined.update(fresh)
    result={"schema_version":1,"generator":str(Path(__file__).relative_to(ROOT)),"generator_sha256":sha(__file__),
            "python":platform.python_version(),
            "dependencies":{"matplotlib":matplotlib.__version__,"numpy":np.__version__,"pandas":pd.__version__},
            "scope":"Plotting only; numerical results from independently scored sealed artifacts.",
            "figures":[combined["upgrade_"+name] for name in FUNCTIONS if "upgrade_"+name in combined]}
    manifest_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if OPTIONS.qa:qa_exports()
    print(json.dumps({"generated":sum(r["status"]=="generated" for r in MANIFEST),"skipped":sum(r["status"]!="generated" for r in MANIFEST)},ensure_ascii=False))


if __name__=="__main__":main()
