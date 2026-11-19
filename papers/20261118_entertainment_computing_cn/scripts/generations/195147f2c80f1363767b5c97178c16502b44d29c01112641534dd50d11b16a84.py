"""Evidence-only figures. No model fitting, scoring or answer-file access.

Explicit paths prevent output from moving when this workspace is copied.
Run with --repo-root REPO --out-root REPO/analysis/.../workspaces/F --qa.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle, Arc
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--repo-root", type=Path, required=True)
PARSER.add_argument("--out-root", type=Path, required=True)
PARSER.add_argument("--baseline-root", type=Path, default=Path(__file__).resolve().parents[1], help="Portable first-round data and font metadata; defaults to the script workspace.")
PARSER.add_argument("--only", default="")
PARSER.add_argument("--qa", action="store_true")
ARGS = PARSER.parse_args()
ROOT, WORK = ARGS.repo_root.resolve(), ARGS.out_root.resolve()
OUT, QA, DATA = WORK/"figures", WORK/"qa", WORK/"data/figure_data"
SOURCE = ROOT/"analysis/paper_upgrade_round2/20261118/workspaces"
RECORD = ROOT/"analysis/paper_upgrade_records/20261118"
for p in (OUT, QA, DATA):
    p.mkdir(parents=True, exist_ok=True)
BASELINE = ARGS.baseline_root.resolve()
for rel in ("data",):
    for source_file in (BASELINE/rel).rglob("*"):
        if source_file.is_file():
            destination=WORK/source_file.relative_to(BASELINE)
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copyfile(source_file,destination)
if not (QA/"font_check.json").exists():
    shutil.copyfile(BASELINE/"qa/font_check.json",QA/"font_check.json")

spec = importlib.util.spec_from_file_location("first_round_figures", Path(__file__).with_name("legacy_reference_figures.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.ROOT, old.PAPER, old.RECORD = ROOT, WORK, RECORD
old.OUT, old.QA, old.DATA = OUT, QA, DATA
old.FONTS = json.loads((QA/"font_check.json").read_text())
old.OPTIONS = argparse.Namespace(
    stats_root=RECORD/"results/statistics/fixed_replay_v2",
    corr_root=RECORD/"results/statistics/matched_train_v3",
    learning_root=RECORD/"results/statistics/learning_curve_v2",
    prospective_root=RECORD/"prospective",
)
old.style()
plt.rcParams.update({"text.color":"#24333F", "axes.labelcolor":"#24333F", "xtick.color":"#24333F", "ytick.color":"#24333F"})
BLUE, ORANGE, GRAY = old.BLUE, old.ORANGE, old.GRAY
INK, GRID, PALE = "#24333F", "#DFE5E9", "#F2F5F7"
TEAL = "#557F75"
save, bind, sha, read_csv, read_json = old.save, old.bind, old.sha, old.read_csv, old.read_json
clean, panel, arrow = old.clean, old.panel, old.arrow


def sealed(path: Path):
    """Verify declared source-output seals before binding the table."""
    path = Path(path)
    seals = [path.parent/"output_seal.json", path.parent/"evidence_seal.json", path.parent/"run_manifest.json"]
    matched = False
    for sp in seals:
        if not sp.exists():
            continue
        d = json.loads(bind(sp).read_text())
        for mapping in (d, d.get("files", {}), d.get("outputs_sha256", {})):
            if path.name in mapping and isinstance(mapping[path.name], str):
                if sha(path) != mapping[path.name]:
                    raise ValueError(f"Source seal mismatch: {path}")
                matched = True
    if not matched and path.parent.name not in {"B", "E"}:
        raise ValueError(f"No declared source seal for {path}")
    bind(path)
    # Keep a portable, hash-bound copy of exactly the table used by the figure.
    key = "round2_"+"_".join(path.relative_to(SOURCE).parts)
    local = DATA/key
    shutil.copyfile(path, local)
    bind(local)
    return path


def table(path, **kwargs):
    return pd.read_csv(sealed(path), float_precision="round_trip", **kwargs)


def diagram(height):
    fig, ax = plt.subplots(figsize=(16/2.54, height/2.54))
    ax.set(xlim=(0,160), ylim=(0,height*10))
    ax.axis("off")
    fig.subplots_adjust(left=.01,right=.99,bottom=.015,top=.985)
    return fig, ax


def text(ax,x,y,s,**kw):
    return ax.text(x,y,s,color=INK,fontsize=8.7,ha="left",va="center",linespacing=1.45,zorder=6,**kw)


def card(ax,x,y,w,h,title,*,body=None,color=BLUE,icon=None,depth=True,fs=9.2):
    """Native card, with a shallow paper edge rather than an equal-depth cube."""
    prefix = title.replace(" ","-")
    if depth:
        shadow = FancyBboxPatch((x+.65,y-.65),w,h,boxstyle="round,pad=0,rounding_size=1.1",facecolor="#E5EAED",edgecolor="none",zorder=1)
        shadow.set_gid(prefix+"-paper-shadow");ax.add_patch(shadow)
        edge=Polygon([(x+w,y+.6),(x+w+.75,y+1.2),(x+w+.75,y+h-1),(x+w,y+h-.5)],facecolor="#D2DFE8" if color==BLUE else "#E8D8C9",edgecolor="none",zorder=2)
        edge.set_gid(prefix+"-paper-edge");ax.add_patch(edge)
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.1",facecolor="white",edgecolor=color,lw=.8,zorder=3)
    p.set_gid(prefix+"-card");ax.add_patch(p)
    ax.plot([x+1,x+w-1],[y+h-6.5,y+h-6.5],color=GRID,lw=.55,zorder=4)
    ax.text(x+3,y+h-3.5,title,color=INK,fontsize=fs,va="center",zorder=6)
    if body:
        ax.text(x+3,y+(h-6.5)/2,body,color=INK,fontsize=8.5,va="center",linespacing=1.5,zorder=6)
    if icon:
        icon(ax,x+3,y+3,w-6,h-11,color)
    return x,y,w,h


def rank_prefix(ax,x,y,w,h,color):
    for j,(rank,length) in enumerate([("T1000",.90),("T2000",.90),("…",.90)]):
        yy=y+h-(j+.6)*h/3
        ax.add_patch(FancyBboxPatch((x,yy-1.8),w,3.6,boxstyle="round,pad=0,rounding_size=.55",facecolor=PALE,edgecolor="none",zorder=4))
        ax.text(x+1.5,yy,rank,fontsize=7.8,color=INK,va="center",zorder=6)
        start=x+w*.42
        ax.plot([start,x+w*length],[yy,yy],color=color,lw=1.05,zorder=5)
        ax.plot([x+w*length],[yy],"o",color=color,ms=2.4,zorder=5)


def metadata(ax,x,y,w,h,color):
    ax.add_patch(Rectangle((x,y),w*.34,h,facecolor=PALE,edgecolor=color,lw=.6,zorder=4))
    ax.plot([x,x+w*.34],[y+h*.7,y+h*.7],color=color,lw=.6,zorder=5)
    for a in [.08,.21]:
        ax.plot([x+w*a,x+w*a],[y+h*.85,y+h*1.08],color=color,lw=.9,zorder=5)
    for dx in [.08,.19,.29]:
        for dy in [.19,.43]:
            ax.add_patch(Circle((x+w*dx,y+h*dy),.4,color=color,zorder=5))
    ax.text(x+w*.43,y+h*.72,"活动／卡牌",fontsize=8.2,va="center",color=INK,zorder=6)
    ax.text(x+w*.43,y+h*.25,"国服发布过滤",fontsize=8.0,va="center",color=INK,zorder=6)


def ledger(ax,x,y,w,h,color):
    for j in [2,1,0]:
        ax.add_patch(FancyBboxPatch((x+.6*j,y+.5*j),w,h,boxstyle="round,pad=0,rounding_size=.5",facecolor="white",edgecolor=GRID,lw=.65,zorder=4+(2-j)*.1))
    for yy in [.22,.5,.78]:
        ax.plot([x+2,x+w*.37],[y+h*yy,y+h*yy],color=color,lw=.8,zorder=5)
        ax.plot([x+w*.45,x+w-2],[y+h*yy,y+h*yy],color=GRID,lw=.7,zorder=5)


def lock(ax,x,y,size,color=BLUE):
    ax.add_patch(Arc((x+size*.5,y+size*.7),size*.58,size*.66,theta1=0,theta2=180,color=color,lw=.9,zorder=7))
    ax.add_patch(FancyBboxPatch((x+size*.12,y),size*.76,size*.63,boxstyle="round,pad=0,rounding_size=.4",facecolor=PALE,edgecolor=color,lw=.8,zorder=7))
    ax.add_patch(Circle((x+size*.5,y+size*.36),size*.055,color=color,zorder=8))


def shield(ax,x,y,size,color=BLUE):
    pts=[(x,y+size),(x+size*.5,y+size*1.12),(x+size,y+size),(x+size*.86,y+size*.34),(x+size*.5,y),(x+size*.14,y+size*.34)]
    ax.add_patch(Polygon(pts,facecolor=PALE,edgecolor=color,lw=.8,zorder=4))
    ax.plot([x+size*.25,x+size*.44,x+size*.78],[y+size*.58,y+size*.36,y+size*.76],color=color,lw=1.1,zorder=5)


def scene_framework():
    bind(ROOT/"AGENTS.md");bind(ROOT/"analysis/v3_next_calibration.py")
    fig,ax=diagram(9.2)
    for x,lab in [(3,"截至观测的本期输入"),(58,"公开信息"),(113,"期初已结束历史")]:
        ax.text(x,88,lab,fontsize=8.8,color=GRAY)
    card(ax,3,52,44,30,"可见排名前缀",icon=rank_prefix)
    card(ax,58,52,44,30,"活动与卡牌",icon=metadata)
    card(ax,113,52,44,30,"已封存误差账本",icon=ledger)
    card(ax,3,14,44,22,"基础终分预测",body="两线 Context400\n其余排名共享树",fs=9.1)
    card(ax,58,14,44,22,"历史偏差校正",body="本期共用期初状态\n更新规则提前固定",color=ORANGE,fs=9.1)
    card(ax,113,14,44,22,"封存本期预测",body="先发出并保留版本\n结束后独立评分",fs=9.1)
    lock(ax,148,17,5)
    arrow(ax,[(25,52),(25,36)])
    arrow(ax,[(80,52),(80,42),(41,42),(41,36)])
    arrow(ax,[(135,52),(135,42),(91,42),(91,36)],color=ORANGE)
    arrow(ax,[(47,25),(58,25)])
    arrow(ax,[(102,25),(113,25)])
    ax.add_patch(FancyBboxPatch((58,1),99,7,boxstyle="round,pad=0,rounding_size=1",facecolor=PALE,edgecolor="none",zorder=2))
    lock(ax,61,2.1,3.8,GRAY)
    ax.text(67,4.6,"本期终分隔离；观测：活动×排名×时刻，评分：活动等权",fontsize=8.15,va="center",color=INK)
    save(fig,"scene_framework","fig:upgrade-scene","Distinguish observable prefixes, released history and the sealed forecast interface.","多排名终分预测的信息接口。排名叠层的等长横条表示相同实际观测截止，不表示分数大小；公开信息按国服发布时间过滤，历史账本只含期初合法释放的已结束活动。本期全时点共享同一期初校正状态；未知终分在预测阶段隔离。轻侧面仅表达对象层次，不编码数据。",kind="mechanism")


def anchor_difference():
    bind(ROOT/"analysis/v3_next_calibration.py")
    fig,ax=diagram(10.0)
    ax.text(3,95,"历史对数误差：正值表示基础预测偏高",fontsize=9,color=INK)
    card(ax,3,62,36,20,"T2000 历史",body="该线全部合法支持",fs=9.0)
    card(ax,3,28,36,20,"T1000 历史",body="同活动、同阶段配对",fs=9.0)
    card(ax,58,62,47,20,"慢锚点 A",fs=9.1)
    card(ax,58,28,47,20,"快差异 D",color=ORANGE,fs=9.1)
    for yy,a,k in [(70,.2,10),(36,.6,5)]:
        ax.text(61,yy,"更新率",fontsize=8.5,color=INK,va="center")
        ax.text(79,yy,rf"$\alpha={a}$",fontsize=9.5,color=INK,va="center")
        ax.text(61,yy-5,"收缩常数",fontsize=8.5,color=INK,va="center")
        ax.text(79,yy-5,rf"$k={k}$",fontsize=9.5,color=INK,va="center")
    # Mathematical variables are separate objects, so CJK never enters MathText.
    card(ax,121,43,36,32,"扣除重组偏差",fs=9.1)
    ax.text(124,60,"T2000",fontsize=8.5,color=INK)
    ax.text(139,60,r"$-A$",fontsize=10,color=BLUE)
    ax.text(124,51,"T1000",fontsize=8.5,color=INK)
    ax.text(139,51,r"$-(A+D)$",fontsize=9.5,color=ORANGE)
    arrow(ax,[(39,72),(58,72)])
    arrow(ax,[(39,38),(58,38)],color=ORANGE)
    arrow(ax,[(44,72),(48,72),(48,44),(58,44)],color=ORANGE)
    ax.text(52,55,"同活动差值",fontsize=8.0,color=GRAY,ha="center",rotation=90)
    arrow(ax,[(105,72),(113,72),(113,63),(121,63)])
    arrow(ax,[(105,38),(113,38),(113,50),(121,50)],color=ORANGE)
    ax.text(3,19,"先截断总偏差，再乘以 exp(−偏差)；最后施加当前分数与增长上界。",fontsize=8.5,color=INK)
    ax.text(3,12,"无成对支持时，T1000回退本线独立暖校正；锚点与差异支持数分别计算。",fontsize=8.4,color=INK)
    ax.plot([3,157],[7.5,7.5],color=GRID,lw=.65)
    ax.text(3,2,"同面板、同滤波、同收缩的代数负控",fontsize=8.3,color=GRAY)
    ax.text(90,2,r"$sF(e_2)+sF(e_1-e_2)=sF(e_1)$",fontsize=10,color=INK,va="baseline")
    save(fig,"anchor_difference","fig:upgrade-anchor","Preserve asymmetric update/support rules and the exact subtraction direction.","原锚点差异校正。T2000慢锚点使用该线全部合法历史，T1000快差异使用同活动同阶段的两线误差差；更新率为0.2/0.6，收缩常数为10/5。图中重组偏差在解码时扣除，T1000缺配对支持时保留本线独立回退。相同面板、线性滤波及收缩下满足底部恒等式。该图不代表另行评估的自适应凸组合。",kind="mechanism")


def prospective_loop():
    bind(ROOT/"analysis/paper_upgrade_prospective.py")
    status=read_json(RECORD/"prospective/status_v2.json","round2_prospective_status_v2.json")
    config=read_json(RECORD/"prospective/registry_v2/registry/config.json","round2_registry_v2_config.json")
    addendum=read_json(SOURCE/"E/confirmation_plan_addendum.json","round2_confirmation_plan_addendum.json")
    if status["events"] != 0 or addendum["target_events"] != 80:
        raise ValueError("Prospective status changed; revise figure deliberately")
    fig,ax=diagram(10.8)
    card(ax,3,72,43,23,"① 配置与来源登记",body="主比较与网格固定\n活动人口名单完整",fs=8.8)
    lock(ax,37,75,5)
    card(ax,59,72,43,23,"② 合格新活动",body="实际开始晚于登记\n缺失与失败均保留",fs=8.8)
    card(ax,115,72,42,23,"③ 准时预测封存",body="物理案例去重\n时间戳、方法与版本",fs=8.8)
    lock(ax,148,75,5)
    card(ax,115,29,42,23,"④ 核实终分到达",body="先预测，后释放答案\n保留实际到达日志",color=ORANGE,fs=8.8)
    card(ax,59,29,43,23,"⑤ 独立活动评分",body="共同案例与活动等权\n完整报告覆盖和漏报",fs=8.8)
    card(ax,3,29,43,23,"⑥ 后续合法反馈",fs=8.8)
    shield(ax,6,33,7)
    ax.text(16,41.5,"结束 < 下期开始",fontsize=8.1,color=INK,va="center")
    ax.text(16,34.5,"到达 < 下期开始",fontsize=8.1,color=INK,va="center")
    arrow(ax,[(46,83.5),(59,83.5)])
    arrow(ax,[(102,83.5),(115,83.5)])
    arrow(ax,[(136,72),(136,52)],color=ORANGE)
    arrow(ax,[(115,40.5),(102,40.5)])
    arrow(ax,[(59,40.5),(46,40.5)])
    arrow(ax,[(24.5,52),(24.5,61),(80.5,61),(80.5,72)],dashed=True)
    ax.text(74,57,"只影响后续活动；原预测封存不变",ha="center",fontsize=8.25,color=GRAY)
    ax.add_patch(FancyBboxPatch((3,5),154,15,boxstyle="round,pad=0,rounding_size=1.2",facecolor=PALE,edgecolor=GRID,lw=.5))
    ax.text(7,14.5,"当前新队列 0 期",fontsize=9.0,color=ORANGE,va="center")
    ax.text(58,14.5,"20期可行性登记保持；80期采集方案待独立激活",fontsize=8.5,color=INK,va="center")
    ax.text(7,8,"双时间守卫同时满足后，反馈才进入下一期的共同期初池。",fontsize=8.3,color=INK)
    save(fig,"prospective_loop","fig:upgrade-prospective","Expose forecast-before-feedback and two actual-time guards without claiming prospective accuracy.","前瞻记录与反馈闭环。活动须实际开始晚于配置登记，预测按物理案例去重后准时封存；终分保留可信来源及实际到达时刻。历史只有在活动结束和终分实际到达均严格早于下一期开始时释放。原20期可行性登记保持，独立80期采集方案尚待激活；当前新队列0期，无未来精度数值。",kind="mechanism",notes=[{"status":status,"pilot_events":config["plan"]["target_events"],"confirmation_target":addendum["target_events"]}])


def matched_correlations():
    f=read_csv(old.OPTIONS.corr_root/"matched_correlation_pairs.csv","matched_correlation_pairs.csv")
    ranks=sorted(set(f.rank_a)|set(f.rank_b));ns=f.n.unique()
    if len(ns)!=1 or len(ranks)!=18:
        raise ValueError("Matched support changed")
    fig,axes=plt.subplots(2,2,figsize=(16/2.54,15.4/2.54))
    cmap=LinearSegmentedColormap.from_list("correlation",[BLUE,"#FAFAFA",ORANGE])
    panels=[("log_final","终分水平"),("remaining_growth_p50","半程剩余增长"),("raw_error_phase1","基础外推误差"),("independent_error_phase1","独立校正后误差")]
    idx={r:i for i,r in enumerate(ranks)}
    for ax,(key,name),letter in zip(axes.flat,panels,"abcd"):
        m=np.eye(18)
        for r in f[f.panel==key].itertuples():
            m[idx[r.rank_a],idx[r.rank_b]]=m[idx[r.rank_b],idx[r.rank_a]]=r.spearman
        edge=np.arange(19)-.5
        im=ax.pcolormesh(edge,edge,m,vmin=-1,vmax=1,cmap=cmap,shading="flat",rasterized=False,antialiased=False)
        yy,xx=np.where(m<0);ax.scatter(xx,yy,marker="x",s=9,linewidths=.45,color=INK)
        ax.set(xlim=(-.5,17.5),ylim=(17.5,-.5),aspect="equal")
        ax.set_xticks(range(18),labels=[str(x) for x in ranks],rotation=90,fontsize=7.8)
        ax.set_yticks(range(18),labels=[str(x) for x in ranks],fontsize=7.8)
        ax.tick_params(length=2,pad=2)
        panel(ax,f"({letter}) {name}")
    fig.subplots_adjust(left=.085,right=.88,bottom=.11,top=.95,wspace=.37,hspace=.38)
    ca=fig.add_axes([.927,.25,.018,.48]);cb=fig.colorbar(im,cax=ca,ticks=[-1,-.5,0,.5,1],label="Spearman 相关系数")
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face")
    fig.text(.49,.042,f"四面板共用 {int(ns[0])} 活动、18排名；×表示负相关；T1500训练缺失",ha="center",fontsize=8.3,color=INK)
    fig.text(.49,.016,"T1000/T2000：Context400；其余排名：共享树。相关性只作描述。",ha="center",fontsize=8.1,color=GRAY)
    save(fig,"matched_correlations","fig:upgrade-correlation","Compare outcome and residual correlation on the identical event/rank support.","同一34期18排名完整面板的Spearman相关。终分、半程剩余增长、中期原始外推误差和独立校正后误差共用相同活动支持；叉号标记负相关。T1500因训练缺失不进入共同面板。T1000/T2000与其余排名基础路线不同，相关结构仅作描述。")


def sensitivity():
    frames={s:read_csv(old.OPTIONS.stats_root/f"{s}_metrics.csv",f"replay_{s}_metrics.csv").set_index("method") for s in ["validation","test"]}
    sweeps=[("anchor_alpha",[.1,.2,.3],.2,"锚点更新率"),("difference_alpha",[.3,.6,.9],.6,"差异更新率"),("anchor_shrinkage",[0,5,10,20],10,"锚点收缩常数"),("difference_shrinkage",[0,2.5,5,10],5,"差异收缩常数"),("minimum_pairs",[1,3,5,10],1,"最低配对活动数")]
    fig,axes=plt.subplots(3,2,figsize=(16/2.54,13.2/2.54))
    vals=[]
    for key,values,base,_ in sweeps:
        names=["legacy_pair" if v==base else f"pair_{key}_{v:g}".replace(".","p") for v in values]
        vals.extend(frames[s].loc[names,"event_macro_mape_pct"].to_numpy() for s in frames)
    low=min(np.min(a) for a in vals);high=max(np.max(a) for a in vals);pad=max(.08,(high-low)*.1)
    for ax,(key,values,base,label),letter in zip(axes.flat,sweeps,"abcde"):
        for split,col,ls,mk,lab in [("validation",BLUE,"--","o","重复使用验证"),("test",ORANGE,"-","s","已探索测试")]:
            names=["legacy_pair" if v==base else f"pair_{key}_{v:g}".replace(".","p") for v in values]
            y=frames[split].loc[names,"event_macro_mape_pct"].to_numpy()
            ax.plot(values,y,color=col,ls=ls,marker=mk,ms=4,label=lab)
        ax.axvline(base,color=GRAY,lw=.7,ls=":")
        ax.set(xticks=values,ylim=(low-pad,high+pad),xlabel=label)
        clean(ax);panel(ax,f"({letter})")
    axes[2,1].axis("off")
    h,l=axes[0,0].get_legend_handles_labels();axes[2,1].legend(h,l,loc="upper left",fontsize=8.7)
    axes[2,1].text(0,.55,"每次仅改变一项\n线段只连接已运行设置\n竖虚线：原配方\n测试结果未用于择参",fontsize=8.6,linespacing=1.7,transform=axes[2,1].transAxes,color=INK,va="top")
    fig.supylabel("活动等权 MAPE／%",x=.015,fontsize=9)
    fig.subplots_adjust(left=.105,right=.985,top=.95,bottom=.075,hspace=.60,wspace=.28)
    save(fig,"sensitivity","fig:upgrade-sensitivity","Retain every discrete one-factor setting on comparable axes without test selection.","完整预登记单因素敏感性。五个面板采用相同MAPE纵轴尺度；每次只改变一个参数，其他设置固定，竖虚线为原配方。点线连接实际运行值，重复使用验证与已探索测试分别标示。测试只作描述，没有按该图选择更新规则。")


def rank_stage():
    f=read_csv(old.OPTIONS.stats_root/"test_stage_rank.csv","test_stage_rank.csv")
    stagecol="stage_id" if "stage_id" in f else "stage"
    stages=["progress_10pct","progress_25pct","progress_50pct","progress_75pct","progress_90pct","progress_95pct"]
    ranks=sorted(f["rank"].unique());mat=np.full((len(ranks),6),np.nan);support=np.zeros_like(mat)
    for i,r in enumerate(ranks):
        for j,s in enumerate(stages):
            q=f[(f["rank"]==r)&(f[stagecol]==s)&f.method.isin(["ewma_reference","legacy_pair"])].set_index("method")
            if set(q.index)=={"ewma_reference","legacy_pair"}:
                mat[i,j]=q.loc["legacy_pair","event_macro_mape_pct"]-q.loc["ewma_reference","event_macro_mape_pct"]
                support[i,j]=q.loc["ewma_reference","events"]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,11.5/2.54),gridspec_kw={"width_ratios":[1,1]})
    cmap=LinearSegmentedColormap.from_list("rankstage",[BLUE,"white",ORANGE]);cmap.set_bad("#E5E5E5")
    lim=max(np.nanmax(np.abs(mat)),.01);edge=np.arange(7)-.5;rows=np.arange(len(ranks)+1)-.5
    im=ax.pcolormesh(edge,rows,np.ma.masked_invalid(mat),vmin=-lim,vmax=lim,cmap=cmap,shading="flat",rasterized=False,antialiased=False)
    sm=bx.pcolormesh(edge,rows,np.ma.masked_where(support==0,support),vmin=0,vmax=16,cmap="Greys",shading="flat",rasterized=False,antialiased=False)
    for a in (ax,bx):
        a.set(xlim=(-.5,5.5),ylim=(len(ranks)-.5,-.5),xticks=range(6),xlabel="活动进度／%")
        a.set_xticklabels(["10","25","50","75","90","95"])
        a.set_yticks(range(len(ranks)),labels=[f"T{r}" for r in ranks],fontsize=8.0)
        a.tick_params(length=2,pad=2)
    for i in range(len(ranks)):
        for j in range(6):
            if support[i,j]:
                bx.text(j,i,f"{support[i,j]:g}",ha="center",va="center",fontsize=8.1,color="white" if support[i,j]>8 else INK)
                v=mat[i,j]
                ax.text(j,i,f"{v:+.2f}" if abs(v)>1e-8 else "·",ha="center",va="center",fontsize=8.0,color="white" if abs(v)>.65*lim else (INK if abs(v)>1e-8 else GRAY))
            else:
                ax.text(j,i,"—",ha="center",va="center",fontsize=8,color=GRAY)
    panel(ax,"(a) 锚点差异−独立／百分点");panel(bx,"(b) 同案例的支持活动数")
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025);sb=fig.colorbar(sm,ax=bx,fraction=.045,pad=.025,ticks=[0,8,16])
    for b in (cb,sb):b.solids.set_rasterized(False);b.solids.set_edgecolor("face")
    fig.subplots_adjust(left=.105,right=.985,top=.93,bottom=.13,wspace=.40)
    fig.text(.5,.025,"· = 0（两方法相同）；— = 缺失；负差表示MAPE较低。总体仍按物理案例去重。",ha="center",fontsize=8.2,color=INK)
    save(fig,"rank_stage","fig:upgrade-rank-stage","Keep all rank/stage effects and their exact activity support visible.","排名与进度条件下的活动等权MAPE差和共同支持。负差表示锚点差异法较低；点号为零差，横线为缺失，T1000/T2000之外两方法输出相同。右侧逐格报告相同案例的支持活动数。时间分面使用网格成员关系，总体按物理案例去重。T1500冷启动与缺失条件完整保留。")


def tail_interval():
    f=read_csv(old.OPTIONS.stats_root/"test_metrics.csv","replay_test_metrics.csv").set_index("method")
    rroot=SOURCE/"E/reliability_002"
    s=table(rroot/"interval_summary.csv").sort_values("nominal_coverage")
    c=table(rroot/"interval_full_group_conditions.csv")
    w=table(rroot/"interval_conditions.csv")
    fig,axes=plt.subplots(2,2,figsize=(16/2.54,14/2.54));ax,bx,cx,dx=axes.flat
    for name in old.COLORS:
        row=f.loc[name];ls,mk=old.STYLES[name]
        ax.plot(range(3),[100*row[k]/f.loc["uncorrected",k] for k in ["event_macro_mape_pct","p90_ape_pct","rmse_ep"]],color=old.COLORS[name],ls=ls,marker=mk,ms=4,label=old.LABELS[name])
    ax.set_xticks(range(3),labels=["MAPE","P90 APE","RMSE"])
    ax.set_ylabel("相对基础预测／%（基础=100）");clean(ax);panel(ax,"(a) 点预测与尾部取舍")
    ax.legend(fontsize=8.0,loc="upper left")
    x=s.nominal_coverage.to_numpy()*100
    for key,lo,hi,col,mk,ls,lab,off in [
        ("event_equal_point_coverage","event_equal_point_coverage_low95","event_equal_point_coverage_high95",BLUE,"o","--","单前缀",-.5),
        ("event_equal_group_joint_coverage","event_equal_group_joint_coverage_low95","event_equal_group_joint_coverage_high95",ORANGE,"s","-","排名×阶段联合",0),
        ("whole_event_joint_coverage","whole_event_cp_low95","whole_event_cp_high95",GRAY,"^",":","整活动联合",.5)]:
        y=s[key].to_numpy()*100
        bx.errorbar(x+off,y,yerr=[y-s[lo].to_numpy()*100,s[hi].to_numpy()*100-y],color=col,marker=mk,ls=ls,capsize=2,elinewidth=.8,label=lab,ms=4)
    bx.plot([77,93],[77,93],color=GRAY,ls=":",lw=.6)
    bx.set(xlim=(77,93),ylim=(-4,105),xticks=[80,90],xlabel="名义覆盖／%",ylabel="活动等权经验覆盖／%")
    bx.legend(fontsize=7.9,loc="center right");clean(bx);panel(bx,"(b) 三种覆盖目标")
    bx.text(80.8,4,"0/16期",fontsize=8,color=GRAY);bx.text(90.8,10,"1/16期",fontsize=8,color=GRAY)
    for nom,col,mk,off in [(.8,BLUE,"o",-.06),(.9,ORANGE,"s",.06)]:
        a=c[(c.dimension=="phase")&(c.nominal_coverage==nom)].copy();a["order"]=a.value.astype(int);a=a.sort_values("order")
        xx=np.arange(3)+off;y=a.event_equal_full_group_joint_coverage.to_numpy()*100
        cx.errorbar(xx,y,yerr=[y-a.low95.to_numpy()*100,a.high95.to_numpy()*100-y],color=col,marker=mk,ls="--" if nom==.8 else "-",capsize=2,elinewidth=.8,label=f"名义{nom*100:g}%",ms=4)
        cx.axhline(nom*100,color=col,lw=.55,ls=":")
    cx.set(xticks=range(3),xticklabels=["早期 <25%\nn=16","中期 25–90%\nn=16","末期 ≥90%\nn=15"],ylim=(65,102),ylabel="完整排名×阶段联合覆盖／%")
    cx.legend(fontsize=8,loc="lower left");clean(cx);panel(cx,"(c) 阶段条件覆盖")
    for i,(src,lab,col,mk) in enumerate([("rank_phase","本排名×阶段",BLUE,"o"),("all_event_phase_pool","跨排名回退池",ORANGE,"s")]):
        a=w[(w.dimension=="source")&(w.value==src)].sort_values("nominal_coverage")
        xx=np.arange(2)+(-.1 if i==0 else .1);y=a.event_equal_relative_width_pct.to_numpy()
        dx.errorbar(xx,y,yerr=[y-a.event_equal_relative_width_pct_low95.to_numpy(),a.event_equal_relative_width_pct_high95.to_numpy()-y],color=col,marker=mk,ls="--" if i==0 else "-",capsize=2,elinewidth=.8,label=lab,ms=4)
        for xx1,row in zip(xx,a.itertuples()):dx.text(xx1,row.event_equal_relative_width_pct+8,f"{row.n_events}期",ha="center",fontsize=8,color=col)
    dx.set(xticks=range(2),xticklabels=["名义80%","名义90%"],ylim=(0,190),ylabel="活动等权平均相对宽度／%")
    dx.legend(fontsize=8,loc="upper left");clean(dx);panel(dx,"(d) 高覆盖回退的宽度代价")
    fig.subplots_adjust(left=.10,right=.985,top=.95,bottom=.11,wspace=.38,hspace=.49)
    fig.text(.5,.026,"16期固定封存预测；组/点误差棒为活动重采样95%区间，整活动为iid二项参考区间。",ha="center",fontsize=8.15,color=INK)
    fig.text(.5,.007,"回退池组联合覆盖为100%，仅4/10期；宽度与覆盖均受排名、阶段及支持组成影响。",ha="center",fontsize=8.0,color=GRAY)
    save(fig,"tail_interval","fig:upgrade-interval","Separate three coverage estimands, phase conditions and the cost of fallback width.","固定封存预测的点/尾部取舍与经验区间可靠性。（a）基础预测归一为100。（b）单前缀、排名×阶段全部前缀联合、整活动联合为不同目标；整活动仅0/16与1/16期覆盖。（c）完整组阶段条件覆盖，阶段按实际进度25%和90%划分。（d）本排名阶段来源与跨排名回退的活动等权相对宽度；回退组联合覆盖均100%，但仅涉及4/10期且较宽。点和组误差棒为活动配对探索性95%重采样区间，整活动为iid二项参考Clopper–Pearson区间，活动时间依赖另作敏感性分析。未来覆盖另待新活动验证。",notes=["Condition tables retain complete groups; no sample selection by support or error.","Finite interval rate100%; unavailable and unbounded cases0 at both levels."])


def adaptive_diagnostics():
    f=table(SOURCE/"B/run_v2_001/summary.csv")
    d=table(SOURCE/"B/lambda_diagnostics.csv")
    bind(SOURCE/"B/run_v2_001/selection_seal.json")
    names=["independent_fixed_recipe","fixed_quarter","fixed_half","fully_shared_pair","risk_k4","risk_k12","risk_recent_k8","risk_gated_k8","same_panel_independent","risk_only_T1000","risk_only_T2000","adjacent_risk_k8"]
    labels=["独立固定配方","固定1/4共享","固定1/2共享","完全共享","风险 k=4","风险 k=12","近期风险 k=8","稳健门控 k=8","同面板独立","仅T1000风险共享","仅T2000风险共享","邻排名风险池"]
    fig=plt.figure(figsize=(16/2.54,15.2/2.54));gs=fig.add_gridspec(3,2,height_ratios=[.7,1.5,1.2],width_ratios=[1.3,1])
    ax=fig.add_subplot(gs[0,:]);ax.set(xlim=(0,160),ylim=(0,24));ax.axis("off")
    card(ax,1,4,36,16,"独立专家 I",body="本线合法误差",depth=False,fs=8.8)
    card(ax,52,4,36,16,"共享专家 S",body="完整成对历史",depth=False,fs=8.8)
    ax.text(106,18,r"$b=(1-\lambda)I+\lambda S$",fontsize=11,color=INK,va="center")
    ax.text(106,8,"权重只由过去已发误差决定",fontsize=8.5,color=INK,va="center")
    arrow(ax,[(37,12),(45,12),(45,23),(99,23),(99,17)])
    arrow(ax,[(88,12),(99,12)])
    panel(ax,"(a) 自适应凸组合与原锚点法分别评估",y=1.04)
    bx=fig.add_subplot(gs[1:,:1]);yy=np.arange(12)
    for block,col,mk,off,lab in [("selection",BLUE,"o",-.10,"次10期：选择"),("diagnostic",ORANGE,"s",.10,"末10期：诊断")]:
        p=f[(f.block==block)&(f.panel=="main_pair")].set_index("candidate").loc[names]
        bx.plot(p.event_macro_mape,yy+off,ls="none",marker=mk,color=col,ms=4,label=lab)
    # All12 candidates appear in their registered order, including exact ties.
    bx.set(yticks=yy,yticklabels=labels,xlabel="两线活动等权 MAPE／%")
    bx.invert_yaxis();clean(bx,"x");panel(bx,"(b) 12个预登记候选")
    bx.legend(fontsize=8.1,loc="lower right",bbox_to_anchor=(1,1.08),ncol=2)
    bx.text(.02,-.14,"选择块选中独立配方；末段不重新选择。",transform=bx.transAxes,fontsize=8.2,color=INK)
    cx=fig.add_subplot(gs[1,1]);p=d[(d.block=="diagnostic")&d.candidate.isin(["risk_k4","risk_k12","risk_recent_k8","risk_gated_k8"])].set_index("candidate")
    ks=["risk_k4","risk_k12","risk_recent_k8","risk_gated_k8"]
    for j,k in enumerate(ks):
        r=p.loc[k];cx.plot([0,r.lambda_max],[j,j],color=GRID,lw=3,zorder=1)
        cx.plot(r.lambda_mean,j,"s" if k=="risk_gated_k8" else "o",color=ORANGE if k=="risk_gated_k8" else BLUE,ms=4,zorder=3)
        cx.text(.62,j,f"{int(r.lambda_positive_states)}/{int(r.task_states)}",fontsize=8.2,va="center",color=INK)
    cx.set(yticks=range(4),yticklabels=["全历史 k4","全历史 k12","近期 k8","门控 k8"],xlim=(-.02,.87),xticks=[0,.3,.6],xlabel="共享权重（均值／最大值）")
    cx.invert_yaxis();clean(cx,"x");panel(cx,"(c) 末段60个任务状态")
    dx=fig.add_subplot(gs[2,1]);dx.axis("off")
    dx.text(0,.98,"稳健开关：末段 0/30 打开",fontsize=9.0,color=ORANGE,transform=dx.transAxes,va="top")
    dx.text(0,.79,"与独立相同来自全部关闭。\n共享尚未显示稳定获益。",fontsize=8.6,color=INK,linespacing=1.6,transform=dx.transAxes,va="top")
    dx.text(0,.43,"近期有效支持约12期；\n删一活动的权重跨度可达0.80。\n状态数不能当独立活动数。",fontsize=8.4,color=INK,linespacing=1.6,transform=dx.transAxes,va="top")
    fig.subplots_adjust(left=.23,right=.985,bottom=.10,top=.95,hspace=.62,wspace=.58)
    save(fig,"adaptive_diagnostics","fig:round2-adaptive","Explain why past-issued adaptive sharing did not displace the independent reference.","训练期自适应共享诊断。35期按实际时间分为15期历史、10期选择、10期末段诊断；两线选择和诊断分别372/352唯一案例。全部12候选按预登记顺序呈现，选择块选中独立固定配方，末段未重选。λ面板按活动×排名×阶段去重为60任务状态；横线至最大λ，点为均值，右侧为正λ状态数。门控末段按活动×阶段0/30打开，与独立相同源于全部关闭。删一跨度是敏感性而非置信区间。新增结果属于训练期回溯时间外推诊断。",notes=["The nominal19-rank panel observes18 ranks; T1500 absent.","B35 and C12 events overlap; their union is35."])


def fairness_diagnostics():
    croot=SOURCE/"C/evidence_001"
    f=table(croot/"C2_parameter_and_MAC_contrasts.csv")
    budget=table(croot/"C2_budget_totals.csv").set_index("structure")
    q=table(SOURCE/"E/statistics_002/C2_paired.csv",dtype={"endpoint":str})
    hist=pd.read_csv(bind(SOURCE/"C/runs/c2_002/history_groups.csv"))
    groups=["N32full","N16span","N16recent"];gl=["32期完整","16期同跨度","16期近期"]
    fig=plt.figure(figsize=(16/2.54,13.2/2.54));gs=fig.add_gridspec(2,2,height_ratios=[1.1,1])
    ax=fig.add_subplot(gs[0,0]);mat=np.zeros((3,3))
    for i,g in enumerate(groups):
        for j,ep in enumerate([32,44,56]):mat[i,j]=f[(f.group==g)&(f.endpoint==ep)].shared60_minus_independent_pp.iloc[0]
    lim=np.max(np.abs(mat));cmap=LinearSegmentedColormap.from_list("fairness",[BLUE,"white",ORANGE])
    im=ax.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,mat,vmin=-lim,vmax=lim,cmap=cmap,shading="flat",rasterized=False,antialiased=False)
    for i in range(3):
        for j in range(3):ax.text(j,i,f"{mat[i,j]:+.2f}",fontsize=9,ha="center",va="center",color="white" if abs(mat[i,j])>.65*lim else INK)
    ax.set(xticks=range(3),xticklabels=["32","44","56"],yticks=range(3),yticklabels=gl,xlabel="训练截止活动序位",ylim=(2.5,-.5))
    panel(ax,"(a) 共享60−独立60／百分点")
    bx=fig.add_subplot(gs[0,1])
    sub=q[(q.endpoint=="all12")&(q.seed=="mean")&(q.kind=="exact_parameter_match")&(q.metric=="event_macro_MAPE_pct")]
    for i,g in enumerate(groups):
        a=sub[sub.left.eq(g+"_independent60_mean")&sub.right.eq(g+"_shared60_mean")]
        for block,col,lw,off in [(1,BLUE,2,-.05),(5,GRAY,.9,.07)]:
            r=a[a.block==block].iloc[0]
            bx.plot([-r.high95,-r.low95],[i+off,i+off],color=col,lw=lw)
        bx.plot(-a.iloc[0].difference,i,"o",color=BLUE,ms=4)
    bx.axvline(0,color=GRAY,ls="--",lw=.7)
    bx.set(yticks=range(3),yticklabels=gl,xlabel="共享60−独立60 MAPE差／百分点",ylim=(2.5,-.5))
    clean(bx,"x");panel(bx,"(b) 合并12查询活动的95%区间")
    bx.plot([],[],color=BLUE,lw=2,label="活动配对");bx.plot([],[],color=GRAY,lw=.9,label="5活动循环块")
    bx.legend(fontsize=8,loc="lower center",bbox_to_anchor=(.5,-.60))
    cx=fig.add_subplot(gs[1,0]);mnames=["independent60","shared60","shared30"];xx=np.arange(3)
    for key,col,mk,ls,lab in [("forward_linear_MACs",BLUE,"o","-","训练前向线性MAC"),("parameter_updates",ORANGE,"s","--","标量参数更新数"),("optimizer_steps",GRAY,"^",":","优化器step次数")]:
        ys=[budget.loc[m,key]/budget.loc["independent60",key] for m in mnames]
        cx.plot(xx,ys,color=col,marker=mk,ls=ls,label=lab,ms=4)
    cx.set(xticks=xx,xticklabels=["独立60轮","共享60轮","共享30轮"],ylim=(0,2.35),xlim=(-.1,3.55),ylabel="计数／独立60轮计数")
    clean(cx);panel(cx,"(c) 相同参数量仍有预算差异")
    for y,col,lab in [(budget.loc["shared30","forward_linear_MACs"]/budget.loc["independent60","forward_linear_MACs"],BLUE,"前向MAC"),(.5,ORANGE,"参数更新"),(.25,GRAY,"step次数")]:
        cx.text(2.14,y,lab,fontsize=8.0,color=col,va="center")
    dx=fig.add_subplot(gs[1,1])
    for g,col,mk,ls,lab in zip(groups,[GRAY,BLUE,ORANGE],["^","o","s"],[":","-","--"],gl):
        a=hist[hist.group==g].sort_values("endpoint")
        dx.plot(a.endpoint,a.span_days,color=col,marker=mk,ls=ls,ms=4,label=lab)
    dx.set(xticks=[32,44,56],xlabel="训练截止活动序位",ylabel="训练活动实际开始跨度／d",ylim=(0,380))
    clean(dx);panel(dx,"(d) 数量与跨度分开记录")
    dx.legend(fontsize=8,loc="lower right")
    fig.subplots_adjust(left=.155,right=.985,top=.94,bottom=.13,wspace=.52,hspace=.82)
    fig.text(.5,.04,"同一单元总参数32d+66（d=56或57）；三种子最终分数均值；108次真实拟合。",ha="center",fontsize=8.2,color=INK)
    fig.text(.5,.012,"每截止块仅4查询活动；三块合计12期，共享60轮赢3/9格，近似MAC共享30轮9/9格较差。",ha="center",fontsize=8.0,color=GRAY)
    save(fig,"fairness_diagnostics","fig:round2-fairness","Separate parameter count, optimization effort, forward compute, history count and span.","容量、预算与时间支持的训练期诊断。（a）同一截止和历史组内总参数精确匹配，格值为共享60轮减独立60轮MAPE。（b）三块合并12查询活动的配对与5活动循环块探索性95%区间，均跨零；源表的独立减共享方向在绘图时取负。（c）训练前向线性MAC、标量参数更新和optimizer.step为不同计数；共享30轮近似匹配MAC，但每参数更新数减半。（d）32完整与16同跨度共享首末训练日期，16近期具有较短跨度；组成和类别支持仍变化。三种子42/123/2026最终分数均值，各单元4查询活动。",notes=["MAC counts exclude backward, activations, bias, optimizer and preprocessing.","Total parameters are matched within a cell, not across different feature vocabularies.","The12 query events are a subset of B35."])


def calibrated_transfer():
    droot=SOURCE/"D"
    prof=table(droot/"calibration_run_v2_001/calibration_profile.csv")
    moments=table(droot/"calibration_run_v2_001/paired_diagnostics.csv")
    sim=table(droot/"simulation_run_v2_001/failure_regions.csv")
    transfer=table(droot/"posthoc_transfer_run_001/event_comparisons.csv").sort_values("query_index")
    summary=json.loads(sealed(droot/"posthoc_transfer_run_001/summary.json").read_text())
    fig,axes=plt.subplots(2,2,figsize=(16/2.54,14.2/2.54));ax,bx,cx,dx=axes.flat
    # Profile the nuisance grid by the preregistered trace(Q)/trace(R) ratio.
    # Ratios on the rational grid can differ by floating arithmetic only. The
    # 1e-12 key is far below the registered grid spacing; raw support points stay
    # unchanged. Save the aggregation so the visual profile is independently
    # inspectable and never creates vertical spikes from duplicate ratios.
    profile_keys=prof.assign(q_r_ratio_key=prof.q_r_ratio.round(12))
    p=profile_keys.groupby("q_r_ratio_key",as_index=False).agg(two_delta_nll=("two_delta_nll","min"),grid_points=("grid_index","size")).sort_values("q_r_ratio_key")
    old.derived(p,"round2_profile_ratio_curve.csv")
    ax.scatter(prof.q_r_ratio,1+prof.two_delta_nll,color="#CCD6DC",s=4,marker=".",label="900格")
    ax.plot(p.q_r_ratio_key,1+p.two_delta_nll,color=BLUE,marker="o",ms=3,label="剖面最小值")
    supported=prof[prof.support]
    ax.scatter(supported.q_r_ratio,1+supported.two_delta_nll,color=ORANGE,s=13,marker="s",label="17支持点",zorder=4)
    ax.axhline(11,color=GRAY,lw=.65,ls=":")
    ax.set_xscale("symlog",linthresh=.5);ax.set_yscale("log")
    ax.set(xlabel="trace(Q) / trace(R)",ylabel=r"$1+2\Delta\mathrm{NLL}$",xlim=(0,float(prof.q_r_ratio.max())*1.1),xticks=[0,1,10,100])
    ax.set_xticklabels(["0","1","10","100"])
    clean(ax);panel(ax,"(a) 受限模型的描述性剖面")
    ax.legend(fontsize=7.5,loc="upper center",bbox_to_anchor=(.5,-.24),ncol=3,handlelength=1.4,columnspacing=.8)
    fig.text(.105,.525,"支持比值1–45；描述性条件集合（未校准置信覆盖）",fontsize=7.8,color=INK,va="top")
    r=moments[(moments.segment=="calibration")&(moments.phase==1)].iloc[0]
    vals=np.array([r.R_psd_00,r.R_psd_11])*1000
    bx.bar([0,1],vals,color=[BLUE,ORANGE],edgecolor=INK,lw=.6,width=.58)
    for j,v in enumerate(vals):bx.text(j,v*.84 if j==0 else v+.05*vals.max(),f"{v:.3f}",fontsize=8.5,ha="center",color="white" if j==0 else INK)
    bx.set(xticks=[0,1],xticklabels=["T1000","T2000"],ylim=(0,vals.max()*1.78),ylabel="观测噪声矩估计对角 ×10³")
    clean(bx);panel(bx,"(b) 等噪声限制与实际矩诊断")
    bx.text(.03,.94,f"对角比 {vals.max()/vals.min():.2f}",transform=bx.transAxes,fontsize=9,color=INK,va="top")
    bx.text(.03,.81,"剖面模型限制两线方差相同；\n它覆盖的机制范围有限。",transform=bx.transAxes,fontsize=8.4,linespacing=1.6,color=INK,va="top")
    methodstyles=[("fully_shared_pair","完全共享",GRAY,"x"),("fixed_half","固定1/2共享",ORANGE,"s"),("risk_recent_k8","近期风险共享",TEAL,"^"),("fitted_fullcov_DLM","拟合DLM",BLUE,"o")]
    # Every registered regime appears once for each method; axes show relative risks.
    for method,lab,col,mk in methodstyles:
        a=sim[(sim.method==method)&sim.metric.isin(["state_mse","synthetic_unbounded_ape_pct"])]
        pivot=a.pivot(index="regime",columns="metric",values="mean")
        baseline=sim[(sim.method=="independent_ewma")&sim.metric.isin(["state_mse","synthetic_unbounded_ape_pct"])].pivot(index="regime",columns="metric",values="mean")
        z=100*(pivot/baseline-1)
        if len(z)!=24:raise ValueError("Expected all24 simulation regimes")
        cx.scatter(z.state_mse,z.synthetic_unbounded_ape_pct,color=col,marker=mk,s=17,linewidths=.7,facecolors="none" if mk!="x" else col,label=lab)
    cx.axhline(0,color=GRAY,lw=.65,ls="--");cx.axvline(0,color=GRAY,lw=.65,ls="--")
    cx.set_xscale("symlog",linthresh=10);cx.set_yscale("symlog",linthresh=10)
    cx.set(xlabel="相对独立的状态MSE变化／%",ylabel="相对独立的无界APE变化／%")
    clean(cx);panel(cx,"(c) 两种损失的方向比较")
    cx.legend(fontsize=7.6,loc="lower right")
    cx.text(.02,.98,"DLM：状态22/24改善\nAPE14/24改善",transform=cx.transAxes,fontsize=7.9,color=INK,ha="left",va="top",linespacing=1.5)
    xx=np.arange(len(transfer));yy=transfer.MAPE_minus_I.to_numpy()
    dx.bar(xx,yy,color=[BLUE if z<0 else ORANGE for z in yy],edgecolor=INK,lw=.5,width=.65)
    dx.axhline(0,color=GRAY,lw=.7)
    dx.axhline(yy.mean(),color=INK,lw=.8,ls="--")
    dx.set(xticks=xx,xticklabels=[str(e) for e in transfer.event],ylabel="固定DLM−独立 MAPE差／百分点",xlabel="末10期活动（按实际开始）",ylim=(min(-.9,float(yy.min())*1.2),float(yy.max())*1.55))
    dx.tick_params(axis="x",labelrotation=55,labelsize=8)
    clean(dx);panel(dx,"(d) 事后固定迁移：真实phase 1")
    m=summary["event_macro"];iv=m["case_mape"]-m["MAPE_minus_I"]
    dx.text(.03,.97,f"DLM {m['case_mape']:.3f}%；独立 {iv:.3f}%\n10活动／120案例；平均差 +{yy.mean():.3f} pp",transform=dx.transAxes,fontsize=8.3,va="top",linespacing=1.5,color=INK)
    fig.subplots_adjust(left=.105,right=.985,top=.94,bottom=.18,wspace=.42,hspace=.62)
    fig.text(.5,.035,"模拟：24条件×300路径，状态MSE与无界APE；真实：10活动、120有界案例。",ha="center",fontsize=8.15,color=INK)
    fig.text(.5,.011,"真实迁移为事后一次固定配方，收益按活动等权MAPE检验。",ha="center",fontsize=8.0,color=GRAY)
    save(fig,"calibrated_transfer","fig:round2-transfer","Expose profile uncertainty, model restriction, metric discordance and failed real transfer.","真实残差约束的机制诊断与固定迁移。（a）900点受限高斯状态模型网格，描绘trace(Q)/trace(R)下的剖面最小值及2ΔNLL≤10的17个描述性支持点；比值按小数12位等价键稳定分组，支持散点保留原值，阈值的置信覆盖尚待校准。（b）前25完整活动对中期的PSD矩估计噪声对角不相同，与剖面采用的等方差限制形成解释边界。（c）四方法各完整24条件，每条件300路径；双轴采用零附近线性、远端对数的对称尺度，左下表示两目标均改善，两种损失按各自口径解释。（d）使用前25期固定参数的一次事后DLM真实迁移，在后10期phase1的120案例上高于独立法；全部活动保留，图中平均差不用于再选择。",notes=["Mixed real-data/synthetic mechanism evidence. No new fits or paths are generated by this script.","Simulation paths are independent within regimes; real activities are the independent observational units.","round(q_r_ratio,12) only stabilizes numerical grouping on the registered discrete grid; all17 support points retain their original ratio coordinates."])


FUNCTIONS=dict(old.FUNCTIONS)
FUNCTIONS.update({f.__name__:f for f in [scene_framework,anchor_difference,prospective_loop,matched_correlations,sensitivity,rank_stage,tail_interval,adaptive_diagnostics,fairness_diagnostics,calibrated_transfer]})

UNITS={
    "scene_framework":"Mechanism only; equal prefix bars indicate a shared observation cutoff, not score magnitude.",
    "information_timeline":"Activity count; chronology is schematic, not to physical time scale.",
    "anchor_difference":"Dimensionless log-error bias; update rates and shrinkage constants dimensionless; support in activities.",
    "sampling_diagnosis":"Observations per activity; observation interval in minutes; observations per day.",
    "activity_forecasts":"Score in million EP; elapsed time in days.",
    "matched_correlations":"Spearman correlation, dimensionless; support in activities.",
    "eventwise_improvement":"MAPE difference in percentage points; cumulative absolute contribution in percent.",
    "sensitivity":"Activity-macro MAPE in percent; rates/shrinkage dimensionless; minimum support in activities.",
    "structural_ablations":"Activity-macro MAPE difference in percentage points; exploratory95% intervals.",
    "training_size":"Training activities; validation activity-macro MAPE in percent.",
    "rank_stage":"MAPE difference in percentage points; progress in percent; support in activities.",
    "neural_comparison":"MAPE difference in percentage points; exploratory95% intervals.",
    "synthetic_sharing":"Dimensionless log-state MSE difference multiplied by1000; variance units0.0001.",
    "tail_interval":"Relative point metrics, coverage and relative interval width in percent; support in activities.",
    "prospective_loop":"Activity counts and strict actual-time inequalities; no prospective accuracy values.",
    "adaptive_diagnostics":"Activity-macro MAPE in percent; mixture weight dimensionless; support in task states or activities as labelled.",
    "fairness_diagnostics":"MAPE difference in percentage points; compute/update ratios dimensionless; actual-start span in days.",
    "calibrated_transfer":"Dimensionless trace ratio and NLL; log-error noise variance times1000; simulation relative risk changes in percent; real MAPE difference in percentage points.",
}


def export_qa():
    old.qa_exports()
    # Fixed Brettel-style linear approximations are visual stress tests, not diagnoses.
    matrices={"deuteranopia":np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]]),
              "protanopia":np.array([[.152286,1.052583,-.204868],[.114503,.786281,.099216],[-.003882,-.048116,1.051998]])}
    for mode,matrix in matrices.items():
        im=Image.open(QA/"upgrade_figure_contact_color.png").convert("RGB")
        arr=np.asarray(im,dtype=float)/255
        lin=np.where(arr<=.04045,arr/12.92,((arr+.055)/1.055)**2.4)
        lin=np.clip(lin@matrix.T,0,1)
        rgb=np.where(lin<=.0031308,lin*12.92,1.055*lin**(1/2.4)-.055)
        Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(QA/f"upgrade_figure_contact_{mode}.png")


def main():
    requested=[x.removeprefix("upgrade_") for x in ARGS.only.split(",") if x]
    names=requested or list(FUNCTIONS)
    if set(names)-set(FUNCTIONS):raise ValueError("Unknown semantic figure")
    manifest_path=QA/"upgrade_figure_manifest.json"
    previous=json.loads(manifest_path.read_text()).get("figures",[]) if manifest_path.exists() else []
    old.MANIFEST.clear()
    generation_source=WORK/"scripts/generations"/(sha(__file__)+".py")
    generation_source.parent.mkdir(parents=True,exist_ok=True)
    if not generation_source.exists():shutil.copyfile(__file__,generation_source)
    for name in names:
        old.ACTIVE_INPUTS.clear()
        bind(generation_source);bind(Path(__file__).with_name("legacy_reference_figures.py"))
        FUNCTIONS[name]()
        print("GENERATED upgrade_"+name,flush=True)
    combined={r["name"]:r for r in previous}
    combined.update({r["name"]:r for r in old.MANIFEST})
    for key,item in combined.items():item["units"]=UNITS[key.removeprefix("upgrade_")]
    result={"schema_version":2,"generator":str(Path(__file__).relative_to(ROOT)),"generator_sha256":sha(__file__),
            "explicit_repo_root":str(ROOT),"explicit_output_root":str(WORK),"python":platform.python_version(),
            "dependencies":{"matplotlib":matplotlib.__version__,"numpy":np.__version__,"pandas":pd.__version__},
            "scope":"Plotting only, no fitting/scoring/answer access. Original15 preserved plus3 question-driven figures.",
            "figures":[combined["upgrade_"+n] for n in FUNCTIONS if "upgrade_"+n in combined]}
    manifest_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if ARGS.qa:export_qa()
    print(json.dumps({"generated":len(old.MANIFEST),"total":len(result["figures"])},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
