"""Publication plots from extracted frozen tables; no model execution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
D, OUT = PAPER/'data', PAPER/'figures'
FONTS = json.loads((PAPER/'qa/font_check.json').read_text())
for key in ['latin','cjk']:
    fm.fontManager.addfont(FONTS[key]['path'])
plt.rcParams.update({'font.family':FONTS['font_family_order'], 'font.size':9,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.linewidth':.7,'lines.linewidth':1.2,'lines.markersize':3.5,
    'legend.frameon':False,'mathtext.fontset':'cm','axes.unicode_minus':True,
    'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})
COLORS = {'uncorrected':'#737373','ewma_reference':'#456D90','legacy_pair':'#B36B3D'}
LABELS = {'uncorrected':'未校正基线','ewma_reference':'独立滚动校正','legacy_pair':'双线组合'}
STYLES = {'uncorrected':(':','x'),'ewma_reference':('--','o'),'legacy_pair':('-','s')}
RECORDS = []


def save(fig, name, purpose, inputs):
    fig.canvas.draw()
    for suffix in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{suffix}',dpi=400,bbox_inches='tight',pad_inches=.05)
    with Image.open(OUT/f'{name}.png') as im:
        im.convert('L').save(PAPER/'qa'/f'{name}_gray.png')
    RECORDS.append({'name':name,'purpose':purpose,'inputs':inputs,'font_families':FONTS['font_family_order'],
                    'fallback':False,'png_dpi':400,'figure_size_cm':(fig.get_size_inches()*2.54).tolist()})
    plt.close(fig)


def protocol():
    fig,(ax,bx)=plt.subplots(2,1,figsize=(16/2.54,9.2/2.54),gridspec_kw={'height_ratios':[1,1.65]})
    ax.broken_barh([(0,32)],(0,1),facecolors='#D8DEE4',edgecolors='white')
    ax.broken_barh([(32,35)],(0,1),facecolors='#95ADB9',edgecolors='white')
    ax.broken_barh([(67,14)],(0,1),facecolors='#C5D5C9',edgecolors='white')
    ax.broken_barh([(81,16)],(0,1),facecolors='#DAB79C',edgecolors='white')
    for a,n,label in [(0,32,'基础热身 32'),(32,35,'因果残差热身 35'),(67,14,'验证 14'),(81,16,'测试 16')]:
        ax.text(a+n/2,.5,label,ha='center',va='center',fontsize=8.5)
    ax.set_xlim(0,97);ax.set_ylim(-.1,1.2);ax.set_yticks([])
    ax.set_xticks([0,32,67,81,97],labels=['1','32','67','81','97'])
    ax.set_xlabel('按国服实际开始时间排序的活动序号')
    ax.text(0,1.08,'(a)',transform=ax.transAxes)
    for s in ['left','right','top']:ax.spines[s].set_visible(False)
    bx.set_xlim(0,10);bx.set_ylim(0,3.3);bx.axis('off')
    def box(x,y,w,h,text,color='#F2F4F5'):
        bx.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#777777',linewidth=.65))
        bx.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5)
    def arrow(a,b):bx.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=.8,color='#555555'))
    box(.1,2.25,2.25,.65,'期初已完成历史\n基础模型与预处理')
    box(.1,.95,2.25,.6,'本期已观测前缀')
    box(3,.95,1.55,.6,'基础预测')
    box(5.2,.95,1.65,.6,'误差校正')
    box(7.65,.95,2.1,.6,'封存本期预测')
    box(5.05,2.25,2,.65,'历史封存预测误差\n固定期初校正状态')
    box(7.65,.02,2.1,.55,'结束后发布真实终分')
    arrow((2.35,1.25),(3,1.25));arrow((2.35,2.55),(3.8,1.6));arrow((4.55,1.25),(5.2,1.25))
    arrow((6.85,1.25),(7.65,1.25));arrow((6.05,2.25),(6.05,1.6))
    arrow((8.7,.95),(8.7,.6));arrow((9.75,.3),(9.96,.3))
    arrow((9.96,.3),(9.96,2.56));arrow((9.96,2.56),(7.1,2.56))
    bx.text(8.6,2.85,'仅用于下一期更新',ha='center',fontsize=8)
    bx.text(.1,3.08,'(b)',fontsize=9)
    fig.subplots_adjust(left=.04,right=.98,top=.96,bottom=.03,hspace=.57)
    save(fig,'causal_evaluation_protocol','Clarify 32+35 warm prequential schedule and forecast-before-feedback order.',
         ['cohort_boundaries.csv','facts.json; source protocol and implementation'])


def correlations():
    fig,axes=plt.subplots(1,2,figsize=(16/2.54,8.4/2.54))
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#E5E5E5')
    for ax,name,letter,n in zip(axes,['train_log_final','train_error_raw_phase0'],['a','b'],[67,35]):
        f=pd.read_csv(D/f'{name}.csv',index_col=0)
        im=ax.imshow(np.ma.masked_invalid(f.to_numpy(float)),vmin=-1,vmax=1,cmap=cmap,aspect='equal',interpolation='none')
        ticks=np.arange(len(f));ax.set_xticks(ticks,labels=f.columns,rotation=90,fontsize=7.7)
        ax.set_yticks(ticks,labels=f.columns,fontsize=7.7);ax.tick_params(length=2,pad=2)
        ax.set_xlabel('排名线');ax.set_ylabel('排名线')
        ax.text(0,1.025,f'({letter})  n = {n}',transform=ax.transAxes,fontsize=9)
        for spine in ax.spines.values():spine.set_linewidth(.4)
    fig.subplots_adjust(left=.055,right=.91,bottom=.22,top=.97,wspace=.35)
    cax=fig.add_axes([.94,.27,.018,.57]);fig.colorbar(im,cax=cax,ticks=[-1,-.5,0,.5,1],label='Spearman')
    save(fig,'rank_level_and_error_correlations','Contrast common score levels with heterogeneous causal prediction errors; T1500 masked.',
         ['train_log_final.csv','train_error_raw_phase0.csv'])


def forest():
    f=pd.read_csv(D/'main_models.csv').set_index('method')
    names=['uncorrected','legacy_pair','previous_neural','previous_selected_graph','shared_single','rank_heads','related_static','related_phase']
    labels=['未校正基线','双线组合（历史最佳）','多线门控网络','图平滑 λ=1','共享编码与共享输出','每线专属头','静态相关线分支','阶段图与注意力门控']
    fig,ax=plt.subplots(figsize=(16/2.54,7.7/2.54));y=np.arange(len(names))
    for i,name in enumerate(names):
        row=f.loc[name];point=row.delta_vs_reference_points;color=COLORS['legacy_pair'] if name=='legacy_pair' else '#456D90'
        ax.plot([row.block5_low,row.block5_high],[i+.09,i+.09],color='#909090',lw=1,solid_capstyle='butt')
        ax.plot([row.iid_low,row.iid_high],[i-.04,i-.04],color=color,lw=2,solid_capstyle='butt')
        ax.plot(point,i-.04,'o',color=color,ms=4,mfc='white' if name=='legacy_pair' else color)
    ax.axvline(0,color='#444444',lw=.7,ls='--');ax.set_yticks(y,labels=labels);ax.invert_yaxis()
    ax.set_xlabel('相对独立滚动校正的 MAPE 差值／百分点')
    ax.plot([],[],color='#456D90',lw=2,label='活动配对 95% 区间')
    ax.plot([],[],color='#909090',lw=1,label='5活动块 95% 区间')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,fontsize=8)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#E4E4E4',lw=.4)
    fig.subplots_adjust(left=.31,right=.98,bottom=.19,top=.83)
    save(fig,'matched_model_error_differences','Show matched case differences and uncertainty, including negative neural outcomes.', ['main_models.csv'])


def temporal():
    f=pd.read_csv(D/'stage_results.csv')
    groups=[('elapsed',[1,3,6,12,24,48,72],'开始后时长／h'),
            ('progress',[10,25,50,75,90,95],'活动进度／%'),
            ('remaining',[48,24,12,6,3,1,.5,.25],'距结束时长／h')]
    fig,axes=plt.subplots(1,3,figsize=(16/2.54,6.4/2.54),sharey=True)
    for ax,(kind,values,label),letter in zip(axes,groups,'abc'):
        stages=[f'{kind}_{v:g}h' if kind!='progress' else f'progress_{v:g}pct' for v in values]
        for name in ['uncorrected','ewma_reference','legacy_pair']:
            part=f[f.method.eq(name)].set_index('stage').reindex(stages)
            ls,mark=STYLES[name];ax.plot(np.arange(len(stages)),part.event_macro_mape_pct,color=COLORS[name],ls=ls,marker=mark,label=LABELS[name],ms=3)
        ax.set_xticks(range(len(values)),labels=[f'{v:g}' for v in values],rotation=45)
        ax.set_xlabel(label);ax.spines[['top','right']].set_visible(False)
        ax.text(.03,.95,f'({letter})',transform=ax.transAxes,va='top')
        ax.grid(axis='y',color='#E4E4E4',lw=.4);ax.set_ylim(0,36)
    axes[0].set_ylabel('活动等权 MAPE／%')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.53,1.01),fontsize=8)
    fig.subplots_adjust(left=.085,right=.98,top=.83,bottom=.24,wspace=.16)
    save(fig,'error_by_forecast_origin','Describe error by three frozen time axes and preserve unavailable early stages as gaps.', ['stage_results.csv'])


def ranks():
    f=pd.read_csv(D/'rank_results.csv');fig,ax=plt.subplots(figsize=(16/2.54,7.5/2.54))
    x=np.arange(len(f))
    for name in ['uncorrected','ewma_reference','legacy_pair']:
        ls,mark=STYLES[name];ax.plot(x,f[name],color=COLORS[name],ls=ls,marker=mark,label=LABELS[name],
            mfc='white' if name=='ewma_reference' else COLORS[name],ms=4)
    ax.set_xticks(x,labels=['T'+str(r) for r in f['rank']],rotation=60,ha='right')
    ax.set_xlabel('排名线');ax.set_ylabel('活动等权 MAPE／%');ax.set_ylim(0,21)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E4E4E4',lw=.4)
    ax.legend(ncol=3,loc='upper right',fontsize=8)
    fig.subplots_adjust(left=.085,right=.98,top=.97,bottom=.27)
    save(fig,'errors_across_all_ranks','Show full rank coverage and concentration of paired correction gains at T1000.', ['rank_results.csv'])


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    protocol();correlations();forest();temporal();ranks()
    (PAPER/'qa/figure_records.json').write_text(json.dumps(RECORDS,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'generated','figures':len(RECORDS),'fonts':FONTS['font_family_order']}))
