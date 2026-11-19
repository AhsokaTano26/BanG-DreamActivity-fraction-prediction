"""Publication figures for the observational activity data paper."""
from pathlib import Path
import gzip,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, dates as mdates
from PIL import Image

P=Path(__file__).resolve().parents[1];D=P/'data';OUT=P/'figures'
FONTS=json.loads((P/'qa/font_check.json').read_text())
for key in ['latin','cjk']:fm.fontManager.addfont(FONTS[key]['path'])
plt.rcParams.update({'font.family':FONTS['font_family_order'],'font.size':9,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.7,
    'lines.linewidth':1.2,'lines.markersize':3.5,'legend.frameon':False,'mathtext.fontset':'cm',
    'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})
BLUE='#456D90';ORANGE='#B36B3D';GREEN='#628575';GRAY='#777777'
RECORDS=[]


def frame(name):return pd.read_csv(D/name)


def polish(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#e8e8e8',linewidth=.45);ax.set_axisbelow(True)


def panel(ax,label):ax.text(0,1.04,label,transform=ax.transAxes,va='bottom')


def save(fig,name,purpose,inputs):
    fig.canvas.draw()
    for suffix in ['pdf','png']:fig.savefig(OUT/f'{name}.{suffix}',dpi=400,bbox_inches='tight',pad_inches=.05)
    with Image.open(OUT/f'{name}.png') as im:im.convert('L').save(P/'qa'/f'{name}_gray.png')
    RECORDS.append({'name':name,'purpose':purpose,'inputs':inputs,'backend':'Matplotlib',
        'font_families':FONTS['font_family_order'],'fallback':False,'png_dpi':400,
        'figure_size_cm':(fig.get_size_inches()*2.54).tolist()})
    plt.close(fig)


def sampling():
    f=frame('rank_summary.csv');y=np.arange(len(f))
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,11/2.54),sharey=True,gridspec_kw={'width_ratios':[1.25,1]})
    ax.barh(y,f.source_curves,color='white',edgecolor=GRAY,height=.72,label='原始曲线')
    ax.barh(y,f.valid_events,color=BLUE,height=.44,label='合格终分')
    for yy,v in zip(y,f.valid_events):ax.text(v+1,yy,str(v),va='center',fontsize=7.5)
    ax.set_yticks(y,labels=['T'+str(x) for x in f['rank']]);ax.invert_yaxis();ax.set_xlim(0,115)
    ax.set_xlabel('活动数');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=2,fontsize=8)
    bx.barh(y,f.median_gap_min,color=GREEN,height=.5,hatch='//',linewidth=.3)
    bx.set_xlabel('曲线内典型间隔 / min');bx.set_xlim(0,35)
    for a in [ax,bx]:a.spines[['top','right']].set_visible(False);a.grid(axis='x',color='#e8e8e8',linewidth=.45);a.set_axisbelow(True)
    panel(ax,'(a)');panel(bx,'(b)');fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.9,wspace=.15)
    save(fig,'sampling_and_rank_coverage','Different ranks have unequal endpoint coverage and sampling resolution',['rank_summary.csv'])


def score_shape():
    common=pd.read_csv(D/'common18_final_scores.csv',index_col=0);r=np.array([int(x) for x in common.columns])
    comp=frame('coverage_median_comparison.csv')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,7.5/2.54),gridspec_kw={'width_ratios':[1.15,1]})
    med=common.median().to_numpy();lo=common.quantile(.25).to_numpy();hi=common.quantile(.75).to_numpy()
    ax.fill_between(r,lo,hi,color=BLUE,alpha=.18);ax.plot(r,med,'o-',color=BLUE,markersize=3)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([10,100,1000,10000,30000],labels=['10','100','1000','10000','30000'])
    ax.tick_params(axis='x',rotation=35);ax.set_xlabel('排名（对数坐标）');ax.set_ylabel('终分 / EP（对数坐标）')
    panel(ax,'(a)');ax.text(.03,.04,f'n = {len(common)}',transform=ax.transAxes)
    x=np.arange(3);width=.24
    legacy=frame('legacy_terminal_scores.csv');legacy=legacy.groupby('rank').final.median().reindex(comp['rank'])
    bx.bar(x-width,legacy.to_numpy()/10000,width,color='white',edgecolor=GRAY,hatch='//',label='旧缓存各自样本')
    bx.bar(x,comp.own_median/10000,width,color='#B9B9B9',edgecolor=GRAY,label='当前各自样本')
    bx.bar(x+width,comp.matched_median/10000,width,color=ORANGE,label=f'当前共同{int(comp.matched_events.iloc[0])}期')
    bx.set_xticks(x,labels=['T1000','T1500','T2000']);bx.set_ylabel('终分中位数 / 万EP');bx.set_ylim(0,680)
    bx.legend(loc='upper right',fontsize=7.5);panel(bx,'(b)')
    for a in [ax,bx]:polish(a)
    fig.subplots_adjust(left=.1,right=.98,bottom=.24,top=.92,wspace=.42)
    save(fig,'rank_shape_and_coverage_effect','Source updates and event coverage change marginal score medians',['common18_final_scores.csv','coverage_median_comparison.csv','legacy_terminal_scores.csv'])


def temporal():
    f=frame('event_score_ratios.csv');f=f[(f.numerator==1000)&(f.denominator==2000)].sort_values('start_ms')
    summary=frame('calendar_ratio_summary.csv');summary=summary[(summary.numerator==1000)&(summary.denominator==2000)]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,7/2.54),gridspec_kw={'width_ratios':[1.4,1]})
    times=pd.to_datetime(f.start_ms,unit='ms',utc=True).dt.tz_convert('Asia/Shanghai')
    ax.plot(times,f.ratio,color='#BEBEBE',linewidth=.7)
    for year,color,marker in [(2024,GRAY,'o'),(2025,BLUE,'s'),(2026,ORANGE,'^')]:
        mask=f.year==year;ax.scatter(times[mask],f.loc[mask,'ratio'],color=color,marker=marker,s=12,label=str(year),zorder=3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x',rotation=35);ax.set_ylabel('终分比 T1000/T2000');ax.set_xlabel('活动实际开始日期')
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.22),ncol=3,fontsize=7,borderaxespad=0);panel(ax,'(a)')
    x=np.arange(3);vals=summary.median_ratio.to_numpy()
    bx.errorbar(x,vals,yerr=np.stack([vals-summary.block5_lower95,summary.block5_upper95-vals]),fmt='o',color=BLUE,capsize=3)
    for xx,upper,n in zip(x,summary.block5_upper95,summary.events):bx.text(xx,upper+.06,f'n = {n}',ha='center',fontsize=8)
    bx.set_xticks(x,labels=summary.year.astype(str));bx.set_xlim(-.45,2.45);bx.set_ylim(1,3.2)
    bx.set_xlabel('相同月份：每年2至10月');bx.set_ylabel('比值中位数与95%块区间');panel(bx,'(b)')
    for a in [ax,bx]:polish(a)
    fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.8,wspace=.35)
    save(fig,'temporal_rank_ratio','The rank ratio changes nonmonotonically in aligned calendar periods',['event_score_ratios.csv','calendar_ratio_summary.csv'])


def growth():
    f=frame('matched_four_rank_growth.csv')
    styles={10:(GRAY,':','x'),1000:(ORANGE,'-','s'),2000:(BLUE,'--','o'),30000:(GREEN,'-.','^')}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,7/2.54))
    for r,(color,ls,marker) in styles.items():
        a=f[(f['rank']==r)&(f.axis=='progress')].sort_values('value')
        ax.plot(a.value*100,a.median_fraction*100,color=color,ls=ls,marker=marker,label='T'+str(r))
        b=f[(f['rank']==r)&(f.axis=='remaining_h')].sort_values('value',ascending=False)
        bx.plot(b.value,b.median_remaining_share*100,color=color,ls=ls,marker=marker)
    ax.set_xlabel('活动进度 / %');ax.set_ylabel('已观测分数占终分 / %');ax.set_ylim(0,105)
    bx.set_xscale('log');bx.invert_xaxis();bx.set_xticks([48,24,12,6,3,1,.5,.25],labels=['48','24','12','6','3','1','0.5','0.25'])
    bx.set_xlabel('名义距结束时长 / h（对数坐标）');bx.set_ylabel('观测后增长占终分 / %')
    for a in [ax,bx]:polish(a)
    panel(ax,'(a)');panel(bx,'(b)');fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.53,1),fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,bottom=.2,top=.84,wspace=.3)
    save(fig,'matched_rank_growth_profiles','Matched four rank panels reveal different accumulation and late growth profiles',['matched_four_rank_growth.csv','matched_growth_event_ids.json'])


def correlations():
    final=pd.read_csv(D/'matched_final_spearman.csv',index_col=0);growth=pd.read_csv(D/'matched_growth_spearman.csv',index_col=0)
    labels=final.columns.tolist();fig,(ax,bx)=plt.subplots(1,2,figsize=(16/2.54,8.4/2.54))
    for a,data in [(ax,final),(bx,growth)]:
        im=a.imshow(data.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r',aspect='equal',interpolation='nearest')
        a.set_xticks(range(18),labels=labels,rotation=90,fontsize=8);a.set_yticks(range(18),labels=labels,fontsize=8)
        a.tick_params(length=1.5,pad=2);a.set_xlabel('排名线')
    ax.set_ylabel('排名线');panel(ax,'(a)');panel(bx,'(b)')
    fig.subplots_adjust(left=.08,right=.91,bottom=.22,top=.94,wspace=.25)
    cax=fig.add_axes([.93,.27,.018,.56]);fig.colorbar(im,cax=cax).set_label('Spearman')
    save(fig,'matched_score_growth_correlations','Score levels and remaining growth have different dependence on the same complete event panel',['matched_final_spearman.csv','matched_growth_spearman.csv'])


def context():
    f=frame('association_coefficients.csv');f=f[f.model=='pooled_final']
    fig,(ax,bx)=plt.subplots(2,1,figsize=(16/2.54,11.5/2.54),gridspec_kw={'height_ratios':[1,1.4]})
    for a,prefix,color in [(ax,'kind=',BLUE),(bx,'band_model=',ORANGE)]:
        x=f[f.term.str.startswith(prefix)].sort_values('beta',ascending=False)
        y=np.arange(len(x));a.axvline(0,color=GRAY,lw=.8,ls='--')
        a.errorbar(x.beta,y,xerr=np.stack([x.beta-x.hac5_lower95,x.hac5_upper95-x.beta]),fmt='o',color=color,capsize=2)
        a.set_yticks(y,labels=[s.split('=',1)[1] for s in x.term]);a.invert_yaxis();a.set_xlabel('条件对数终分差与95%区间')
        a.spines[['top','right']].set_visible(False);a.grid(axis='x',color='#e8e8e8',linewidth=.45);a.set_axisbelow(True)
    panel(ax,'(a)');panel(bx,'(b)');fig.subplots_adjust(left=.3,right=.97,bottom=.1,top=.96,hspace=.55)
    save(fig,'conditional_type_band_associations','Conditional type and band associations with event cluster temporal uncertainty',['association_coefficients.csv','models.json'])


def content():
    f=frame('character_exposure.csv');meta=json.load(gzip.open(D/'source_metadata.json.gz','rt'))
    bands=sorted(f.band_id.unique());counts=np.full((len(bands),5),np.nan);names=np.empty((len(bands),5),object);names[:]=''
    labels=[]
    for i,b in enumerate(bands):
        labels.append(meta['bands'][str(int(b))]['bandName'][3])
        a=f[f.band_id==b].sort_values('character');assert len(a)==5
        for j,row in enumerate(a.itertuples()):counts[i,j]=row.events;names[i,j]=row.name.replace(' ','')+'\n'+str(row.events)
    monthly=frame('card_releases_complete_months.csv')
    fig,(ax,bx)=plt.subplots(2,1,figsize=(16/2.54,12/2.54),gridspec_kw={'height_ratios':[2,1]})
    im=ax.imshow(counts,cmap='Blues',vmin=0,vmax=max(counts.ravel()),aspect='auto',interpolation='nearest')
    for i in range(len(bands)):
        for j in range(5):ax.text(j,i,names[i,j],ha='center',va='center',fontsize=8,color='white' if counts[i,j]>=13 else 'black')
    ax.set_yticks(range(len(bands)),labels=labels);ax.set_xticks([]);ax.tick_params(length=0)
    panel(ax,'(a)')
    x=np.arange(len(monthly));other=monthly.cards-monthly.five_star
    bx.bar(x,other,color='#BBBBBB',width=.75,label='其余星级')
    bx.bar(x,monthly.five_star,bottom=other,color=BLUE,hatch='//',width=.75,label='5星')
    ix=np.arange(0,len(x),3);bx.set_xticks(ix,labels=monthly.year_month.iloc[ix],rotation=40,ha='right')
    bx.set_ylabel('国服发布卡牌数');bx.set_xlabel('完整月份');bx.set_ylim(bottom=0);bx.legend(loc='upper right',ncol=2,fontsize=8)
    polish(bx);panel(bx,'(b)');fig.subplots_adjust(left=.26,right=.97,bottom=.16,top=.96,hspace=.4)
    save(fig,'character_exposure_and_card_releases','Observed bonus character exposure and release metadata describe content supply',['character_exposure.csv','card_releases_complete_months.csv','source_metadata.json.gz'])


def main():
    OUT.mkdir(exist_ok=True)
    for fn in [sampling,score_shape,temporal,growth,correlations,context,content]:fn()
    (P/'qa/figure_records.json').write_text(json.dumps(RECORDS,ensure_ascii=False,indent=2)+'\n')
    print('Created',len(RECORDS),'vector figures and 400 dpi previews.')


if __name__=='__main__':main()
