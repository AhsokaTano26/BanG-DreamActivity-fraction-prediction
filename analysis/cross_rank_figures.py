"""Standalone scientific figures from saved cross-rank statistics, never refit models."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch,Rectangle
import numpy as np
import pandas as pd
from analysis.cross_rank_statistics import OUTPUT,RANKS,sha


def figures(source=OUTPUT,fonts=Path('/tmp/bangdream-cross-rank-fonts.json')):
    source=Path(source);output=source/'figures'
    if output.exists():raise FileExistsError('Preserve previous figure export')
    output.mkdir();font=json.loads(Path(fonts).read_text())
    shutil.copyfile(fonts,output/'font_check.json')
    result=json.loads((source/'results.json').read_text())
    cmap=LinearSegmentedColormap.from_list('rank_corr',['#456D90','#fafafa','#A66D47']);cmap.set_bad('#e5e5e5')
    style={'font.family':font['font_family_order'],'font.size':9,'axes.labelsize':9,
           'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.7,
           'mathtext.fontset':'cm','pdf.fonttype':42,'ps.fonttype':42,
           'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
           'legend.frameon':False,'legend.fontsize':8,'axes.unicode_minus':True}
    panels=['blind_current_p50','blind_increment_p50_p75','train_log_final','train_final_partial',
            'train_error_raw_phase0','train_error_independent_phase0','train_error_raw_phase1_lag1']
    receipts=[]
    def save(fig,name):
        fig.canvas.draw()
        fig.savefig(output/f'{name}.pdf',bbox_inches='tight',pad_inches=.05)
        fig.savefig(output/f'{name}.png',dpi=400,bbox_inches='tight',pad_inches=.05)
        plt.close(fig)
    with plt.rc_context(style):
        for panel in panels:
            frame=pd.read_csv(source/'matrices'/f'{panel}.csv',index_col=0)
            values=frame.to_numpy(float)
            fig,ax=plt.subplots(figsize=(17.5/2.54,16.5/2.54),layout='constrained')
            im=ax.imshow(np.ma.masked_invalid(values),vmin=-1,vmax=1,cmap=cmap,interpolation='nearest',aspect='equal')
            ax.set_xticks(range(len(RANKS)),[f'T{r}' for r in RANKS],rotation=60,ha='right')
            ax.set_yticks(range(len(RANKS)),[f'T{r}' for r in RANKS])
            lag=panel.endswith('lag1')
            ax.set_xlabel('本期排名线' if lag else '排名线')
            ax.set_ylabel('上期排名线' if lag else '排名线')
            if np.isnan(values).any():
                for j in np.where(np.isnan(values).all(axis=0))[0]:
                    ax.add_patch(Rectangle((j-.5,-.5),1,len(RANKS),facecolor='none',edgecolor='#aaa',hatch='///',lw=0))
                    ax.add_patch(Rectangle((-.5,j-.5),len(RANKS),1,facecolor='none',edgecolor='#aaa',hatch='///',lw=0))
                ax.legend(handles=[Patch(facecolor='#e5e5e5',hatch='///',edgecolor='#aaa',label='无训练数据')],loc='upper left',bbox_to_anchor=(0,1.06),borderaxespad=0)
            cb=fig.colorbar(im,ax=ax,fraction=.042,pad=.025,shrink=.88)
            cb.set_label('Partial Spearman correlation' if panel=='train_final_partial' else 'Spearman correlation')
            save(fig,panel)
            info=next(x for x in result['panels'] if x['panel']==panel)
            receipts.append({'figure':panel,'input':f'matrices/{panel}.csv','input_sha256':sha(source/'matrices'/f'{panel}.csv'),
                             'scope':info['scope'],'common_activity_units':info['common_events'],'missing_is_blank_hatched':True})
        fig,ax=plt.subplots(figsize=(13/2.54,8/2.54),layout='constrained')
        summary=pd.read_csv(source/'panel_summary.csv').set_index('panel')
        for key,label,color,marker,offset in [('within_band','区间内','#456D90','o',-.04),('between_band','跨区间','#A66D47','s',.04)]:
            rows=[summary.loc[f'train_error_raw_phase{p}'] for p in range(3)]
            y=np.array([r[key+'_median'] for r in rows]);ci=np.array([json.loads(r[key+'_block5_95ci']) for r in rows])
            ax.errorbar(np.arange(3)+offset,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),fmt=marker+'-',color=color,
                        lw=1.2,capsize=3,markersize=4,label=label)
        ax.set_xticks(range(3),['早期(<25%)','中期(25–90%)','末期(≥90%)'])
        ax.set_ylabel('Spearman correlation');ax.set_xlabel('活动阶段')
        ax.axhline(0,color='#888',lw=.7,ls=':');ax.set_ylim(-.15,1.0);ax.legend(loc='upper right')
        save(fig,'error_correlation_by_phase')
        # Actual grayscale QA drawn from the same reviewed values, not a replacement output.
        frame=pd.read_csv(source/'matrices/train_error_raw_phase0.csv',index_col=0)
        fig,ax=plt.subplots(figsize=(17.5/2.54,16.5/2.54),layout='constrained')
        gray=plt.colormaps['Greys'].copy();gray.set_bad('#e5e5e5')
        im=ax.imshow(np.ma.masked_invalid(frame.to_numpy(float)),vmin=-1,vmax=1,cmap=gray)
        ax.set_xticks(range(len(RANKS)),[f'T{r}' for r in RANKS],rotation=60,ha='right')
        ax.set_yticks(range(len(RANKS)),[f'T{r}' for r in RANKS]);ax.set_xlabel('排名线');ax.set_ylabel('排名线')
        fig.colorbar(im,ax=ax,fraction=.042,pad=.025,shrink=.88).set_label('Spearman correlation')
        fig.savefig(output/'grayscale_qa.png',dpi=300,bbox_inches='tight',pad_inches=.05);plt.close(fig)
    shutil.copyfile(__file__,output/'source.py')
    (output/'figure_sources.json').write_text(json.dumps({'receipts':receipts,'font':font,'style':style,
        'errorbar':'median within/between predefined rank bands, activity-block5 percentile95% intervals; marginal exploratory intervals',
        'source_sha256':sha(Path(__file__))},indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'exported_figures':len(panels)+1,'directory':str(output)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=OUTPUT);p.add_argument('--fonts',type=Path,default=Path('/tmp/bangdream-cross-rank-fonts.json'))
    a=p.parse_args();figures(a.source,a.fonts)
