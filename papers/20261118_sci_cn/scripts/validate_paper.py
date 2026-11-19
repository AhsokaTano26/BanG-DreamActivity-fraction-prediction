"""Validate manuscript numbers against frozen extracted evidence and source hashes."""
from pathlib import Path
import hashlib,json,re,subprocess
import xml.etree.ElementTree as ET
import pandas as pd

P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]


def main():
    tex=(P/'main.tex').read_text();facts=json.loads((P/'data/facts.json').read_text())
    checks=[]
    def check(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    manifest=json.loads((P/'data/source_manifest.json').read_text())
    check(all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in manifest.items()),'all_original_source_hashes_unchanged')
    labels=set(re.findall(r'\\label\{([^}]+)\}',tex))
    references=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex))
    check(references<=labels,'all_native_cross_references_resolve')
    bib=(P/'references.bib').read_text();bibkeys=set(re.findall(r'@\w+\{([^,]+)',bib))
    cites={k for group in re.findall(r'\\citep\{([^}]+)\}',tex) for k in group.split(',')}
    check(cites==bibkeys and len(bibkeys)==19,'all_19_bibliography_entries_are_used')
    check(tex.count('\\begin{equation}')==8,'eight_native_numbered_equations')
    check(tex.count('\\begin{figure}')==5,'five_native_numbered_figures')
    check(tex.count('\\begin{table}')==7,'seven_native_numbered_tables')
    for path in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex):
        check((P/path).is_file(),'vector_figure_exists:'+path)
    tables={}
    for content in re.findall(r'\\begin\{table\}.*?\\end\{table\}',tex,re.S):
        name=re.search(r'\\label\{tab:([^}]+)\}',content)[1]
        data=content.split('\\midrule')[1].split('\\bottomrule')[0]
        tables[name]=[[v.strip().removesuffix('\\\\').strip() for v in line.split('&')]
                      for line in data.splitlines() if '&' in line]
    ranks=pd.read_csv(P/'data/rank_results.csv')
    for row,(_,reference) in zip(tables['allranks'],ranks.iterrows()):
        check(row[0]=='T'+str(int(reference['rank'])),'rank_identity:'+row[0])
        expected=[f'{reference[k]:.3f}' for k in ['uncorrected','ewma_reference','legacy_pair']]
        check(row[-3:]==expected,'rank_three_scores:'+row[0])
        check(int(row[2])==reference.test_curves and int(row[3])==reference.test_cases,'rank_coverage:'+row[0])
    check(len(tables['allranks'])==19 and ranks.test_cases.sum()==5220,'full_19_rank_panel_and_5220_cases')
    check((ranks.loc[~ranks['rank'].isin([1000,2000]),'legacy_pair'].to_numpy()==
           ranks.loc[~ranks['rank'].isin([1000,2000]),'ewma_reference'].to_numpy()).all(),'unchanged_other_17_ranks')
    models=pd.read_csv(P/'data/main_models.csv')
    for row,(_,ref) in zip(tables['main'],models.iterrows()):
        check(row[-3:]==[f'{ref[c]:.3f}' for c in ['validation_mape_pct','test_mape_pct','original2_test_mape_pct']],
              'main_three_scores:'+ref.method)
    mapping=['catboost_rank','dual_tcn_mean','phase_route','retrieval__retrieval_residual','blend_new_25',
        'growth__growth_dual_clock','growth__growth_progress','growth__next_growth_progress_aux025',
        'foundation__foundation_tabpfn_v2_context','foundation__foundation_tabpfn_v2',
        'foundation__foundation_chronos2','foundation__foundation_timesfm25']
    static=pd.read_csv(P/'data/static_v2_models.csv').set_index('method')
    for row,key in zip(tables['static'],mapping):
        ref=static.loc[key]
        check(row[-2:]==[f'{ref.validation_mape_pct:.3f}',f'{ref.test_mape_pct:.3f}'],'static_validation_and_test:'+key)
        if 'chronos' not in key and 'timesfm' not in key:
            check(row[-3]==f'{ref.train_mape_pct:.3f}','static_train_diagnostic:'+key)
        else:check(row[-3]=='—','zero_shot_training_not_fabricated:'+key)
    stats=pd.read_csv(P/'data/correlation_summary.csv').set_index('panel')
    panels=['train_log_final','train_final_partial','train_remaining_growth_p50','train_error_raw_phase0',
            'train_error_raw_phase1','train_error_raw_phase2','train_error_raw_phase0_lag1']
    for row,key in zip(tables['correlation'],panels):
        ref=stats.loc[key]
        expected=[f'{ref[c]:.3f}'.replace('-','−') for c in ['within_band_median','between_band_median']]
        check(row[-2:]==expected,'correlation_summary:'+key)
    comparisons=facts['task_matched_comparisons'].values()
    pairs=[('shared_single','ewma_reference'),('rank_heads','shared_single'),('related_static','rank_heads'),('related_phase','related_static')]
    for row,(method,control) in zip(tables['ablation'],pairs):
        ref=next(x for x in comparisons if x['method']==method and x['reference']==control)
        numbers=[float(x.replace('−','-')) for s in row[1:] for x in re.findall(r'[+−-]?\d+\.\d+',s)]
        expected=[ref['mean_difference_points'],*ref['paired_95_ci'],*ref['circular_block_95_ci']['5']]
        check(numbers==[float(f'{x:.3f}') for x in expected],'paired_ablation:'+method)
    check(len(pd.read_csv(P/'data/supplement_all_329_results.csv'))==329,'all_329_results_preserved')
    check(all(f'{facts["derived_relative_reductions_pct"][k]:.2f}' in tex for k in ['warm_vs_raw','pair_vs_raw','pair_vs_warm']),
          'relative_reductions_not_confused_with_percentage_points')
    check('历史最佳并用于生产' not in tex or '事前' in tex,'selection_roles_documented')
    log=(P/'output/main.log').read_text(errors='replace')
    forbidden=['Overfull','Missing character','undefined references','undefined citations','Citation(s) may have changed',
               'Label(s) may have changed','Unknown CJK family']
    check(not any(s in log for s in forbidden),'final_tex_no_clipping_missing_glyphs_or_unresolved_references')
    subprocess.run(['pdftotext','-bbox-layout',str(P/'output/main.pdf'),str(P/'qa/pdf_bbox.html')],check=True)
    subprocess.run(['pdftotext','-layout',str(P/'output/main.pdf'),str(P/'qa/native_pdf.txt')],check=True)
    tree=ET.parse(P/'qa/pdf_bbox.html');pages=[x for x in tree.iter() if x.tag.endswith('page')]
    for i,page in enumerate(pages,1):
        w,h=float(page.attrib['width']),float(page.attrib['height'])
        words=[x for x in page.iter() if x.tag.endswith('word')]
        check(all(float(x.attrib['xMin'])>=20 and float(x.attrib['xMax'])<=w-20 and
                  float(x.attrib['yMin'])>=15 and float(x.attrib['yMax'])<=h-15 for x in words),'printed_bounds_page:'+str(i))
    pdftext=(P/'qa/native_pdf.txt').read_text()
    check('�' not in pdftext and '??' not in pdftext,'pdf_text_has_no_replacement_characters_or_missing_refs')
    check('5220' in pdftext and '6.540' in pdftext and 'T1500' in pdftext,'key_results_present_in_actual_pdf')
    result={'status':'pass','checks':len(checks),'pages':len(pages),'source_files_verified':len(manifest),
            'equations':8,'tables':7,'figures':5,'references':19,'details':checks,
            'scope':'Numerical, provenance, structure and print bounds; visual page inspection is recorded separately.'}
    (P/'qa/numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['details','scope']}))


if __name__=='__main__':main()
