from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .utils import load_json, dump_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs-root',default='outputs/runs'); ap.add_argument('--output',default='outputs/aggregate'); args=ap.parse_args()
    root=Path(args.runs_root); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    frames=[]; summaries=[]
    for p in sorted(root.rglob('per_image_metrics.csv')):
        run=p.parent; cfg=load_json(run/'run_config.json'); df=pd.read_csv(p)
        df['experiment']=cfg['experiment']; df['model_arg']=cfg['model']; df['prior_kind']=cfg['prior_kind']; df['prior_depth']=cfg['prior_depth']; df['split_sha256']=cfg['split_sha256']; df['run_dir']=str(run)
        frames.append(df)
        sp=run/'outer_test_summary.json'
        if sp.exists(): summaries.append(load_json(sp)|{'experiment':cfg['experiment'],'run_dir':str(run),'prior_kind':cfg['prior_kind'],'prior_depth':cfg['prior_depth']})
    if not frames: raise RuntimeError(f'No per_image_metrics.csv under {root}')
    all_df=pd.concat(frames,ignore_index=True); all_df.to_csv(out/'all_per_image_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'all_run_summaries.csv',index=False)
    checks=[]
    keys=['experiment','model','budget','selection_rule','seed']
    for key,g in all_df.groupby(keys,dropna=False):
        dup=int(g.duplicated(['case_id']).sum()); n=int(g.case_id.nunique()); folds=int(g.fold.nunique())
        checks.append(dict(zip(keys,key))|{'n_rows':len(g),'n_unique_cases':n,'n_folds':folds,'duplicate_cases':dup,'complete_oof_999':bool(n==999 and dup==0 and folds==5)})
    ck=pd.DataFrame(checks); ck.to_csv(out/'oof_completeness.csv',index=False)
    bad=ck[~ck.complete_oof_999]
    dump_json({'n_runs':len(summaries),'n_metric_rows':len(all_df),'n_oof_groups':len(ck),'n_incomplete_oof_groups':len(bad)},out/'collection_summary.json')
    print(ck.to_string(index=False))
    if len(bad): print('\nWARNING: incomplete OOF groups exist; statistics scripts will only use complete groups unless --allow-incomplete is set.')

if __name__=='__main__': main()
