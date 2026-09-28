from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ci',default='outputs/statistics/bootstrap_ci_core_stage1.csv'); ap.add_argument('--output',default='outputs/figures'); args=ap.parse_args()
    df=pd.read_csv(args.ci); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for metric in ['dice','hd95','asd']:
        g=df[df.metric==metric].copy()
        if len(g)==0: continue
        fig,ax=plt.subplots(figsize=(8.2,4.8)); x=range(len(g)); y=g['mean'].to_numpy(); lo=y-g['bootstrap95_low'].to_numpy(); hi=g['bootstrap95_high'].to_numpy()-y
        ax.errorbar(list(x),y,yerr=[lo,hi],fmt='o',capsize=4,linestyle='none')
        ax.set_xticks(list(x)); ax.set_xticklabels(g.model,rotation=25,ha='right'); ax.set_ylabel(metric.upper()); ax.set_title(f'{metric.upper()} with image-level bootstrap 95% CI')
        fig.tight_layout(); fig.savefig(out/f'comparison_{metric}.png',dpi=300); fig.savefig(out/f'comparison_{metric}.pdf'); plt.close(fig)

if __name__=='__main__': main()
