from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from .utils import load_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs-root',default='outputs/runs'); ap.add_argument('--lock',default='outputs/PROTOCOL_FROZEN.json'); args=ap.parse_args()
    lock=Path(args.lock)
    if not lock.exists(): raise RuntimeError('Protocol must be frozen before formal outer-test evaluation.')
    frozen=load_json(lock); frozen_runs=sorted(frozen['runs'], key=lambda r:r['run_dir']); allowed={r['run_dir'] for r in frozen_runs}
    discovered={str(p.parent) for p in Path(args.runs_root).rglob('REFIT_COMPLETE')}
    unexpected=sorted(discovered-allowed)
    missing=sorted(allowed-discovered)
    if unexpected: raise RuntimeError(f'Unexpected refit runs appeared after freeze: {unexpected[:5]}')
    if missing: raise RuntimeError(f'Frozen refit runs are missing: {missing[:5]}')
    for frozen_entry in frozen_runs:
        rd=Path(frozen_entry['run_dir'])
        if (rd/'OUTER_TEST_COMPLETE').exists(): print('[SKIP locked]',rd); continue
        c=load_json(rd/'run_config.json')
        cmd=[sys.executable,'-m','curvrevision.train_cv','--data-root',str(c['data_root']),'--split-json',str(c['split_json']),'--output-root',str(c['output_root']),
             '--experiment',str(c['experiment']),'--model',str(c['model']),'--selection-rule',str(c.get('selection_rule','best_dice')),'--prior-kind',str(c['prior_kind']),'--prior-depth',str(c['prior_depth']),'--budget',str(c['budget']),
             '--seed',str(c['seed']),'--phase','test','--image-size',str(c['image_size']),'--base-channels',str(c['base_channels']),'--batch-size',str(c['batch_size']),
             '--eval-batch-size',str(c['eval_batch_size']),'--stage1-epochs',str(c['stage1_epochs']),'--stage2-epochs',str(c['stage2_epochs']),'--threshold',str(c['threshold']),
             '--protocol-lock',str(lock)]
        if c.get('amp'): cmd.append('--amp')
        print('[OUTER-TEST]',rd,flush=True); subprocess.run(cmd,check=True)

if __name__=='__main__': main()
