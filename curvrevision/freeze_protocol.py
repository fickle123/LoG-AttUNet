from __future__ import annotations
import argparse, hashlib
from pathlib import Path
from .utils import dump_json, load_json, sha256_file, environment_report


def tree_hash(paths):
    h=hashlib.sha256(); entries=[]
    for p in sorted(paths,key=lambda x:str(x)):
        digest=sha256_file(p); rel=str(p); h.update(rel.encode()); h.update(digest.encode()); entries.append({'path':rel,'sha256':digest})
    return h.hexdigest(),entries


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs-root',default='outputs/runs'); ap.add_argument('--split-root',default='outputs/splits'); ap.add_argument('--matrix-report',default='outputs/formal_matrix_check.json'); ap.add_argument('--lock',default='outputs/PROTOCOL_FROZEN.json'); args=ap.parse_args()
    lock=Path(args.lock)
    if lock.exists(): raise RuntimeError(f'Protocol already frozen: {lock}')
    matrix_path=Path(args.matrix_report)
    if not matrix_path.exists():
        raise RuntimeError(f'Run scripts/16_check_formal_matrix.sh before protocol freeze: {matrix_path}')
    matrix=load_json(matrix_path)
    if matrix.get('status')!='PASS' or int(matrix.get('missing_count',-1))!=0 or int(matrix.get('malformed_count',-1))!=0 or int(matrix.get('extra_refit_runs_not_in_formal_matrix',-1))!=0:
        raise RuntimeError(f'Formal matrix report is not a clean PASS: {matrix_path}')
    root=Path(args.runs_root)
    if list(root.rglob('OUTER_TEST_COMPLETE')):
        raise RuntimeError('Cannot create a clean protocol freeze after outer-test results already exist in the formal runs root.')
    refits=sorted(root.rglob('REFIT_COMPLETE'))
    if not refits: raise RuntimeError('No completed refit runs found.')
    runs=[]
    for marker in refits:
        rd=marker.parent; cfg=rd/'run_config.json'; sel=rd/'selection.json'; ck=rd/'final_outer_train_model.pth'
        if not (cfg.exists() and sel.exists() and ck.exists()): raise RuntimeError(f'incomplete formal run: {rd}')
        c=load_json(cfg); runs.append({'run_dir':str(rd),'experiment':c['experiment'],'model_label':c['model_label'],'budget':c['budget'],'selection_rule':c.get('selection_rule','best_dice'),'seed':c['seed'],
                                      'split_sha256':c['split_sha256'],'run_config_sha256':sha256_file(cfg),'selection_sha256':sha256_file(sel),'checkpoint_sha256':sha256_file(ck)})
    code_files=list(Path('curvrevision').glob('*.py'))+list(Path('configs').glob('*'))+list(Path('scripts').glob('*.sh'))+list(Path('docs').glob('*.md'))+[Path('requirements.txt'),Path('README.md')]
    code_files=[p for p in code_files if p.exists()]
    code_hash,code_entries=tree_hash(code_files)
    split_files=sorted(Path(args.split_root).glob('fold_*.json'))
    if len(split_files)!=5: raise RuntimeError(f'Expected five outer split JSONs, found {len(split_files)}')
    split_hash,split_entries=tree_hash(split_files)
    if len(runs) != int(matrix.get('expected_refit_runs', -1)):
        raise RuntimeError(f'Frozen run count {len(runs)} does not match matrix expectation {matrix.get("expected_refit_runs")}')
    obj={'status':'FROZEN_BEFORE_OUTER_TEST','formal_runs_root':args.runs_root,'n_refit_runs':len(runs),'code_tree_sha256':code_hash,'split_tree_sha256':split_hash,
         'code_files':code_entries,'split_files':split_entries,'runs':runs,'environment_at_freeze':environment_report(),
         'formal_matrix_report':str(matrix_path),'formal_matrix_report_sha256':sha256_file(matrix_path),'formal_matrix_expected_runs':int(matrix['expected_refit_runs']),
         'rule':'After this file exists, formal develop/refit is prohibited. Outer-test may be evaluated once per frozen run only.'}
    dump_json(obj,lock); print(f'FROZEN: {lock}'); print(f'formal refit runs: {len(runs)}'); print(f'code sha256: {code_hash}'); print(f'split sha256: {split_hash}')

if __name__=='__main__': main()
