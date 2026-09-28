#!/usr/bin/env bash
set -euo pipefail
export DATA_ROOT="${DATA_ROOT:-../fetal_dataset}"
python - <<'PY'
from pathlib import Path
import torch
from curvrevision.data import discover_hc18_cases, robust_metadata_table, find_metadata_columns
from curvrevision.model import build_model
from curvrevision.utils import environment_report
root=Path(__import__('os').environ.get('DATA_ROOT','../fetal_dataset'))
cases=discover_hc18_cases(root)
df=robust_metadata_table(root); cols=find_metadata_columns(df)
assert len(cases)==999
assert cols['name'] is not None and cols['hc'] is not None and (cols['pixel'] is not None or (cols['sx'] and cols['sy']))
print('DATA OK:', root, 'cases=',len(cases), 'metadata columns=',cols)
print('ENV:', environment_report())
x=torch.randn(1,1,64,64)
for name in ['unet','attention_unet','resunet','attention_resunet','full_standard','full_log_attention_only','full_log_shallow']:
    m=build_model(name,base=4).eval()
    with torch.inference_mode(): y=m(x)
    assert y['logits'].shape==(1,1,64,64)
    print(name, 'params=',sum(p.numel() for p in m.parameters()))
std=build_model('full_standard',base=4); log=build_model('full_log_attention_only',base=4)
assert sum(p.numel() for p in std.parameters())==sum(p.numel() for p in log.parameters()), 'matched attention controls must have identical trainable capacity'
assert std.boundary_prior is False and log.boundary_prior is False
print('MATCHED CONTROL CHECK OK')
PY
mkdir -p outputs
python - <<'PY'
from curvrevision.utils import dump_json,environment_report
dump_json(environment_report(),'outputs/preflight_environment.json')
PY
echo "PRE-FLIGHT COMPLETE"
