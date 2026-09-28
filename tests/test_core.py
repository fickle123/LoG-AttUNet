import numpy as np
import torch

from curvrevision.metrics import per_case_metrics
from curvrevision.model import PassThroughGate, build_model
from curvrevision.losses import stage1_loss


def test_models():
    x=torch.randn(2,1,64,64); y=torch.zeros(2,1,64,64); y[:,:,16:48,16:48]=1; sdf=torch.zeros_like(y)
    names=['unet','attention_unet','resunet','attention_resunet','full_standard','full_log_attention_only','full_log_shallow']
    for n in names:
        m=build_model(n,base=4); out=m(x); assert out['logits'].shape==y.shape
        l,_=stage1_loss(out,y,sdf,1,m.multi_task); assert torch.isfinite(l)
    std=build_model('full_standard',base=4); log=build_model('full_log_attention_only',base=4)
    assert sum(p.numel() for p in std.parameters()) == sum(p.numel() for p in log.parameters())
    assert std.boundary_prior is False and log.boundary_prior is False
    assert isinstance(build_model('resunet',base=4).d1.gate, PassThroughGate)
    assert isinstance(build_model('unet',base=4).d1.gate, PassThroughGate)
    assert build_model('prior_ablation',prior_kind='none',base=4).d1.gate.prior_kind=='none'
    # Region-only controls must not carry unused multi-task heads.
    assert build_model('unet',base=4).sdf is None
    assert build_model('resunet',base=4).boundary is None


def test_metrics_empty_penalty():
    gt=np.zeros((32,32),np.uint8); gt[8:24,8:24]=1; pred=np.zeros_like(gt)
    m=per_case_metrics(pred,gt); assert m['empty_pred_failure']; assert m['hd']>32
    perfect=per_case_metrics(gt,gt); assert perfect['dice']==1 and perfect['hd']==0 and perfect['hd95']==0


if __name__=='__main__':
    test_models(); test_metrics_empty_penalty(); print('ALL CORE TESTS PASSED')
