from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, cin:int, cout:int):
        super().__init__()
        self.c1=nn.Conv2d(cin,cout,3,padding=1,bias=False); self.n1=nn.InstanceNorm2d(cout,affine=True)
        self.c2=nn.Conv2d(cout,cout,3,padding=1,bias=False); self.n2=nn.InstanceNorm2d(cout,affine=True)
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1,bias=False)
    def forward(self,x):
        y=F.relu(self.n1(self.c1(x)),inplace=True); y=self.n2(self.c2(y)); return F.relu(y+self.skip(x),inplace=True)


class ConvBlock(nn.Module):
    def __init__(self, cin:int, cout:int):
        super().__init__()
        self.c1=nn.Conv2d(cin,cout,3,padding=1,bias=False); self.n1=nn.InstanceNorm2d(cout,affine=True)
        self.c2=nn.Conv2d(cout,cout,3,padding=1,bias=False); self.n2=nn.InstanceNorm2d(cout,affine=True)
    def forward(self,x):
        x=F.relu(self.n1(self.c1(x)),inplace=True)
        return F.relu(self.n2(self.c2(x)),inplace=True)


class FixedPrior(nn.Module):
    def __init__(self, kind:str):
        super().__init__(); self.kind=kind
        g=torch.tensor([1.,4.,6.,4.,1.]); g=(g[:,None]*g[None,:]); g/=g.sum()
        lap=torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])
        sx=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8.0
        sy=sx.t()
        self.register_buffer('g',g[None,None]); self.register_buffer('lap',lap[None,None]); self.register_buffer('sx',sx[None,None]); self.register_buffer('sy',sy[None,None])
    def dw(self,x,k,p): return F.conv2d(x,k.expand(x.shape[1],1,*k.shape[-2:]),padding=p,groups=x.shape[1])
    def forward(self,x):
        if self.kind in ('none','standard','passthrough'): return torch.zeros((x.shape[0],1,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        if self.kind=='log':
            r=self.dw(self.dw(x,self.g,2),self.lap,1).abs().mean(1,keepdim=True)
        elif self.kind=='laplacian':
            r=self.dw(x,self.lap,1).abs().mean(1,keepdim=True)
        elif self.kind=='sobel':
            gx=self.dw(x,self.sx,1); gy=self.dw(x,self.sy,1); r=torch.sqrt(gx.square()+gy.square()+1e-8).mean(1,keepdim=True)
        elif self.kind=='gaussian_gradient':
            z=self.dw(x,self.g,2); gx=self.dw(z,self.sx,1); gy=self.dw(z,self.sy,1); r=torch.sqrt(gx.square()+gy.square()+1e-8).mean(1,keepdim=True)
        else: raise ValueError(self.kind)
        mu=r.mean((2,3),keepdim=True); sd=r.std((2,3),keepdim=True,unbiased=False)
        z=(r-mu)/(sd+1e-6)
        return torch.sigmoid(2.5*(z-0.5)).square()


class AttentionGate(nn.Module):
    def __init__(self, xch:int, gch:int, hidden:int|None=None, prior_kind:str='standard'):
        super().__init__(); hidden=hidden or max(8,xch//2); self.prior_kind=prior_kind
        self.wx=nn.Sequential(nn.Conv2d(xch,hidden,1,bias=False),nn.InstanceNorm2d(hidden,affine=True))
        self.wg=nn.Sequential(nn.Conv2d(gch,hidden,1,bias=False),nn.InstanceNorm2d(hidden,affine=True))
        self.prior=FixedPrior(prior_kind)
        self.psi=nn.Conv2d(hidden,1,1)
        self.last_prior=None
    def forward(self,x,g):
        if self.prior_kind == 'passthrough':
            k=torch.zeros((x.shape[0],1,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
            self.last_prior=k
            return x,k
        f=F.relu(self.wx(x)+self.wg(g),inplace=True)
        k=self.prior(f)
        self.last_prior=k
        if self.prior_kind not in ('none','standard'):
            f=f*(1.0+k)
        a=torch.sigmoid(self.psi(f))
        return x*a, k


class PassThroughGate(nn.Module):
    """Unconditional skip connection with no hidden/unused attention parameters."""
    def forward(self, x, g):
        k=torch.zeros((x.shape[0],1,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        return x,k


class UpBlock(nn.Module):
    def __init__(self, in_ch:int, skip_ch:int, out_ch:int, prior_kind:str, block_cls=ResidualBlock):
        super().__init__(); self.up=nn.ConvTranspose2d(in_ch,out_ch,2,stride=2); self.gate=PassThroughGate() if prior_kind=='passthrough' else AttentionGate(skip_ch,out_ch,prior_kind=prior_kind); self.block=block_cls(out_ch+skip_ch,out_ch)
    def forward(self,x,skip):
        g=self.up(x)
        if g.shape[-2:]!=skip.shape[-2:]: g=F.interpolate(g,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        s,k=self.gate(skip,g); return self.block(torch.cat([g,s],1)),k


def prior_levels(depth:str, prior_kind:str) -> list[str]:
    kinds=['standard']*4
    mapping={'shallow':[0,1],'middle':[1,2],'deep':[2,3],'all':[0,1,2,3],'none':[]}
    for i in mapping[depth]: kinds[i]=prior_kind
    return kinds


class CurvatureAttUNet(nn.Module):
    def __init__(self, base:int=40, prior_kind:str='log', prior_depth:str='shallow', multi_task:bool=True, use_attention:bool=True, boundary_prior:bool=True, residual:bool=True):
        super().__init__(); c=[base,base*2,base*4,base*8,base*16]
        self.multi_task=multi_task; self.boundary_prior=boundary_prior; self.pool=nn.MaxPool2d(2)
        block_cls=ResidualBlock if residual else ConvBlock
        self.e1=block_cls(1,c[0]); self.e2=block_cls(c[0],c[1]); self.e3=block_cls(c[1],c[2]); self.e4=block_cls(c[2],c[3]); self.b=block_cls(c[3],c[4])
        levels=prior_levels(prior_depth,prior_kind) if use_attention else ['passthrough']*4
        self.d4=UpBlock(c[4],c[3],c[3],levels[3],block_cls); self.d3=UpBlock(c[3],c[2],c[2],levels[2],block_cls); self.d2=UpBlock(c[2],c[1],c[1],levels[1],block_cls); self.d1=UpBlock(c[1],c[0],c[0],levels[0],block_cls)
        self.main=nn.Conv2d(c[0],1,1)
        if self.multi_task:
            self.aux2=nn.Conv2d(c[1],1,1); self.aux3=nn.Conv2d(c[2],1,1)
            self.sdf=nn.Sequential(nn.Conv2d(c[0],c[0]//2,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c[0]//2,1,1),nn.Tanh())
            self.boundary=nn.Sequential(nn.Conv2d(c[0]+1,c[0],3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c[0],c[0]//2,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c[0]//2,1,1))
        else:
            self.aux2=None; self.aux3=None; self.sdf=None; self.boundary=None
    def encode(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); b=self.b(self.pool(e4)); return e1,e2,e3,e4,b
    def forward(self,x):
        e1,e2,e3,e4,b=self.encode(x); d4,k4=self.d4(b,e4); d3,k3=self.d3(d4,e3); d2,k2=self.d2(d3,e2); d1,k1=self.d1(d2,e1)
        z0=self.main(d1)
        # The submitted final model feeds the shallowest LoG response to the boundary head.
        # Controlled prior/depth experiments can set boundary_prior=False so the ONLY changed path is attention.
        if self.boundary_prior and k1 is not None:
            k=k1
            if k.shape[-2:]!=d1.shape[-2:]: k=F.interpolate(k,size=d1.shape[-2:],mode='bilinear',align_corners=False)
        else:
            k=torch.zeros_like(z0)
        if self.multi_task:
            delta=self.boundary(torch.cat([d1,k],1))
            z=z0+delta
            aux2=F.interpolate(self.aux2(d2),size=z.shape[-2:],mode='bilinear',align_corners=False)
            aux3=F.interpolate(self.aux3(d3),size=z.shape[-2:],mode='bilinear',align_corners=False)
            sdf=self.sdf(d1)
        else:
            delta=torch.zeros_like(z0); z=z0; aux2=None; aux3=None; sdf=None
        return {'logits':z,'main_logits':z0,'aux2':aux2,'aux3':aux3,'sdf':sdf,'boundary_residual':delta,'kappa':k}
    def freeze_encoder(self):
        for module in [self.e1,self.e2,self.e3,self.e4,self.b]:
            for p in module.parameters(): p.requires_grad=False


def build_model(name:str, prior_kind:str='log', prior_depth:str='shallow', base:int=40):
    name=name.lower()
    # Controlled U-Net-family baselines. These are NOT claimed to be official repository implementations.
    if name in ('unet','unet_control'):
        return CurvatureAttUNet(base,prior_kind='standard',prior_depth='none',multi_task=False,use_attention=False,boundary_prior=False,residual=False)
    if name in ('attention_unet','attunet_control'):
        return CurvatureAttUNet(base,prior_kind='standard',prior_depth='none',multi_task=False,use_attention=True,boundary_prior=False,residual=False)
    if name in ('resunet','resunet_control'):
        return CurvatureAttUNet(base,prior_kind='standard',prior_depth='none',multi_task=False,use_attention=False,boundary_prior=False)
    if name in ('attunet','attention_resunet','attention_resunet_control'):
        return CurvatureAttUNet(base,prior_kind='standard',prior_depth='none',multi_task=False,use_attention=True,boundary_prior=False)
    # Matched full multi-task controls. Boundary prior is disabled in both, isolating attention guidance.
    if name=='full_standard':
        return CurvatureAttUNet(base,prior_kind='standard',prior_depth='none',multi_task=True,use_attention=True,boundary_prior=False)
    if name=='full_log_attention_only':
        return CurvatureAttUNet(base,prior_kind='log',prior_depth='shallow',multi_task=True,use_attention=True,boundary_prior=False)
    # Submitted final architecture: LoG shallow attention + LoG response supplied to boundary residual head.
    if name=='full_log_shallow':
        return CurvatureAttUNet(base,prior_kind='log',prior_depth='shallow',multi_task=True,use_attention=True,boundary_prior=True)
    # Prior/depth studies isolate only the attention path to avoid an output-head confound.
    if name=='prior_ablation':
        return CurvatureAttUNet(base,prior_kind=prior_kind,prior_depth='shallow',multi_task=True,use_attention=True,boundary_prior=False)
    if name=='depth_ablation':
        return CurvatureAttUNet(base,prior_kind='log',prior_depth=prior_depth,multi_task=True,use_attention=True,boundary_prior=False)
    raise ValueError(f'Unknown model {name}')
