from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import torch
from .model import build_model
from .utils import dump_json, environment_report


def params(model): return sum(p.numel() for p in model.parameters())


def profile_flops(model,device,size):
    # torch profiler captures both module convolutions and functional depthwise convolutions used by the fixed priors.
    acts=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda': acts.append(torch.profiler.ProfilerActivity.CUDA)
    x=torch.randn(1,1,size,size,device=device)
    model.eval()
    with torch.inference_mode(), torch.profiler.profile(activities=acts,with_flops=True,record_shapes=False) as prof:
        model(x)
        if device.type=='cuda': torch.cuda.synchronize()
    flops=sum((e.flops or 0) for e in prof.key_averages())
    return float(flops)


def latency(model,device,size,batch,warmup,iters,amp):
    x=torch.randn(batch,1,size,size,device=device); model.eval()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(amp and device.type=='cuda')): model(x)
        if device.type=='cuda': torch.cuda.synchronize()
        t0=time.perf_counter()
        for _ in range(iters):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(amp and device.type=='cuda')): model(x)
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-t0
    ms=elapsed/iters*1000; throughput=batch*iters/elapsed; peak=(torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else float('nan'))
    return ms,throughput,peak


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='outputs/efficiency'); ap.add_argument('--image-size',type=int,default=256); ap.add_argument('--base-channels',type=int,default=40)
    ap.add_argument('--models',nargs='+',default=['unet','attention_unet','resunet','attention_resunet','full_standard','full_log_attention_only','full_log_shallow']); ap.add_argument('--warmup',type=int,default=50); ap.add_argument('--iters',type=int,default=200); ap.add_argument('--throughput-batch',type=int,default=8); ap.add_argument('--amp',action='store_true'); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
    for name in args.models:
        model=build_model(name,base=args.base_channels).to(device)
        f=profile_flops(model,device,args.image_size)
        l1,tp1,mem1=latency(model,device,args.image_size,1,args.warmup,args.iters,args.amp)
        lb,tp,mem=latency(model,device,args.image_size,args.throughput_batch,args.warmup,args.iters,args.amp)
        rows.append({'model':name,'parameters':params(model),'parameters_m':params(model)/1e6,'flops':f,'gflops':f/1e9,'batch1_latency_ms':l1,'batch1_throughput_images_s':tp1,
                     f'batch{args.throughput_batch}_latency_ms':lb,f'batch{args.throughput_batch}_throughput_images_s':tp,'peak_gpu_memory_mb':max(mem1,mem),
                     'image_size':args.image_size,'amp':args.amp,'device':str(device)})
        print(rows[-1],flush=True); del model
        if device.type=='cuda': torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'efficiency.csv',index=False); dump_json(environment_report(),out/'hardware_environment.json')

if __name__=='__main__': main()
