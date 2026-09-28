from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import HC18Dataset, Case, discover_hc18_cases
from .losses import stage1_loss, stage2_loss
from .metrics import per_case_metrics
from .model import build_model
from .utils import atomic_torch_save, dump_json, environment_report, load_json, seed_everything, sha256_file


def model_label(model: str, prior_kind: str, prior_depth: str) -> str:
    if model == "prior_ablation":
        return f"prior_{prior_kind}"
    if model == "depth_ablation":
        return f"depth_{prior_depth}"
    return model


def selected_cases(all_cases: list[Case], ids: Iterable[str]) -> list[Case]:
    mapping = {c.case_id: c for c in all_cases}
    ids = list(ids)
    missing = [x for x in ids if x not in mapping]
    if missing:
        raise KeyError(f"Split contains unknown case IDs: {missing[:5]}")
    return [mapping[x] for x in ids]


def loader(cases, image_size, batch_size, augment, seed, shuffle):
    ds = HC18Dataset(cases, image_size=image_size, augment=augment, seed=seed, with_sdf=True)
    g = torch.Generator(); g.manual_seed(seed)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                    pin_memory=torch.cuda.is_available(), generator=g, drop_last=False)
    return ds, ld


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(enabled and device.type == "cuda"))


def optimizer_stage1(model, lr=1e-4, wd=1e-5):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)


def optimizer_stage2(model, mode: str, boundary_lr=1e-5, decoder_lr=5e-6, wd=1e-6):
    if mode == "frozen_encoder":
        model.freeze_encoder()
        fast, slow = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("gate", "main", "aux", "sdf", "boundary")):
                fast.append(p)
            else:
                slow.append(p)
        groups=[]
        if fast: groups.append({"params":fast,"lr":boundary_lr})
        if slow: groups.append({"params":slow,"lr":decoder_lr})
        return torch.optim.AdamW(groups, weight_decay=wd)
    if mode == "full":
        # Fair matched-budget continuation: all parameters remain trainable, but
        # the exact same two-rate grouping used by the submitted second stage is
        # retained.  This avoids giving the proposed model a uniquely favorable
        # optimizer while still allowing every comparator the same 8-epoch
        # optimization opportunity.
        for p in model.parameters():
            p.requires_grad=True
        fast, slow = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in ("gate", "main", "aux", "sdf", "boundary")):
                fast.append(p)
            else:
                slow.append(p)
        groups=[]
        if fast: groups.append({"params":fast,"lr":boundary_lr})
        if slow: groups.append({"params":slow,"lr":decoder_lr})
        return torch.optim.AdamW(groups, weight_decay=wd)
    raise ValueError(mode)


def train_epoch(model, ld, optimizer, scaler, device, amp, epoch_one_based, stage):
    model.train()
    ld.dataset.set_epoch(epoch_one_based)
    # Make shuffle order an explicit function of (training seed, epoch), so an
    # interrupted/resumed run is byte-for-byte protocol-equivalent to an
    # uninterrupted run rather than restarting the DataLoader RNG stream.
    if getattr(ld, "generator", None) is not None:
        ld.generator.manual_seed(int(ld.dataset.seed) * 1_000_003 + int(epoch_one_based) * 10_009)
    sums={}; n=0
    for batch in ld:
        x=batch["image"].to(device, non_blocking=True); y=batch["mask"].to(device, non_blocking=True); sdf=batch["sdf"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            out=model(x)
            if stage == 1:
                loss, parts = stage1_loss(out, y, sdf, epoch_one_based, multi_task=model.multi_task)
            else:
                loss, parts = stage2_loss(out, y, sdf, multi_task=model.multi_task)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        bs=x.shape[0]; n += bs
        sums["loss"]=sums.get("loss",0.0)+float(loss.detach())*bs
        for k,v in parts.items(): sums[k]=sums.get(k,0.0)+float(v)*bs
    return {k:v/max(n,1) for k,v in sums.items()}


@torch.inference_mode()
def evaluate(model, ld, device, amp, threshold=0.5, save_predictions: Path|None=None):
    model.eval(); rows=[]
    if save_predictions is not None: save_predictions.mkdir(parents=True,exist_ok=True)
    for batch in ld:
        x=batch["image"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            out=model(x); prob=torch.sigmoid(out["logits"])
        pred=(prob >= threshold).cpu().numpy()[:,0].astype(np.uint8)
        gt=(batch["mask"].numpy()[:,0] >= .5).astype(np.uint8)
        for i,cid in enumerate(batch["case_id"]):
            met=per_case_metrics(pred[i],gt[i]); met["case_id"]=str(cid); rows.append(met)
            if save_predictions is not None:
                cv2.imwrite(str(save_predictions/f"{cid}.png"), pred[i]*255)
    df=pd.DataFrame(rows)
    numeric=["dice","iou","hd","hd95","asd"]
    summary={f"mean_{c}":float(df[c].mean()) for c in numeric}
    summary.update({f"median_{c}":float(df[c].median()) for c in numeric})
    summary["n_cases"]=int(len(df)); summary["empty_pred_failures"]=int(df["empty_pred_failure"].sum())
    return df,summary


def write_history(rows, path):
    pd.DataFrame(rows).to_csv(path,index=False)


def selection_from_history(rows, dice_tolerance=0.0003):
    if not rows: raise RuntimeError("empty history")
    best=max(rows,key=lambda r:(r["val_dice"],-r["val_hd"]))
    dmax=max(r["val_dice"] for r in rows)
    eligible=[r for r in rows if r["val_dice"] >= dmax-dice_tolerance]
    constrained=min(eligible,key=lambda r:(r["val_hd"],-r["val_dice"]))
    return {
        "primary_best_dice_epoch":int(best["epoch"]),
        "primary_best_dice":float(best["val_dice"]),
        "primary_hd_at_best_dice":float(best["val_hd"]),
        "sensitivity_constrained_hd_epoch":int(constrained["epoch"]),
        "sensitivity_constrained_hd":float(constrained["val_hd"]),
        "sensitivity_dice":float(constrained["val_dice"]),
        "dice_tolerance":float(dice_tolerance),
    }


def save_checkpoint(model, path, extra):
    atomic_torch_save({"model":model.state_dict(), **extra}, path)


def load_checkpoint(model, path, device):
    obj=torch.load(path,map_location=device,weights_only=False)
    state=obj["model"] if isinstance(obj,dict) and "model" in obj else obj
    model.load_state_dict(state,strict=True)
    return obj


def develop(args, run_dir, split, all_cases, device):
    seed_everything(args.seed)
    model=build_model(args.model,args.prior_kind,args.prior_depth,args.base_channels).to(device)
    tr_cases=selected_cases(all_cases,split["inner_train"]); va_cases=selected_cases(all_cases,split["inner_val"])
    _,tr_ld=loader(tr_cases,args.image_size,args.batch_size,True,args.seed,True)
    _,va_ld=loader(va_cases,args.image_size,args.eval_batch_size,False,args.seed,False)
    opt=optimizer_stage1(model,args.lr_start,args.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.stage1_epochs,1),eta_min=args.lr_end)
    scaler=torch.amp.GradScaler("cuda",enabled=(args.amp and device.type=="cuda"))
    hist_path=run_dir/"develop_stage1_history.csv"; history=pd.read_csv(hist_path).to_dict('records') if hist_path.exists() else []
    best_path=run_dir/"develop_stage1_best_dice.pth"; last_path=run_dir/"develop_stage1_last.pth"
    start_ep=1
    if last_path.exists():
        obj=torch.load(last_path,map_location=device,weights_only=False); model.load_state_dict(obj['model']); opt.load_state_dict(obj['optimizer']); sch.load_state_dict(obj['scheduler']); scaler.load_state_dict(obj['scaler']); start_ep=int(obj['epoch'])+1
        print(f"[resume develop stage1] epoch {start_ep}",flush=True)
    elif len(history) >= args.stage1_epochs and best_path.exists():
        # Reuse a byte-for-byte compatible inner-development run (used by matched-budget/original-stage2 branches).
        start_ep=args.stage1_epochs+1
        print(f"[reuse develop stage1] history/checkpoint already complete; skip {args.stage1_epochs} epochs",flush=True)
    best_dice=max([float(r['val_dice']) for r in history],default=-1.0)
    for ep in range(start_ep,args.stage1_epochs+1):
        parts=train_epoch(model,tr_ld,opt,scaler,device,args.amp,ep,1)
        _,sv=evaluate(model,va_ld,device,args.amp,args.threshold)
        row={"epoch":ep,"lr":float(opt.param_groups[0]["lr"]),**{f"train_{k}":v for k,v in parts.items()},
             "val_dice":sv["mean_dice"],"val_iou":sv["mean_iou"],"val_hd":sv["mean_hd"],"val_hd95":sv["mean_hd95"],"val_asd":sv["mean_asd"],"val_failures":sv["empty_pred_failures"]}
        history=[r for r in history if int(r['epoch'])<ep]+[row]; write_history(history,hist_path)
        if row["val_dice"] > best_dice:
            best_dice=row["val_dice"]; save_checkpoint(model,best_path,{"epoch":ep,"phase":"develop_stage1","val":sv})
        sch.step()
        atomic_torch_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),'epoch':ep},last_path)
        print(f"[develop stage1] ep={ep:02d}/{args.stage1_epochs} loss={parts['loss']:.5f} Dice={sv['mean_dice']*100:.4f}% HD={sv['mean_hd']:.4f} HD95={sv['mean_hd95']:.4f}",flush=True)
    sel1=selection_from_history(history,args.dice_tolerance)
    if not best_path.exists(): raise RuntimeError('missing develop best-Dice checkpoint')
    if int(torch.load(best_path,map_location="cpu",weights_only=False)["epoch"]) != sel1["primary_best_dice_epoch"]:
        raise RuntimeError("best-Dice checkpoint/history mismatch")

    selection={"stage1":sel1,"stage2":None,"budget":args.budget,"stage2_mode":None}
    if args.budget != "stage1":
        mode="full" if args.budget=="matched48" else "frozen_encoder"
        hist2_path=run_dir/"develop_stage2_history.csv"; history2=pd.read_csv(hist2_path).to_dict('records') if hist2_path.exists() else []
        best2_path=run_dir/"develop_stage2_best_dice.pth"; last2_path=run_dir/"develop_stage2_last.pth"
        if last2_path.exists():
            obj=torch.load(last2_path,map_location=device,weights_only=False); model.load_state_dict(obj['model'])
            opt2=optimizer_stage2(model,mode,args.stage2_boundary_lr,args.stage2_decoder_lr,args.stage2_weight_decay); sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=max(args.stage2_epochs,1),eta_min=args.lr_end)
            opt2.load_state_dict(obj['optimizer']); sch2.load_state_dict(obj['scheduler']); scaler.load_state_dict(obj['scaler']); start2=int(obj['epoch'])+1
            print(f"[resume develop stage2:{mode}] epoch {start2}",flush=True)
        else:
            load_checkpoint(model,best_path,device); opt2=optimizer_stage2(model,mode,args.stage2_boundary_lr,args.stage2_decoder_lr,args.stage2_weight_decay)
            sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=max(args.stage2_epochs,1),eta_min=args.lr_end); start2=1
        best2=max([float(r['val_dice']) for r in history2],default=-1.0)
        for ep in range(start2,args.stage2_epochs+1):
            parts=train_epoch(model,tr_ld,opt2,scaler,device,args.amp,ep,2)
            _,sv=evaluate(model,va_ld,device,args.amp,args.threshold)
            row={"epoch":ep,"lr":float(opt2.param_groups[0]["lr"]),**{f"train_{k}":v for k,v in parts.items()},
                 "val_dice":sv["mean_dice"],"val_iou":sv["mean_iou"],"val_hd":sv["mean_hd"],"val_hd95":sv["mean_hd95"],"val_asd":sv["mean_asd"],"val_failures":sv["empty_pred_failures"]}
            history2=[r for r in history2 if int(r['epoch'])<ep]+[row]; write_history(history2,hist2_path)
            if row["val_dice"] > best2:
                best2=row["val_dice"]; save_checkpoint(model,best2_path,{"epoch":ep,"phase":"develop_stage2","val":sv,"stage1_epoch":sel1["primary_best_dice_epoch"]})
            sch2.step(); atomic_torch_save({'model':model.state_dict(),'optimizer':opt2.state_dict(),'scheduler':sch2.state_dict(),'scaler':scaler.state_dict(),'epoch':ep},last2_path)
            print(f"[develop stage2:{mode}] ep={ep:02d}/{args.stage2_epochs} loss={parts['loss']:.5f} Dice={sv['mean_dice']*100:.4f}% HD={sv['mean_hd']:.4f}",flush=True)
        selection["stage2"]=selection_from_history(history2,args.dice_tolerance); selection["stage2_mode"]=mode
    dump_json(selection,run_dir/"selection.json")
    for q in (run_dir/"develop_stage1_last.pth",run_dir/"develop_stage2_last.pth"):
        if q.exists(): q.unlink()
    (run_dir/"DEVELOP_COMPLETE").write_text("inner validation only\n")
    return selection

def chosen_epoch(selection_block: dict, rule: str) -> int:
    if rule == "best_dice": return int(selection_block["primary_best_dice_epoch"])
    if rule == "constrained_hd": return int(selection_block["sensitivity_constrained_hd_epoch"])
    raise ValueError(rule)

def refit(args, run_dir, split, all_cases, device):
    sel_path=run_dir/"selection.json"
    if not sel_path.exists(): raise FileNotFoundError(f"Run develop first: {sel_path}")
    selection=load_json(sel_path); seed_everything(args.seed)
    model=build_model(args.model,args.prior_kind,args.prior_depth,args.base_channels).to(device)
    cases=selected_cases(all_cases,split["outer_train"]); _,ld=loader(cases,args.image_size,args.batch_size,True,args.seed,True)
    scaler=torch.amp.GradScaler("cuda",enabled=(args.amp and device.type=="cuda")); n1=chosen_epoch(selection["stage1"],args.selection_rule)
    n2=chosen_epoch(selection["stage2"],args.selection_rule) if selection.get('stage2') else 0; mode=selection.get('stage2_mode')
    last=run_dir/"refit_last.pth"; hist_path=run_dir/"refit_history.csv"; hist=pd.read_csv(hist_path).to_dict('records') if hist_path.exists() else []
    resume_obj=torch.load(last,map_location=device,weights_only=False) if last.exists() else None
    if resume_obj is not None: model.load_state_dict(resume_obj['model']); scaler.load_state_dict(resume_obj['scaler']); print(f"[resume refit] stage={resume_obj['stage']} epoch={resume_obj['epoch']}",flush=True)

    # Stage 1
    if resume_obj is None or int(resume_obj['stage'])==1:
        opt=optimizer_stage1(model,args.lr_start,args.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.stage1_epochs,1),eta_min=args.lr_end); start1=1
        if resume_obj is not None:
            opt.load_state_dict(resume_obj['optimizer']); sch.load_state_dict(resume_obj['scheduler']); start1=int(resume_obj['epoch'])+1
        for ep in range(start1,n1+1):
            parts=train_epoch(model,ld,opt,scaler,device,args.amp,ep,1); sch.step(); hist=[r for r in hist if not (int(r['stage'])==1 and int(r['epoch'])>=ep)]+[{"stage":1,"epoch":ep,**parts}]; write_history(hist,hist_path)
            atomic_torch_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),'stage':1,'epoch':ep},last)
            print(f"[refit stage1] ep={ep:02d}/{n1} loss={parts['loss']:.5f}",flush=True)
    # Stage 2, if selected by the inner-development procedure.
    if n2>0:
        opt2=optimizer_stage2(model,mode,args.stage2_boundary_lr,args.stage2_decoder_lr,args.stage2_weight_decay); sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=max(args.stage2_epochs,1),eta_min=args.lr_end); start2=1
        if resume_obj is not None and int(resume_obj['stage'])==2:
            opt2.load_state_dict(resume_obj['optimizer']); sch2.load_state_dict(resume_obj['scheduler']); start2=int(resume_obj['epoch'])+1
        for ep in range(start2,n2+1):
            parts=train_epoch(model,ld,opt2,scaler,device,args.amp,ep,2); sch2.step(); hist=[r for r in hist if not (int(r['stage'])==2 and int(r['epoch'])>=ep)]+[{"stage":2,"epoch":ep,**parts}]; write_history(hist,hist_path)
            atomic_torch_save({'model':model.state_dict(),'optimizer':opt2.state_dict(),'scheduler':sch2.state_dict(),'scaler':scaler.state_dict(),'stage':2,'epoch':ep},last)
            print(f"[refit stage2:{mode}] ep={ep:02d}/{n2} loss={parts['loss']:.5f}",flush=True)
    ckpt=run_dir/"final_outer_train_model.pth"; save_checkpoint(model,ckpt,{"selection":selection,"phase":"refit","seed":args.seed})
    manifest={"checkpoint":str(ckpt),"checkpoint_sha256":sha256_file(ckpt),"outer_train_cases":len(cases),"selection":selection}; dump_json(manifest,run_dir/"refit_manifest.json")
    if last.exists(): last.unlink()
    if not args.keep_develop_checkpoints:
        for q in (run_dir/"develop_stage1_best_dice.pth",run_dir/"develop_stage2_best_dice.pth"):
            if q.exists(): q.unlink()
    (run_dir/"REFIT_COMPLETE").write_text("trained without outer-test access\n")
    return manifest

def test_outer(args, run_dir, split, all_cases, device):
    lock=Path(args.protocol_lock)
    if not lock.exists(): raise RuntimeError(f"Formal outer-test is locked until the protocol is frozen: {lock}")
    frozen=load_json(lock)
    entries={r["run_dir"]:r for r in frozen.get("runs",[])}
    if str(run_dir) not in entries: raise RuntimeError(f"Run is not listed in frozen protocol manifest: {run_dir}")
    # Refuse evaluation if code/splits/checkpoints changed after the pre-test freeze.
    for e in frozen.get("code_files",[]):
        p=Path(e["path"]);
        if (not p.exists()) or sha256_file(p)!=e["sha256"]: raise RuntimeError(f"Code changed after protocol freeze: {p}")
    for e in frozen.get("split_files",[]):
        p=Path(e["path"]);
        if (not p.exists()) or sha256_file(p)!=e["sha256"]: raise RuntimeError(f"Split changed after protocol freeze: {p}")
    marker=run_dir/"OUTER_TEST_COMPLETE"
    if marker.exists() and not args.force_test:
        raise RuntimeError(f"Outer test already completed. Refusing to re-evaluate without --force-test: {marker}")
    ckpt=run_dir/"final_outer_train_model.pth"
    if not ckpt.exists(): raise FileNotFoundError(f"Run refit first: {ckpt}")
    frozen_ckpt=entries[str(run_dir)].get("checkpoint_sha256")
    if frozen_ckpt and sha256_file(ckpt)!=frozen_ckpt: raise RuntimeError(f"Final checkpoint changed after protocol freeze: {ckpt}")
    seed_everything(args.seed)
    model=build_model(args.model,args.prior_kind,args.prior_depth,args.base_channels).to(device)
    load_checkpoint(model,ckpt,device)
    cases=selected_cases(all_cases,split["outer_test"]); _,ld=loader(cases,args.image_size,args.eval_batch_size,False,args.seed,False)
    pred_dir=run_dir/"predictions_256"
    df,s=evaluate(model,ld,device,args.amp,args.threshold,pred_dir)
    df.insert(1,"fold",int(split["fold"])); df.insert(2,"seed",args.seed); df.insert(3,"model",model_label(args.model,args.prior_kind,args.prior_depth)); df.insert(4,"budget",args.budget); df.insert(5,"selection_rule",args.selection_rule)
    df.to_csv(run_dir/"per_image_metrics.csv",index=False)
    summary={**s,"fold":int(split["fold"]),"seed":args.seed,"model":model_label(args.model,args.prior_kind,args.prior_depth),"budget":args.budget,"selection_rule":args.selection_rule,"threshold":args.threshold,
             "segmentation_postprocess":"none","empty_prediction_policy":"image diagonal penalty","checkpoint_sha256":sha256_file(ckpt),"protocol_lock_sha256":sha256_file(lock)}
    dump_json(summary,run_dir/"outer_test_summary.json")
    marker.write_text("outer test evaluated once; do not use for tuning\n")
    print("[outer-test]",summary,flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description="Leakage-resistant HC18 nested outer-test training/evaluation")
    ap.add_argument("--data-root",default="../fetal_dataset")
    ap.add_argument("--split-json",required=True)
    ap.add_argument("--output-root",default="outputs/runs")
    ap.add_argument("--experiment",default="core")
    ap.add_argument("--model",required=True,choices=["unet","attention_unet","resunet","attention_resunet","attunet","full_standard","full_log_attention_only","full_log_shallow","prior_ablation","depth_ablation"])
    ap.add_argument("--prior-kind",default="log",choices=["none","standard","sobel","laplacian","gaussian_gradient","log"])
    ap.add_argument("--prior-depth",default="shallow",choices=["none","shallow","middle","deep","all"])
    ap.add_argument("--budget",default="stage1",choices=["stage1","matched48","original48"]); ap.add_argument("--selection-rule",default="best_dice",choices=["best_dice","constrained_hd"])
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--phase",required=True,choices=["develop","refit","test","all"])
    ap.add_argument("--image-size",type=int,default=256); ap.add_argument("--base-channels",type=int,default=40)
    ap.add_argument("--stage1-epochs",type=int,default=40); ap.add_argument("--stage2-epochs",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--eval-batch-size",type=int,default=8)
    ap.add_argument("--lr-start",type=float,default=1e-4); ap.add_argument("--lr-end",type=float,default=1e-6); ap.add_argument("--weight-decay",type=float,default=1e-5)
    ap.add_argument("--stage2-boundary-lr",type=float,default=1e-5); ap.add_argument("--stage2-decoder-lr",type=float,default=5e-6); ap.add_argument("--stage2-weight-decay",type=float,default=1e-6)
    ap.add_argument("--threshold",type=float,default=0.5); ap.add_argument("--dice-tolerance",type=float,default=0.0003)
    ap.add_argument("--amp",action="store_true"); ap.add_argument("--force-test",action="store_true"); ap.add_argument("--keep-develop-checkpoints",action="store_true"); ap.add_argument("--protocol-lock",default="outputs/PROTOCOL_FROZEN.json")
    args=ap.parse_args()
    if args.budget=="original48" and args.model!="full_log_shallow":
        raise ValueError("original48 is reserved for the submitted full_log_shallow frozen-encoder method. Use matched48 for comparator fairness.")
    split_path=Path(args.split_json); split=load_json(split_path); all_cases=discover_hc18_cases(args.data_root)
    label=model_label(args.model,args.prior_kind,args.prior_depth)
    run_dir=Path(args.output_root)/args.experiment/label/f"budget_{args.budget}"/f"selection_{args.selection_rule}"/f"seed_{args.seed}"/f"fold_{split['fold']}"
    run_dir.mkdir(parents=True,exist_ok=True)
    config={**vars(args),"split_sha256":sha256_file(split_path),"run_dir":str(run_dir),"model_label":label,"environment":environment_report()}
    config_path=run_dir/"run_config.json"
    if config_path.exists():
        old=load_json(config_path)
        immutable=["data_root","split_sha256","model","prior_kind","prior_depth","budget","selection_rule","seed","image_size","base_channels","stage1_epochs","stage2_epochs","batch_size","eval_batch_size","lr_start","lr_end","weight_decay","stage2_boundary_lr","stage2_decoder_lr","stage2_weight_decay","threshold","dice_tolerance","amp"]
        changed={k:(old.get(k),config.get(k)) for k in immutable if old.get(k)!=config.get(k)}
        if changed: raise RuntimeError(f"Run directory config mismatch; refusing to mix experiments: {changed}")
    else: dump_json(config,config_path)
    lock=Path(args.protocol_lock)
    if lock.exists() and args.phase in ("develop","refit","all"):
        raise RuntimeError(f"Protocol is frozen; formal training/refit is prohibited. Do not tune after seeing outer-test data. Lock: {lock}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run={run_dir}\ndevice={device}\nphase={args.phase}\nsplit={split_path} sha256={config['split_sha256']}",flush=True)
    if args.phase in ("develop","all"): develop(args,run_dir,split,all_cases,device)
    if args.phase in ("refit","all"): refit(args,run_dir,split,all_cases,device)
    if args.phase in ("test","all"): test_outer(args,run_dir,split,all_cases,device)

if __name__=="__main__": main()
