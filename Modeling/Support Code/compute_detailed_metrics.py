#!/usr/bin/env python3
# compute_detailed_metrics.py

# =============================================================
# Nombre del archivo: compute_detailed_metrics.py
# Autores: Bárbara Paola Alcántara Vega
#           
# Descripción:
#Calcula Loss, MAE, Bias, Varianza, Accuracy y sus contrapartes 
# en validation
#Soporta:
# - usar archivos preds_*.npz si ya existen
# - usar múltiples .npz (ensemble) para estimar Varianza entre 
#   modelos
# - usar checkpoint + MC Dropout para estimar Varianza (si tu
#   modelo tiene dropout)
#Salida: imprime resumen y guarda npz/report en out_dir
#
# Dependencias: argparse, pathlib, numpy, math, sklearn, json,
#               torch, glob, os, typing
#     
# =============================================================


import argparse
from pathlib import Path
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
import json
import torch
import torch.nn as nn
import glob
import os
from typing import Optional

def find_npz_in_run(run_dir: str, split: str) -> Optional[Path]:
    """
    Search recursively for preds_{split}.npz (or *preds*_{split}.npz) under run_dir.
    Returns newest file by modification time or None.
    """
    patterns = [
        os.path.join(run_dir, f"**/preds_{split}.npz"),
        os.path.join(run_dir, f"**/*preds*_{split}.npz"),
        os.path.join(run_dir, f"preds_{split}.npz"),
        os.path.join(run_dir, f"*preds*_{split}.npz"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p, recursive=True))
    matches = sorted(set(matches))
    if not matches:
        return None
    # choose newest by mtime
    newest = max(matches, key=lambda p: os.path.getmtime(p))
    return Path(newest)

def load_npz_if_exists(p: Path):
    if p is None:
        return None
    if not Path(p).exists():
        return None
    d = np.load(p, allow_pickle=True)
    preds = np.array(d["preds"])
    labels = np.array(d["labels"]) if "labels" in d else None
    paths = None
    if "paths" in d:
        # convert paths to plain strings
        raw_paths = np.array(d["paths"])
        # sometimes saved as bytes
        paths = [str(x) for x in raw_paths]
    return {"preds": preds, "labels": labels, "paths": paths, "npz_path": str(p)}

def nearest_quarter(arr):
    return np.round(np.array(arr) / 0.25) * 0.25

def compute_basic_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    exact_acc = (nearest_quarter(y_pred) == y_true).mean()
    return {"mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2), "exact_acc": float(exact_acc)}

def compute_bias_variance_from_ensemble(preds_ensemble, y_true):
    mean_pred = preds_ensemble.mean(axis=0)
    var_pred = preds_ensemble.var(axis=0, ddof=0)
    bias = mean_pred - y_true
    bias_abs_mean = np.mean(np.abs(bias))
    var_mean = np.mean(var_pred)
    return {"bias_per_sample": bias, "var_per_sample": var_pred, "bias_abs_mean": float(bias_abs_mean), "var_mean": float(var_mean), "mean_pred": mean_pred}

def safe_print_dict(d: dict):
    for k,v in d.items():
        print(f"{k}: {v}")

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If run_dir provided and explicit npz args are default/absent, try to locate proper npz inside run_dir
    if args.out_dir:
        # only override if user didn't explicitly pass a specific path (i.e., they are still default or empty)
        if not args.train_npz or args.train_npz.strip() == "":
            found = find_npz_in_run(args.out_dir, "train")
            if found:
                print(f"[auto] found train npz in run_dir: {found}")
                args.train_npz = str(found)
        if not args.val_npz or args.val_npz.strip() == "":
            found = find_npz_in_run(args.out_dir, "val")
            if found:
                print(f"[auto] found val npz in run_dir: {found}")
                args.val_npz = str(found)
        if not args.test_npz or args.test_npz.strip() == "":
            found = find_npz_in_run(args.out_dir, "test")
            if found:
                print(f"[auto] found test npz in run_dir: {found}")
                args.test_npz = str(found)

    # Try to load preds files first
    train_npz = load_npz_if_exists(Path(args.train_npz)) if args.train_npz else None
    val_npz = load_npz_if_exists(Path(args.val_npz)) if args.val_npz else None
    test_npz = load_npz_if_exists(Path(args.test_npz)) if args.test_npz else None

    # Show what we actually loaded (helps debug wrong-run issue)
    if train_npz:
        print(f"Loaded train npz from: {train_npz.get('npz_path')}; samples={len(train_npz['preds'])}")
    if val_npz:
        print(f"Loaded val npz from: {val_npz.get('npz_path')}; samples={len(val_npz['preds'])}")
    if test_npz:
        print(f"Loaded test npz from: {test_npz.get('npz_path')}; samples={len(test_npz['preds'])}")

    summary = {}

    # If ensemble files provided (multiple npz), build preds_ensemble for test/val
    if args.ensemble_files:
        ens_preds_list = []
        labels_ref = None
        for p in args.ensemble_files:
            d = load_npz_if_exists(Path(p))
            if d is None:
                raise RuntimeError(f"Ensemble file not found: {p}")
            ens_preds_list.append(d["preds"])
            if labels_ref is None:
                labels_ref = d["labels"]
            else:
                if not np.array_equal(labels_ref, d["labels"]):
                    raise RuntimeError("Ensemble files have differing labels arrays - align before using ensemble_files.")
        preds_ensemble = np.stack(ens_preds_list, axis=0)
        emb = compute_bias_variance_from_ensemble(preds_ensemble, labels_ref)
        basic = compute_basic_metrics(labels_ref, emb["mean_pred"])
        summary["ensemble"] = {"basic": basic, "bias_abs_mean": emb["bias_abs_mean"], "var_mean": emb["var_mean"]}
        np.savez(out_dir / "ensemble_bias_var.npz", bias=emb["bias_per_sample"], var=emb["var_per_sample"], mean_pred=emb["mean_pred"], labels=labels_ref)

    # If preds_val exists compute metrics and bias/var fallback
    if val_npz:
        y_val = val_npz["labels"]
        yhat = val_npz["preds"]
        basic = compute_basic_metrics(y_val, yhat)
        var_across_dataset = float(np.var(yhat))
        bias_mean = float(np.mean(yhat - y_val))
        summary["val"] = {"basic": basic, "bias_mean": bias_mean, "var_across_dataset": var_across_dataset}
        preds_round = nearest_quarter(yhat)
        summary["val"]["classification_report"] = classification_report([str(x) for x in y_val], [str(x) for x in preds_round], zero_division=0, digits=4)
        np.savez(out_dir / "val_detailed.npz", preds=yhat, labels=y_val, preds_round=preds_round, paths=val_npz.get("paths", None))
    if train_npz:
        y_tr = train_npz["labels"]
        yhat_tr = train_npz["preds"]
        basic = compute_basic_metrics(y_tr, yhat_tr)
        var_across_dataset = float(np.var(yhat_tr))
        bias_mean = float(np.mean(yhat_tr - y_tr))
        summary["train"] = {"basic": basic, "bias_mean": bias_mean, "var_across_dataset": var_across_dataset}
        np.savez(out_dir / "train_detailed.npz", preds=yhat_tr, labels=y_tr, paths=train_npz.get("paths", None))

    if test_npz:
        y_ts = test_npz["labels"]
        yhat_ts = test_npz["preds"]
        basic = compute_basic_metrics(y_ts, yhat_ts)
        bias_mean = float(np.mean(yhat_ts - y_ts))
        var_across_dataset = float(np.var(yhat_ts))
        summary["test"] = {"basic": basic, "bias_mean": bias_mean, "var_across_dataset": var_across_dataset}
        np.savez(out_dir / "test_detailed.npz", preds=yhat_ts, labels=y_ts, paths=test_npz.get("paths", None))

    # If user provided single checkpoint and requested MC dropout estimation
    if args.checkpoint and args.mc_dropout:
        try:
            import timm
            has_timm = True
        except Exception:
            has_timm = False
        model_name = args.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if has_timm:
            model = timm.create_model(model_name, pretrained=False, num_classes=1)
        else:
            from torchvision import models
            if "resnet" in model_name:
                m = models.resnet50(pretrained=False)
            else:
                m = models.resnet50(pretrained=False)
            in_f = m.fc.in_features
            m.fc = nn.Sequential(nn.Linear(in_f,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,1))
            model = m
        chk = torch.load(args.checkpoint, map_location=device)
        if isinstance(chk, dict) and ("model_state" in chk or "model_state_dict" in chk):
            if "model_state" in chk:
                model.load_state_dict(chk["model_state"])
            else:
                model.load_state_dict(chk["model_state_dict"])
        elif isinstance(chk, dict):
            # maybe the checkpoint is a dict with other keys but contains state_dict
            if "state_dict" in chk:
                model.load_state_dict(chk["state_dict"])
            else:
                # try direct load - may raise
                try:
                    model.load_state_dict(chk)
                except Exception as e:
                    raise RuntimeError(f"Unable to load checkpoint state dict: {e}")
        else:
            model.load_state_dict(chk)
        model.to(device)
        model.train()  # enable dropout; WARNING: BatchNorm will also be in train mode

        # loader_fn uses the paths saved in the test/val npz we actually loaded (ensures same images)
        def loader_fn_all():
            from PIL import Image
            arrs = []
            img_paths = []
            # priority: test_npz.paths -> val_npz.paths
            if test_npz and test_npz.get("paths"):
                img_paths = test_npz["paths"]
            elif val_npz and val_npz.get("paths"):
                img_paths = val_npz["paths"]
            else:
                raise RuntimeError("MC Dropout requires saved 'paths' in preds_test.npz or preds_val.npz to reload images.")
            # convert to list of strings if not already
            img_paths = [str(p) for p in img_paths]
            tf = None
            try:
                from torchvision import transforms
                tf = transforms.Compose([transforms.Resize(int(args.img_size*1.05)), transforms.CenterCrop(args.img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            except Exception:
                tf = None
            for p in img_paths:
                img = Image.open(p).convert("RGB")
                if tf:
                    x = tf(img).unsqueeze(0).numpy()
                else:
                    x = np.array(img).transpose(2,0,1).astype(np.float32)/255.0
                    x = x[np.newaxis,...]
                arrs.append(x)
            return arrs

        T = args.mc_T
        print(f"[MC Dropout] running T={T} forward passes (this may take a while)")
        preds_T_list = []
        with torch.no_grad():
            for t in range(T):
                batch_preds = []
                arrs = loader_fn_all()
                for x in arrs:
                    x_tensor = torch.from_numpy(x).to(device)
                    out = model(x_tensor).cpu().numpy().reshape(-1)
                    batch_preds.append(out)
                batch_preds = np.concatenate(batch_preds, axis=0)
                preds_T_list.append(batch_preds)
        preds_T = np.stack(preds_T_list, axis=0)
        mean_pred = preds_T.mean(axis=0)
        var_pred = preds_T.var(axis=0, ddof=0)
        labels = None
        if test_npz and test_npz.get("labels") is not None:
            labels = test_npz["labels"]
        elif val_npz and val_npz.get("labels") is not None:
            labels = val_npz["labels"]
        if labels is not None:
            emb = compute_bias_variance_from_ensemble(preds_T, labels)
            summary["mc_dropout"] = {"bias_abs_mean": emb["bias_abs_mean"], "var_mean": emb["var_mean"], "mean_pred_basic": compute_basic_metrics(labels, emb["mean_pred"])}
            np.savez(out_dir / "mc_dropout_preds.npz", preds_T=preds_T, mean_pred=mean_pred, var_pred=var_pred, labels=labels)
        else:
            summary["mc_dropout"] = {"note": "no labels found to compute bias; saved preds_T only"}
            np.savez(out_dir / "mc_dropout_preds.npz", preds_T=preds_T, mean_pred=mean_pred, var_pred=var_pred)

    # Save summary json
    with open(out_dir / "detailed_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", out_dir / "detailed_metrics_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str,
                   default=r"C:\Users\Artur\OneDrive\Documents\MuuMetrics Pytorch\outputs_ft",
                   help="Path to the folder containing preds_*.npz files")
    p.add_argument("--out_dir", type=str,
                   default=r"C:\Users\Artur\OneDrive\Documents\MuuMetrics Pytorch\diag_out",
                   help="Directory to save diagnostic results")
    args = p.parse_args()

    # Auto-locate preds files in run_dir
    from pathlib import Path
    run_dir = Path(args.run_dir)
    val_npz = run_dir / "preds_val.npz"
    test_npz = run_dir / "preds_test.npz"
    train_npz = run_dir / "preds_train.npz"
    # Build fake argparse namespace expected by main()
    class A: pass
    a = A()
    a.train_npz = str(train_npz) if train_npz.exists() else None
    a.val_npz = str(val_npz) if val_npz.exists() else None
    a.test_npz = str(test_npz) if test_npz.exists() else None
    a.ensemble_files = None
    a.checkpoint = None
    a.model = "convnext_base"
    a.mc_dropout = False
    a.mc_T = 30
    a.img_size = 384
    a.out_dir = args.out_dir
    main(a)

