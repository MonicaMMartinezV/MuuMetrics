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
#Salida: imprime resumen y guarda npz/report en outDir
#
# Dependencias: argparse, pathlib, numpy, math, sklearn, json,
#               torch, glob, os, typing
#     
# =============================================================


import argparse
from pathlib import Path
import numpy as np
import math
from sklearn.metrics import meanAbsoluteError, meanSquaredError, r2Score, classification_report
import json
import torch
import torch.nn as nn
import glob
import os
from typing import Optional

def findNpzInRun(runDir: str, split: str) -> Optional[Path]:
    """
    Buscar recursivamente por preds_{split}.npz ó *preds*_{slpit}.npz bajo runDir
    Devuelve el archivo más reciente por modificación, tiempo o None
    """
    patterns = [
        os.path.join(runDir, f"**/preds_{split}.npz"),
        os.path.join(runDir, f"**/*preds*_{split}.npz"),
        os.path.join(runDir, f"preds_{split}.npz"),
        os.path.join(runDir, f"*preds*_{split}.npz"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p, recursive=True))
    matches = sorted(set(matches))
    if not matches:
        return None
    # escoger el más nuevo por tiempo
    newest = max(matches, key=lambda p: os.path.getmtime(p))
    return Path(newest)

def loadNpzIfExists(p: Path):
    if p is None:
        return None
    if not Path(p).exists():
        return None
    d = np.load(p, allow_pickle=True)
    preds = np.array(d["preds"])
    labels = np.array(d["labels"]) if "labels" in d else None
    paths = None
    if "paths" in d:
        # convertir paths a strings 
        raw_paths = np.array(d["paths"])
        # a veces se salvan como bytes
        paths = [str(x) for x in raw_paths]
    return {"preds": preds, "labels": labels, "paths": paths, "npz_path": str(p)}

def nearestQuarter(arr):
    return np.round(np.array(arr) / 0.25) * 0.25

def computeBasicMetrics(yTrue, y_pred):
    mae = meanAbsoluteError(yTrue, y_pred)
    mse = meanSquaredError(yTrue, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2Score(yTrue, y_pred)
    exact_acc = (nearestQuarter(y_pred) == yTrue).mean()
    return {"mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2), "exact_acc": float(exact_acc)}

def computeBiasVarianceFromEnsemble(predsEnsemble, yTrue):
    meanPred = predsEnsemble.mean(axis=0)
    varPred = predsEnsemble.var(axis=0, ddof=0)
    bias = meanPred - yTrue
    biasAbsMean = np.mean(np.abs(bias))
    varMean = np.mean(varPred)
    return {"bias_per_sample": bias, "var_per_sample": varPred, "biasAbsMean": float(biasAbsMean), "varMean": float(varMean), "meanPred": meanPred}

def safePrintDict(d: dict):
    for k,v in d.items():
        print(f"{k}: {v}")

def main(args):
    outDir = Path(args.outDir)
    outDir.mkdir(parents=True, exist_ok=True)

    # si runDir da un npz, los argumentos son default/absent, intentar localizar el npz propio dentro de runDir
    if args.outDir:
        # solo hacer override si el usuario no ingresa un path específico (si todos siguen siendo el default o están vacíos)
        if not args.trainNpz or args.trainNpz.strip() == "":
            found = findNpzInRun(args.outDir, "train")
            if found:
                print(f"[auto] found train npz in runDir: {found}")
                args.trainNpz = str(found)
        if not args.valNpz or args.valNpz.strip() == "":
            found = findNpzInRun(args.outDir, "val")
            if found:
                print(f"[auto] found val npz in runDir: {found}")
                args.valNpz = str(found)
        if not args.testNpz or args.testNpz.strip() == "":
            found = findNpzInRun(args.outDir, "test")
            if found:
                print(f"[auto] found test npz in runDir: {found}")
                args.testNpz = str(found)

    # Intentar cargar los archivos de preds primero
    trainNpz = loadNpzIfExists(Path(args.trainNpz)) if args.trainNpz else None
    valNpz = loadNpzIfExists(Path(args.valNpz)) if args.valNpz else None
    testNpz = loadNpzIfExists(Path(args.testNpz)) if args.testNpz else None

    # mostrar lo que se cargó (por si aparece un wrong-run)
    if trainNpz:
        print(f"Loaded train npz from: {trainNpz.get('npz_path')}; samples={len(trainNpz['preds'])}")
    if valNpz:
        print(f"Loaded val npz from: {valNpz.get('npz_path')}; samples={len(valNpz['preds'])}")
    if testNpz:
        print(f"Loaded test npz from: {testNpz.get('npz_path')}; samples={len(testNpz['preds'])}")

    summary = {}

    # Si los archivos de ensemble (varios npz) fueron provistos, construye predsEnsemble para test/val
    if args.ensemble_files:
        ensPredsList = []
        labelsRef = None
        for p in args.ensemble_files:
            d = loadNpzIfExists(Path(p))
            if d is None:
                raise RuntimeError(f"Ensemble file not found: {p}")
            ensPredsList.append(d["preds"])
            if labelsRef is None:
                labelsRef = d["labels"]
            else:
                if not np.array_equal(labelsRef, d["labels"]):
                    raise RuntimeError("Ensemble files have differing labels arrays - align before using ensemble_files.")
        predsEnsemble = np.stack(ensPredsList, axis=0)
        emb = computeBiasVarianceFromEnsemble(predsEnsemble, labelsRef)
        basic = computeBasicMetrics(labelsRef, emb["meanPred"])
        summary["ensemble"] = {"basic": basic, "biasAbsMean": emb["biasAbsMean"], "varMean": emb["varMean"]}
        np.savez(outDir / "ensemble_bias_var.npz", bias=emb["bias_per_sample"], var=emb["var_per_sample"], meanPred=emb["meanPred"], labels=labelsRef)

    # si preds_val existe, calcular las métricas y el bias/var fallback
    if valNpz:
        yVal = valNpz["labels"]
        yhat = valNpz["preds"]
        basic = computeBasicMetrics(yVal, yhat)
        varAcrossDataset = float(np.var(yhat))
        biasMean = float(np.mean(yhat - yVal))
        summary["val"] = {"basic": basic, "biasMean": biasMean, "varAcrossDataset": varAcrossDataset}
        predsRound = nearestQuarter(yhat)
        summary["val"]["classification_report"] = classification_report([str(x) for x in yVal], [str(x) for x in predsRound], zero_division=0, digits=4)
        np.savez(outDir / "val_detailed.npz", preds=yhat, labels=yVal, predsRound=predsRound, paths=valNpz.get("paths", None))
    if trainNpz:
        yTr = trainNpz["labels"]
        yhatTr = trainNpz["preds"]
        basic = computeBasicMetrics(yTr, yhatTr)
        varAcrossDataset = float(np.var(yhatTr))
        biasMean = float(np.mean(yhatTr - yTr))
        summary["train"] = {"basic": basic, "biasMean": biasMean, "varAcrossDataset": varAcrossDataset}
        np.savez(outDir / "train_detailed.npz", preds=yhatTr, labels=yTr, paths=trainNpz.get("paths", None))

    if testNpz:
        yTs = testNpz["labels"]
        yhatTs = testNpz["preds"]
        basic = computeBasicMetrics(yTs, yhatTs)
        biasMean = float(np.mean(yhatTs - yTs))
        varAcrossDataset = float(np.var(yhatTs))
        summary["test"] = {"basic": basic, "biasMean": biasMean, "varAcrossDataset": varAcrossDataset}
        np.savez(outDir / "test_detailed.npz", preds=yhatTs, labels=yTs, paths=testNpz.get("paths", None))

    # Solo Si se incluyó el single checkpoint y se pidió la estimación del MC dropout
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
            # En caso de que el checkpoint se haya puesto como dict con otras llaves pero coniene state_dict
            if "state_dict" in chk:
                model.load_state_dict(chk["state_dict"])
            else:
                # Intentar una carga directa
                try:
                    model.load_state_dict(chk)
                except Exception as e:
                    raise RuntimeError(f"Unable to load checkpoint state dict: {e}")
        else:
            model.load_state_dict(chk)
        model.to(device)
        model.train()  # enable dropout; PERO, CUIDADO: BatchNorm TAMBIÉN ESTARÁ EN MODO ENTRENAMIENTO

        # loader_fn usa los paths guardados en el test/val npz que se cargan (para asegurarnos que sean las mismas imágenes)
        def loaderFnAll():
            from PIL import Image
            arrs = []
            imgPaths = []
            # prioridad: testNpz.paths -> valNpz.paths
            if testNpz and testNpz.get("paths"):
                imgPaths = testNpz["paths"]
            elif valNpz and valNpz.get("paths"):
                imgPaths = valNpz["paths"]
            else:
                raise RuntimeError("MC Dropout requires saved 'paths' in predsTest.npz or preds_val.npz to reload images.")
            # convertir la lista de cadenas si es que no lo estaban ya
            imgPaths = [str(p) for p in imgPaths]
            tf = None
            try:
                from torchvision import transforms
                tf = transforms.Compose([transforms.Resize(int(args.img_size*1.05)), transforms.CenterCrop(args.img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            except Exception:
                tf = None
            for p in imgPaths:
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
        predsT_list = []
        with torch.no_grad():
            for t in range(T):
                batch_preds = []
                arrs = loaderFnAll()
                for x in arrs:
                    x_tensor = torch.from_numpy(x).to(device)
                    out = model(x_tensor).cpu().numpy().reshape(-1)
                    batch_preds.append(out)
                batch_preds = np.concatenate(batch_preds, axis=0)
                predsT_list.append(batch_preds)
        predsT = np.stack(predsT_list, axis=0)
        meanPred = predsT.mean(axis=0)
        varPred = predsT.var(axis=0, ddof=0)
        labels = None
        if testNpz and testNpz.get("labels") is not None:
            labels = testNpz["labels"]
        elif valNpz and valNpz.get("labels") is not None:
            labels = valNpz["labels"]
        if labels is not None:
            emb = computeBiasVarianceFromEnsemble(predsT, labels)
            summary["mc_dropout"] = {"biasAbsMean": emb["biasAbsMean"], "varMean": emb["varMean"], "meanPred_basic": computeBasicMetrics(labels, emb["meanPred"])}
            np.savez(outDir / "mc_dropout_preds.npz", predsT=predsT, meanPred=meanPred, varPred=varPred, labels=labels)
        else:
            summary["mc_dropout"] = {"note": "no labels found to compute bias; saved predsT only"}
            np.savez(outDir / "mc_dropout_preds.npz", predsT=predsT, meanPred=meanPred, varPred=varPred)

    # guardar el resumen (summary) como json
    with open(outDir / "detailed_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", outDir / "detailed_metrics_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runDir", type=str,
                   default=r"C:\Users\Artur\OneDrive\Documents\MuuMetrics Pytorch\outputs_ft",
                   help="Path to the folder containing preds_*.npz files")
    p.add_argument("--outDir", type=str,
                   default=r"C:\Users\Artur\OneDrive\Documents\MuuMetrics Pytorch\diag_out",
                   help="Directory to save diagnostic results")
    args = p.parse_args()

    # Auto-localizar archivos de preds en runDir
    from pathlib import Path
    runDir = Path(args.runDir)
    valNpz = runDir / "preds_val.npz"
    testNpz = runDir / "predsTest.npz"
    trainNpz = runDir / "predsTrain.npz"
    # Crea un argparse namespace "falso" porque es lo que espera el main()
    class A: pass
    a = A()
    a.trainNpz = str(trainNpz) if trainNpz.exists() else None
    a.valNpz = str(valNpz) if valNpz.exists() else None
    a.testNpz = str(testNpz) if testNpz.exists() else None
    a.ensemble_files = None
    a.checkpoint = None
    a.model = "convnext_base"
    a.mc_dropout = False
    a.mc_T = 30
    a.img_size = 384
    a.outDir = args.outDir
    main(a)