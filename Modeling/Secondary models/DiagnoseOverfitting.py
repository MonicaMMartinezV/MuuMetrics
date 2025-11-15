#!/usr/bin/env python3
# diagnose_overfitting.py

# =============================================================
# Nombre del archivo: diagnose_overfitting.py
# Autores: Bárbara Paola Alcántara Vega
#           
# Descripción:
#Fine-tune ConvNeXt for regression (optimized for RTX 4090).
#Diagnóstico de overfitting / underfitting y calidad del modelo.
#Incluye:
# - Lectura de history.json y preds_*.npz
# - Cálculo de métricas MAE, RMSE, R2
# - Gráficas: train vs val MAE, histogramas y scatter true vs pred
# - Diagnóstico textual con recomendaciones
#
# Dependencias: os, numpy, pathlib, typing, PIL, torch, argparse,
#               json, matplotlib, sklearn, math, sys
#     
# =============================================================


import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import meanAbsoluteError, meanSquaredError, r2_score, classification_report, confusion_matrix
import math
import sys
import os
import argparse

def main(args):
    print(f"Run directory: {args.run_dir}")
    print(f"Output directory: {args.out_dir}")

def loadHistory(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def safeLoadNpz(path: Path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: np.array(v) for k, v in d.items()}

def computeRegMetrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    mae = meanAbsoluteError(y_true, y_pred)
    try:
        rmse = meanSquaredError(y_true, y_pred, squared=False)
    except TypeError:
        # versiones anteriores de sklearn no soportan `squared` kwarg
        rmse = math.sqrt(meanSquaredError(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def classificationFromReg(preds, labels):
    predsRounded = np.round(np.array(preds) / 0.25) * 0.25
    unique = sorted(np.unique(labels))
    return predsRounded, unique

def briefDiagnosis(trainMae_series, valMae_series, finalTrainMetrics=None, finalValMetrics=None):
    diagnosis = []
    if trainMae_series is None or valMae_series is None:
        diagnosis.append("No train/val MAE series available.")
        return diagnosis
    tr_last = trainMae_series[-1]
    val_last = valMae_series[-1]
    gap = val_last - tr_last
    diagnosis.append(f"Final train MAE: {tr_last:.4f}, val MAE: {val_last:.4f}, gap={gap:.4f}")
    if gap > 0.05 and gap / max(tr_last, 1e-8) > 0.2:
        diagnosis.append("Fuerte overfitting: val MAE mucho mayor que train MAE.")
    elif gap > 0.02:
        diagnosis.append("Leve overfitting detectado.")
    elif gap < -0.02:
        diagnosis.append("Posible data leakage: val MAE menor que train MAE.")
    else:
        diagnosis.append("Sin señales claras de overfitting.")
    return diagnosis

def saveTextReport(out_path: Path, lines):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            # Reemplaza los caracteres que no son codificables por cp1252 con '?'
            safe_ln = ln.encode("utf-8", errors="replace").decode("utf-8")
            f.write(safe_ln + "\n")


def plotScatter(y_true, y_pred, out_path):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', label="Ideal (y=x)")
    r2 = r2_score(y_true, y_pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Scatter True vs Pred (R²={r2:.3f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {out_path}")

def main(args):
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = loadHistory(run_dir / "history.json")
    reportLines = []

    # Gráfica de entrenamiento
    if history:
        trainMae = history.get("trainMae") or history.get("train_loss")
        valMae = history.get("valMae") or history.get("val_loss")
        if trainMae and valMae:
            plt.figure(figsize=(8,5))
            plt.plot(trainMae, label="train")
            plt.plot(valMae, label="val")
            plt.xlabel("Epoch"); plt.ylabel("MAE/Loss")
            plt.legend(); plt.grid(True)
            plt.title("Train vs Val Curve")
            p = out_dir / "train_val_curve.png"
            plt.savefig(p, bbox_inches="tight"); plt.close()
            print(f"Saved learning curve to {p}")

    # Carga preds
    predsVal = safeLoadNpz(run_dir / "predsVal.npz")
    predsTrain = safeLoadNpz(run_dir / "predsTrain.npz")
    predsTest = safeLoadNpz(run_dir / "predsTest.npz")

    finalTrainMetrics = finalValMetrics = None

    if predsTrain and "preds" in predsTrain:
        m = computeRegMetrics(predsTrain["labels"], predsTrain["preds"])
        reportLines.append(f"Train: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}")
        finalTrainMetrics = m

    if predsVal and "preds" in predsVal:
        m = computeRegMetrics(predsVal["labels"], predsVal["preds"])
        reportLines.append(f"Val:   MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}")
        finalValMetrics = m

        predsRound, unique = classificationFromReg(predsVal["preds"], predsVal["labels"])
        cr = classification_report([str(x) for x in predsVal["labels"]],
                                   [str(x) for x in predsRound],
                                   digits=4, zero_division=0)
        reportLines.append("Classification report (val, nearest class):")
        reportLines.append(cr)
        cm = confusion_matrix([str(x) for x in predsVal["labels"]],
                              [str(x) for x in predsRound],
                              labels=[str(u) for u in unique])
        reportLines.append("Confusion matrix (rows=true, cols=pred):")
        reportLines.append(np.array2string(cm, separator=", "))

        # scatter true vs pred
        plotScatter(predsVal["labels"], predsVal["preds"], out_dir / "scatter_true_vs_pred.png")

    # Diagnóstico
    if history and "trainMae" in history and "valMae" in history:
        diag = briefDiagnosis(history["trainMae"], history["valMae"],
                               finalTrainMetrics, finalValMetrics)
        reportLines.append("=== DIAGNOSIS ===")
        reportLines.extend(diag)

    saveTextReport(out_dir / "overfitting_report.txt", reportLines)
    print("\n".join(reportLines))
    print(f"Saved report to {out_dir / 'overfitting_report.txt'}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir",
        type=str,
        default=r"C:\Users\Artur\OneDrive\Documents\MuuMetrics Pytorch\outputs_ft",
        help="Path to the training run directory"
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=r"C:\Users\Artur\OneDrive\Documents\MuuMetrics Pytorch\diag_out2",
        help="Directory to save diagnostic results"
    )
    args = p.parse_args()

    # Se crea automáticamente el output directory si es que no existe
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)