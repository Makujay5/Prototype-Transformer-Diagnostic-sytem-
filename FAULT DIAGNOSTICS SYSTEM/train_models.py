"""
=============================================================================
  SMART FAULT DIAGNOSTIC SYSTEM — PROTOTYPE
  Module 2: ML Training Pipeline
=============================================================================
  Prototype: 1 kVA single-phase transformer, 240V→12V
  Sensors  : ACS712-20A, ZMPT101B, DS18B20 (×2), HC-SR04
  Classes  : 5 (Normal, Overheating, Low Oil, Overcurrent/Short, Voltage Fault)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

from data_generator import (
    generate_prototype_data, load_and_preprocess,
    FAULT_LABELS, N_CLASSES,
    ACS712_MAX_A, ZMPT101B_MAX_V, PROTO_RATED_I, PROTO_RATED_V
)

warnings.filterwarnings("ignore")
SEED = 42

MODEL_DIR = "./models"
PLOT_DIR  = "./plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

CLASS_NAMES = list(FAULT_LABELS.values())
PALETTE     = ["#2ECC71","#E74C3C","#F39C12","#9B59B6","#3498DB"]


# ── Confusion Matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax, annot_kws={"size":10})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"cm_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [Plot] Confusion matrix → {path}")


# ── Feature Importance ────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(range(len(imp)), imp[idx], color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importances — {model_name}\n(Prototype: 1kVA, 240V→12V)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"feat_imp_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [Plot] Feature importance → {path}")


# ── Model Comparison ──────────────────────────────────────────────────────────
def plot_model_comparison(results):
    names  = list(results.keys())
    acc    = [results[n]["accuracy"]  for n in names]
    f1_mac = [results[n]["f1_macro"]  for n in names]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    b1 = ax.bar(x-w/2, acc,    w, label="Accuracy",   color="#2196F3", edgecolor="white")
    b2 = ax.bar(x+w/2, f1_mac, w, label="F1 (macro)", color="#4CAF50", edgecolor="white")
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.80, 1.01); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Prototype Fault Classifier\n"
                 "(1 kVA Single-Phase Transformer | ACS712 + ZMPT101B)",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Model comparison → {path}")


# ── ROC Curves ────────────────────────────────────────────────────────────────
def plot_roc(model, X_test, y_test, model_name):
    y_bin   = label_binarize(y_test, classes=list(range(N_CLASSES)))
    y_score = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_score[:,i])
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i],
                label=f"{CLASS_NAMES[i]}  (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate",fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name}\n(Prototype Transformer)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right",fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"roc_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [Plot] ROC curves → {path}")


# ── Sensor Distributions ──────────────────────────────────────────────────────
def plot_sensor_distributions(df):
    sensors = ["v_primary_V","v_secondary_V","i_load_A",
               "temp_winding_C","temp_oil_C","oil_level_pct",
               "v_ratio_dev_pct","load_pct"]
    sensor_labels = [
        "Primary Voltage (V)\n[ZMPT101B — 0–250V]",
        "Secondary Voltage (V)\n[ZMPT101B — 0–25V]",
        "Load Current (A)\n[ACS712 — 0–20A]",
        "Winding Temp (°C)\n[DS18B20]",
        "Oil Temp (°C)\n[DS18B20]",
        "Oil Level (%)\n[HC-SR04]",
        "Voltage Ratio Dev (%)",
        "Load % of ACS712 Max",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, (sensor, label) in enumerate(zip(sensors, sensor_labels)):
        for j, (lbl, name) in enumerate(FAULT_LABELS.items()):
            axes[i].hist(df[df["fault_label"]==lbl][sensor],
                         bins=30, alpha=0.55, color=PALETTE[j],
                         label=name, edgecolor="none")
        axes[i].set_title(label, fontsize=8, fontweight="bold")
        axes[i].set_xlabel("Value", fontsize=7)
        axes[i].tick_params(labelsize=7)
        if i == 0:
            axes[i].legend(fontsize=6)
    fig.suptitle("Prototype Sensor Distributions by Fault Class\n"
                 "1 kVA Transformer | ACS712-20A | ZMPT101B | DS18B20 | HC-SR04",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "sensor_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot] Sensor distributions → {path}")


# ── Evaluate model ────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    print(f"\n{'='*58}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*58}")
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 (macro) : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), model_name)
    plot_roc(model, X_test, y_test, model_name)
    if feature_names and hasattr(model, "feature_importances_"):
        plot_feature_importance(model, feature_names, model_name)
    return {"accuracy": acc, "f1_macro": f1}


# ── Main ──────────────────────────────────────────────────────────────────────
def train_all_models():
    print("\n" + "="*58)
    print("  PROTOTYPE TRANSFORMER FAULT DIAGNOSTIC SYSTEM")
    print("  1 kVA | 240V→12V | ACS712-20A | ZMPT101B")
    print("="*58)

    print("\n[Step 1] Generating prototype dataset ...")
    df = generate_prototype_data(n_samples_per_class=1000,
                                  save_csv=True, output_dir=".")

    print("\n[Step 2] Preprocessing ...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, scaler) = load_and_preprocess(df)

    X_tr = np.vstack([X_train, X_val])
    y_tr = np.concatenate([y_train, y_val])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=4, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            decision_function_shape="ovr", probability=True,
            class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1,
            max_depth=4, subsample=0.85,
            random_state=SEED),
        "ANN (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=0.001,
            max_iter=400, early_stopping=True,
            validation_fraction=0.1, random_state=SEED),
    }

    results = {}
    trained = {}
    print("\n[Step 3] Training models ...")
    for name, clf in models.items():
        print(f"\n  ► {name} ...")
        clf.fit(X_tr, y_tr)
        trained[name] = clf
        results[name] = evaluate_model(clf, X_test, y_test, name,
            feature_names=(feature_names if hasattr(clf,"feature_importances_") else None))
        joblib.dump(clf, os.path.join(MODEL_DIR, f"{name.replace(' ','_')}.pkl"))

    plot_model_comparison(results)
    plot_sensor_distributions(df)

    best_name  = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = trained[best_name]
    print(f"\n{'*'*58}")
    print(f"  ★ Best Model : {best_name}")
    print(f"  Accuracy     : {results[best_name]['accuracy']:.4f}")
    print(f"  F1 (macro)   : {results[best_name]['f1_macro']:.4f}")
    print(f"{'*'*58}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    meta = {
        "best_model_name": best_name,
        "feature_names":   feature_names,
        "class_names":     CLASS_NAMES,
        "results":         {k:{kk:vv for kk,vv in v.items()} for k,v in results.items()},
        "prototype_specs": {
            "transformer":     "1 kVA single-phase step-down",
            "primary_V":       230,
            "secondary_V":     12,
            "rated_current_A": PROTO_RATED_I,
            "current_sensor":  "ACS712-20A (max 20A)",
            "voltage_sensor":  "ZMPT101B (max 250V)",
            "temp_sensor":     "DS18B20 (x2)",
            "oil_sensor":      "HC-SR04 ultrasonic",
        }
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "metadata.pkl"))
    print(f"\n  Models saved → {MODEL_DIR}/")
    return trained, results, feature_names, scaler, X_test, y_test


if __name__ == "__main__":
    train_all_models()
