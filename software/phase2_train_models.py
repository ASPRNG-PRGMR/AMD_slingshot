"""
HeteroWise — Phase 2: ML Prediction Engine
Trains RandomForest regressors to predict energy consumption on CPU/GPU/NPU.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATASET ──────────────────────────────────────────────────────────────
df = pd.read_csv('/home/noobiegg/Documents/AMD_shit/workload_dataset.csv')
print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns\n")

# ── FEATURES & TARGETS ────────────────────────────────────────────────────────
FEATURES = ['flops', 'batch_size', 'precision_enc', 'param_count']
TARGETS   = {
    'cpu': 'cpu_energy_j',
    'gpu': 'gpu_energy_j',
    'npu': 'npu_energy_j',
}

X = df[FEATURES].copy()
# Log-scale FLOPs and params for better ML learning
X['log_flops']  = np.log10(X['flops'])
X['log_params'] = np.log10(X['param_count'])
X['log_batch']  = np.log2(X['batch_size'])
X_final_features = ['log_flops', 'log_batch', 'precision_enc', 'log_params']
X = X[X_final_features]

# ── TRAIN/TEST SPLIT ──────────────────────────────────────────────────────────
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

models = {}
results = {}
print("=" * 60)
print("TRAINING REGRESSORS")
print("=" * 60)

for name, target_col in TARGETS.items():
    y = df[target_col]
    y_log = np.log1p(y)  # log-transform for skewed targets

    y_train = y_log.iloc[X_train.index]
    y_test  = y_log.iloc[X_test.index]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Back-transform for real-scale metrics
    mse     = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
    rmse    = np.sqrt(mse)
    r2      = r2_score(y_test, y_pred)

    models[name] = model
    results[name] = {'RMSE': rmse, 'R2': r2}

    print(f"\n{name.upper()} Energy Predictor:")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f} J")

    # Feature importances
    fi = pd.Series(model.feature_importances_, index=X_final_features).sort_values(ascending=False)
    print(f"  Feature importance:")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 30)
        print(f"    {feat:<15} {bar}  {imp:.3f}")

# ── SAVE MODELS ───────────────────────────────────────────────────────────────
from pathlib import Path
import pickle

# Create project directory inside user's home
base_dir = Path.home() / "HeteroWise" / "models"
base_dir.mkdir(parents=True, exist_ok=True)

# Save trained models
for name, model in models.items():
    with open(base_dir / f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Also save feature list for consistent prediction
model_meta = {
    'features': X_final_features,
    'targets': TARGETS,
}

with open(base_dir / "model_meta.pkl", "wb") as f:
    pickle.dump(model_meta, f)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

for name, res in results.items():
    print(f"  {name.upper():5s}  R²={res['R2']:.4f}  RMSE={res['RMSE']:.4f} J")

print(f"\n✅ Models saved to: {base_dir}")

# ── QUICK SANITY CHECK ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY CHECK — Predict a few samples")
print("=" * 60)

test_cases = [
    {'name': 'Tiny inference (1B FLOPs, batch=1, INT8)',
     'flops':1e9, 'batch':1, 'prec':0.25, 'params':5e5},
    {'name': 'Medium model (50B FLOPs, batch=16, FP16)',
     'flops':5e10, 'batch':16, 'prec':0.5, 'params':2e8},
    {'name': 'Large training (500B FLOPs, batch=128, FP32)',
     'flops':5e11, 'batch':128, 'prec':1.0, 'params':1e9},
]

for tc in test_cases:
    x = pd.DataFrame([[
        np.log10(tc['flops']),
        np.log2(tc['batch']),
        tc['prec'],
        np.log10(tc['params'])
    ]], columns=X_final_features)

    preds = {}
    for name, model in models.items():
        preds[name] = float(np.expm1(model.predict(x)[0]))

    best = min(preds, key=preds.get)
    print(f"\n{tc['name']}")
    for hw, e in preds.items():
        marker = " ◄ BEST" if hw == best else ""
        print(f"  {hw.upper():5s}: {e:.4f} J{marker}")
