"""
HeteroWise — Phase 1: Synthetic System Modeling
Balanced CPU/GPU/NPU dataset with realistic energy competition.

Decision regions (by design):
  CPU wins:  FLOPs < ~1B  AND  batch <= 4   (startup overhead dominates GPU/NPU)
  NPU wins:  FLOPs 1B–50B AND  batch <= 32  AND  precision INT8/FP16
  GPU wins:  FLOPs > 50B  OR   batch > 64   (massive parallelism pays off)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

# ── INPUT FEATURES ──────────────────────────────────────────────────────────
flops    = np.exp(np.random.uniform(np.log(1e6), np.log(5e11), N))
batch    = np.random.choice([1,2,4,8,16,32,64,128,256], N)
prec_lbl = np.random.choice(['FP32','FP16','INT8'], N, p=[0.4,0.4,0.2])
prec_enc = np.where(prec_lbl=='FP32', 1.0, np.where(prec_lbl=='FP16', 0.5, 0.25))
params   = flops * np.random.uniform(5e-4, 5e-3, N)

def noise(n, std=0.08):
    return np.clip(np.random.normal(1.0, std, n), 0.75, 1.35)

# ── LATENCY (seconds) ──────────────────────────────────────────────────────
# CPU: no startup, good at sequential small work
cpu_lat = (flops * prec_enc) / (4e10 * (1 + 0.2*np.log2(batch+1)))
cpu_lat *= noise(N)

# GPU: excellent throughput but 5ms fixed startup; needs large work to amortize
GPU_STARTUP = 0.005
gpu_lat = (flops * prec_enc) / (1e13 * (1 + 1.5*np.log2(batch+1))) + GPU_STARTUP
gpu_lat *= noise(N)

# NPU: very efficient for INT8/FP16 medium workloads, but saturates at huge scale
NPU_STARTUP    = 0.0003
npu_prec_pen   = np.where(prec_lbl=='FP32', 3.0, np.where(prec_lbl=='FP16', 1.0, 0.6))
npu_sat        = np.maximum(1.0, (flops / 2e10)**0.4)   # saturates above 20B FLOPs
npu_lat = (flops * prec_enc) / (2e12) * npu_prec_pen * npu_sat + NPU_STARTUP
npu_lat *= noise(N)

# ── ENERGY (Joules = latency × TDP) ────────────────────────────────────────
cpu_energy = cpu_lat * 45.0  * noise(N, 0.05)
gpu_energy = gpu_lat * 200.0 * noise(N, 0.05)
npu_energy = npu_lat * 8.0   * noise(N, 0.05)

# ── DATAFRAME ──────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'flops':         flops,
    'batch_size':    batch.astype(float),
    'precision':     prec_lbl,
    'precision_enc': prec_enc,
    'param_count':   params,
    'cpu_latency_s': cpu_lat,
    'gpu_latency_s': gpu_lat,
    'npu_latency_s': npu_lat,
    'cpu_energy_j':  cpu_energy,
    'gpu_energy_j':  gpu_energy,
    'npu_energy_j':  npu_energy,
})

df['best_accelerator'] = df[['cpu_energy_j','gpu_energy_j','npu_energy_j']].idxmin(axis=1)\
    .str.replace('_energy_j','').str.upper()

df.to_csv('/home/noobiegg/Documents/AMD_shit/workload_dataset.csv', index=False)
print(f"✅ Dataset saved: {len(df)} samples\n")
print("Best accelerator distribution:")
print(df['best_accelerator'].value_counts())
print(f"\nEnergy stats (Joules):")
print(df[['cpu_energy_j','gpu_energy_j','npu_energy_j']].describe().round(4))

# Show some CPU-winning rows
cpu_rows = df[df['best_accelerator']=='CPU'].head(3)
print(f"\nSample CPU-winning rows:")
print(cpu_rows[['flops','batch_size','precision','cpu_energy_j','gpu_energy_j','npu_energy_j']].round(4))
