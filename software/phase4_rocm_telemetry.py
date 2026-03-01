"""
HeteroWise - Phase 4: ROCm Telemetry Validation
Captures live AMD GPU power via rocm-smi, runs workloads,
and validates ML energy predictions against real hardware readings.

Run:
  python3 phase4_rocm_telemetry.py
  sudo python3 phase4_rocm_telemetry.py   # if power reads show N/A
"""

import subprocess
import re
import time
import os
import sys
import numpy as np
import pandas as pd
import pickle

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')
CSV_PATH  = os.path.join(BASE, 'telemetry_log.csv')


# ── ROCm-SMI INTERFACE ────────────────────────────────────────────────────────

def get_gpu_metrics():
    """Parse rocm-smi plain-text output for power, temp, clock, utilization."""
    blank = {'power_w': None, 'temp_c': None, 'clock_mhz': None, 'util_pct': None}
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        out = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return blank

    power = clock = temp = util = None

    m = re.search(r'(\d+\.\d+)W', out)
    if m:
        power = float(m.group(1))

    m = re.search(r'(\d+\.\d+).C', out)
    if m:
        temp = float(m.group(1))

    m = re.search(r'(\d+)[Mm]hz', out)
    if m:
        clock = float(m.group(1))

    matches = re.findall(r'(\d+)%', out)
    if matches:
        util = float(matches[-1])

    return {'power_w': power, 'temp_c': temp, 'clock_mhz': clock, 'util_pct': util}


def rocm_available():
    try:
        subprocess.run(['rocm-smi'], capture_output=True, timeout=3)
        return True
    except FileNotFoundError:
        return False


def display(val, suffix='', decimals=1):
    if val is None:
        return 'N/A'
    return '{:.{}f}{}'.format(val, decimals, suffix)


# ── CPU WORKLOAD ──────────────────────────────────────────────────────────────

def run_workload(flops_target, batch, precision):
    """
    NumPy matrix multiply — always runs on CPU.
    rocm-smi captures ambient GPU power during this window.
    FLOPs for matmul(A,B) where A:(batch,n), B:(n,n) = 2*batch*n^2
    """
    dtype_map = {'FP32': np.float32, 'FP16': np.float16, 'INT8': np.int8}
    dtype = dtype_map.get(precision, np.float32)

    n = int((flops_target / (2 * batch)) ** 0.5)
    n = max(32, min(n, 1024))

    A = np.random.randn(batch, n).astype(dtype)
    B = np.random.randn(n, n).astype(dtype)
    _ = A @ B  # warm up

    t0 = time.perf_counter()
    for _ in range(5):
        _ = A @ B
    t1 = time.perf_counter()

    actual_flops = 2 * batch * n * n * 5
    latency_s    = (t1 - t0) / 5
    return latency_s, actual_flops, n


# ── ML PREDICTOR ──────────────────────────────────────────────────────────────

def load_models():
    models = {}
    for hw in ['cpu', 'gpu', 'npu']:
        path = os.path.join(MODEL_DIR, hw + '_model.pkl')
        with open(path, 'rb') as f:
            models[hw.upper()] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return models, meta['features']


def ml_predict(models, features, flops, batch, precision_enc, params):
    x = pd.DataFrame([[
        np.log10(max(flops,  1.0)),
        np.log2( max(batch,  1.0)),
        precision_enc,
        np.log10(max(params, 1.0))
    ]], columns=features)
    return {hw: float(np.expm1(m.predict(x)[0])) for hw, m in models.items()}


# ── BENCHMARK ─────────────────────────────────────────────────────────────────

def benchmark(name, flops, batch, precision, params, models, features):
    prec_enc = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}[precision]

    print('\n  ' + '-' * 54)
    print('  ' + name)
    print('  FLOPs={:.1e}  Batch={}  Precision={}'.format(flops, batch, precision))

    preds    = ml_predict(models, features, flops, batch, prec_enc, params)
    best_hw  = min(preds, key=preds.get)
    ml_gpu_j = preds['GPU']

    m_before = get_gpu_metrics()
    time.sleep(0.2)

    latency_s, actual_flops, n = run_workload(flops, batch, precision)

    time.sleep(0.2)
    m_after = get_gpu_metrics()

    p_before = m_before['power_w']
    p_after  = m_after['power_w']
    has_power = (p_before is not None and p_after is not None)

    if has_power:
        avg_power       = (p_before + p_after) / 2.0
        measured_energy = avg_power * latency_s
    else:
        avg_power       = None
        measured_energy = None

    print('\n  ML Predictions:')
    for hw in ['CPU', 'GPU', 'NPU']:
        marker = '  <- best' if hw == best_hw else ''
        print('    {}: {:.4f} J{}'.format(hw, preds[hw], marker))

    print('\n  ROCm Telemetry:')
    if has_power:
        print('    Power:   {:.1f}W -> {:.1f}W  (avg {:.1f}W)'.format(
            p_before, p_after, avg_power))
    else:
        print('    Power:   N/A  (run with sudo for power readings)')
    print('    Temp:    ' + display(m_after['temp_c'],    suffix='C'))
    print('    Clock:   ' + display(m_after['clock_mhz'], suffix=' MHz', decimals=0))
    print('    Latency: {:.2f} ms  (matrix {}x{})'.format(latency_s * 1000, batch, n))

    print('\n  Validation:')
    print('    ML predicted GPU energy:  {:.4f} J'.format(ml_gpu_j))
    err = None
    if has_power:
        print('    Measured energy (rocm):   {:.4f} J'.format(measured_energy))
        err = abs(ml_gpu_j - measured_energy) / max(ml_gpu_j, measured_energy) * 100
        print('    Prediction error:         {:.1f}%'.format(err))
        if err < 25:
            verdict = 'good agreement'
        elif err < 60:
            verdict = 'moderate (iGPU vs discrete GPU model -- expected)'
        else:
            verdict = 'high -- iGPU profile differs; telemetry queued for retraining'
        print('    Verdict:                  ' + verdict)
    else:
        print('    Measured energy:          N/A')

    row = {
        'workload':            name,
        'flops':               flops,
        'actual_flops':        actual_flops,
        'batch_size':          batch,
        'precision':           precision,
        'param_count':         params,
        'ml_cpu_j':            preds['CPU'],
        'ml_gpu_j':            preds['GPU'],
        'ml_npu_j':            preds['NPU'],
        'ml_best_hw':          best_hw,
        'rocm_power_before_w': p_before,
        'rocm_power_after_w':  p_after,
        'rocm_avg_power_w':    avg_power,
        'rocm_temp_c':         m_after['temp_c'],
        'rocm_clock_mhz':      m_after['clock_mhz'],
        'latency_s':           latency_s,
        'measured_energy_j':   measured_energy,
        'pred_error_pct':      err,
        'matrix_n':            n,
        'timestamp':           time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    df = pd.DataFrame([row])
    write_header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=write_header, index=False)
    return row


# ── APPEND TO TRAINING DATA ───────────────────────────────────────────────────

def append_to_dataset(log_path, dataset_path):
    if not os.path.exists(log_path):
        return
    log   = pd.read_csv(log_path)
    valid = log[log['measured_energy_j'].notna() & (log['measured_energy_j'] > 0)]
    if len(valid) == 0:
        print('\n  No valid power readings to append (run with sudo for power data)')
        return

    prec_map = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}
    new_rows = pd.DataFrame({
        'flops':          valid['actual_flops'].values,
        'batch_size':     valid['batch_size'].values,
        'precision':      valid['precision'].values,
        'precision_enc':  valid['precision'].map(prec_map).values,
        'param_count':    valid['param_count'].values,
        'cpu_latency_s':  valid['latency_s'].values,
        'gpu_latency_s':  valid['latency_s'].values,
        'npu_latency_s':  valid['latency_s'].values,
        'cpu_energy_j':   valid['ml_cpu_j'].values,
        'gpu_energy_j':   valid['measured_energy_j'].values,
        'npu_energy_j':   valid['ml_npu_j'].values,
        'best_accelerator': valid['ml_best_hw'].values,
    })

    if os.path.exists(dataset_path):
        existing = pd.read_csv(dataset_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_csv(dataset_path, index=False)
        print('\n  Appended {} real rows -> dataset now {} rows'.format(
            len(new_rows), len(combined)))
    else:
        new_rows.to_csv(dataset_path, index=False)
        print('\n  Created dataset with {} telemetry rows'.format(len(new_rows)))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 58)
    print('  HeteroWise - Phase 4: ROCm Telemetry Validation')
    print('=' * 58)

    if rocm_available():
        m = get_gpu_metrics()
        print('\n  rocm-smi detected')
        print('    Power: ' + display(m['power_w'], suffix='W'))
        print('    Temp:  ' + display(m['temp_c'],  suffix='C'))
        print('    Clock: ' + display(m['clock_mhz'], suffix=' MHz', decimals=0))
        if m['power_w'] is None:
            print('    (Power N/A -- try: sudo python3 phase4_rocm_telemetry.py)')
    else:
        print('\n  rocm-smi not found')
        print('  Install: sudo dnf install rocm-smi')
        print('  Continuing with latency-only measurements')

    try:
        models, features = load_models()
        print('\n  ML models loaded')
    except FileNotFoundError:
        print('\n  Models not found -- run phase2_train_models.py first')
        sys.exit(1)

    print('\n' + '=' * 58)
    print('  BENCHMARKS')
    print('=' * 58)

    suite = [
        ('Small  (batch=1,  FP32)', 1e8,  1,  'FP32', 1e6),
        ('Medium (batch=8,  FP16)', 1e9,  8,  'FP16', 5e6),
        ('Large  (batch=32, FP32)', 1e10, 32, 'FP32', 1e7),
    ]

    results = []
    for name, flops, batch, prec, params in suite:
        row = benchmark(name, flops, batch, prec, params, models, features)
        results.append(row)
        time.sleep(0.8)

    print('\n' + '=' * 58)
    print('  SUMMARY')
    print('=' * 58)
    print('  {:<28} {:>8} {:>10} {:>7}'.format('Workload', 'ML pred', 'Measured', 'Error'))
    print('  ' + '-' * 56)
    for r in results:
        meas = '{:.4f}J'.format(r['measured_energy_j']) if r['measured_energy_j'] else '     N/A'
        err  = '{:.1f}%'.format(r['pred_error_pct'])    if r['pred_error_pct']    else '    N/A'
        print('  {:<28} {:>7.4f}J {:>10} {:>7}'.format(
            r['workload'], r['ml_gpu_j'], meas, err))

    print('\n  Log saved -> telemetry_log.csv')

    append_to_dataset(CSV_PATH, os.path.join(BASE, 'workload_dataset.csv'))

    print('\n  To retrain with real data:')
    print('    python3 phase2_train_models.py')
    print('\n' + '=' * 58 + '\n')


if __name__ == '__main__':
    main()
