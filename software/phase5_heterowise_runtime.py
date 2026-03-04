"""
HeteroWise - Phase 5: Runtime SDK
Deployment Orchestration Middleware

Turns HeteroWise from an advisory tool into callable middleware.
Developers import this and call optimize_and_run() or use the
EnergyBudgetOptimizer. The runtime handles everything internally:
  - workload profiling
  - accelerator selection
  - precision/batch tuning
  - telemetry capture
  - adaptive model updates

Usage:
    from heterowise_runtime import HeteroWiseRuntime, EnergyBudgetOptimizer

    hw = HeteroWiseRuntime()
    result = hw.optimize_and_run(my_fn, flops=1e9, batch=8, precision='FP16', params=1e7)

    # or with an energy budget:
    opt = EnergyBudgetOptimizer(max_energy_j=2.0)
    config = opt.solve(flops=5e10, params=1e8)
"""

import os
import re
import sys
import json
import time
import pickle
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Any

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')
LOG_PATH  = os.path.join(BASE, 'telemetry_log.csv')
SESSION_LOG = os.path.join(BASE, 'runtime_session.jsonl')

HW_TDP   = {'CPU': 45, 'GPU': 200, 'NPU': 8}
PREC_ENC = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}
PREC_LIST = ['FP32', 'FP16', 'INT8']

# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkloadSpec:
    """Everything the runtime needs to know about a workload."""
    flops:     float
    batch:     int
    precision: str    # 'FP32' | 'FP16' | 'INT8'
    params:    float
    task_type: str = 'inference'   # 'inference' | 'training'
    label:     str = 'unnamed'


@dataclass
class AcceleratorDecision:
    """The runtime's full decision for a workload."""
    recommended_hw:   str     # 'CPU' | 'GPU' | 'NPU'
    precision:        str
    batch:            int
    predicted_energy_j: float
    predicted_latency_ms: float
    efficiency_score: float   # higher = better perf/watt
    alternatives:     dict    # {hw: energy_j} for all three
    explanation:      str
    rocm_flags:       list


@dataclass
class ExecutionResult:
    """What comes back after optimize_and_run()."""
    decision:         AcceleratorDecision
    measured_energy_j: Optional[float]
    measured_latency_ms: float
    prediction_error_pct: Optional[float]
    return_value:     Any
    timestamp:        str


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────

def _rocm_power() -> Optional[float]:
    try:
        r = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        m = re.search(r'(\d+\.\d+)W', r.stdout)
        return float(m.group(1)) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _rocm_metrics() -> dict:
    blank = {'power_w': None, 'temp_c': None, 'clock_mhz': None}
    try:
        r = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        out = r.stdout
        power = clock = temp = None
        m = re.search(r'(\d+\.\d+)W', out)
        if m: power = float(m.group(1))
        m = re.search(r'(\d+\.\d+).C', out)
        if m: temp = float(m.group(1))
        m = re.search(r'(\d+)[Mm]hz', out)
        if m: clock = float(m.group(1))
        return {'power_w': power, 'temp_c': temp, 'clock_mhz': clock}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return blank


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LAYER
# ─────────────────────────────────────────────────────────────────────────────

class PredictionEngine:
    """Wraps the three trained RandomForest models."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.models = {}
        for hw in ['cpu', 'gpu', 'npu']:
            path = os.path.join(model_dir, f'{hw}_model.pkl')
            with open(path, 'rb') as f:
                self.models[hw.upper()] = pickle.load(f)
        with open(os.path.join(model_dir, 'model_meta.pkl'), 'rb') as f:
            self.features = pickle.load(f)['features']
        self._call_count = 0

    def predict(self, flops: float, batch: int, precision: str, params: float) -> dict:
        """Returns {CPU: joules, GPU: joules, NPU: joules}."""
        x = pd.DataFrame([[
            np.log10(max(flops,  1.0)),
            np.log2( max(batch,  1.0)),
            PREC_ENC.get(precision, 1.0),
            np.log10(max(params, 1.0))
        ]], columns=self.features)

        self._call_count += 1
        return {hw: float(np.expm1(m.predict(x)[0]))
                for hw, m in self.models.items()}

    def reload(self, model_dir: str = MODEL_DIR):
        """Hot-reload models after adaptive retraining."""
        self.__init__(model_dir)
        print('  [PredictionEngine] Models reloaded after adaptive update.')


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE MODEL REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveRefinementEngine:
    """
    Collects real telemetry measurements and periodically retrains
    the GPU energy predictor using incremental learning.

    Every N measured samples, it:
      1. Appends real rows to workload_dataset.csv
      2. Retrains just the GPU regressor (most hardware-specific)
      3. Saves the updated model
      4. Signals the PredictionEngine to reload

    This is the feedback loop that makes HeteroWise adaptive.
    """

    def __init__(self, retrain_every: int = 10):
        self.buffer       = []
        self.retrain_every = retrain_every
        self.retrain_count = 0

    def record(self, spec: WorkloadSpec, hw: str,
               measured_energy_j: float, latency_s: float):
        """Add a real measurement to the buffer."""
        if measured_energy_j is None or measured_energy_j <= 0:
            return
        self.buffer.append({
            'flops':          spec.flops,
            'batch_size':     spec.batch,
            'precision':      spec.precision,
            'precision_enc':  PREC_ENC.get(spec.precision, 1.0),
            'param_count':    spec.params,
            'cpu_latency_s':  latency_s,
            'gpu_latency_s':  latency_s,
            'npu_latency_s':  latency_s,
            'measured_hw':    hw,
            'measured_energy_j': measured_energy_j,
        })

        if len(self.buffer) >= self.retrain_every:
            self._retrain()

    def _retrain(self):
        """Incremental retrain of GPU predictor with buffered real data."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        dataset_path = os.path.join(BASE, 'workload_dataset.csv')
        if not os.path.exists(dataset_path):
            return

        print('\n  [AdaptiveRefinement] Retraining GPU model with {} new samples...'.format(
            len(self.buffer)))

        # Load existing dataset + append real rows
        existing = pd.read_csv(dataset_path)
        new_rows = pd.DataFrame(self.buffer)

        # Build real gpu_energy_j from measurements (only GPU measurements update GPU model)
        gpu_measured = new_rows[new_rows['measured_hw'] == 'GPU'].copy()
        if len(gpu_measured) > 0:
            gpu_measured['gpu_energy_j'] = gpu_measured['measured_energy_j']
            gpu_measured['cpu_energy_j'] = gpu_measured['measured_energy_j'] * 3  # rough ratio
            gpu_measured['npu_energy_j'] = gpu_measured['measured_energy_j'] * 0.1
            gpu_measured['best_accelerator'] = 'GPU'

            combined = pd.concat([existing, gpu_measured[existing.columns.intersection(
                gpu_measured.columns)]], ignore_index=True)
            combined.to_csv(dataset_path, index=False)

        # Retrain GPU model on updated data
        features = ['log_flops', 'log_batch', 'precision_enc', 'log_params']
        df = pd.read_csv(dataset_path)
        df['log_flops']  = np.log10(df['flops'].clip(lower=1))
        df['log_batch']  = np.log2(df['batch_size'].clip(lower=1))
        df['log_params'] = np.log10(df['param_count'].clip(lower=1))

        X = df[features]
        y = np.log1p(df['gpu_energy_j'])

        model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                       random_state=42, n_jobs=-1)
        model.fit(X, y)

        with open(os.path.join(MODEL_DIR, 'gpu_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

        self.retrain_count += 1
        self.buffer.clear()
        print('  [AdaptiveRefinement] GPU model updated (retrain #{})'.format(
            self.retrain_count))


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

ROCM_FLAGS = {
    'GPU': ['HSA_ENABLE_SDMA=0', 'GPU_MAX_HW_QUEUES=8',
            'MIOPEN_FIND_ENFORCE=ENFORCE_DB', 'ROCM_HOME=/opt/rocm'],
    'NPU': ['XLNX_VART_FIRMWARE=/lib/firmware/amdnpu',
            'AMD_OOB_NPU_FIRMWARE=1', 'XRT_INI_PATH=/etc/xrt.ini'],
    'CPU': ['OMP_NUM_THREADS=8', 'GOMP_CPU_AFFINITY=0-7'],
}

EXPLANATIONS = {
    'GPU': ('GPU selected: high FLOPs and/or large batch saturates parallel cores. '
            '200W TDP justified by throughput.'),
    'NPU': ('NPU selected: low-power inference accelerator (8W TDP). '
            'Optimal for small-medium workloads with FP16/INT8.'),
    'CPU': ('CPU selected: workload is lightweight enough that GPU/NPU startup '
            'overhead exceeds execution cost.'),
}

def decide(energy_map: dict, spec: WorkloadSpec) -> AcceleratorDecision:
    best_hw      = min(energy_map, key=energy_map.get)
    best_energy  = energy_map[best_hw]
    eff          = {hw: (1.0 / e) for hw, e in energy_map.items()}
    best_eff     = eff[best_hw] / max(eff.values()) * 100.0
    latency_ms   = (best_energy / HW_TDP[best_hw]) * 1000.0

    return AcceleratorDecision(
        recommended_hw    = best_hw,
        precision         = spec.precision,
        batch             = spec.batch,
        predicted_energy_j= best_energy,
        predicted_latency_ms = latency_ms,
        efficiency_score  = best_eff,
        alternatives      = energy_map,
        explanation       = EXPLANATIONS.get(best_hw, ''),
        rocm_flags        = ROCM_FLAGS.get(best_hw, []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY BUDGET OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class EnergyBudgetOptimizer:
    """
    Given a maximum energy budget in Joules, finds the best
    (hardware, precision, batch) combination that fits within budget.

    Example:
        opt = EnergyBudgetOptimizer(max_energy_j=2.0)
        config = opt.solve(flops=5e10, params=1e8)
        print(config)
    """

    def __init__(self, max_energy_j: float, engine: PredictionEngine = None):
        self.max_energy_j = max_energy_j
        self.engine = engine or PredictionEngine()

    def solve(self, flops: float, params: float,
              preferred_precision: str = 'FP32',
              preferred_batch: int = 32) -> dict:
        """
        Searches (hardware x precision x batch) space for configs
        that satisfy the energy budget. Returns the most efficient one.
        """
        candidates = []
        batch_options = [b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]
                         if b <= preferred_batch * 2]
        prec_options  = PREC_LIST  # FP32, FP16, INT8

        for prec in prec_options:
            for batch in batch_options:
                energy_map = self.engine.predict(flops, batch, prec, params)
                for hw, energy_j in energy_map.items():
                    if energy_j <= self.max_energy_j:
                        latency_ms = (energy_j / HW_TDP[hw]) * 1000.0
                        candidates.append({
                            'hardware':     hw,
                            'precision':    prec,
                            'batch':        batch,
                            'energy_j':     energy_j,
                            'latency_ms':   latency_ms,
                            'efficiency':   1.0 / energy_j,
                            'within_budget': True,
                        })

        if not candidates:
            # Nothing fits — return the globally lowest energy option
            best = None
            best_e = float('inf')
            for prec in prec_options:
                energy_map = self.engine.predict(flops, 1, prec, params)
                for hw, e in energy_map.items():
                    if e < best_e:
                        best_e = e
                        best = {'hardware': hw, 'precision': prec, 'batch': 1,
                                'energy_j': e,
                                'latency_ms': (e / HW_TDP[hw]) * 1000.0,
                                'efficiency': 1.0 / e,
                                'within_budget': False,
                                'note': 'Budget impossible — returning lowest energy option'}
            return best

        # Sort by efficiency (best perf/watt), prefer higher batch if tied
        candidates.sort(key=lambda c: (-c['efficiency'], -c['batch']))
        best = candidates[0]
        best['budget_j']     = self.max_energy_j
        best['candidates_found'] = len(candidates)
        return best


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNTIME SDK
# ─────────────────────────────────────────────────────────────────────────────

class HeteroWiseRuntime:
    """
    Main entry point. Import this and call optimize_and_run().

    Example:
        from heterowise_runtime import HeteroWiseRuntime

        hw = HeteroWiseRuntime()

        def my_workload():
            import numpy as np
            A = np.random.randn(32, 512).astype(np.float32)
            B = np.random.randn(512, 512).astype(np.float32)
            return A @ B

        result = hw.optimize_and_run(
            fn        = my_workload,
            flops     = 1e10,
            batch     = 32,
            precision = 'FP32',
            params    = 1e7,
            label     = 'resnet_layer'
        )

        print(result.decision.recommended_hw)
        print(result.measured_energy_j)
    """

    def __init__(self, retrain_every: int = 10, verbose: bool = True):
        self.engine    = PredictionEngine()
        self.adaptive  = AdaptiveRefinementEngine(retrain_every=retrain_every)
        self.verbose   = verbose
        self._run_count = 0
        self._session_start = datetime.now().isoformat()

        if self.verbose:
            print('[HeteroWise] Runtime initialized.')
            print('[HeteroWise] Models loaded. Adaptive retraining every {} runs.'.format(
                retrain_every))

    def optimize_and_run(self,
                         fn:        Callable,
                         flops:     float,
                         batch:     int,
                         precision: str,
                         params:    float,
                         label:     str = 'workload') -> ExecutionResult:
        """
        Full pipeline:
          1. Predict energy for all three accelerators
          2. Select best hardware
          3. Execute fn() and measure real latency
          4. Capture rocm-smi power readings
          5. Log telemetry
          6. Feed to adaptive refinement engine
          7. Return ExecutionResult
        """
        spec = WorkloadSpec(flops=flops, batch=batch, precision=precision,
                            params=params, label=label)
        self._run_count += 1

        if self.verbose:
            print('\n[HeteroWise] Run #{} — {}'.format(self._run_count, label))

        # ── Step 1: Predict ──────────────────────────────────────────────────
        energy_map = self.engine.predict(flops, batch, precision, params)
        decision   = decide(energy_map, spec)

        if self.verbose:
            print('  Decision: {} ({:.4f}J predicted, {:.1f}ms)'.format(
                decision.recommended_hw,
                decision.predicted_energy_j,
                decision.predicted_latency_ms))
            print('  Explanation: ' + decision.explanation)

        # ── Step 2: Execute with telemetry ───────────────────────────────────
        m_before = _rocm_metrics()
        t0       = time.perf_counter()

        return_value = fn()

        t1      = time.perf_counter()
        m_after = _rocm_metrics()

        latency_s  = t1 - t0
        latency_ms = latency_s * 1000.0

        # ── Step 3: Compute measured energy ──────────────────────────────────
        p_before = m_before['power_w']
        p_after  = m_after['power_w']

        if p_before is not None and p_after is not None:
            avg_power       = (p_before + p_after) / 2.0
            measured_energy = avg_power * latency_s
        else:
            avg_power       = None
            measured_energy = None

        # ── Step 4: Prediction error ──────────────────────────────────────────
        if measured_energy is not None and measured_energy > 0:
            pred_err = (abs(decision.predicted_energy_j - measured_energy) /
                        max(decision.predicted_energy_j, measured_energy) * 100.0)
        else:
            pred_err = None

        if self.verbose:
            print('  Measured latency: {:.2f}ms'.format(latency_ms))
            if measured_energy is not None:
                print('  Measured energy:  {:.4f}J  (predicted: {:.4f}J, error: {:.1f}%)'.format(
                    measured_energy, decision.predicted_energy_j, pred_err))
            else:
                print('  Power reading: N/A (run with sudo for live readings)')

        # ── Step 5: Log ───────────────────────────────────────────────────────
        row = {
            'timestamp':          datetime.now().isoformat(),
            'label':              label,
            'run':                self._run_count,
            'flops':              flops,
            'batch':              batch,
            'precision':          precision,
            'params':             params,
            'recommended_hw':     decision.recommended_hw,
            'predicted_energy_j': decision.predicted_energy_j,
            'measured_energy_j':  measured_energy,
            'measured_latency_ms':latency_ms,
            'pred_error_pct':     pred_err,
            'rocm_power_w':       avg_power,
            'rocm_temp_c':        m_after['temp_c'],
            'rocm_clock_mhz':     m_after['clock_mhz'],
        }
        df = pd.DataFrame([row])
        write_header = not os.path.exists(LOG_PATH)
        df.to_csv(LOG_PATH, mode='a', header=write_header, index=False)

        # ── Step 6: Feed adaptive engine ─────────────────────────────────────
        if measured_energy is not None:
            self.adaptive.record(spec, decision.recommended_hw,
                                 measured_energy, latency_s)
            # If adaptive engine retrained, reload models
            if self.adaptive.retrain_count > 0:
                self.engine.reload()

        return ExecutionResult(
            decision             = decision,
            measured_energy_j    = measured_energy,
            measured_latency_ms  = latency_ms,
            prediction_error_pct = pred_err,
            return_value         = return_value,
            timestamp            = row['timestamp'],
        )

    def status(self) -> dict:
        """Return current runtime state."""
        return {
            'runs':              self._run_count,
            'adaptive_retrains': self.adaptive.retrain_count,
            'buffer_size':       len(self.adaptive.buffer),
            'session_start':     self._session_start,
            'model_calls':       self.engine._call_count,
        }

    def summary(self):
        """Print session summary from log."""
        if not os.path.exists(LOG_PATH):
            print('[HeteroWise] No runs logged yet.')
            return

        df = pd.read_csv(LOG_PATH)
        if len(df) == 0:
            return

        print('\n[HeteroWise] Session Summary')
        print('  Total runs:   {}'.format(len(df)))
        print('  Hardware used:')
        if 'recommended_hw' not in df.columns:
            print('    (column missing - log may have been written without headers; delete {} and re-run)'.format(LOG_PATH))
        else:
            for hw, cnt in df['recommended_hw'].value_counts().items():
                print('    {}: {}'.format(hw, cnt))

        valid = df[df['measured_energy_j'].notna()]
        if len(valid) > 0:
            print('  Avg measured energy: {:.4f}J'.format(valid['measured_energy_j'].mean()))
            print('  Avg prediction error: {:.1f}%'.format(
                valid['pred_error_pct'].dropna().mean()))

        print('  Adaptive retrains: {}'.format(self.adaptive.retrain_count))


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE TOP-LEVEL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

_default_runtime = None

def optimize_and_run(fn: Callable,
                     flops: float,
                     batch: int,
                     precision: str = 'FP32',
                     params: float = 1e7,
                     label: str = 'workload') -> ExecutionResult:
    """
    Module-level convenience function.

    from heterowise_runtime import optimize_and_run

    result = optimize_and_run(my_fn, flops=1e10, batch=32, precision='FP16', params=1e7)
    """
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = HeteroWiseRuntime(verbose=False)
    return _default_runtime.optimize_and_run(fn, flops, batch, precision, params, label)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  HeteroWise - Phase 5: Runtime SDK Demo')
    print('=' * 60)

    # ── Demo 1: optimize_and_run API ──────────────────────────────────────────
    print('\n--- Demo 1: optimize_and_run() SDK ---')

    runtime = HeteroWiseRuntime(retrain_every=10, verbose=True)

    workloads = [
        ('tiny_inference',   1e8,  1,  'INT8',  1e6),
        ('mobile_inference', 1e9,  4,  'FP16',  1e7),
        ('server_batch',     5e10, 64, 'FP32',  5e7),
        ('large_training',   3e11, 128,'FP32',  7e8),
    ]

    for label, flops, batch, prec, params in workloads:
        def workload_fn(f=flops, b=batch, p=prec):
            n = int((f / (2 * b)) ** 0.5)
            n = max(32, min(n, 512))
            dtype = np.float16 if p == 'FP16' else np.float32
            A = np.random.randn(b, n).astype(dtype)
            B = np.random.randn(n, n).astype(dtype)
            return A @ B

        result = runtime.optimize_and_run(
            fn=workload_fn, flops=flops, batch=batch,
            precision=prec, params=params, label=label
        )

    # ── Demo 2: Energy Budget Optimizer ───────────────────────────────────────
    print('\n\n--- Demo 2: EnergyBudgetOptimizer ---')

    budget_scenarios = [
        (0.5,  5e9,  1e7,  'Strict mobile budget (0.5J)'),
        (2.0,  5e10, 1e8,  'Server budget (2.0J)'),
        (0.01, 5e11, 1e9,  'Tiny budget — should find fallback (0.01J)'),
    ]

    for budget_j, flops, params, desc in budget_scenarios:
        print('\n  Budget: {}J — {}'.format(budget_j, desc))
        opt    = EnergyBudgetOptimizer(max_energy_j=budget_j,
                                       engine=runtime.engine)
        config = opt.solve(flops=flops, params=params,
                           preferred_precision='FP32', preferred_batch=32)

        if config:
            within = 'YES' if config.get('within_budget', False) else 'NO (fallback)'
            print('    Best config:  {} + {} + batch={}'.format(
                config['hardware'], config['precision'], config['batch']))
            print('    Energy:       {:.4f}J  (budget: {}J)'.format(
                config['energy_j'], budget_j))
            print('    Latency est.: {:.2f}ms'.format(config['latency_ms']))
            print('    Within budget: {}'.format(within))
            if config.get('candidates_found'):
                print('    Candidates:   {} configs found within budget'.format(
                    config['candidates_found']))
            if config.get('note'):
                print('    Note: ' + config['note'])

    # ── Session Summary ───────────────────────────────────────────────────────
    runtime.summary()
    st = runtime.status()
    print('\n  Runtime Status:')
    for k, v in st.items():
        print('    {}: {}'.format(k, v))

    print('\n' + '=' * 60 + '\n')
