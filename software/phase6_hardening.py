"""
HeteroWise - Phase 6: Production Hardening
Thermal Awareness + Stack Positioning + Runtime Extensions

Add this to heterowise_runtime.py by importing:
    from phase6_hardening import ThermalAwareRuntime, StackPositioner

Or run standalone:
    python3 phase6_hardening.py
"""

import os
import re
import sys
import time
import subprocess
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')

HW_TDP   = {'CPU': 45, 'GPU': 200, 'NPU': 8}
PREC_ENC = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}

# ─────────────────────────────────────────────────────────────────────────────
# THERMAL AWARENESS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThermalState:
    temp_c:           Optional[float]
    clock_mhz:        Optional[float]
    power_w:          Optional[float]
    throttle_active:  bool
    thermal_headroom: Optional[float]   # degrees below threshold
    recommendation:   str               # 'normal' | 'caution' | 'throttle'


class ThermalAwarenessEngine:
    """
    Reads live GPU temperature via rocm-smi and applies a thermal penalty
    to GPU efficiency scores when the chip is running hot.

    Thresholds (tuned for AMD iGPU / Vega):
      < 70°C  — normal operation, no penalty
      70-80°C — caution zone, 25% GPU efficiency penalty
      > 80°C  — thermal throttle, 60% GPU efficiency penalty
                (system biases toward CPU or NPU to cool down)

    This makes HeteroWise energy + thermal aware — not just energy aware.
    """

    TEMP_NORMAL   = 70.0   # degrees C
    TEMP_CAUTION  = 80.0
    PENALTY_CAUTION  = 0.25   # reduce GPU efficiency score by 25%
    PENALTY_THROTTLE = 0.60   # reduce GPU efficiency score by 60%

    def __init__(self, warn_threshold: float = 70.0, throttle_threshold: float = 80.0):
        self.warn_threshold     = warn_threshold
        self.throttle_threshold = throttle_threshold
        self._last_reading      = None
        self._last_ts           = 0

    def read(self) -> ThermalState:
        """Read current thermal state from rocm-smi."""
        now = time.time()

        # Cache readings for 2 seconds to avoid hammering rocm-smi
        if self._last_reading and (now - self._last_ts) < 2.0:
            return self._last_reading

        temp = power = clock = None
        try:
            r = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
            out = r.stdout
            m = re.search(r'(\d+\.\d+).C', out)
            if m: temp = float(m.group(1))
            m = re.search(r'(\d+\.\d+)W', out)
            if m: power = float(m.group(1))
            m = re.search(r'(\d+)[Mm]hz', out)
            if m: clock = float(m.group(1))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if temp is None:
            state = ThermalState(
                temp_c=None, clock_mhz=clock, power_w=power,
                throttle_active=False, thermal_headroom=None,
                recommendation='normal'
            )
        elif temp >= self.throttle_threshold:
            state = ThermalState(
                temp_c=temp, clock_mhz=clock, power_w=power,
                throttle_active=True,
                thermal_headroom=-(temp - self.throttle_threshold),
                recommendation='throttle'
            )
        elif temp >= self.warn_threshold:
            state = ThermalState(
                temp_c=temp, clock_mhz=clock, power_w=power,
                throttle_active=False,
                thermal_headroom=self.throttle_threshold - temp,
                recommendation='caution'
            )
        else:
            state = ThermalState(
                temp_c=temp, clock_mhz=clock, power_w=power,
                throttle_active=False,
                thermal_headroom=self.throttle_threshold - temp,
                recommendation='normal'
            )

        self._last_reading = state
        self._last_ts = now
        return state

    def apply_thermal_penalty(self, energy_map: dict, thermal: ThermalState) -> dict:
        """
        Returns an adjusted energy_map where GPU energy is increased
        (penalized) based on thermal state.

        Higher penalty = GPU looks less attractive = system routes away from GPU.
        This is the key Phase 6 addition: thermal routing, not just energy routing.
        """
        adjusted = dict(energy_map)

        if thermal.recommendation == 'caution':
            # GPU is hot — make it look 25% more expensive
            adjusted['GPU'] = energy_map['GPU'] * (1.0 + self.PENALTY_CAUTION)
        elif thermal.recommendation == 'throttle':
            # GPU is very hot — make it look 60% more expensive
            adjusted['GPU'] = energy_map['GPU'] * (1.0 + self.PENALTY_THROTTLE)

        return adjusted

    def status_line(self, state: ThermalState) -> str:
        if state.temp_c is None:
            return 'Thermal: N/A (rocm-smi unavailable)'
        icons = {'normal': 'OK', 'caution': 'WARM', 'throttle': 'HOT'}
        icon  = icons.get(state.recommendation, '?')
        line  = '[{}] {:.1f}C'.format(icon, state.temp_c)
        if state.power_w:
            line += '  {:.1f}W'.format(state.power_w)
        if state.clock_mhz:
            line += '  {:.0f}MHz'.format(state.clock_mhz)
        if state.thermal_headroom is not None and state.recommendation != 'throttle':
            line += '  ({:.1f}C headroom)'.format(state.thermal_headroom)
        elif state.recommendation == 'throttle':
            line += '  (THROTTLING: GPU penalized)'
        return line


# ─────────────────────────────────────────────────────────────────────────────
# THERMAL-AWARE RUNTIME (extends Phase 5 runtime)
# ─────────────────────────────────────────────────────────────────────────────

class ThermalAwareRuntime:
    """
    Drop-in extension of Phase 5 HeteroWiseRuntime that adds thermal
    awareness to every decision.

    Usage:
        from phase6_hardening import ThermalAwareRuntime

        runtime = ThermalAwareRuntime(thermal_warn=70, thermal_throttle=80)
        result  = runtime.recommend(flops=5e10, batch=32, precision='FP32', params=1e7)
    """

    def __init__(self,
                 thermal_warn:     float = 70.0,
                 thermal_throttle: float = 80.0,
                 verbose:          bool  = True):
        # Load models
        self.models = {}
        for hw in ['cpu', 'gpu', 'npu']:
            with open(os.path.join(MODEL_DIR, f'{hw}_model.pkl'), 'rb') as f:
                self.models[hw.upper()] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'model_meta.pkl'), 'rb') as f:
            self.features = pickle.load(f)['features']

        self.thermal  = ThermalAwarenessEngine(thermal_warn, thermal_throttle)
        self.verbose  = verbose
        self._history = []

    def _predict(self, flops, batch, precision, params) -> dict:
        x = pd.DataFrame([[
            np.log10(max(flops,  1.0)),
            np.log2( max(batch,  1.0)),
            PREC_ENC.get(precision, 1.0),
            np.log10(max(params, 1.0))
        ]], columns=self.features)
        return {hw: float(np.expm1(m.predict(x)[0]))
                for hw, m in self.models.items()}

    def recommend(self, flops: float, batch: int,
                  precision: str, params: float,
                  label: str = '') -> dict:
        """
        Returns recommendation dict with thermal-adjusted decision.
        """
        # 1. Raw ML prediction
        raw_energy = self._predict(flops, batch, precision, params)

        # 2. Read thermal state
        thermal_state = self.thermal.read()

        # 3. Apply thermal penalty to GPU
        adj_energy = self.thermal.apply_thermal_penalty(raw_energy, thermal_state)

        # 4. Decide on adjusted values
        best_raw = min(raw_energy, key=raw_energy.get)
        best_adj = min(adj_energy, key=adj_energy.get)
        thermal_rerouted = (best_raw != best_adj)

        result = {
            'label':             label,
            'recommended_hw':    best_adj,
            'thermal_rerouted':  thermal_rerouted,
            'thermal_state':     thermal_state.recommendation,
            'temp_c':            thermal_state.temp_c,
            'raw_best':          best_raw,
            'energy_raw':        raw_energy,
            'energy_adjusted':   adj_energy,
            'predicted_energy_j':adj_energy[best_adj],
            'precision':         precision,
            'batch':             batch,
        }

        if self.verbose:
            self._print_result(result, thermal_state)

        self._history.append(result)
        return result

    def _print_result(self, r: dict, thermal: ThermalState):
        label = '  [{}]'.format(r['label']) if r['label'] else ''
        print('\n  Workload{}'.format(label))
        print('  Thermal:  {}'.format(self.thermal.status_line(thermal)))
        print('  Raw predictions:')
        for hw in ['CPU', 'GPU', 'NPU']:
            raw = r['energy_raw'][hw]
            adj = r['energy_adjusted'][hw]
            pen = '  (+{:.0f}% thermal penalty)'.format(
                (adj/raw - 1)*100) if adj != raw else ''
            print('    {}: {:.4f}J{}'.format(hw, adj, pen))

        reroute_note = ''
        if r['thermal_rerouted']:
            reroute_note = '  [REROUTED from {} due to thermal]'.format(r['raw_best'])
        print('  Decision: {}{}'.format(r['recommended_hw'], reroute_note))
        print('  Energy:   {:.4f}J'.format(r['predicted_energy_j']))


# ─────────────────────────────────────────────────────────────────────────────
# STACK POSITIONER — architectural documentation utility
# ─────────────────────────────────────────────────────────────────────────────

class StackPositioner:
    """
    Prints HeteroWise's precise position in the AMD software stack.
    Use this output in presentations and documentation.
    """

    STACK = [
        ('Application Layer',
         'User model code (PyTorch, ONNX, TensorFlow)',
         False),
        ('HeteroWise Runtime SDK',
         'heterowise_runtime.py  —  THIS IS WHERE WE SIT',
         True),
        ('ML Framework',
         'PyTorch / ONNX Runtime / Tensorflow',
         False),
        ('ROCm / CPU Backend',
         'ROCm HIP, MIOpen, rocBLAS  |  CPU: OpenMP, BLAS',
         False),
        ('AMD Hardware',
         'Ryzen CPU  |  Radeon GPU (ROCm)  |  Ryzen AI NPU',
         False),
    ]

    AMD_PRODUCTS = {
        'CPU Runtime':  'AMD Ryzen (Zen4) — OpenMP threading, BLAS',
        'GPU Runtime':  'AMD Radeon via ROCm — HIP, MIOpen, rocBLAS',
        'NPU Runtime':  'AMD Ryzen AI — ONNX Runtime + VitisAI EP',
        'Telemetry':    'ROCm-SMI — live power, temp, clock, utilization',
        'Future':       'ROCm Telemetry API, Vitis AI quantizer, XDNA SDK',
    }

    POSITIONING = {
        'what_it_is':     'AI-aware heterogeneous execution middleware',
        'what_it_is_not': [
            'Not OS-level (no kernel modifications)',
            'Not firmware-level (no hardware register access)',
            'Not a compiler (no code transformation)',
            'Not a runtime dispatcher in the traditional sense',
        ],
        'what_it_does':   [
            'Pre-execution energy prediction via ML',
            'Cross-accelerator efficiency comparison',
            'Thermal-aware routing via rocm-smi',
            'Constraint-aware planning (EnergyBudgetOptimizer)',
            'Adaptive model refinement from real telemetry',
            'Deployment code generation (ROCm flags, ONNX EP config)',
        ],
    }

    @classmethod
    def print_stack(cls):
        print('\n' + '=' * 62)
        print('  HeteroWise — Software Stack Position')
        print('=' * 62)
        for i, (layer, detail, is_us) in enumerate(cls.STACK):
            marker = '  >>>' if is_us else '     '
            print('{} {}'.format(marker, layer))
            print('       {}'.format(detail))
            if i < len(cls.STACK) - 1:
                print('          |')
                print('          v')
        print('=' * 62)

    @classmethod
    def print_amd_products(cls):
        print('\n  AMD Products / Technologies Used:')
        for k, v in cls.AMD_PRODUCTS.items():
            print('    {:15s}  {}'.format(k + ':', v))

    @classmethod
    def print_positioning(cls):
        print('\n  What HeteroWise IS:')
        print('    ' + cls.POSITIONING['what_it_is'])
        print('\n  What it DOES:')
        for item in cls.POSITIONING['what_it_does']:
            print('    + ' + item)
        print('\n  What it is NOT:')
        for item in cls.POSITIONING['what_it_is_not']:
            print('    - ' + item)

    @classmethod
    def print_business_model(cls):
        print('\n' + '=' * 62)
        print('  HeteroWise — Business Model')
        print('=' * 62)

        print('''
  DEVELOPMENT COST
  ─────────────────────────────────────────────────────
  3 engineers x 6 months x $8,000/month (loaded)  = $144,000
  Cloud training infrastructure                    =  $10,000
  ──────────────────────────────────────────────────────────
  Total                                            = $154,000

  REVENUE OPTIONS
  ─────────────────────────────────────────────────────
  Option A — OEM Licensing  (recommended for AMD)
    $2 per AMD-powered device
    AMD ships ~50M APUs/year
    1% penetration = 500,000 units/year
    Revenue = $1,000,000/year

  Option B — Enterprise License
    $50,000 per enterprise deployment
    Use case: data centers with AMD EPYC + Instinct
    20 customers = $1,000,000/year
    ROI for customer: if 5% power saving on 1MW DC
    at $0.10/kWh = $43,800 saved/year -> positive ROI

  Option C — ROCm Plugin / Freemium SDK
    Core SDK: free (drives AMD ecosystem adoption)
    Paid: adaptive dashboard, telemetry API, SLA support
    $500/month per team = $6,000/year per customer
    500 teams = $3,000,000 ARR

  POSITIONING IN AMD ECOSYSTEM
  ─────────────────────────────────────────────────────
  + Ryzen AI SDK companion tool
  + ROCm plugin (energy-aware profiling layer)
  + Edge AI optimization for AMD-powered devices
  + Data center energy orchestration for EPYC clusters
''')


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '=' * 62)
    print('  HeteroWise - Phase 6: Production Hardening')
    print('=' * 62)

    # ── Stack positioning ─────────────────────────────────────────────────────
    StackPositioner.print_stack()
    StackPositioner.print_amd_products()
    StackPositioner.print_positioning()
    StackPositioner.print_business_model()

    # ── Thermal awareness demo ────────────────────────────────────────────────
    print('\n' + '=' * 62)
    print('  THERMAL AWARENESS DEMO')
    print('=' * 62)
    print('\n  Running with real rocm-smi thermal data...')

    runtime = ThermalAwareRuntime(
        thermal_warn=70.0,
        thermal_throttle=80.0,
        verbose=True
    )

    # Simulate three scenarios
    scenarios = [
        ('tiny_inference',   1e8,  1,   'INT8',  1e6),
        ('medium_inference', 5e9,  16,  'FP16',  5e7),
        ('large_training',   3e11, 128, 'FP32',  7e8),
    ]

    for label, flops, batch, prec, params in scenarios:
        runtime.recommend(flops=flops, batch=batch,
                          precision=prec, params=params, label=label)

    # Thermal reroute demo
    # At flops=8e10 / batch=32 / FP32: GPU wins raw (1.20J) vs NPU (1.84J)
    # At 60% thermal penalty: GPU -> 1.92J > NPU 1.84J => system reroutes to NPU
    print('\n\n  --- Simulated Thermal Throttle Scenario (GPU at 85C) ---')
    print('  Workload: 8e10 FLOPs, batch=32, FP32')
    print('  GPU normally wins here. Watch what happens at 85C.\n')

    runtime_hot = ThermalAwareRuntime(thermal_warn=70, thermal_throttle=80, verbose=False)
    fake_hot = ThermalState(
        temp_c=85.0, clock_mhz=800.0, power_w=22.0,
        throttle_active=True, thermal_headroom=-5.0,
        recommendation='throttle'
    )
    raw = runtime_hot._predict(8e10, 32, 'FP32', 1e7)
    adj = runtime_hot.thermal.apply_thermal_penalty(raw, fake_hot)
    best_raw = min(raw, key=raw.get)
    best_adj = min(adj, key=adj.get)

    print('  Thermal status:  85.0C  22.0W  800MHz  (THROTTLING)')
    for hw in ['CPU', 'GPU', 'NPU']:
        pen = '  (+60% thermal penalty)' if adj[hw] != raw[hw] else ''
        print('  {} energy: {:.4f}J{}'.format(hw, adj[hw], pen))
    print('  Raw decision:     {} ({:.4f}J)'.format(best_raw, raw[best_raw]))
    print('  Thermal decision: {} ({:.4f}J)'.format(best_adj, adj[best_adj]))
    if best_raw != best_adj:
        print('  => REROUTED from {} to {} -- GPU too hot, NPU takes over'.format(
            best_raw, best_adj))
    else:
        print('  => No reroute needed at current thermal penalty')

    print('\n' + '=' * 62 + '\n')
