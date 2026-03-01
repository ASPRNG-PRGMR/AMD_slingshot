"""
HeteroWise - Phase 7: Hardware-Ready Telemetry Abstraction
Simulated MCU Provider + Abstract Telemetry Interface

Architecture:
    Application
         |
    HeteroWise Runtime
         |
    TelemetryInterface (abstract)
        /              \\
  RocmProvider     SimulatedMCUProvider
  (real hardware)  (physics-based simulation)

The SimulatedMCU models:
  - Per-rail power (CPU rail, GPU rail, NPU rail, system rail)
  - Thermal inertia: temp(t+1) = temp(t) + alpha*(power - cooling)
  - Gaussian noise on all readings
  - Clock throttle under thermal load
  - Load-dependent current draw

This means the system behaves as if a Cortex-M0+ with INA226
power monitors is streaming real rail data.

Run:
    python3 phase7_telemetry.py
    python3 phase7_telemetry.py --mode rocm     # use real rocm-smi
    python3 phase7_telemetry.py --mode sim       # simulated MCU
    python3 phase7_telemetry.py --mode demo      # side-by-side comparison
"""

import os
import re
import sys
import abc
import math
import time
import random
import pickle
import subprocess
import numpy as np
import pandas as pd
from typing   import Optional
from dataclasses import dataclass, field

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')
PREC_ENC  = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}
HW_TDP    = {'CPU': 45, 'GPU': 200, 'NPU': 8}


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY DATA CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryReading:
    """
    Structured telemetry snapshot.
    Matches what a Cortex-M0+ INA226 monitor board would stream.
    """
    # Per-rail power (Watts)
    cpu_power_w:    float = 0.0
    gpu_power_w:    float = 0.0
    npu_power_w:    float = 0.0
    system_power_w: float = 0.0   # SoC background + memory + IO

    # Thermal
    temp_c:         float = 0.0
    clock_mhz:      float = 0.0
    throttle_active: bool = False

    # Derived
    total_power_w:  float = 0.0
    timestamp:      float = field(default_factory=time.time)
    source:         str   = 'unknown'   # 'rocm' | 'simulated_mcu' | 'unavailable'

    def __post_init__(self):
        if self.total_power_w == 0.0:
            self.total_power_w = (self.cpu_power_w + self.gpu_power_w +
                                  self.npu_power_w + self.system_power_w)


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryProvider(abc.ABC):
    """
    Abstract telemetry interface.
    Any provider — real hardware or simulation — must implement this.
    The runtime only ever talks to this interface.
    """

    @abc.abstractmethod
    def read(self, workload_flops: float = 0.0,
             workload_batch: int = 1,
             active_hw: str = 'CPU') -> TelemetryReading:
        """
        Return a full TelemetryReading.
        workload_flops / batch / active_hw are hints for simulation;
        real providers ignore them and read live hardware.
        """
        ...

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    def available(self) -> bool:
        """Return True if this provider can actually deliver readings."""
        return True


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 1: ROCm-SMI (real hardware)
# ─────────────────────────────────────────────────────────────────────────────

class RocmTelemetryProvider(TelemetryProvider):
    """
    Reads live AMD GPU metrics via rocm-smi.
    This is the production provider for AMD hardware.
    """

    def name(self) -> str:
        return 'ROCm-SMI (real AMD GPU)'

    def available(self) -> bool:
        try:
            subprocess.run(['rocm-smi'], capture_output=True, timeout=3)
            return True
        except FileNotFoundError:
            return False

    def read(self, workload_flops: float = 0.0,
             workload_batch: int = 1,
             active_hw: str = 'GPU') -> TelemetryReading:

        try:
            r = subprocess.run(['rocm-smi'], capture_output=True,
                               text=True, timeout=5)
            out = r.stdout

            power = temp = clock = None
            m = re.search(r'(\d+\.\d+)W', out)
            if m: power = float(m.group(1))
            m = re.search(r'(\d+\.\d+).C', out)
            if m: temp = float(m.group(1))
            m = re.search(r'(\d+)[Mm]hz', out)
            if m: clock = float(m.group(1))

            # rocm-smi gives aggregate GPU power
            # Estimate per-rail split (GPU is primary consumer)
            gpu_w    = power or 0.0
            cpu_w    = 8.0    # Ryzen idle estimate
            sys_w    = 5.0    # memory + IO
            throttle = (temp or 0.0) > 80.0

            return TelemetryReading(
                cpu_power_w    = cpu_w,
                gpu_power_w    = gpu_w,
                npu_power_w    = 0.0,
                system_power_w = sys_w,
                temp_c         = temp or 0.0,
                clock_mhz      = clock or 0.0,
                throttle_active= throttle,
                source         = 'rocm',
            )

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return TelemetryReading(source='unavailable')


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 2: Simulated MCU (physics-based)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedMCUTelemetryProvider(TelemetryProvider):
    """
    Physics-inspired simulation of a Cortex-M0+ INA226 power monitor board.

    Models:
      - Per-rail power: CPU, GPU, NPU, system
      - Thermal inertia: temp rises under load, falls when idle
        temp(t+1) = temp(t) + alpha*(power/TDP - cooling_factor)
      - Realistic noise on all readings (INA226 ±0.5% typical)
      - Clock throttle: clock drops when temp > 80C
      - Load-dependent behavior: bigger workloads draw more power

    This allows the system to demonstrate hardware-ready behavior
    without physical MCU hardware.
    """

    # INA226 measurement noise (±0.5%)
    NOISE_PCT = 0.015

    # Thermal model constants
    THERMAL_ALPHA        = 0.08    # heating rate coefficient
    THERMAL_COOLING      = 0.04    # passive cooling coefficient
    AMBIENT_TEMP         = 35.0    # degrees C ambient
    THROTTLE_TEMP        = 82.0    # throttle kicks in
    THROTTLE_CLOCK_RATIO = 0.6     # clock drops to 60% when throttling

    # Nominal per-rail power at idle (Watts)
    IDLE = {
        'cpu': 6.0,
        'gpu': 4.0,
        'npu': 0.5,
        'sys': 4.5,
    }

    # TDP per rail
    TDP = {
        'cpu': 45.0,
        'gpu': 35.0,   # iGPU, not discrete
        'npu': 8.0,
        'sys': 8.0,
    }

    # Nominal clock at base frequency
    BASE_CLOCK_MHZ = 1600.0
    MAX_CLOCK_MHZ  = 2400.0

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self._temp        = self.AMBIENT_TEMP + random.uniform(5, 10)
        self._last_ts     = time.time()
        self._load_factor = 0.0   # 0.0 = idle, 1.0 = full load
        self._reading_idx = 0

    def name(self) -> str:
        return 'Simulated MCU (INA226 + Cortex-M0+ model)'

    def _noise(self, value: float) -> float:
        """Apply INA226-class measurement noise."""
        return value * (1.0 + random.uniform(-self.NOISE_PCT, self.NOISE_PCT))

    def _compute_load_factor(self, flops: float, batch: int, hw: str) -> float:
        """
        Compute 0-1 load factor from workload characteristics.
        Drives both power and thermal models.
        """
        if flops <= 0:
            return 0.05  # idle

        # Log-normalize FLOPs against TDP reference
        flops_norm  = min(math.log10(max(flops, 1)) / 12.0, 1.0)
        batch_norm  = min(math.log2(max(batch, 1)) / 8.0,   1.0)  # max at batch=256

        hw_factors = {'GPU': 1.0, 'CPU': 0.7, 'NPU': 0.5}
        hw_scale   = hw_factors.get(hw, 0.7)

        return min(flops_norm * 0.6 + batch_norm * 0.4, 1.0) * hw_scale

    def _update_thermal(self, total_power: float, dt: float):
        """
        Thermal inertia model:
          temp(t+dt) = temp(t) + alpha * (power_ratio - cooling) * dt_factor
        """
        max_power   = sum(self.TDP.values())
        power_ratio = total_power / max_power
        dt_factor   = min(dt / 0.1, 3.0)   # cap at 3x step

        delta = self.THERMAL_ALPHA * (power_ratio - self.THERMAL_COOLING) * dt_factor
        self._temp = max(
            self.AMBIENT_TEMP,
            self._temp + delta + random.uniform(-0.05, 0.05)
        )

    def _compute_per_rail(self, flops: float, batch: int,
                          active_hw: str) -> dict:
        """
        Compute realistic per-rail power draw.
        Active hardware draws proportionally to load.
        Inactive rails draw idle + small coupling noise.
        """
        lf = self._compute_load_factor(flops, batch, active_hw)
        self._load_factor = lf

        rails = {}
        for rail in ['cpu', 'gpu', 'npu', 'sys']:
            hw_key   = rail.upper() if rail != 'sys' else 'SYS'
            is_active = (rail.upper() == active_hw)

            if is_active:
                # Active rail: idle + load contribution up to TDP
                load_power = self.IDLE[rail] + lf * (self.TDP[rail] - self.IDLE[rail])
            else:
                # Inactive rail: idle + small sympathetic draw
                sympathetic = lf * 0.08 * self.TDP[rail]
                load_power  = self.IDLE[rail] + sympathetic

            rails[rail] = self._noise(load_power)

        return rails

    def _clock_speed(self) -> float:
        """
        Clock speed with thermal throttle model.
        Drops proportionally when above throttle temp.
        """
        if self._temp >= self.THROTTLE_TEMP:
            throttle_ratio = max(
                self.THROTTLE_CLOCK_RATIO,
                1.0 - (self._temp - self.THROTTLE_TEMP) * 0.02
            )
            base = self.BASE_CLOCK_MHZ * throttle_ratio
        else:
            # Mild boost below throttle
            headroom   = (self.THROTTLE_TEMP - self._temp) / self.THROTTLE_TEMP
            boost      = 1.0 + headroom * 0.3
            base       = min(self.BASE_CLOCK_MHZ * boost, self.MAX_CLOCK_MHZ)

        return self._noise(base)

    def read(self, workload_flops: float = 0.0,
             workload_batch: int = 1,
             active_hw: str = 'GPU') -> TelemetryReading:

        now = time.time()
        dt  = now - self._last_ts
        self._last_ts  = now
        self._reading_idx += 1

        rails     = self._compute_per_rail(workload_flops, workload_batch, active_hw)
        total_w   = sum(rails.values())
        self._update_thermal(total_w, dt)

        clock     = self._clock_speed()
        throttle  = self._temp >= self.THROTTLE_TEMP

        return TelemetryReading(
            cpu_power_w    = round(rails['cpu'], 3),
            gpu_power_w    = round(rails['gpu'], 3),
            npu_power_w    = round(rails['npu'], 3),
            system_power_w = round(rails['sys'], 3),
            total_power_w  = round(total_w, 3),
            temp_c         = round(self._temp, 2),
            clock_mhz      = round(clock, 1),
            throttle_active= throttle,
            source         = 'simulated_mcu',
        )

    def heat_up(self, degrees: float = 20.0):
        """Test helper: artificially raise temperature."""
        self._temp += degrees

    def cool_down(self):
        """Test helper: reset to ambient."""
        self._temp = self.AMBIENT_TEMP + 5.0


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 3: Auto-detect (tries ROCm first, falls back to simulated)
# ─────────────────────────────────────────────────────────────────────────────

class AutoTelemetryProvider(TelemetryProvider):
    """
    Automatically selects the best available provider:
      1. ROCm-SMI if rocm-smi is installed and responds
      2. SimulatedMCU otherwise

    This is the recommended default for most use cases.
    """

    def __init__(self):
        rocm = RocmTelemetryProvider()
        if rocm.available():
            self._provider = rocm
            print('  [Telemetry] ROCm-SMI detected — using real AMD GPU telemetry')
        else:
            self._provider = SimulatedMCUTelemetryProvider()
            print('  [Telemetry] ROCm-SMI not found — using SimulatedMCU provider')

    def name(self) -> str:
        return 'Auto ({})'.format(self._provider.name())

    def read(self, **kwargs) -> TelemetryReading:
        return self._provider.read(**kwargs)


def make_provider(mode: str = 'auto') -> TelemetryProvider:
    """
    Factory function. Use in HeteroWiseRuntime:
      runtime = HeteroWiseRuntime(telemetry_mode='simulated')
    """
    mode = mode.lower()
    if mode == 'rocm':
        return RocmTelemetryProvider()
    elif mode in ('sim', 'simulated', 'mcu'):
        return SimulatedMCUTelemetryProvider()
    else:
        return AutoTelemetryProvider()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 RUNTIME (integrates telemetry provider into decision loop)
# ─────────────────────────────────────────────────────────────────────────────

class Phase7Runtime:
    """
    Full runtime with pluggable telemetry provider.
    Replaces the hardcoded rocm-smi calls in heterowise_runtime.py
    with the abstract TelemetryProvider interface.

    Usage:
        from phase7_telemetry import Phase7Runtime

        # Use simulated MCU
        runtime = Phase7Runtime(telemetry_mode='simulated')
        result  = runtime.run(flops=5e10, batch=64, precision='FP32', params=1e7)

        # Use real ROCm hardware
        runtime = Phase7Runtime(telemetry_mode='rocm')
        result  = runtime.run(flops=5e10, batch=64, precision='FP32', params=1e7)
    """

    def __init__(self, telemetry_mode: str = 'auto', verbose: bool = True):
        self.telemetry  = make_provider(telemetry_mode)
        self.verbose    = verbose
        self._models    = {}
        self._features  = None
        self._load_models()

    def _load_models(self):
        for hw in ['cpu', 'gpu', 'npu']:
            with open(os.path.join(MODEL_DIR, f'{hw}_model.pkl'), 'rb') as f:
                self._models[hw.upper()] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'model_meta.pkl'), 'rb') as f:
            self._features = pickle.load(f)['features']

    def _predict(self, flops, batch, precision, params) -> dict:
        x = pd.DataFrame([[
            np.log10(max(flops,  1.0)),
            np.log2( max(batch,  1.0)),
            PREC_ENC.get(precision, 1.0),
            np.log10(max(params, 1.0))
        ]], columns=self._features)
        return {hw: float(np.expm1(m.predict(x)[0]))
                for hw, m in self._models.items()}

    def run(self, flops: float, batch: int, precision: str,
            params: float, label: str = '') -> dict:
        """
        Full pipeline with telemetry provider.
        Returns result dict with decision + measured telemetry.
        """
        # 1. Predict
        energy_map = self._predict(flops, batch, precision, params)
        best_hw    = min(energy_map, key=energy_map.get)

        # 2. Read telemetry before
        t_before   = self.telemetry.read(workload_flops=flops,
                                          workload_batch=batch,
                                          active_hw=best_hw)
        t0 = time.perf_counter()

        # 3. Simulate workload execution
        import numpy as _np
        n    = max(32, min(int((flops / (2 * batch)) ** 0.5), 512))
        A    = _np.random.randn(batch, n).astype(_np.float32)
        B    = _np.random.randn(n, n).astype(_np.float32)
        _    = A @ B

        t1 = time.perf_counter()

        # 4. Read telemetry after
        t_after  = self.telemetry.read(workload_flops=flops,
                                        workload_batch=batch,
                                        active_hw=best_hw)
        latency_s = t1 - t0

        # 5. Compute measured energy from active rail
        rail_map  = {'CPU': 'cpu_power_w', 'GPU': 'gpu_power_w', 'NPU': 'npu_power_w'}
        rail_attr = rail_map.get(best_hw, 'total_power_w')
        p_before  = getattr(t_before, rail_attr, t_before.total_power_w)
        p_after   = getattr(t_after,  rail_attr, t_after.total_power_w)
        avg_power = (p_before + p_after) / 2.0
        measured_j = avg_power * latency_s

        pred_j    = energy_map[best_hw]
        err_pct   = (abs(pred_j - measured_j) / max(pred_j, measured_j) * 100.0
                     if measured_j > 0 else None)

        result = {
            'label':          label,
            'hardware':       best_hw,
            'precision':      precision,
            'batch':          batch,
            'predicted_j':    pred_j,
            'measured_j':     measured_j,
            'latency_ms':     latency_s * 1000,
            'error_pct':      err_pct,
            'telemetry_src':  t_after.source,
            'rail_breakdown': {
                'cpu_w':  t_after.cpu_power_w,
                'gpu_w':  t_after.gpu_power_w,
                'npu_w':  t_after.npu_power_w,
                'sys_w':  t_after.system_power_w,
                'total_w':t_after.total_power_w,
            },
            'temp_c':         t_after.temp_c,
            'clock_mhz':      t_after.clock_mhz,
            'throttle':       t_after.throttle_active,
            'alternatives':   energy_map,
        }

        if self.verbose:
            self._print_result(result)

        return result

    def _print_result(self, r: dict):
        label = ' [{}]'.format(r['label']) if r['label'] else ''
        src_tag = '(simulated MCU)' if r['telemetry_src'] == 'simulated_mcu' else '(ROCm-SMI)'

        print('\n  Workload{}'.format(label))
        print('  Telemetry: {} {}'.format(r['telemetry_src'], src_tag))
        print('  Decision:  {} — {:.4f}J predicted'.format(r['hardware'], r['predicted_j']))
        print('  Measured:  {:.4f}J  ({:.2f}ms)'.format(r['measured_j'], r['latency_ms']))
        if r['error_pct'] is not None:
            print('  Error:     {:.1f}%'.format(r['error_pct']))

        rb = r['rail_breakdown']
        print('  Rail breakdown:')
        print('    CPU: {:5.2f}W  GPU: {:5.2f}W  NPU: {:5.2f}W  SYS: {:5.2f}W  Total: {:5.2f}W'.format(
            rb['cpu_w'], rb['gpu_w'], rb['npu_w'], rb['sys_w'], rb['total_w']))
        throttle_str = '  *** THROTTLING ***' if r['throttle'] else ''
        print('  Temp: {:.1f}C  Clock: {:.0f}MHz{}'.format(
            r['temp_c'], r['clock_mhz'], throttle_str))


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo_provider(provider: TelemetryProvider, label: str):
    print('\n  Provider: {}'.format(label))
    print('  ' + '-' * 54)

    workloads = [
        (1e8,  1,   'CPU', 'idle   '),
        (1e9,  8,   'NPU', 'small  '),
        (5e10, 64,  'GPU', 'medium '),
        (3e11, 128, 'GPU', 'heavy  '),
    ]

    for flops, batch, hw, tag in workloads:
        r = provider.read(workload_flops=flops,
                          workload_batch=batch,
                          active_hw=hw)
        print('  {} hw={} | CPU:{:5.1f}W  GPU:{:5.1f}W  NPU:{:4.1f}W  SYS:{:4.1f}W | '
              'total:{:5.1f}W  temp:{:.1f}C  clock:{:.0f}MHz{}'.format(
              tag, hw,
              r.cpu_power_w, r.gpu_power_w, r.npu_power_w, r.system_power_w,
              r.total_power_w, r.temp_c, r.clock_mhz,
              ' [THROTTLE]' if r.throttle_active else ''))


def _demo_thermal_inertia(provider: SimulatedMCUTelemetryProvider):
    print('\n  Thermal Inertia Demo (watching temp rise and fall under load)')
    print('  ' + '-' * 54)

    steps = [
        (3e11, 128, 'GPU', 'heavy load'),
        (3e11, 128, 'GPU', 'heavy load'),
        (3e11, 128, 'GPU', 'heavy load'),
        (1e8,  1,   'CPU', 'cool-down '),
        (1e8,  1,   'CPU', 'cool-down '),
        (1e8,  1,   'CPU', 'cool-down '),
    ]

    for flops, batch, hw, tag in steps:
        time.sleep(0.05)
        r = provider.read(workload_flops=flops, workload_batch=batch, active_hw=hw)
        bar_len = int((r.temp_c - 35) / 1.5)
        bar_len = max(0, min(bar_len, 40))
        bar = '|' + '#' * bar_len + '.' * (40 - bar_len) + '|'
        print('  {} {}  {:.1f}C  {:.0f}MHz  GPU:{:.1f}W{}'.format(
            tag, bar, r.temp_c, r.clock_mhz, r.gpu_power_w,
            '  THROTTLE' if r.throttle_active else ''))


def _demo_throttle(provider: SimulatedMCUTelemetryProvider):
    print('\n  Thermal Throttle Demo (heating GPU to throttle point)')
    print('  ' + '-' * 54)
    provider.heat_up(50.0)  # force into throttle zone

    for i in range(4):
        r = provider.read(workload_flops=5e10, workload_batch=64, active_hw='GPU')
        print('  step {} | temp:{:.1f}C  clock:{:.0f}MHz  gpu:{:.1f}W  throttle:{}'.format(
            i+1, r.temp_c, r.clock_mhz, r.gpu_power_w, r.throttle_active))

    provider.cool_down()
    print('  [cool_down() called]')
    r = provider.read(workload_flops=1e8, workload_batch=1, active_hw='CPU')
    print('  After cooldown | temp:{:.1f}C  clock:{:.0f}MHz'.format(
        r.temp_c, r.clock_mhz))


if __name__ == '__main__':
    mode = 'demo'
    if '--mode' in sys.argv:
        idx  = sys.argv.index('--mode')
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'demo'

    print('\n' + '=' * 60)
    print('  HeteroWise - Phase 7: Telemetry Abstraction Layer')
    print('=' * 60)

    if mode == 'rocm':
        print('\n--- ROCm-SMI Provider ---')
        p = RocmTelemetryProvider()
        if not p.available():
            print('  rocm-smi not available. Install: sudo dnf install rocm-smi')
            sys.exit(1)
        _demo_provider(p, p.name())

    elif mode == 'sim':
        print('\n--- Simulated MCU Provider ---')
        p = SimulatedMCUTelemetryProvider(seed=42)
        _demo_provider(p, p.name())
        _demo_thermal_inertia(SimulatedMCUTelemetryProvider(seed=7))
        _demo_throttle(SimulatedMCUTelemetryProvider(seed=99))

    else:
        # Full demo: architecture + both providers + full runtime
        print('\n--- Architecture ---')
        print('''
      Application Layer
            |
      Phase7Runtime
            |
      TelemetryProvider (abstract)
           / \\
    ROCm     SimulatedMCU
    (real)   (INA226 + Cortex-M0+ model)
        ''')

        print('--- Provider Comparison ---')
        rocm_p = RocmTelemetryProvider()
        sim_p  = SimulatedMCUTelemetryProvider(seed=42)

        if rocm_p.available():
            _demo_provider(rocm_p, rocm_p.name())
        else:
            print('\n  ROCm-SMI: not available (install rocm-smi to enable)')

        _demo_provider(sim_p, sim_p.name())

        print('\n\n--- Thermal Inertia Model ---')
        _demo_thermal_inertia(SimulatedMCUTelemetryProvider(seed=7))

        print('\n\n--- Throttle Demo ---')
        _demo_throttle(SimulatedMCUTelemetryProvider(seed=99))

        print('\n\n--- Full Phase7Runtime (simulated mode) ---')
        runtime = Phase7Runtime(telemetry_mode='simulated', verbose=True)

        workloads = [
            (1e8,  1,   'INT8', 1e6,  'tiny_edge'),
            (1e9,  8,   'FP16', 5e6,  'mobile'),
            (5e10, 64,  'FP32', 5e7,  'server'),
            (3e11, 128, 'FP32', 7e8,  'training'),
        ]
        for flops, batch, prec, params, label in workloads:
            runtime.run(flops=flops, batch=batch, precision=prec,
                        params=params, label=label)

    print('\n' + '=' * 60 + '\n')
