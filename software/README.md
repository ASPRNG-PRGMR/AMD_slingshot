# ⚡ HeteroWise
### AI-Powered Heterogeneous Workload Energy Optimizer

An ML-based advisory and orchestration engine that predicts energy consumption across AMD CPU, GPU, and NPU — recommends the most efficient accelerator — and in Phase 5 exposes a developer SDK with adaptive model refinement and energy budget constraints.

---

## Project Structure

```
heterowise/
├── phase1_generate_data.py    # Generate 1,500 synthetic workload samples
├── phase2_train_models.py     # Train 3 RandomForest regressors (CPU/GPU/NPU)
├── phase3_dashboard.py        # Streamlit interactive UI
├── phase3_cli_demo.py         # CLI version (no Streamlit needed)
├── phase4_rocm_telemetry.py   # Live AMD GPU validation via rocm-smi
├── heterowise_runtime.py      # Phase 5: Runtime SDK + EnergyBudgetOptimizer
├── rocm_integration.py        # ROCm deployment strategy generator
├── requirements.txt
├── workload_dataset.csv       # Created by Phase 1, grows with real telemetry
├── telemetry_log.csv          # Created by Phase 4 and Phase 5
└── models/                    # Created by Phase 2
    ├── cpu_model.pkl
    ├── gpu_model.pkl
    ├── npu_model.pkl
    └── model_meta.pkl
```

---

## Setup

```bash
pip install -r requirements.txt
sudo dnf install rocm-smi        # Fedora — for live GPU power readings
rocm-smi                         # verify AMD GPU is detected
```

---

## Run Order

### Phase 1 — Generate synthetic dataset
```bash
python3 phase1_generate_data.py
```
Creates `workload_dataset.csv` with 1,500 samples across CPU/GPU/NPU decision regions.

### Phase 2 — Train ML models
```bash
python3 phase2_train_models.py
```
Trains 3 RandomForest regressors, saves to `models/`.

### Phase 3 — Decision engine

CLI (no extra dependencies):
```bash
python3 phase3_cli_demo.py
```

Streamlit UI:
```bash
streamlit run phase3_dashboard.py
```

### Phase 4 — Live ROCm telemetry validation
```bash
sudo python3 phase4_rocm_telemetry.py
```
Reads live AMD GPU power via `rocm-smi`, runs workloads, compares measured energy against ML predictions, appends real data to training set.

> `sudo` required on Fedora for power sensor reads. Without it the script still runs — latency is measured but power shows N/A.

### Phase 5 — Runtime SDK
```bash
python3 heterowise_runtime.py
```
Runs the full SDK demo: `optimize_and_run()` pipeline across four workloads, then the `EnergyBudgetOptimizer` across three budget scenarios.

### Retrain after collecting real data
```bash
python3 phase2_train_models.py
```
Re-run after Phase 4 or Phase 5 to incorporate real hardware measurements into the models.

---

## Phase 5 — SDK Usage

### Basic: optimize_and_run()

```python
from heterowise_runtime import HeteroWiseRuntime

runtime = HeteroWiseRuntime()

def my_workload():
    import numpy as np
    A = np.random.randn(32, 512).astype('float32')
    B = np.random.randn(512, 512).astype('float32')
    return A @ B

result = runtime.optimize_and_run(
    fn        = my_workload,
    flops     = 1e10,
    batch     = 32,
    precision = 'FP32',
    params    = 1e7,
    label     = 'my_model_layer'
)

print(result.decision.recommended_hw)     # 'GPU'
print(result.decision.predicted_energy_j) # 1.0309
print(result.measured_latency_ms)         # actual wall-clock ms
print(result.measured_energy_j)           # real Joules if rocm-smi available
```

### Module-level shortcut

```python
from heterowise_runtime import optimize_and_run

result = optimize_and_run(my_fn, flops=1e9, batch=8, precision='FP16', params=1e7)
```

### Energy Budget Optimizer

Set a maximum energy budget in Joules. The optimizer searches across all (hardware × precision × batch) combinations and returns the most efficient config that fits.

```python
from heterowise_runtime import EnergyBudgetOptimizer

opt    = EnergyBudgetOptimizer(max_energy_j=2.0)
config = opt.solve(flops=5e10, params=1e8, preferred_precision='FP32', preferred_batch=32)

print(config['hardware'])    # 'NPU'
print(config['precision'])   # 'INT8'
print(config['batch'])       # 64
print(config['energy_j'])    # 0.0568
print(config['within_budget'])  # True
print(config['candidates_found'])  # 42
```

If no combination fits the budget, returns the globally lowest energy option and sets `within_budget: False`.

---

## Architecture

```
Application Layer
       ↓
HeteroWise Runtime SDK  (heterowise_runtime.py)
       ↓
Prediction Engine       (3x RandomForestRegressor)
       ↓
Decision Engine         (energy comparison + efficiency ranking)
       ↓
Deployment Strategy     (rocm_integration.py — flags, code templates)
       ↓
ROCm / CPU Runtime
       ↓
Telemetry Collector     (rocm-smi power + latency)
       ↓
Adaptive Refinement     (buffer → retrain GPU model → reload)
```

The adaptive loop: every N measured runs, `AdaptiveRefinementEngine` retrains the GPU predictor with real hardware data and hot-reloads it into the running runtime — no restart needed.

---

## What Each Phase Does

| Phase | File | What it does |
|-------|------|-------------|
| 1 | `phase1_generate_data.py` | Synthetic dataset — 1,500 samples with physics-based energy formulas |
| 2 | `phase2_train_models.py` | RandomForest training — R² 0.997/0.796/0.995 for CPU/GPU/NPU |
| 3 | `phase3_dashboard.py` / `phase3_cli_demo.py` | UI + CLI decision engine |
| 4 | `phase4_rocm_telemetry.py` | Live rocm-smi validation, real energy measurement |
| 5 | `heterowise_runtime.py` | SDK, adaptive retraining, energy budget optimizer |
| — | `rocm_integration.py` | ROCm/NPU/CPU deployment code + env flag generator |

---

## Model Performance

| Model | R² | RMSE | Key feature |
|-------|----|------|-------------|
| CPU energy | 0.9971 | 8.97 J | FLOPs (96.7%) |
| GPU energy | 0.7962 | 0.14 J | FLOPs (69.4%) + batch (16.4%) |
| NPU energy | 0.9947 | 0.39 J | FLOPs (78.3%) + precision (21.5%) |

Input features: `log(FLOPs)`, `log(batch)`, `precision_enc`, `log(params)`

---

## Hardware Model

| | CPU | GPU | NPU |
|--|-----|-----|-----|
| Simulated throughput | 40 GFLOP/s | 5 TFLOP/s | 600 GFLOP/s |
| TDP | 45W | 200W | 8W |
| Startup cost | none | 5ms | 0.3ms |
| Scales with batch | weakly | strongly | no |
| Best precision | FP32 | FP16/FP32 | INT8/FP16 |

Target: AMD Ryzen CPU · AMD Radeon GPU (ROCm) · AMD Ryzen AI NPU

---

## Troubleshooting

**Power reads show N/A in Phase 4 or 5:**
```bash
sudo python3 phase4_rocm_telemetry.py
sudo python3 heterowise_runtime.py
```

**rocm-smi not found:**
```bash
sudo dnf install rocm-smi    # Fedora
sudo apt install rocm-smi    # Ubuntu/Debian
```

**Models not found:**
```bash
python3 phase1_generate_data.py
python3 phase2_train_models.py
```
Must run Phases 1 and 2 before 3, 4, or 5.

**High prediction error in Phase 4/5:**
Expected for integrated Vega iGPU — the synthetic model was tuned for discrete AMD GPU TDP profiles. Run Phase 4 to collect real measurements, then retrain with Phase 2.

---

## Stack

- Python 3.10+
- scikit-learn — RandomForestRegressor
- NumPy + Pandas — data generation, feature engineering
- Matplotlib — charts
- Streamlit — dashboard UI
- rocm-smi — live AMD GPU telemetry (system package, no pip install)

---

## What This Is Not

HeteroWise is a **software advisory and orchestration layer** — not a kernel dispatcher, OS scheduler, or compiler. It does not intercept system calls or modify how the OS routes compute. It advises, configures, and measures. The gap to a full production system is real hardware telemetry at scale and an optional runtime hook that calls the SDK automatically at model invocation time.

---

## Phase 6 — Production Hardening

```bash
python3 phase6_hardening.py
```

### What it adds

**Thermal Awareness Engine** — reads live GPU temperature via `rocm-smi`. If the GPU exceeds configurable thresholds, it applies an energy penalty to GPU efficiency scores, causing the decision engine to route away from the GPU automatically:

| Temperature | Mode | GPU penalty |
|-------------|------|-------------|
| < 70°C | Normal | None |
| 70–80°C | Caution | +25% (GPU looks more expensive) |
| > 80°C | Throttle | +60% (system routes to NPU/CPU) |

Example: at 8×10¹⁰ FLOPs / batch=32 / FP32, GPU wins at 1.20J. At 85°C with 60% penalty, GPU becomes 1.92J vs NPU at 1.84J — system reroutes to NPU automatically.

**ThermalAwareRuntime** — drop-in replacement for Phase 5 runtime with thermal routing built in:

```python
from phase6_hardening import ThermalAwareRuntime

runtime = ThermalAwareRuntime(thermal_warn=70, thermal_throttle=80)
result  = runtime.recommend(flops=8e10, batch=32, precision='FP32', params=1e7)
# If GPU is running hot -> automatically routes to NPU
```

**StackPositioner** — architectural documentation utility:

```python
from phase6_hardening import StackPositioner
StackPositioner.print_stack()          # full layer diagram
StackPositioner.print_amd_products()   # AMD tech used per layer
StackPositioner.print_business_model() # cost + revenue breakdown
```

### Full architecture (Phase 6 final form)

```
Application Layer          (user model code)
         |
HeteroWise Runtime SDK     (heterowise_runtime.py / phase6_hardening.py)
         |
Prediction Engine          (3x RandomForest — CPU/GPU/NPU energy)
         |
Decision Engine            (energy + thermal ranking)
         |
Thermal Awareness          (rocm-smi temp -> penalty -> reroute)
         |
Deployment Strategy        (rocm_integration.py)
         |
ROCm / CPU / NPU Runtime   (hardware execution)
         |
Telemetry Collector        (rocm-smi power + latency)
         |
Adaptive Refinement        (retrain GPU model -> hot reload)
```

---

## Phase 7 — Telemetry Abstraction Layer

```bash
python3 phase7_telemetry.py               # full demo (auto-detects provider)
python3 phase7_telemetry.py --mode sim    # simulated MCU only
python3 phase7_telemetry.py --mode rocm   # real rocm-smi only
```

### Architecture

```
Application Layer
      |
Phase7Runtime
      |
TelemetryProvider  (abstract interface)
     /       \
ROCm          SimulatedMCU
Provider      Provider
(real AMD)    (INA226 + Cortex-M0+ physics model)
```

Any provider can be swapped in with a single argument — the runtime is identical regardless.

### Providers

**`RocmTelemetryProvider`** — reads live AMD GPU data via `rocm-smi`. Estimates per-rail split from aggregate GPU power reading.

**`SimulatedMCUTelemetryProvider`** — physics-based simulation of a Cortex-M0+ board with INA226 current monitors. Models:
- Per-rail power: CPU rail, GPU rail, NPU rail, system rail independently
- Thermal inertia: `temp(t+1) = temp(t) + alpha*(power_ratio - cooling) * dt`
- INA226-class noise: ±1.5% on all readings
- Clock throttle: clock drops proportionally when temp > 82°C
- Load-dependent draw: bigger FLOPs + larger batch = higher rail current

**`AutoTelemetryProvider`** — tries ROCm first, falls back to SimulatedMCU automatically.

### Using in your own code

```python
from phase7_telemetry import Phase7Runtime, make_provider

# Simulated MCU mode
runtime = Phase7Runtime(telemetry_mode='simulated')
result  = runtime.run(flops=5e10, batch=64, precision='FP32', params=1e7)

# Access per-rail breakdown
print(result['rail_breakdown'])
# {'cpu_w': 8.9, 'gpu_w': 30.0, 'npu_w': 1.0, 'sys_w': 5.0, 'total_w': 45.0}

# Or just get a raw reading
provider = make_provider('simulated')
reading  = provider.read(workload_flops=3e11, workload_batch=128, active_hw='GPU')
print(reading.gpu_power_w)   # 32.3W
print(reading.temp_c)        # 43.1C
print(reading.throttle_active)  # False

# Force a thermal throttle scenario
from phase7_telemetry import SimulatedMCUTelemetryProvider
p = SimulatedMCUTelemetryProvider()
p.heat_up(50.0)              # artificially heat to throttle zone
r = p.read(workload_flops=5e10, workload_batch=64, active_hw='GPU')
print(r.clock_mhz)           # ~1276MHz (throttled from 1850MHz)
print(r.throttle_active)     # True
```

### What the throttle demo shows

At 92°C (simulated), clock drops from ~1850MHz to ~1276MHz. Combined with Phase 6's thermal penalty, the decision engine simultaneously:
1. Detects high temperature via `TelemetryProvider.read()`
2. Applies +60% energy penalty to GPU efficiency score
3. Routes workload to NPU or CPU instead
4. Continues logging real rail-level power data

This is the full energy + thermal co-optimization loop.
