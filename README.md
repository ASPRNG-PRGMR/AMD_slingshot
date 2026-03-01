# ⚡ HeteroWise

**AI-Powered Heterogeneous Workload Energy Optimizer for AMD Platforms**

HeteroWise is a telemetry-aware, machine learning–driven middleware that predicts energy consumption across AMD CPU, GPU (ROCm), and NPU accelerators and dynamically selects the most energy-efficient execution target.

The system integrates predictive modeling, real hardware telemetry, constraint-aware optimization, and adaptive refinement to maximize performance-per-watt in heterogeneous compute environments.

---

## 🚀 Motivation

Modern AI workloads run on heterogeneous systems (CPU + GPU + NPU), yet:

- Accelerator selection is often static or heuristic-based
- Energy constraints are rarely enforced deterministically
- Thermal conditions are not proactively considered
- Real hardware telemetry is underutilized

HeteroWise addresses these gaps by introducing AI-driven, constraint-aware workload orchestration aligned with AMD's heterogeneous compute ecosystem.

---

## 🧠 Core Capabilities

- ML-based energy prediction (CPU / GPU / NPU)
- Real-time ROCm telemetry validation
- Deterministic energy budget enforcement
- Thermal-aware dynamic rerouting
- Adaptive telemetry-driven model refinement
- Developer SDK (`optimize_and_run()`)

---

## 🏗 System Architecture

```
Application Layer
        ↓
HeteroWise Runtime SDK
        ↓
Prediction Engine (3x ML Models)
        ↓
Decision Engine
        ↓
Energy Budget Optimizer
        ↓
Thermal Awareness Layer
        ↓
ROCm / CPU / NPU Execution
        ↓
Telemetry Collector
        ↓
Adaptive Refinement Engine
```

Telemetry abstraction layer supports:
- **ROCm-SMI provider** — real hardware
- **Simulated hardware provider** — for testing

---

## 🔄 Workflow

1. Extract workload features (FLOPs, batch, precision, params)
2. Predict energy for CPU, GPU, NPU
3. Rank accelerators by predicted energy
4. Enforce energy budget constraints
5. Execute workload
6. Measure real telemetry (ROCm-SMI)
7. Log and adapt model
8. Reroute under thermal thresholds

---

## 🧪 Project Phases

| Phase | Description |
|-------|-------------|
| Phase 1 | **Synthetic Workload Modeling** — Generated 1,500 physics-inspired workload samples across heterogeneous regions |
| Phase 2 | **ML Training** — Trained RandomForest regressors for CPU, GPU, and NPU energy prediction |
| Phase 3 | **Decision Engine** — Accelerator recommendation via CLI and dashboard |
| Phase 4 | **ROCm Telemetry Validation** — Live GPU power measurement using rocm-smi |
| Phase 5 | **Runtime SDK** — `optimize_and_run()` interface + EnergyBudgetOptimizer |
| Phase 6 | **Thermal Awareness** — Dynamic rerouting under thermal stress |
| Phase 7 | **Telemetry Abstraction** — Hardware-ready telemetry provider interface |

---

## 📦 Repository Structure

```
heterowise/
│
├── phase1_generate_data.py
├── phase2_train_models.py
├── phase3_dashboard.py
├── phase3_cli_demo.py
├── phase4_rocm_telemetry.py
├── heterowise_runtime.py
├── rocm_integration.py
├── models/
├── workload_dataset.csv
├── telemetry_log.csv
└── requirements.txt
```

---


## ⚙ Hardware Setup (TI Code Composer Studio)

Firmware is pre-configured. No code editing required.

### 🔌 Step 1 — Open Firmware in CCS

1. Launch TI Code Composer Studio (CCS)
2. Go to: **File → Open Folder**
3. Select: `VoltGuard/firmware/`
4. Ensure the project appears in Project Explorer

### ⚡ Step 2 — Flash the MCU

1. Connect the MSPM0G3507 LaunchPad via USB
2. Expand the project
3. Right-click: main_project
4. Select: Flash
5. The firmware will be programmed onto the microcontroller
6. Ensure the device exits debug mode and runs normally after flashing

### UART Verification

1. Connect the device via USB
2. Open Device Manager and locate the COM port labeled **Texas Instruments** — note the port number (e.g. `COM6`)
3. Open a serial console (inside CCS or external) and configure it with the COM port you located above:

---

## 🖥 Software Setup (Python Backend)

### Requirements

- Python 3.10+
- AMD GPU (for ROCm telemetry phase)
- Fedora / Ubuntu with `rocm-smi` installed

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install ROCm-SMI (Linux)

```bash
# Fedora
sudo dnf install rocm-smi

# Ubuntu / Debian
sudo apt install rocm-smi
```

### Phase 1 — Generate Dataset

```bash
python phase1_generate_data.py
```

### Phase 2 — Train Models

```bash
python phase2_train_models.py
```

### Phase 3 — Run Decision Engine

```bash
# CLI
python phase3_cli_demo.py

# Dashboard
streamlit run phase3_dashboard.py
```

### Phase 4 — ROCm Telemetry Validation

```bash
# Install (root environment, required for power readings)
sudo -H python3 -m pip install -r phase4_dependencies/requirements.txt

# Run (latency only)
python phase4_rocm_telemetry.py

# Run (with live GPU power telemetry)
sudo python phase4_rocm_telemetry.py
```

### Phase 5 — Runtime SDK Demo

```bash
python heterowise_runtime.py
```
---

```
Port:      COM6  (use the Texas Instruments port from Device Manager)
Baud Rate: 115200
(Keep other settings default)
```

You should see real-time telemetry output.

---

---

## 📊 Model Performance

| Model | R²    | Key Feature        |
|-------|-------|--------------------|
| CPU   | ~0.99 | FLOPs              |
| GPU   | ~0.79 | FLOPs + batch      |
| NPU   | ~0.99 | FLOPs + precision  |

> \* GPU variance reflects real-world telemetry noise; adaptive refinement improves alignment.

---

## 🖥 Optional Hardware Extension

HeteroWise supports an optional **MSPM0-based energy console**:

- Live energy + temperature display
- Budget control via joystick
- UART runtime integration

This module enhances visualization and edge deployment capability but does not control compute routing.

---

## 💡 AMD Technologies Used

- AMD Ryzen CPU (heterogeneous environment)
- AMD Radeon GPU
- ROCm runtime
- ROCm-SMI telemetry interface

HeteroWise enhances the AMD compute ecosystem through intelligent accelerator orchestration.

---

## 💼 Business Model (Concept)

| Tier               | Pricing (INR)               |
| ------------------ | --------------------------- |
| OEM Licensing      | ₹166 – ₹415 per device      |
| Enterprise License | ₹20.75 Lakhs – ₹62.25 Lakhs |
| Developer SDK      | ₹16,517 per year            |

> Initial development estimate: ~₹1.33 Crore

---

*⚡ HeteroWise — Constraint-aware heterogeneous AI orchestration for AMD platforms.*
