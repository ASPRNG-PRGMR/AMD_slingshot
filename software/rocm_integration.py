"""
HeteroWise — ROCm Integration Layer
Level 1: Architectural Integration

This module sits between the Decision Engine and the actual runtime,
translating recommendations into hardware-specific deployment strategies.

Architecture:
  Frontend
      ↓
  Prediction Engine  (phase2_train_models.py)
      ↓
  Decision Engine    (phase3_dashboard.py)
      ↓
  Deployment Strategy Generator  ← THIS FILE
      ↓
  [CPU Runtime] OR [ROCm GPU Runtime] OR [NPU SDK Runtime]
"""

from dataclasses import dataclass, field
from typing import Optional
import json

# ── LEVEL 1: ARCHITECTURAL — Data Contracts ───────────────────────────────────

@dataclass
class WorkloadProfile:
    """Input contract for any workload entering HeteroWise."""
    flops: float
    batch_size: int
    precision: str          # 'FP32' | 'FP16' | 'INT8'
    param_count: float
    task_type: str = 'inference'  # 'inference' | 'training' | 'fine_tuning'
    latency_critical: bool = False
    max_energy_budget_j: Optional[float] = None


@dataclass
class EnergyPrediction:
    """Structured output from the ML prediction engine."""
    cpu_energy_j: float
    gpu_energy_j: float
    npu_energy_j: float
    cpu_latency_ms: float
    gpu_latency_ms: float
    npu_latency_ms: float
    recommended_hw: str     # 'CPU' | 'GPU' | 'NPU'
    confidence: float       # 0.0–1.0


@dataclass
class DeploymentStrategy:
    """
    Full deployment strategy output — what the system gives to a developer.
    This is the bridge between ML recommendation and actual runtime code.
    """
    target_hardware: str                    # 'CPU' | 'ROCm_GPU' | 'NPU'
    precision_recommendation: str           # 'FP32' | 'FP16' | 'INT8'
    batch_recommendation: int
    mixed_precision: bool
    estimated_energy_j: float
    estimated_latency_ms: float
    runtime_config: dict = field(default_factory=dict)
    deployment_commands: list = field(default_factory=list)
    rocm_flags: list = field(default_factory=list)
    explanation: str = ""


# ── LEVEL 2: DEPLOYMENT STRATEGY — ROCm-Aware Generator ──────────────────────

class DeploymentStrategyGenerator:
    """
    Translates a HeteroWise EnergyPrediction into a concrete deployment
    strategy with ROCm-specific configuration, environment flags,
    and ready-to-run code templates.

    This is the key architectural layer that makes HeteroWise actionable,
    not just advisory.
    """

    # ROCm environment variables for different optimization profiles
    ROCM_BASE_ENV = {
        'HSA_ENABLE_SDMA': '0',          # Disable SDMA for better PCIe transfer
        'GPU_MAX_HW_QUEUES': '8',         # Maximize hardware queue depth
        'ROCM_HOME': '/opt/rocm',
    }

    ROCM_FP16_ENV = {
        **ROCM_BASE_ENV,
        'MIOPEN_FIND_ENFORCE': 'ENFORCE_DB',  # Use cached kernel configs
        'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM': '1',
    }

    ROCM_INT8_ENV = {
        **ROCM_BASE_ENV,
        'ROCBLAS_TENSILE_LIBPATH': '/opt/rocm/lib/rocblas/library',
        'PYTORCH_MIOPEN_SUGGEST_NHWC': '1',   # NHWC format faster for INT8 conv
    }

    NPU_ENV = {
        'XLNX_VART_FIRMWARE': '/lib/firmware/amdnpu',
        'AMD_OOB_NPU_FIRMWARE': '1',
        'XRT_INI_PATH': '/etc/xrt.ini',
    }

    def generate(self, profile: WorkloadProfile, prediction: EnergyPrediction) -> DeploymentStrategy:
        """Main entry point — generate a full deployment strategy."""
        hw = prediction.recommended_hw

        if hw == 'GPU':
            return self._rocm_strategy(profile, prediction)
        elif hw == 'NPU':
            return self._npu_strategy(profile, prediction)
        else:
            return self._cpu_strategy(profile, prediction)

    # ── ROCm GPU Strategy ─────────────────────────────────────────────────────

    def _rocm_strategy(self, profile: WorkloadProfile, pred: EnergyPrediction) -> DeploymentStrategy:
        """
        Generate a ROCm-optimized GPU deployment strategy.
        Includes: precision tuning, MIOpen hints, PyTorch ROCm config.
        """
        # Precision upgrade recommendation
        rec_precision = profile.precision
        mixed = False

        if profile.precision == 'FP32' and profile.task_type == 'inference':
            rec_precision = 'FP16'   # Safe for inference, 2x throughput
            mixed = True

        # ROCm environment flags based on precision
        env_flags = self.ROCM_INT8_ENV if rec_precision == 'INT8' else self.ROCM_FP16_ENV
        rocm_flags = [f"{k}={v}" for k, v in env_flags.items()]

        # Batch size tuning for ROCm (power-of-2, multiple of warp size 64)
        rec_batch = self._tune_batch_for_gpu(profile.batch_size)

        runtime_config = {
            'backend': 'rocm',
            'device': 'cuda',   # PyTorch uses 'cuda' API even for ROCm
            'precision': rec_precision,
            'batch_size': rec_batch,
            'use_amp': mixed,                    # Automatic Mixed Precision
            'miopen_exhaustive_search': False,   # True only for production
            'rocm_version_required': '5.7+',
            'amd_gpu_targets': ['gfx1030', 'gfx1100', 'gfx1101'],  # RDNA2/3
        }

        # Deployable code snippet
        commands = self._rocm_pytorch_snippet(rec_precision, rec_batch, mixed)

        energy_j = pred.gpu_energy_j
        if mixed and profile.precision == 'FP32':
            energy_j *= 0.55  # FP16 typically ~45% energy reduction

        explanation = (
            f"ROCm GPU selected: high FLOPs ({profile.flops:.1e}) with large batch ({rec_batch}) "
            f"saturates GPU parallelism. "
        )
        if mixed:
            explanation += f"Precision upgraded FP32→FP16 via AMP: ~45% energy saving on GPU. "
        explanation += f"ROCm 5.7+ with MIOpen kernel caching enabled."

        return DeploymentStrategy(
            target_hardware='ROCm_GPU',
            precision_recommendation=rec_precision,
            batch_recommendation=rec_batch,
            mixed_precision=mixed,
            estimated_energy_j=energy_j,
            estimated_latency_ms=pred.gpu_latency_ms,
            runtime_config=runtime_config,
            deployment_commands=commands,
            rocm_flags=rocm_flags,
            explanation=explanation,
        )

    def _rocm_pytorch_snippet(self, precision: str, batch: int, amp: bool) -> list:
        lines = [
            "# HeteroWise ROCm Deployment — Generated Configuration",
            "import torch",
            "",
            "# ROCm device setup",
            "assert torch.cuda.is_available(), 'ROCm not detected. Install: https://rocm.docs.amd.com'",
            "device = torch.device('cuda')  # ROCm uses CUDA API",
            "model = model.to(device)",
        ]
        if precision == 'FP16':
            lines += [
                "",
                "# FP16 inference (HeteroWise recommendation)",
                "model = model.half()",
                "with torch.no_grad():",
                f"    inputs = inputs.half().to(device)",
                f"    outputs = model(inputs)  # batch={batch}",
            ]
        elif amp:
            lines += [
                "",
                "# AMP (Auto Mixed Precision) for training/inference",
                "scaler = torch.cuda.amp.GradScaler()",
                "with torch.cuda.amp.autocast():",
                f"    outputs = model(inputs.to(device))  # batch={batch}",
            ]
        else:
            lines += [
                "",
                f"with torch.no_grad():",
                f"    outputs = model(inputs.to(device))  # batch={batch}",
            ]
        lines += [
            "",
            "# ROCm environment (set before launching):",
            "# export HSA_ENABLE_SDMA=0",
            "# export GPU_MAX_HW_QUEUES=8",
            "# export MIOPEN_FIND_ENFORCE=ENFORCE_DB",
        ]
        return lines

    # ── NPU Strategy ──────────────────────────────────────────────────────────

    def _npu_strategy(self, profile: WorkloadProfile, pred: EnergyPrediction) -> DeploymentStrategy:
        """
        Generate an AMD Ryzen AI NPU deployment strategy.
        Uses Vitis AI / ONNX Runtime with VitisAI EP.
        """
        # NPU strongly prefers INT8 or FP16
        rec_precision = 'INT8' if profile.precision in ('INT8', 'FP16') else 'FP16'
        mixed = (profile.precision == 'FP32')

        runtime_config = {
            'backend': 'ryzen_ai_npu',
            'execution_provider': 'VitisAIExecutionProvider',
            'config_file': 'vaip_config.json',
            'precision': rec_precision,
            'batch_size': min(profile.batch_size, 4),  # NPU pipeline limited
            'cache_dir': './npu_kernel_cache',
            'ryzen_ai_sdk_required': '1.1+',
            'npu_targets': ['PHX', 'HPT'],  # Phoenix, Hawk Point APUs
        }

        commands = [
            "# HeteroWise NPU Deployment — Ryzen AI SDK",
            "# Step 1: Install Ryzen AI SDK",
            "# https://ryzenai.docs.amd.com/en/latest/inst.html",
            "",
            "import onnxruntime as ort",
            "",
            "# Step 2: Load with VitisAI Execution Provider",
            "providers = [",
            "    ('VitisAIExecutionProvider', {",
            "        'config_file': 'vaip_config.json',",
            "        'cacheDir': './npu_kernel_cache',",
            "        'cacheKey': 'heterowise_npu_model',",
            "    })",
            "]",
            "sess = ort.InferenceSession('model.onnx', providers=providers)",
            "",
            "# Step 3: Run inference",
            f"# Recommended batch: {min(profile.batch_size, 4)} (NPU pipeline optimized)",
            "outputs = sess.run(None, {'input': input_array})",
            "",
            "# Step 4: Export PyTorch model to ONNX first",
            "# torch.onnx.export(model, dummy_input, 'model.onnx',",
            "#     opset_version=13, input_names=['input'], output_names=['output'])",
        ]

        env_flags = [f"{k}={v}" for k, v in self.NPU_ENV.items()]

        energy_j = pred.npu_energy_j
        if rec_precision == 'INT8' and profile.precision == 'FP32':
            energy_j *= 0.4  # INT8 quantization ~60% energy reduction on NPU

        explanation = (
            f"NPU selected: low-power inference accelerator (8W TDP) optimal for "
            f"batch={min(profile.batch_size,4)} with {rec_precision} precision. "
            f"AMD Ryzen AI NPU via ONNX Runtime + VitisAI EP. "
        )
        if mixed:
            explanation += "FP32→INT8 quantization recommended: ~60% NPU energy reduction."

        return DeploymentStrategy(
            target_hardware='NPU',
            precision_recommendation=rec_precision,
            batch_recommendation=min(profile.batch_size, 4),
            mixed_precision=mixed,
            estimated_energy_j=energy_j,
            estimated_latency_ms=pred.npu_latency_ms,
            runtime_config=runtime_config,
            deployment_commands=commands,
            rocm_flags=env_flags,
            explanation=explanation,
        )

    # ── CPU Strategy ──────────────────────────────────────────────────────────

    def _cpu_strategy(self, profile: WorkloadProfile, pred: EnergyPrediction) -> DeploymentStrategy:
        """
        AMD Ryzen CPU deployment — uses PyTorch CPU + OpenMP threading.
        """
        import os
        cpu_threads = min(profile.batch_size * 2, 16)

        runtime_config = {
            'backend': 'cpu',
            'precision': profile.precision,
            'batch_size': profile.batch_size,
            'num_threads': cpu_threads,
            'use_openmp': True,
            'ryzen_target': 'Zen4+',
        }

        commands = [
            "# HeteroWise CPU Deployment — AMD Ryzen Optimized",
            "import torch",
            f"torch.set_num_threads({cpu_threads})",
            "torch.set_num_interop_threads(2)",
            "",
            "# CPU inference",
            "model.eval()",
            "with torch.no_grad():",
            f"    outputs = model(inputs)  # batch={profile.batch_size}",
            "",
            "# Environment:",
            f"# export OMP_NUM_THREADS={cpu_threads}",
            "# export GOMP_CPU_AFFINITY='0-15'  # Pin to Zen4 cores",
        ]

        explanation = (
            f"CPU selected: workload is compute-light enough ({profile.flops:.1e} FLOPs, "
            f"batch={profile.batch_size}) that GPU/NPU startup overhead exceeds execution cost. "
            f"AMD Ryzen Zen4 with {cpu_threads} OpenMP threads recommended."
        )

        return DeploymentStrategy(
            target_hardware='CPU',
            precision_recommendation=profile.precision,
            batch_recommendation=profile.batch_size,
            mixed_precision=False,
            estimated_energy_j=pred.cpu_energy_j,
            estimated_latency_ms=pred.cpu_latency_ms,
            runtime_config=runtime_config,
            deployment_commands=commands,
            rocm_flags=[f"OMP_NUM_THREADS={cpu_threads}", "GOMP_CPU_AFFINITY=0-15"],
            explanation=explanation,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tune_batch_for_gpu(self, batch: int) -> int:
        """Round batch to nearest power of 2, minimum 32 for ROCm efficiency."""
        if batch < 32:
            return 32
        # Round to nearest power of 2
        p = 1
        while p < batch:
            p *= 2
        return min(p, 256)


# ── LEVEL 3: FUTURE EXTENSION — ROCm Telemetry Stub ──────────────────────────

class ROCmTelemetryCollector:
    """
    FUTURE EXTENSION: Real hardware benchmarking via ROCm SMI.

    When running on actual AMD GPU hardware, this replaces synthetic
    predictions with real telemetry, enabling continuous model retraining.

    Not active in prototype — requires:
      - AMD GPU with ROCm 5.7+
      - rocm-smi installed (/opt/rocm/bin/rocm-smi)
      - sudo access for power readings
    """

    def __init__(self):
        self.rocm_available = self._check_rocm()

    def _check_rocm(self) -> bool:
        import subprocess
        try:
            result = subprocess.run(
                ['rocm-smi', '--version'],
                capture_output=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def collect_benchmark(self, model_fn, inputs, device_id: int = 0) -> dict:
        """
        STUB: In production, runs model and reads live power/latency from ROCm SMI.

        Returns synthetic data in prototype mode.
        """
        if not self.rocm_available:
            return self._synthetic_telemetry()

        # Production path (requires real AMD GPU):
        # import subprocess, time
        # start_power = self._read_rocm_power(device_id)
        # t0 = time.perf_counter()
        # _ = model_fn(inputs)
        # t1 = time.perf_counter()
        # end_power = self._read_rocm_power(device_id)
        # return {'latency_ms': (t1-t0)*1000, 'power_w': (start_power+end_power)/2, ...}

        return self._synthetic_telemetry()

    def _read_rocm_power(self, device_id: int) -> float:
        """Read instantaneous GPU power via rocm-smi."""
        import subprocess
        result = subprocess.run(
            ['rocm-smi', f'--device={device_id}', '--showpower', '--json'],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        return float(data['card0']['Average Graphics Package Power (W)'])

    def _synthetic_telemetry(self) -> dict:
        return {
            'source': 'synthetic',
            'rocm_available': False,
            'note': 'Install ROCm 5.7+ and AMD GPU for live telemetry',
            'rocm_install': 'https://rocm.docs.amd.com/en/latest/deploy/linux/index.html',
        }

    def get_retraining_sample(self, profile: 'WorkloadProfile', real_energy_j: float,
                               real_latency_ms: float) -> dict:
        """
        FUTURE: Feed real measurements back to retrain the ML models.
        This closes the loop: prototype → production telemetry → better predictions.
        """
        return {
            'flops': profile.flops,
            'batch_size': profile.batch_size,
            'precision_enc': {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}[profile.precision],
            'param_count': profile.param_count,
            'measured_energy_j': real_energy_j,
            'measured_latency_ms': real_latency_ms,
            'source': 'rocm_telemetry',
        }


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pickle
    import numpy as np
    import pandas as pd

    print("\n" + "="*65)
    print("  HeteroWise — ROCm Integration Layer Demo")
    print("="*65)

    # Load trained models
    models = {}
    for hw in ['cpu', 'gpu', 'npu']:
        with open(f'models/{hw}_model.pkl', 'rb') as f:
            models[hw.upper()] = pickle.load(f)
    with open('models/model_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    features = meta['features']

    HW_TDP = {'CPU': 45, 'GPU': 200, 'NPU': 8}

    def run_pipeline(name, flops, batch, precision, params):
        print(f"\n{'─'*65}")
        print(f"  Workload: {name}")

        prec_enc = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}[precision]
        x = pd.DataFrame([[np.log10(flops), np.log2(batch), prec_enc, np.log10(params)]],
                         columns=features)

        energies = {hw: float(np.expm1(m.predict(x)[0])) for hw, m in models.items()}
        best_hw = min(energies, key=energies.get)

        pred = EnergyPrediction(
            cpu_energy_j=energies['CPU'], gpu_energy_j=energies['GPU'], npu_energy_j=energies['NPU'],
            cpu_latency_ms=energies['CPU']/HW_TDP['CPU']*1000,
            gpu_latency_ms=energies['GPU']/HW_TDP['GPU']*1000,
            npu_latency_ms=energies['NPU']/HW_TDP['NPU']*1000,
            recommended_hw=best_hw, confidence=0.92
        )

        profile = WorkloadProfile(flops=flops, batch_size=batch, precision=precision,
                                   param_count=params)

        # Generate deployment strategy
        gen = DeploymentStrategyGenerator()
        strategy = gen.generate(profile, pred)

        print(f"\n  ► Target Hardware:    {strategy.target_hardware}")
        print(f"  ► Precision:          {strategy.precision_recommendation}"
              f"{'  (upgraded)' if strategy.mixed_precision else ''}")
        print(f"  ► Batch:              {strategy.batch_recommendation}")
        print(f"  ► Est. Energy:        {strategy.estimated_energy_j:.4f} J")
        print(f"  ► Est. Latency:       {strategy.estimated_latency_ms:.2f} ms")
        print(f"\n  Explanation: {strategy.explanation}")

        print(f"\n  ROCm/Runtime Flags:")
        for flag in strategy.rocm_flags[:4]:
            print(f"    export {flag}")

        print(f"\n  Deployment Code:")
        for line in strategy.deployment_commands[:8]:
            print(f"    {line}")
        if len(strategy.deployment_commands) > 8:
            print(f"    ... (+{len(strategy.deployment_commands)-8} more lines)")

    run_pipeline("Tiny INT8 inference",   flops=5e8,  batch=1,   precision='INT8',  params=4e6)
    run_pipeline("Medium FP16 inference", flops=5e9,  batch=8,   precision='FP16',  params=1e8)
    run_pipeline("Large FP32 training",   flops=3e11, batch=128, precision='FP32',  params=7e8)

    # Show telemetry stub
    print(f"\n{'─'*65}")
    print("  ROCm Telemetry Status:")
    telemetry = ROCmTelemetryCollector()
    result = telemetry.collect_benchmark(None, None)
    print(f"    ROCm detected: {result['rocm_available']}")
    print(f"    Mode: {result['source']}")
    print(f"    Note: {result['note']}")
    print(f"\n{'='*65}\n")
