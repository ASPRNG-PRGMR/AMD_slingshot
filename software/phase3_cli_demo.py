"""
HeteroWise — CLI Demo (Phase 3 without Streamlit)
Runs the full decision engine and prints recommendations.
"""

import numpy as np
import pandas as pd
import pickle
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')

HW_TDP = {'CPU': 45, 'GPU': 200, 'NPU': 8}
HW_COLORS = {'CPU': '\033[94m', 'GPU': '\033[92m', 'NPU': '\033[93m'}
RESET = '\033[0m'
BOLD  = '\033[1m'

def load_models():
    models = {}
    for hw in ['cpu', 'gpu', 'npu']:
        with open(f'{MODEL_DIR}/{hw}_model.pkl', 'rb') as f:
            models[hw.upper()] = pickle.load(f)
    with open(f'{MODEL_DIR}/model_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return models, meta

def predict(models, features, flops, batch, precision_enc, params):
    x = pd.DataFrame([[
        np.log10(flops),
        np.log2(batch),
        precision_enc,
        np.log10(params)
    ]], columns=features)
    return {hw: float(np.expm1(m.predict(x)[0])) for hw, m in models.items()}

def bar(val, max_val, width=30, char='█'):
    filled = int((val / max_val) * width)
    return char * filled + '░' * (width - filled)

def recommend(energy, latency):
    best = min(energy, key=energy.get)
    worst = max(energy, key=energy.get)
    savings = (1 - energy[best] / energy[worst]) * 100

    print(f"\n{'─'*60}")
    print(f"  {'ENERGY CONSUMPTION':30s}  {'LATENCY':12s}  {'EFFICIENCY':10s}")
    print(f"{'─'*60}")

    max_e = max(energy.values())
    for hw in ['CPU', 'GPU', 'NPU']:
        e = energy[hw]
        l = latency[hw] * 1000
        eff = (1/e) / (1/min(energy.values())) * 100
        marker = f" ◄ BEST" if hw == best else ""
        c = HW_COLORS[hw]
        print(f"  {c}{hw}{RESET}  {bar(e, max_e, 20):<22}  "
              f"{e:.4f} J    {l:.2f} ms    {eff:.1f}%{BOLD if hw==best else ''}{marker}{RESET}")
    print(f"{'─'*60}")
    print(f"\n  {BOLD}⚡ RECOMMENDATION: {HW_COLORS[best]}{best}{RESET}{BOLD}{RESET}")
    print(f"  Saves {savings:.1f}% energy vs worst option\n")
    return best

def run_demo():
    print(f"\n{BOLD}{'='*60}")
    print("  ⚡ HeteroWise — AI Energy Optimizer  (CLI Demo)")
    print(f"{'='*60}{RESET}")

    models, meta = load_models()
    features = meta['features']
    prec_map = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}

    test_cases = [
        {
            'name': '🔬 Tiny Edge Inference',
            'desc': 'MobileNet-class, real-time on edge device',
            'flops': 5e8, 'batch': 1, 'precision': 'INT8', 'params': 4e6
        },
        {
            'name': '📱 Mobile App Inference',
            'desc': 'GPT-2 small inference, FP16',
            'flops': 5e9, 'batch': 4, 'precision': 'FP16', 'params': 1.2e8
        },
        {
            'name': '🖥️  Server Batch Inference',
            'desc': 'ResNet-50 batch processing, FP32',
            'flops': 8e10, 'batch': 64, 'precision': 'FP32', 'params': 2.5e7
        },
        {
            'name': '🏋️  Large Model Training',
            'desc': 'Transformer fine-tuning, large batch',
            'flops': 3e11, 'batch': 128, 'precision': 'FP32', 'params': 7e8
        },
        {
            'name': '🚀 Hyperscale Training',
            'desc': 'LLM pre-training step',
            'flops': 5e11, 'batch': 256, 'precision': 'FP32', 'params': 7e9
        },
    ]

    for tc in test_cases:
        print(f"\n  {BOLD}{tc['name']}{RESET}")
        print(f"  {tc['desc']}")
        print(f"  FLOPs={tc['flops']:.1e}  Batch={tc['batch']}  "
              f"Precision={tc['precision']}  Params={tc['params']:.1e}")

        enc = prec_map[tc['precision']]
        energy = predict(models, features, tc['flops'], tc['batch'], enc, tc['params'])
        latency = {hw: e / HW_TDP[hw] for hw, e in energy.items()}
        recommend(energy, latency)

    # Interactive mode
    print(f"\n{'='*60}")
    print("  🎮 INTERACTIVE MODE — Enter your workload")
    print(f"{'='*60}")
    try:
        flops_e = float(input("\n  FLOPs (e.g. 1e9): ") or "1e9")
        batch   = int(input("  Batch size (1/2/4/8/16/32/64/128/256): ") or "8")
        prec    = input("  Precision (FP32/FP16/INT8) [FP16]: ").strip() or "FP16"
        params_e= float(input("  Param count (e.g. 1e7): ") or "1e7")

        enc = prec_map.get(prec.upper(), 0.5)
        print(f"\n  Running prediction for your workload...")
        energy  = predict(models, features, flops_e, batch, enc, params_e)
        latency = {hw: e / HW_TDP[hw] for hw, e in energy.items()}
        recommend(energy, latency)

    except (EOFError, KeyboardInterrupt):
        print("\n  [Skipping interactive mode]")

    print(f"\n{'='*60}")
    print("  ✅ HeteroWise demo complete!")
    print(f"  📊 Run: streamlit run phase3_dashboard.py  for the full UI")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    run_demo()
