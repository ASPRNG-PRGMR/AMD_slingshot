"""
HeteroWise — Phase 3: Decision Engine + Dashboard
Streamlit app for interactive accelerator recommendation.

Run with: streamlit run phase3_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeteroWise — AI Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .recommend-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 2px solid #667eea;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .winner-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base =  Path(__file__).resolve().parent
    model_dir = base / 'models'
    models = {}
    for hw in ['cpu', 'gpu', 'npu']:
        with open(f'{model_dir}/{hw}_model.pkl', 'rb') as f:
            models[hw] = pickle.load(f)
    with open(f'{model_dir}/model_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return models, meta

models, meta = load_models()
FEATURES = meta['features']

HW_COLORS = {
    'CPU': '#3b82f6',   # blue
    'GPU': '#10b981',   # green
    'NPU': '#f59e0b',   # amber
}
HW_TDP = {'CPU': 45, 'GPU': 200, 'NPU': 8}

# ── PREDICTION FUNCTION ────────────────────────────────────────────────────────
def predict_energy(flops, batch, precision_enc, params):
    x = pd.DataFrame([[
        np.log10(flops),
        np.log2(batch),
        precision_enc,
        np.log10(params)
    ]], columns=FEATURES)

    results = {}
    for hw in ['cpu', 'gpu', 'npu']:
        energy_j = float(np.expm1(models[hw].predict(x)[0]))
        results[hw.upper()] = energy_j
    return results

def get_latency_estimate(flops, batch, precision_enc, energy_j, tdp):
    """Back-calculate approximate latency from energy and TDP."""
    return energy_j / tdp

# ── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-header">⚡ HeteroWise</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Heterogeneous Workload Energy Optimizer · AMD CPU · GPU · NPU</div>',
                unsafe_allow_html=True)

st.divider()

# ── SIDEBAR INPUT ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Workload Configuration")
    st.caption("Define your AI workload characteristics")

    st.subheader("Compute Intensity")
    flops_exp = st.slider(
        "FLOPs (log scale)",
        min_value=6.0, max_value=11.7, value=9.0, step=0.1,
        help="Total floating point operations"
    )
    flops = 10 ** flops_exp
    flops_display = f"{flops:.2e}"
    st.caption(f"**{flops_display}** FLOPs")

    batch = st.select_slider(
        "Batch Size",
        options=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        value=16
    )

    st.subheader("Precision")
    precision_label = st.radio(
        "Numerical Precision",
        options=['FP32', 'FP16', 'INT8'],
        index=1,
        horizontal=True,
        help="Lower precision = faster + more energy-efficient on NPU/GPU"
    )
    prec_map = {'FP32': 1.0, 'FP16': 0.5, 'INT8': 0.25}
    precision_enc = prec_map[precision_label]

    st.subheader("Model Size")
    params_exp = st.slider(
        "Parameter Count (log scale)",
        min_value=4.0, max_value=10.0, value=7.0, step=0.1
    )
    params = 10 ** params_exp
    st.caption(f"**{params:.2e}** parameters")

    st.divider()
    st.caption("🧠 Predictions powered by RandomForest regressors trained on 1,500 synthetic workload samples")

# ── MAIN PREDICTION ───────────────────────────────────────────────────────────
energy = predict_energy(flops, batch, precision_enc, params)

# Compute efficiency = performance-per-watt proxy = 1/energy
efficiency = {hw: 1.0 / e for hw, e in energy.items()}
best_hw = min(energy, key=energy.get)
best_energy = energy[best_hw]

# Compute latency estimates
latency = {hw: get_latency_estimate(flops, batch, precision_enc, energy[hw], HW_TDP[hw])
           for hw in energy}

# ── RECOMMENDATION BOX ────────────────────────────────────────────────────────
st.markdown("## 🎯 Recommendation")

rcol1, rcol2 = st.columns([3, 2])

with rcol1:
    # Why this recommendation
    why_map = {
        'CPU': (
            "**CPU** wins for this workload because the compute intensity is low enough "
            "that GPU/NPU startup and idle overhead exceed the CPU's sequential execution cost. "
            "For tiny FLOPs with small batches, the CPU avoids expensive hardware initialization."
        ),
        'GPU': (
            "**GPU** wins for this workload due to massive parallelism. "
            "Large FLOPs and/or large batch sizes allow the GPU to saturate its thousands of cores, "
            "making the high TDP (200W) worth it through extreme throughput."
        ),
        'NPU': (
            "**NPU** wins for this workload as a low-power inference accelerator. "
            "With FP16/INT8 precision and moderate FLOPs, the NPU's 8W TDP creates "
            "dramatically lower energy consumption than the GPU's 200W overhead."
        ),
    }

    st.markdown(f"""
    <div class="recommend-box">
        <div style="font-size:0.9rem; color:#6b7280; margin-bottom:0.5rem">RECOMMENDED ACCELERATOR</div>
        <div class="winner-badge">⚡ {best_hw}</div>
        <div style="margin-top:1rem; color:#374151; font-size:0.95rem">
            Predicted energy: <strong>{best_energy:.4f} J</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(why_map.get(best_hw, ""))

with rcol2:
    # Energy savings vs worst option
    worst_energy = max(energy.values())
    savings_pct = (1 - best_energy / worst_energy) * 100

    st.metric("Energy vs Worst Option", f"{savings_pct:.1f}% less energy")

    second_best = sorted(energy, key=energy.get)[1]
    savings_vs2 = (1 - best_energy / energy[second_best]) * 100
    st.metric(f"vs {second_best}", f"{savings_vs2:.1f}% less energy")

    st.metric("Best Latency Est.", f"{latency[best_hw]*1000:.2f} ms")

st.divider()

# ── DETAILED COMPARISON ───────────────────────────────────────────────────────
st.markdown("## 📊 Accelerator Comparison")

tab1, tab2, tab3 = st.tabs(["Energy Comparison", "Performance-per-Watt", "Workload Profile"])

with tab1:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#f8fafc')

    hw_names = list(energy.keys())
    energies = list(energy.values())
    colors = [HW_COLORS[hw] for hw in hw_names]

    # Bar chart
    bars = ax1.bar(hw_names, energies, color=colors, width=0.5, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Energy (Joules)', fontsize=11)
    ax1.set_title('Predicted Energy Consumption', fontsize=13, fontweight='bold', pad=15)
    ax1.set_facecolor('#ffffff')

    for bar, val in zip(bars, energies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(energies)*0.01,
                 f'{val:.4f}J', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight winner
    winner_idx = hw_names.index(best_hw)
    bars[winner_idx].set_edgecolor('#1a1a2e')
    bars[winner_idx].set_linewidth(3)
    ax1.text(bars[winner_idx].get_x() + bars[winner_idx].get_width()/2.,
             energies[winner_idx] / 2, '★', ha='center', va='center',
             fontsize=20, color='white', fontweight='bold')

    # Latency comparison
    lats = [latency[hw]*1000 for hw in hw_names]
    bars2 = ax2.bar(hw_names, lats, color=colors, width=0.5, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Estimated Latency (ms)', fontsize=11)
    ax2.set_title('Estimated Inference Latency', fontsize=13, fontweight='bold', pad=15)
    ax2.set_facecolor('#ffffff')
    for bar, val in zip(bars2, lats):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(lats)*0.01,
                 f'{val:.2f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')

    eff_vals = [efficiency[hw] for hw in hw_names]
    # Normalize to best = 100
    max_eff = max(eff_vals)
    eff_norm = [e/max_eff * 100 for e in eff_vals]

    bars = ax.barh(hw_names, eff_norm, color=colors, height=0.4, edgecolor='white', linewidth=2)
    ax.set_xlabel('Relative Efficiency (higher = better)', fontsize=11)
    ax.set_title('Performance-per-Watt Score\n(Efficiency = 1/Energy, normalized to best=100)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 115)

    for bar, val, raw in zip(bars, eff_norm, eff_vals):
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=12, fontweight='bold')

    # Highlight winner
    winner_idx2 = hw_names.index(best_hw)
    bars[winner_idx2].set_edgecolor('#1a1a2e')
    bars[winner_idx2].set_linewidth(3)

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    # Show a radar-style comparison table
    st.markdown("#### Summary Table")
    summary_df = pd.DataFrame({
        'Accelerator': hw_names,
        'Energy (J)': [f"{energy[hw]:.4f}" for hw in hw_names],
        'Latency (ms)': [f"{latency[hw]*1000:.2f}" for hw in hw_names],
        'TDP (W)': [HW_TDP[hw] for hw in hw_names],
        'Efficiency Score': [f"{e:.1f}" for e in eff_norm],
        'Recommended': ['✅ YES' if hw == best_hw else '—' for hw in hw_names],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("#### Your Workload Profile")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.info(f"""
        **Compute**
        - FLOPs: `{flops:.2e}`
        - Batch Size: `{batch}`
        - Precision: `{precision_label}` (enc: {precision_enc})
        """)
    with pcol2:
        st.info(f"""
        **Model**
        - Parameters: `{params:.2e}`
        - Compute/Param ratio: `{flops/params:.1f}`
        - Workload class: `{"Large-batch training" if batch > 32 and flops > 1e10 else "Inference" if batch <= 16 else "Medium batch inference"}`
        """)

    # Feature importance context
    st.markdown("#### What drives predictions most?")
    fi_data = {
        'Feature': ['log(FLOPs)', 'Precision', 'log(Batch)', 'log(Params)'],
        'CPU Importance': [0.967, 0.025, 0.006, 0.002],
        'GPU Importance': [0.694, 0.063, 0.164, 0.080],
        'NPU Importance': [0.783, 0.215, 0.000, 0.002],
    }
    fi_df = pd.DataFrame(fi_data)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    x = np.arange(len(fi_data['Feature']))
    w = 0.25
    ax2.bar(x - w, fi_df['CPU Importance'], w, label='CPU', color=HW_COLORS['CPU'], alpha=0.85)
    ax2.bar(x,     fi_df['GPU Importance'], w, label='GPU', color=HW_COLORS['GPU'], alpha=0.85)
    ax2.bar(x + w, fi_df['NPU Importance'], w, label='NPU', color=HW_COLORS['NPU'], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fi_data['Feature'])
    ax2.set_ylabel('Importance')
    ax2.set_title('Model Feature Importance by Accelerator', fontweight='bold')
    ax2.legend()
    ax2.set_facecolor('#ffffff')
    fig2.patch.set_facecolor('#f8fafc')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("""
**HeteroWise** · Phase 3 Complete · Advisory layer only — not runtime dispatch  
Built with Python · Scikit-learn · Streamlit · Synthetic AMD CPU/GPU/NPU simulation data
""")
