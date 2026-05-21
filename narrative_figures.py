#!/usr/bin/env python3
"""
Three-figure narrative for TFG presentation:
  Fig 1 — Per-patient delta boxplots       "Effect is consistent within patient"
  Fig 2 — LOPO vs Within-patient accuracy  "Generalization fails"
  Fig 3 — Alpha vs Beta scatter            "Two clusters / subtypes"
"""

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import ranksums, pearsonr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.2,
})

# ── CONFIG ───────────────────────────────────────────────────────────────────
FILTERED_DIR = r'C:\Users\arana\OneDrive - UIC\MATLAB\r8gmbtv7w2-3\Filtered Data'

FS          = 500
WIN_SAMPLES = int(4.5 * FS)
STRIDE_SAMP = int(1.0 * FS)
EEG_COLS    = {'F3': 4, 'F4': 5, 'Fz': 16}
LABEL_COL   = 60
BANDS       = {'Delta':(1,4), 'Theta':(4,8), 'Alpha':(8,13), 'Beta':(13,30)}
USE_COLS    = sorted(set(EEG_COLS.values()) | {LABEL_COL})
_COL_MAP    = {orig: i for i, orig in enumerate(USE_COLS)}
EEG_INDICES = [_COL_MAP[v] for v in EEG_COLS.values()]
LBL_I       = _COL_MAP[LABEL_COL]

PATIENTS = {
    '001': [os.path.join(FILTERED_DIR, '001', f'task_{t}.txt') for t in range(1,5)],
    '003': [os.path.join(FILTERED_DIR, '003', f'task_{t}.txt') for t in range(1,5)],
    '004': [os.path.join(FILTERED_DIR, '004', f'task_{t}.txt') for t in range(1,6)],
    '006': [os.path.join(FILTERED_DIR, '006', f'task_{t}.txt') for t in range(1,5)],
    '007': [os.path.join(FILTERED_DIR, '007', f'task_{t}.txt') for t in range(1,5)],
    '008': ([os.path.join(FILTERED_DIR, '008', 'OFF_1', f'task_{t}.txt') for t in range(1,6)] +
            [os.path.join(FILTERED_DIR, '008', 'OFF_2', f'task_{t}.txt') for t in range(1,5)]),
    # '009' excluded — severe class imbalance
    '010': [os.path.join(FILTERED_DIR, '010', f'task_{t}.txt') for t in range(1,5)],
    '011': [os.path.join(FILTERED_DIR, '011', f'task_{t}.txt') for t in range(1,5)],
    '012': [os.path.join(FILTERED_DIR, '012', f'task_{t}.txt') for t in range(1,5)],
}

# Pre-computed results (from multi_patient_pipeline.py, n=9 patients — P009 excluded)
WITHIN_ACC = {'001':0.809,'003':0.816,'004':0.798,'006':0.663,'007':0.662,
              '008':0.736,'010':0.641,'011':0.729,'012':0.645}
LOPO_ACC   = {'001':0.578,'003':0.224,'004':0.664,'006':0.474,'007':0.500,
              '008':0.603,'010':0.500,'011':0.547,'012':0.402}

# Cohen's d mean across F3/F4/Fz (non-overlapping windows, n=9 patients — P009 excluded)
ALPHA_D = {'001':-0.226,'003':0.418,'004':0.071,'006':-0.064,'007':0.032,
           '008':0.286,'010':-0.070,'011':0.160,'012':0.495}
BETA_D  = {'001':0.152,'003':0.338,'004':-0.224,'006':-0.253,'007':-0.146,
           '008':0.385,'010':-0.059,'011':0.110,'012':0.338}

SUBTYPE_A = ['003','008','011','012']
SUBTYPE_B = ['006','010']
COL_A = '#E76F51'; COL_B = '#457B9D'

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def cohens_d(a, b):
    s = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0

def load_patient(file_list):
    chunks = []
    for fp in file_list:
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp, header=None, usecols=USE_COLS, on_bad_lines='skip')
        chunks.append(df[USE_COLS].values.astype(np.float64))
    return np.vstack(chunks) if chunks else None

def delta_windows(data):
    """Returns per-window delta power (mean F3/F4/Fz) and labels."""
    n = len(data)
    pwr, lbl = [], []
    for start in range(0, n - WIN_SAMPLES + 1, STRIDE_SAMP):
        chunk = data[start: start + WIN_SAMPLES]
        label = 1 if np.mean(chunk[:, LBL_I]) >= 0.5 else 0
        bp = []
        for col_idx in EEG_INDICES:
            sig = chunk[:, col_idx]
            f, pxx = scipy.signal.welch(sig, fs=FS, nperseg=FS, noverlap=FS//2)
            mask = (f >= 1) & (f <= 4)
            bp.append(10 * np.log10(np.mean(pxx[mask]) + 1e-12))
        pwr.append(np.mean(bp))
        lbl.append(label)
    return np.array(pwr), np.array(lbl)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading delta power per patient...")
delta_data = {}
for pid, files in PATIENTS.items():
    data = load_patient(files)
    if data is None: continue
    pwr, lbl = delta_windows(data)
    delta_data[pid] = (pwr, lbl)
    print(f"  P{pid}  Normal={np.sum(lbl==0)}  FoG={np.sum(lbl==1)}")

pids = sorted(delta_data.keys())

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Per-patient delta boxplots
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: per-patient delta boxplots...")
fig1, axes = plt.subplots(3, 3, figsize=(18, 12))
fig1.suptitle("Figure 1 — Frontal Delta Power: Normal vs FoG per Patient\n"
              "(mean across F3 / F4 / Fz,  4.5 s windows)", fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, pid in enumerate(pids):
    ax = axes[i]
    pwr, lbl = delta_data[pid]
    norm_p = pwr[lbl == 0]
    fog_p  = pwr[lbl == 1]

    subtype = 'A' if pid in SUBTYPE_A else ('B' if pid in SUBTYPE_B else '?')
    edge_col = COL_A if subtype == 'A' else COL_B

    bp = ax.boxplot([norm_p, fog_p], positions=[1, 2], widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    bp['boxes'][0].set_facecolor('#AEC6E8'); bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor('#F4A582'); bp['boxes'][1].set_alpha(0.8)

    # p-value and Cohen's d
    _, p_val = ranksums(norm_p, fog_p)
    d_val    = cohens_d(fog_p, norm_p)
    y_top    = max(np.percentile(norm_p, 95), np.percentile(fog_p, 95))
    y_ann    = y_top + 0.4
    ax.plot([1, 2], [y_ann, y_ann], 'k-', linewidth=1)
    ax.text(1.5, y_ann + 0.15,
            f"{sig_stars(p_val)}  d={d_val:+.2f}",
            ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks([1, 2]); ax.set_xticklabels(['Normal', 'FoG'])
    ax.set_title(f'Patient {pid}  [Subtype {subtype}]', fontweight='bold',
                 color=edge_col)
    ax.set_ylabel('Delta power (dB)') if i % 4 == 0 else None
    ax.set_xlim(0.4, 2.6)

    # Shade box border by subtype
    for spine in ax.spines.values():
        spine.set_edgecolor(edge_col); spine.set_linewidth(1.8)

handles = [
    mpatches.Patch(facecolor='#AEC6E8', label='Normal'),
    mpatches.Patch(facecolor='#F4A582', label='FoG'),
    Line2D([0],[0], color=COL_A, linewidth=2.5, label='Subtype A (alpha/beta ↑)'),
    Line2D([0],[0], color=COL_B, linewidth=2.5, label='Subtype B (alpha/beta ↓)'),
]
fig1.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
            bbox_to_anchor=(0.5, -0.01), frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])
p1 = r'C:\Users\arana\OneDrive - UIC\MATLAB\narrative_fig1_per_patient_delta.png'
fig1.savefig(p1, dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved -> {p1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — LOPO vs Within-patient bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: LOPO vs within-patient...")
fig2, ax = plt.subplots(figsize=(11, 6))
fig2.suptitle("Figure 2 — Subject-Specific Models vs Cross-Patient Generalization (LOPO)\n"
              "Random Forest, frontal band power features (F3/F4/Fz)",
              fontsize=13, fontweight='bold')

x   = np.arange(len(pids))
w   = 0.35
within_vals = [WITHIN_ACC[p] for p in pids]
lopo_vals   = [LOPO_ACC[p]   for p in pids]

bars_w = ax.bar(x - w/2, within_vals, w, label='Within-patient (5-fold CV)',
                color='#2A9D8F', alpha=0.85, zorder=3)
bars_l = ax.bar(x + w/2, lopo_vals,   w, label='LOPO (cross-patient)',
                color='#E9C46A', alpha=0.85, zorder=3)

# Value labels on bars
for bar in bars_w:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars_l:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

# Chance line
ax.axhline(0.5, color='#E63946', linestyle='--', linewidth=1.8,
           label='Chance level (0.50)', zorder=4)

# Mean lines
ax.axhline(np.mean(within_vals), color='#2A9D8F', linestyle=':',
           linewidth=1.5, alpha=0.7, label=f'Mean within = {np.mean(within_vals):.2f}')
ax.axhline(np.mean(lopo_vals), color='#E9C46A', linestyle=':',
           linewidth=1.5, alpha=0.7, label=f'Mean LOPO = {np.mean(lopo_vals):.2f}')

# Delta annotation for each patient
for i, pid in enumerate(pids):
    gain = WITHIN_ACC[pid] - LOPO_ACC[pid]
    mid  = (WITHIN_ACC[pid] + LOPO_ACC[pid]) / 2
    ax.annotate('', xy=(x[i] + w/2, LOPO_ACC[pid] + 0.01),
                xytext=(x[i] - w/2, WITHIN_ACC[pid] - 0.01),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([f'P{p}' for p in pids], fontsize=11)
ax.set_ylim(0, 1.08); ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Patient', fontsize=12)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
p2 = r'C:\Users\arana\OneDrive - UIC\MATLAB\narrative_fig2.2_lopo_vs_within.png'
fig2.savefig(p2, dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved -> {p2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Alpha vs Beta Cohen's d scatter — two clusters
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: alpha vs beta scatter...")
fig3, ax = plt.subplots(figsize=(8, 7))
fig3.suptitle("Figure 3 — Alpha vs Beta Power Change During FoG\n"
              "Cohen's d (FoG − Normal), mean across F3/F4/Fz",
              fontsize=13, fontweight='bold')

# Quadrant shading
ax.axvspan(-1.5, 0, ymin=0.5, ymax=1.0, color='#457B9D', alpha=0.07)
ax.axvspan(0, 1.5, ymin=0.0, ymax=0.5, color='#457B9D', alpha=0.07)
ax.axvspan(0, 1.5, ymin=0.5, ymax=1.0, color='#E76F51', alpha=0.07)
ax.axvspan(-1.5, 0, ymin=0.0, ymax=0.5, color='#E76F51', alpha=0.07)

ax.axvline(0, color='gray', linewidth=1.2, linestyle='--', zorder=2)
ax.axhline(0, color='gray', linewidth=1.2, linestyle='--', zorder=2)

# Regression line
a_vals = np.array([ALPHA_D[p] for p in pids])
b_vals = np.array([BETA_D[p]  for p in pids])
r_val, p_val = pearsonr(a_vals, b_vals)
x_line = np.linspace(min(a_vals)-0.05, max(a_vals)+0.05, 100)
m, c   = np.polyfit(a_vals, b_vals, 1)
ax.plot(x_line, m*x_line + c, color='#333333', linewidth=1.5,
        linestyle='-', alpha=0.5, zorder=2, label=f'r = {r_val:.3f},  p = {p_val:.4f}')

# Patient dots
for pid in pids:
    a = ALPHA_D[pid]; b = BETA_D[pid]
    if pid in SUBTYPE_A:
        color, marker, subtype_label = COL_A, 'o', 'A'
    elif pid in SUBTYPE_B:
        color, marker, subtype_label = COL_B, 's', 'B'
    else:
        color, marker, subtype_label = '#999999', '^', '?'

    ax.scatter(a, b, color=color, marker=marker, s=120, zorder=5,
               edgecolors='white', linewidths=0.8)
    offset_x = 0.015; offset_y = 0.018
    ax.text(a + offset_x, b + offset_y, f'P{pid}',
            fontsize=10, fontweight='bold', color=color, zorder=6)

# Quadrant labels
ax.text( 0.55,  0.55, 'Subtype A\n(alpha ↑, beta ↑)', transform=ax.transAxes,
        fontsize=10, color=COL_A, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COL_A, alpha=0.8))
ax.text( 0.18,  0.18, 'Subtype B\n(alpha ↓, beta ↓)', transform=ax.transAxes,
        fontsize=10, color=COL_B, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COL_B, alpha=0.8))

ax.set_xlabel("Alpha  Cohen's d  (FoG − Normal)", fontsize=12)
ax.set_ylabel("Beta  Cohen's d  (FoG − Normal)",  fontsize=12)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

plt.tight_layout()
p3 = r'C:\Users\arana\OneDrive - UIC\MATLAB\narrative_fig3_alpha_beta_scatter.png'
fig3.savefig(p3, dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved -> {p3}")
print("\nAll 3 narrative figures saved.")
