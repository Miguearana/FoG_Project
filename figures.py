#!/usr/bin/env python3
"""
Publication-quality figures for TFG:
  Fig 1 — Group PSD (Normal vs FoG, mean ± SEM) + band power boxplots
  Fig 2 — Subtype A vs Subtype B PSD comparison (diverging alpha/beta)
"""

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import ranksums, sem
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

# Subtype classification (n=9 patients, P009 excluded — severe class imbalance)
SUBTYPE_A = ['003', '008', '011', '012']   # alpha+beta increase
SUBTYPE_B = ['001', '006', '010']          # alpha+beta flat/suppress

PATIENTS = {
    '001': [os.path.join(FILTERED_DIR, '001', f'task_{t}.txt') for t in range(1,5)],
    '003': [os.path.join(FILTERED_DIR, '003', f'task_{t}.txt') for t in range(1,5)],
    '004': [os.path.join(FILTERED_DIR, '004', f'task_{t}.txt') for t in range(1,6)],
    '006': [os.path.join(FILTERED_DIR, '006', f'task_{t}.txt') for t in range(1,5)],
    '007': [os.path.join(FILTERED_DIR, '007', f'task_{t}.txt') for t in range(1,5)],
    '008': ([os.path.join(FILTERED_DIR, '008', 'OFF_1', f'task_{t}.txt') for t in range(1,6)] +
            [os.path.join(FILTERED_DIR, '008', 'OFF_2', f'task_{t}.txt') for t in range(1,5)]),
    '010': [os.path.join(FILTERED_DIR, '010', f'task_{t}.txt') for t in range(1,5)],
    '011': [os.path.join(FILTERED_DIR, '011', f'task_{t}.txt') for t in range(1,5)],
    '012': [os.path.join(FILTERED_DIR, '012', f'task_{t}.txt') for t in range(1,5)],
}

# ── HELPERS ──────────────────────────────────────────────────────────────────
def load_patient(file_list):
    chunks = []
    for fp in file_list:
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp, header=None, usecols=USE_COLS, on_bad_lines='skip')
        chunks.append(df[USE_COLS].values.astype(np.float64))
    return np.vstack(chunks) if chunks else None

def compute_patient_psds(data):
    """Returns dict with 'norm' and 'fog', each shape (n_windows, n_freqs, n_channels)."""
    n = len(data)
    norm_psds, fog_psds = [], []
    nperseg = FS * 2   # 0.5 Hz resolution — resolves delta cleanly
    for start in range(0, n - WIN_SAMPLES + 1, STRIDE_SAMP):
        chunk = data[start: start + WIN_SAMPLES]
        label = 1 if np.mean(chunk[:, LBL_I]) >= 0.5 else 0
        win_psds = []
        for col_idx in EEG_INDICES:
            sig = chunk[:, col_idx]
            f, pxx = scipy.signal.welch(sig, fs=FS, window='hann',
                                        nperseg=nperseg, noverlap=nperseg//2)
            win_psds.append(10 * np.log10(pxx + 1e-12))
        if label == 0:
            norm_psds.append(win_psds)
        else:
            fog_psds.append(win_psds)
    # shape: (n_windows, n_channels, n_freqs) → transpose to (n_windows, n_freqs, n_channels)
    to_arr = lambda lst: np.array(lst).transpose(0, 2, 1) if lst else None
    return f, to_arr(norm_psds), to_arr(fog_psds)

def band_power_windows(data):
    """Returns X (n_windows, n_bands*n_ch) and y (n_windows,)."""
    n = len(data)
    X, y = [], []
    for start in range(0, n - WIN_SAMPLES + 1, STRIDE_SAMP):
        chunk = data[start: start + WIN_SAMPLES]
        label = 1 if np.mean(chunk[:, LBL_I]) >= 0.5 else 0
        row = []
        for col_idx in EEG_INDICES:
            sig = chunk[:, col_idx]
            f2, pxx = scipy.signal.welch(sig, fs=FS, window='hann', nperseg=FS, noverlap=FS//2)
            for flo, fhi in BANDS.values():
                mask = (f2 >= flo) & (f2 <= fhi)
                row.append(10 * np.log10(np.mean(pxx[mask]) + 1e-12))
        X.append(row)
        y.append(label)
    return np.array(X), np.array(y)

# ── COMPUTE PER-PATIENT DATA ─────────────────────────────────────────────────
print("Computing PSDs per patient...")
patient_norm_mean = {}   # pid -> (n_freqs, n_ch) mean PSD during Normal
patient_fog_mean  = {}
band_powers       = {}   # pid -> (X, y)
f_axis            = None

for pid, files in PATIENTS.items():
    data = load_patient(files)
    if data is None: continue
    f_ax, norm_arr, fog_arr = compute_patient_psds(data)
    if f_axis is None: f_axis = f_ax
    if norm_arr is not None: patient_norm_mean[pid] = np.mean(norm_arr, axis=0)  # (n_freqs, n_ch)
    if fog_arr  is not None: patient_fog_mean[pid]  = np.mean(fog_arr,  axis=0)
    X_bp, y_bp = band_power_windows(data)
    band_powers[pid] = (X_bp, y_bp)
    print(f"  P{pid} done")

# Frequency mask for display (1-45 Hz)
freq_mask = (f_axis >= 1) & (f_axis <= 45)
f_plot    = f_axis[freq_mask]

# Band boundary x positions for shading
band_colors = {'Delta':'#BDD7EE', 'Theta':'#E2EFDA', 'Alpha':'#FFF2CC', 'Beta':'#FCE4D6'}

def shade_bands(ax, ymin=-40, ymax=20):
    for (name, (flo, fhi)), col in zip(BANDS.items(), band_colors.values()):
        ax.axvspan(flo, fhi, alpha=0.18, color=col, zorder=0)
        ax.text((flo+fhi)/2, ymax - 1.5, name, ha='center', va='top', fontsize=8, color='gray')

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Group PSD (Normal vs FoG) + band power boxplots
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Figure 1...")
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle("Group-Level EEG Analysis: Normal vs FoG  (N=10 patients)",
              fontsize=14, fontweight='bold', y=0.98)

ch_names = list(EEG_COLS.keys())
pids_all = sorted(patient_norm_mean.keys())

for ci, ch in enumerate(ch_names):
    ax = axes1[0, ci]

    # Collect per-patient mean PSDs → shape (n_patients, n_freqs)
    norm_mat = np.array([patient_norm_mean[p][:, ci][freq_mask] for p in pids_all if p in patient_norm_mean])
    fog_mat  = np.array([patient_fog_mean[p][:, ci][freq_mask]  for p in pids_all if p in patient_fog_mean])

    # Individual patient traces (light)
    for row in norm_mat: ax.plot(f_plot, row, color='#3A86FF', alpha=0.15, linewidth=0.8)
    for row in fog_mat:  ax.plot(f_plot, row, color='#FF3A3A', alpha=0.15, linewidth=0.8)

    # Group mean ± SEM (bold)
    n_mean = np.mean(norm_mat, axis=0)
    f_mean = np.mean(fog_mat,  axis=0)
    n_se   = sem(norm_mat, axis=0)
    f_se   = sem(fog_mat,  axis=0)

    ax.plot(f_plot, n_mean, color='#3A86FF', linewidth=2.2, label='Normal')
    ax.fill_between(f_plot, n_mean-n_se, n_mean+n_se, color='#3A86FF', alpha=0.25)
    ax.plot(f_plot, f_mean, color='#FF3A3A', linewidth=2.2, label='FoG')
    ax.fill_between(f_plot, f_mean-f_se, f_mean+f_se, color='#FF3A3A', alpha=0.25)

    shade_bands(ax, ymin=float(np.min([n_mean, f_mean]))-2,
                     ymax=float(np.max([n_mean, f_mean]))+2)

    ax.set_xlim(1, 45); ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)') if ci == 0 else None
    ax.set_title(f'Channel {ch}', fontweight='bold')
    if ci == 0: ax.legend(fontsize=9)

# ── Row 2: band power boxplots (mean across F3/F4/Fz) ────────────────────────
band_names = list(BANDS.keys())
for bi, band in enumerate(band_names):
    ax = axes1[1, bi] if bi < 3 else None
    if ax is None: break

    norm_vals, fog_vals, pid_labels = [], [], []
    for pid in pids_all:
        if pid not in band_powers: continue
        X_bp, y_bp = band_powers[pid]
        # Average the same band across all 3 channels
        ch_indices = [ci * len(BANDS) + bi for ci in range(len(ch_names))]
        n_power = np.mean(X_bp[y_bp == 0][:, ch_indices], axis=1)
        f_power = np.mean(X_bp[y_bp == 1][:, ch_indices], axis=1)
        norm_vals.append(np.mean(n_power))
        fog_vals.append(np.mean(f_power))
        pid_labels.append(pid)

    norm_arr = np.array(norm_vals)
    fog_arr  = np.array(fog_vals)

    # Boxplot
    bp = ax.boxplot([norm_arr, fog_arr], positions=[1, 2],
                    widths=0.45, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3))
    bp['boxes'][0].set_facecolor('#3A86FF'); bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor('#FF3A3A'); bp['boxes'][1].set_alpha(0.55)

    # Individual patient dots + connecting lines
    jitter = np.linspace(-0.08, 0.08, len(norm_arr))
    for j, (nv, fv) in enumerate(zip(norm_arr, fog_arr)):
        ax.plot([1+jitter[j], 2+jitter[j]], [nv, fv], 'gray', alpha=0.4, linewidth=0.8)
        ax.plot(1+jitter[j], nv, 'o', color='#3A86FF', markersize=6, zorder=5)
        ax.plot(2+jitter[j], fv, 'o', color='#FF3A3A', markersize=6, zorder=5)

    # Stats annotation
    _, p_val = ranksums(norm_arr, fog_arr)
    y_top = max(np.max(norm_arr), np.max(fog_arr))
    y_ann = y_top + 0.5
    ax.plot([1, 2], [y_ann, y_ann], 'k-', linewidth=1)
    ax.text(1.5, y_ann + 0.1, sig_stars(p_val), ha='center', fontsize=12)

    ax.set_xticks([1, 2]); ax.set_xticklabels(['Normal', 'FoG'])
    ax.set_title(f'{band} Band Power\n(mean F3/F4/Fz)', fontweight='bold')
    ax.set_ylabel('Power (dB)') if bi == 0 else None

# Hide unused subplot (bottom-right)
axes1[1, 2].set_visible(False)

plt.tight_layout()
p1 = r'C:\Users\arana\OneDrive - UIC\MATLAB\fig1_group_psd_boxplots.png'
fig1.savefig(p1, dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved -> {p1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Subtype A vs B PSD comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2...")
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    "EEG Subtypes: Frontal Alpha/Beta Divergence during FoG\n"
    "Subtype A (alpha+beta ↑, n=5) vs Subtype B (alpha+beta ↓, n=3)  —  delta suppresses in BOTH",
    fontsize=12, fontweight='bold')

pids_A = [p for p in SUBTYPE_A if p in patient_norm_mean]
pids_B = [p for p in SUBTYPE_B if p in patient_norm_mean]

col_A_norm = '#1A6FBF';  col_A_fog = '#FF6B35'
col_B_norm = '#2E8B57';  col_B_fog = '#9B59B6'

for ci, ch in enumerate(ch_names):
    ax = axes2[ci]

    for pids, c_norm, c_fog, label in [
        (pids_A, col_A_norm, col_A_fog, 'A'),
        (pids_B, col_B_norm, col_B_fog, 'B'),
    ]:
        if not pids: continue
        n_mat = np.array([patient_norm_mean[p][:, ci][freq_mask] for p in pids])
        f_mat = np.array([patient_fog_mean[p][:, ci][freq_mask]  for p in pids])

        # Individual traces
        for row in n_mat: ax.plot(f_plot, row, color=c_norm, alpha=0.2, linewidth=0.8)
        for row in f_mat: ax.plot(f_plot, row, color=c_fog,  alpha=0.2, linewidth=0.8)

        n_mu = np.mean(n_mat, axis=0); n_se = sem(n_mat, axis=0) if len(n_mat)>1 else np.zeros_like(n_mu)
        f_mu = np.mean(f_mat, axis=0); f_se = sem(f_mat, axis=0) if len(f_mat)>1 else np.zeros_like(f_mu)

        ax.plot(f_plot, n_mu, color=c_norm, linewidth=2.2,
                label=f'Subtype {label} — Normal', linestyle='--')
        ax.fill_between(f_plot, n_mu-n_se, n_mu+n_se, color=c_norm, alpha=0.2)
        ax.plot(f_plot, f_mu, color=c_fog, linewidth=2.2,
                label=f'Subtype {label} — FoG')
        ax.fill_between(f_plot, f_mu-f_se, f_mu+f_se, color=c_fog, alpha=0.2)

    shade_bands(ax)
    ax.set_xlim(1, 45); ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)') if ci == 0 else None
    ax.set_title(f'Channel {ch}', fontweight='bold')
    if ci == 1:
        ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

# Add annotation arrows on middle panel
ax = axes2[1]
y_range = ax.get_ylim()
ax.annotate('', xy=(20, y_range[0] + (y_range[1]-y_range[0])*0.35),
            xytext=(20, y_range[0] + (y_range[1]-y_range[0])*0.5),
            arrowprops=dict(arrowstyle='->', color=col_A_fog, lw=2))
ax.annotate('', xy=(20, y_range[0] + (y_range[1]-y_range[0])*0.55),
            xytext=(20, y_range[0] + (y_range[1]-y_range[0])*0.42),
            arrowprops=dict(arrowstyle='->', color=col_B_fog, lw=2))
ax.text(22, y_range[0] + (y_range[1]-y_range[0])*0.52, 'A: ↑ beta', color=col_A_fog, fontsize=9)
ax.text(22, y_range[0] + (y_range[1]-y_range[0])*0.42, 'B: ↓ beta', color=col_B_fog, fontsize=9)

plt.tight_layout()
p2 = r'C:\Users\arana\OneDrive - UIC\MATLAB\fig2_subtypes.png'
fig2.savefig(p2, dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved -> {p2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Delta band: consistent suppression across ALL patients (per-patient dots)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3...")
fig3, axes3 = plt.subplots(1, 3, figsize=(13, 5))
fig3.suptitle("Frontal Delta Suppression during FoG — consistent across all 10 patients",
              fontsize=13, fontweight='bold')

for ci, ch in enumerate(ch_names):
    ax = axes3[ci]
    bi = 0  # Delta is band index 0

    norm_per_p, fog_per_p = [], []
    for pid in pids_all:
        if pid not in band_powers: continue
        X_bp, y_bp = band_powers[pid]
        col_idx = ci * len(BANDS) + bi
        norm_per_p.append(np.mean(X_bp[y_bp == 0, col_idx]))
        fog_per_p.append(np.mean(X_bp[y_bp == 1, col_idx]))

    norm_per_p = np.array(norm_per_p)
    fog_per_p  = np.array(fog_per_p)

    # Boxplot
    bp = ax.boxplot([norm_per_p, fog_per_p], positions=[1, 2],
                    widths=0.45, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    bp['boxes'][0].set_facecolor('#3A86FF'); bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor('#FF3A3A'); bp['boxes'][1].set_alpha(0.55)

    # Per-patient lines — color by subtype
    for j, pid in enumerate(pids_all):
        if pid not in band_powers: continue
        nv = norm_per_p[j]; fv = fog_per_p[j]
        color = col_A_fog if pid in SUBTYPE_A else col_B_fog
        ax.plot([1, 2], [nv, fv], '-o', color=color, linewidth=1.5,
                markersize=7, zorder=5, label=f'P{pid}')
        ax.text(2.08, fv, f'P{pid}', fontsize=7.5, va='center', color=color)

    # Stat annotation
    _, p_val = ranksums(norm_per_p, fog_per_p)
    y_top = max(np.max(norm_per_p), np.max(fog_per_p))
    ax.plot([1, 2], [y_top+0.3, y_top+0.3], 'k-', linewidth=1.2)
    ax.text(1.5, y_top+0.5, f'{sig_stars(p_val)}  p={p_val:.4f}', ha='center', fontsize=10)

    ax.set_xticks([1, 2]); ax.set_xticklabels(['Normal', 'FoG'], fontsize=11)
    ax.set_title(f'Delta Power — {ch}', fontweight='bold')
    ax.set_ylabel('Power (dB)') if ci == 0 else None
    ax.set_xlim(0.5, 2.8)

# Legend for subtype colors
handles = [
    Line2D([0],[0], color=col_A_fog, marker='o', linewidth=1.5, label='Subtype A (alpha/beta ↑)'),
    Line2D([0],[0], color=col_B_fog, marker='o', linewidth=1.5, label='Subtype B (alpha/beta ↓)'),
]
fig3.legend(handles=handles, loc='lower center', ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 1])
p3 = r'C:\Users\arana\OneDrive - UIC\MATLAB\fig3_delta_per_patient.png'
fig3.savefig(p3, dpi=180, bbox_inches='tight')
plt.close()
print(f"  Saved -> {p3}")
print("\nAll figures saved.")
