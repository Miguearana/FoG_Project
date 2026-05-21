#!/usr/bin/env python3
"""
Per-patient EEG signal quality metrics on F3/F4/Fz.
  1. HF/LF ratio  — high-frequency (40-100 Hz) vs EEG-band (1-40 Hz) power
                    elevated ratio = muscle/noise contamination
  2. Kurtosis      — heavy tails indicate spike artifacts (clean EEG ~3)
  3. Outlier rate  — fraction of samples exceeding ±150 µV
  4. Median abs amplitude
"""

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

FILTERED_DIR = r'C:\Users\arana\OneDrive - UIC\MATLAB\r8gmbtv7w2-3\Filtered Data'

FS       = 500
EEG_COLS = {'F3': 4, 'F4': 5, 'Fz': 16}
USE_COLS = sorted(set(EEG_COLS.values()))
_IDX     = {orig: i for i, orig in enumerate(USE_COLS)}

PATIENTS = {
    '001': [os.path.join(FILTERED_DIR, '001', f'task_{t}.txt') for t in range(1,5)],
    '003': [os.path.join(FILTERED_DIR, '003', f'task_{t}.txt') for t in range(1,5)],
    '004': [os.path.join(FILTERED_DIR, '004', f'task_{t}.txt') for t in range(1,6)],
    '006': [os.path.join(FILTERED_DIR, '006', f'task_{t}.txt') for t in range(1,5)],
    '007': [os.path.join(FILTERED_DIR, '007', f'task_{t}.txt') for t in range(1,5)],
    '008': ([os.path.join(FILTERED_DIR, '008', 'OFF_1', f'task_{t}.txt') for t in range(1,6)] +
            [os.path.join(FILTERED_DIR, '008', 'OFF_2', f'task_{t}.txt') for t in range(1,5)]),
    '009': [os.path.join(FILTERED_DIR, '009', f'task_{t}.txt') for t in range(1,7)],
    '010': [os.path.join(FILTERED_DIR, '010', f'task_{t}.txt') for t in range(1,5)],
    '011': [os.path.join(FILTERED_DIR, '011', f'task_{t}.txt') for t in range(1,5)],
    '012': [os.path.join(FILTERED_DIR, '012', f'task_{t}.txt') for t in range(1,5)],
}

def hf_lf_ratio(sig):
    f, pxx = scipy.signal.welch(sig, fs=FS, nperseg=FS)
    lf = np.mean(pxx[(f >= 1)  & (f <= 40)])
    hf = np.mean(pxx[(f >= 40) & (f <= 100)])
    return hf / (lf + 1e-12)

def load_signals(file_list):
    chunks = []
    for fp in file_list:
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp, header=None, usecols=USE_COLS, on_bad_lines='skip')
        chunks.append(df[USE_COLS].values.astype(np.float64))
    return np.vstack(chunks) if chunks else None

print(f"{'Patient':<10} {'Channel':<6} {'HF/LF ratio':>12} {'Kurtosis':>10} {'Outlier%':>10} {'Median|amp|':>12}")
print("-" * 62)

results = {}
for pid, files in PATIENTS.items():
    data = load_signals(files)
    if data is None: continue
    row = {}
    for ch_name, col in EEG_COLS.items():
        sig = data[:, _IDX[col]]
        hflf    = hf_lf_ratio(sig)
        kurt    = kurtosis(sig, fisher=True)   # excess kurtosis; Gaussian=0
        outlier = np.mean(np.abs(sig) > 150) * 100
        med_amp = np.median(np.abs(sig))
        row[ch_name] = dict(hflf=hflf, kurt=kurt, outlier=outlier, med_amp=med_amp)
        print(f"  P{pid:<7} {ch_name:<6} {hflf:>12.4f} {kurt:>10.2f} {outlier:>9.2f}% {med_amp:>11.2f}")
    results[pid] = row

# ── SUMMARY: mean across channels per patient ────────────────────────────────
print("\n" + "=" * 62)
print("  PATIENT SUMMARY (mean across F3/F4/Fz)")
print("=" * 62)
print(f"  {'Patient':<10} {'HF/LF':>8} {'Kurtosis':>10} {'Outlier%':>10}  Flag")
print(f"  {'-'*54}")

quality_flags = {}
for pid, row in results.items():
    mean_hflf    = np.mean([row[c]['hflf']    for c in EEG_COLS])
    mean_kurt    = np.mean([row[c]['kurt']    for c in EEG_COLS])
    mean_outlier = np.mean([row[c]['outlier'] for c in EEG_COLS])

    flags = []
    if mean_hflf    > 0.15:  flags.append('HIGH-NOISE')
    if mean_kurt    > 10:    flags.append('SPIKY')
    if mean_outlier > 1.0:   flags.append('OUTLIERS')
    flag_str = ', '.join(flags) if flags else 'OK'
    quality_flags[pid] = flag_str

    print(f"  P{pid:<9} {mean_hflf:>8.4f} {mean_kurt:>10.2f} {mean_outlier:>9.2f}%  {flag_str}")

# ── PLOT ─────────────────────────────────────────────────────────────────────
pids  = list(results.keys())
hflfs = [np.mean([results[p][c]['hflf']    for c in EEG_COLS]) for p in pids]
kurts = [np.mean([results[p][c]['kurt']    for c in EEG_COLS]) for p in pids]
outs  = [np.mean([results[p][c]['outlier'] for c in EEG_COLS]) for p in pids]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("EEG Signal Quality per Patient (F3/F4/Fz mean)", fontsize=12)

for ax, vals, title, thresh, unit in zip(
    axes,
    [hflfs, kurts, outs],
    ['HF/LF Power Ratio\n(muscle noise indicator)', 'Excess Kurtosis\n(spike artifacts)', 'Outlier Rate\n(|amp| > 150 µV)'],
    [0.15, 10, 1.0],
    ['ratio', 'kurtosis', '%']
):
    colors = ['tomato' if v > thresh else 'steelblue' for v in vals]
    ax.bar([f'P{p}' for p in pids], vals, color=colors)
    ax.axhline(thresh, color='red', linestyle='--', linewidth=1, label=f'threshold={thresh}')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(unit)
    ax.legend(fontsize=8)

plt.tight_layout()
save_path = r'C:\Users\arana\OneDrive - UIC\MATLAB\signal_quality.png'
plt.savefig(save_path, dpi=150)
plt.close()
print(f"\nPlot saved -> {save_path}")
