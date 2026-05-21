#!/usr/bin/env python3
"""
Group-level band power statistics across all patients.
For each channel × band: Wilcoxon rank-sum + Cohen's d (Normal vs FoG windows).
Uses the same sliding-window features already extracted by multi_patient_pipeline.py.
"""

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

FILTERED_DIR = r'C:\Users\arana\OneDrive - UIC\MATLAB\r8gmbtv7w2-3\Filtered Data'

FS          = 500
WIN_SAMPLES = int(4.5 * FS)
STRIDE_SAMP = int(1.0 * FS)
EEG_COLS    = {
    'F3': 4,  'F4': 5,  'Fz': 16,   # Frontal
    'C3': 6,  'C4': 7,  'Cz': 17,   # Central (motor cortex)
    'P3': 8,  'P4': 9,  'Pz': 18,   # Parietal
    'O1': 10, 'O2': 11,              # Occipital
}
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
    '009': [os.path.join(FILTERED_DIR, '009', f'task_{t}.txt') for t in range(1,7)],
    '010': [os.path.join(FILTERED_DIR, '010', f'task_{t}.txt') for t in range(1,5)],
    '011': [os.path.join(FILTERED_DIR, '011', f'task_{t}.txt') for t in range(1,5)],
    '012': [os.path.join(FILTERED_DIR, '012', f'task_{t}.txt') for t in range(1,5)],
}

def band_power_db(sig, fmin, fmax):
    f, pxx = scipy.signal.welch(sig, fs=FS, window='hann', nperseg=FS, noverlap=FS//2)
    mask = (f >= fmin) & (f <= fmax)
    return 10.0 * np.log10(np.mean(pxx[mask]) + 1e-12)

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

def load_and_extract(file_list):
    chunks = []
    for fp in file_list:
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, header=None, usecols=USE_COLS, on_bad_lines='skip')
        chunks.append(df[USE_COLS].values.astype(np.float64))
    if not chunks:
        return None, None
    data = np.vstack(chunks)
    n = len(data)
    feats, labels = [], []
    for start in range(0, n - WIN_SAMPLES + 1, STRIDE_SAMP):
        chunk = data[start: start + WIN_SAMPLES]
        label = 1 if np.mean(chunk[:, LBL_I]) >= 0.5 else 0
        row = []
        for col_idx in EEG_INDICES:
            sig = chunk[:, col_idx]
            for flo, fhi in BANDS.values():
                row.append(band_power_db(sig, flo, fhi))
        feats.append(row)
        labels.append(label)
    return np.array(feats), np.array(labels)

feat_names = [f"{ch}_{band}" for ch in EEG_COLS for band in BANDS]

# ── PER-PATIENT STATS ────────────────────────────────────────────────────────
print("Running per-patient statistics...\n")
all_patient_stats = {}

for pid, files in PATIENTS.items():
    X, y = load_and_extract(files)
    if X is None or np.sum(y==1) < 5 or np.sum(y==0) < 5:
        continue

    X_norm = X[y == 0]
    X_fog  = X[y == 1]
    stats  = {}
    for i, fname in enumerate(feat_names):
        _, p = ranksums(X_norm[:, i], X_fog[:, i])
        d    = cohens_d(X_fog[:, i], X_norm[:, i])  # FoG - Normal (negative = suppression)
        stats[fname] = dict(p=p, d=d, n_norm=len(X_norm), n_fog=len(X_fog))
    all_patient_stats[pid] = stats

# ── PRINT PER-PATIENT TABLE ──────────────────────────────────────────────────
for pid, stats in all_patient_stats.items():
    n_norm = list(stats.values())[0]['n_norm']
    n_fog  = list(stats.values())[0]['n_fog']
    print(f"P{pid}  (Normal={n_norm}, FoG={n_fog})")
    print(f"  {'Feature':<14} {'Cohen d':>8}  {'p-value':>10}  Sig?")
    print(f"  {'-'*44}")
    for fname, s in stats.items():
        sig  = '***' if s['p'] < 0.001 else ('**' if s['p'] < 0.01 else ('*' if s['p'] < 0.05 else ''))
        flag = '<-- SUPPRESSION' if s['d'] < -0.3 and s['p'] < 0.05 else (
               '<-- INCREASE'   if s['d'] >  0.3 and s['p'] < 0.05 else '')
        print(f"  {fname:<14} {s['d']:>+8.3f}  {s['p']:>10.4f}  {sig:<3} {flag}")
    print()

# ── GROUP-LEVEL SUMMARY ──────────────────────────────────────────────────────
print("=" * 60)
print("  GROUP SUMMARY: how many patients show significant effect?")
print("=" * 60)
print(f"  {'Feature':<14}  {'N sig (p<0.05)':>14}  {'Mean d':>8}  {'Direction'}")
print(f"  {'-'*56}")

group_rows = []
for fname in feat_names:
    ds   = [all_patient_stats[p][fname]['d'] for p in all_patient_stats]
    ps   = [all_patient_stats[p][fname]['p'] for p in all_patient_stats]
    n_sig = sum(p < 0.05 for p in ps)
    mean_d = np.mean(ds)
    direction = 'suppressed' if mean_d < -0.2 else ('increased' if mean_d > 0.2 else 'mixed')
    group_rows.append((fname, n_sig, mean_d, direction))
    print(f"  {fname:<14}  {n_sig:>5}/{len(all_patient_stats):<8}  {mean_d:>+8.3f}  {direction}")

# ── HEATMAP ──────────────────────────────────────────────────────────────────
channels = list(EEG_COLS.keys())
bands    = list(BANDS.keys())
pids     = list(all_patient_stats.keys())

# Cohen's d matrix: rows=features, cols=patients
d_matrix = np.array([
    [all_patient_stats[p][f]['d'] for p in pids]
    for f in feat_names
])
p_matrix = np.array([
    [all_patient_stats[p][f]['p'] for p in pids]
    for f in feat_names
])

fig, ax = plt.subplots(figsize=(len(pids)*1.1 + 2, max(7, len(feat_names) * 0.38 + 1)))
im = ax.imshow(d_matrix, cmap='RdBu', vmin=-1.2, vmax=1.2, aspect='auto')
plt.colorbar(im, ax=ax, label="Cohen's d  (FoG - Normal)")

ax.set_xticks(range(len(pids)));   ax.set_xticklabels([f'P{p}' for p in pids])
ax.set_yticks(range(len(feat_names))); ax.set_yticklabels(feat_names, fontsize=9)
ax.set_title("Band Power Effect Size (Cohen's d) per Patient\nBlue = suppression during FoG  |  Red = increase during FoG",
             fontsize=11)

# Mark significant cells with *
for i in range(len(feat_names)):
    for j in range(len(pids)):
        if p_matrix[i, j] < 0.05:
            ax.text(j, i, '*', ha='center', va='center', fontsize=13, color='black')

plt.tight_layout()
save_path = r'C:\Users\arana\OneDrive - UIC\MATLAB\group_stats_all_regions_heatmap.png'
plt.savefig(save_path, dpi=150)
plt.close()
print(f"\nHeatmap saved -> {save_path}")

# ── GROUP-LEVEL SIGNIFICANCE: ONE-SAMPLE WILCOXON SIGNED-RANK + BH-FDR ──────
from scipy.stats import wilcoxon

print("\n" + "=" * 70)
print("  GROUP-LEVEL TEST: one-sample Wilcoxon signed-rank on Cohen's d")
print("  H0: median d = 0  (n=10 patients, independent observations)")
print("  Multiple comparisons: Benjamini-Hochberg FDR across 44 tests")
print("=" * 70)

raw_pvals, mean_ds = [], []
for fname in feat_names:
    ds = np.array([all_patient_stats[p][fname]['d'] for p in all_patient_stats])
    mean_ds.append(np.mean(ds))
    if len(np.unique(ds)) < 2:
        raw_pvals.append(1.0)
    else:
        try:
            _, p = wilcoxon(ds, alternative='two-sided', zero_method='wilcox')
        except Exception:
            p = 1.0
        raw_pvals.append(p)

# BH-FDR correction (manual — no statsmodels dependency)
n_tests     = len(raw_pvals)
order       = np.argsort(raw_pvals)
pvals_s     = np.array(raw_pvals)[order]
pvals_adj_s = np.minimum.accumulate(
    (n_tests / np.arange(n_tests, 0, -1)) * pvals_s[::-1]
)[::-1]
pvals_adj_s = np.minimum(pvals_adj_s, 1.0)
pvals_fdr   = np.empty(n_tests); pvals_fdr[order] = pvals_adj_s
reject      = pvals_fdr < 0.05

print(f"\n  {'Feature':<14}  {'N sig/8':>7}  {'Mean d':>8}  {'p (raw)':>10}  {'p (FDR)':>10}  Result")
print(f"  {'-'*70}")

for band in BANDS:
    print(f"\n  [{band}]")
    for ch in EEG_COLS:
        fname = f"{ch}_{band}"
        i     = feat_names.index(fname)
        ds    = np.array([all_patient_stats[p][fname]['d'] for p in all_patient_stats])
        n_sig_pat = sum(all_patient_stats[p][fname]['p'] < 0.05 for p in all_patient_stats)
        result = 'SIGNIFICANT *' if reject[i] else 'ns'
        print(f"  {fname:<14}  {n_sig_pat:>3}/8    {mean_ds[i]:>+8.3f}  "
              f"{raw_pvals[i]:>10.4f}  {pvals_fdr[i]:>10.4f}  {result}")

n_proven = int(np.sum(reject))
print(f"\n  -- SUMMARY --------------------------------------------------")
print(f"  Proven significant after FDR: {n_proven} / {n_tests} channel-band pairs")
print(f"  Note: with n=10, Wilcoxon min achievable p ≈ 0.002 (all same direction)")
print(f"        FDR threshold becomes ~p_raw < 0.05 × rank/44 per feature")
