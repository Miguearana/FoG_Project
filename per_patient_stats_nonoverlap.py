#!/usr/bin/env python3
"""
Per-patient intra-patient statistics using NON-OVERLAPPING windows.

Windows are non-overlapping (stride = window length = 4.5s) so every window
is temporally independent. This makes the Wilcoxon rank-sum test valid without
any autocorrelation concern.

For each patient:
  - Wilcoxon rank-sum: Normal windows vs FoG windows, per channel-band pair
  - Cohen's d: effect size and direction
  - BH-FDR correction across the 44 channel-band pairs tested within that patient

Output: which features survive FDR for each patient, + a per-patient heatmap.
"""

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
FILTERED_DIR = r'C:\Users\arana\OneDrive - UIC\MATLAB\r8gmbtv7w2-3\Filtered Data'
FS          = 500
WIN_SAMPLES = int(4.5 * FS)   # 2250 samples = 4.5 s
STRIDE_SAMP = WIN_SAMPLES      # NON-OVERLAPPING: stride = full window
FOG_THRESH  = 0.5

EEG_COLS = {
    'F3': 4,  'F4': 5,  'Fz': 16,
    'C3': 6,  'C4': 7,  'Cz': 17,
    'P3': 8,  'P4': 9,  'Pz': 18,
    'O1': 10, 'O2': 11,
}
LABEL_COL = 60
BANDS     = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}

USE_COLS    = sorted(set(EEG_COLS.values()) | {LABEL_COL})
_COL_MAP    = {orig: i for i, orig in enumerate(USE_COLS)}
EEG_INDICES = [_COL_MAP[v] for v in EEG_COLS.values()]
LBL_I       = _COL_MAP[LABEL_COL]

PATIENTS = {
    '001': [os.path.join(FILTERED_DIR, '001', f'task_{t}.txt') for t in range(1, 5)],
    '003': [os.path.join(FILTERED_DIR, '003', f'task_{t}.txt') for t in range(1, 5)],
    '004': [os.path.join(FILTERED_DIR, '004', f'task_{t}.txt') for t in range(1, 6)],
    '006': [os.path.join(FILTERED_DIR, '006', f'task_{t}.txt') for t in range(1, 5)],
    '007': [os.path.join(FILTERED_DIR, '007', f'task_{t}.txt') for t in range(1, 5)],
    '008': ([os.path.join(FILTERED_DIR, '008', 'OFF_1', f'task_{t}.txt') for t in range(1, 6)] +
            [os.path.join(FILTERED_DIR, '008', 'OFF_2', f'task_{t}.txt') for t in range(1, 5)]),
    # P009 excluded: severe class imbalance (dataset paper recommendation, Zhang et al. 2022) + specificity loss in multimodal (supervisor recommendation)
    '010': [os.path.join(FILTERED_DIR, '010', f'task_{t}.txt') for t in range(1, 5)],
    '011': [os.path.join(FILTERED_DIR, '011', f'task_{t}.txt') for t in range(1, 5)],
    '012': [os.path.join(FILTERED_DIR, '012', f'task_{t}.txt') for t in range(1, 5)],
}

SUBTYPE = {
    '001': 'B?',
    '003': 'A', '004': 'B*', '006': 'B',
    '007': 'anomaly?',
    '008': 'A', '010': 'B',
    '011': 'A', '012': 'A',
}

feat_names = [f"{ch}_{band}" for ch in EEG_COLS for band in BANDS]

# ── HELPERS ──────────────────────────────────────────────────────────────────

def band_power_db(sig, fmin, fmax):
    f, pxx = scipy.signal.welch(sig, fs=FS, window='hann',
                                nperseg=FS, noverlap=FS // 2)
    mask = (f >= fmin) & (f <= fmax)
    return 10.0 * np.log10(np.mean(pxx[mask]) + 1e-12)


def cohens_d(fog_vals, norm_vals):
    pooled = np.sqrt((np.std(fog_vals, ddof=1)**2 + np.std(norm_vals, ddof=1)**2) / 2)
    return (np.mean(fog_vals) - np.mean(norm_vals)) / pooled if pooled > 0 else 0.0


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (reject array, adjusted p-values)."""
    n       = len(pvals)
    order   = np.argsort(pvals)
    ps      = np.array(pvals)[order]
    adj     = np.minimum.accumulate((n / np.arange(n, 0, -1)) * ps[::-1])[::-1]
    adj     = np.minimum(adj, 1.0)
    out_adj = np.empty(n); out_adj[order] = adj
    return out_adj < alpha, out_adj


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
    n    = len(data)
    feats, labels = [], []
    for start in range(0, n - WIN_SAMPLES + 1, STRIDE_SAMP):   # non-overlapping
        chunk = data[start: start + WIN_SAMPLES]
        label = 1 if np.mean(chunk[:, LBL_I]) >= FOG_THRESH else 0
        row   = []
        for col_idx in EEG_INDICES:
            sig = chunk[:, col_idx]
            for flo, fhi in BANDS.values():
                row.append(band_power_db(sig, flo, fhi))
        feats.append(row)
        labels.append(label)
    return np.array(feats), np.array(labels)


# ── EXTRACT & ANALYSE ────────────────────────────────────────────────────────
print("Extracting features (non-overlapping windows, 4.5s each) ...")
print("=" * 70)

all_results = {}   # pid -> {fname: {p, p_fdr, d, reject}}

for pid, files in PATIENTS.items():
    X, y = load_and_extract(files)
    if X is None:
        print(f"  P{pid}: no data\n"); continue

    n_norm = int(np.sum(y == 0))
    n_fog  = int(np.sum(y == 1))

    if n_norm < 5 or n_fog < 5:
        print(f"  P{pid}: too few FoG windows ({n_fog}) -- skipped\n"); continue

    # Per-feature Wilcoxon + Cohen's d
    raw_ps, ds = [], []
    for i in range(len(feat_names)):
        norm_v = X[y == 0, i]
        fog_v  = X[y == 1, i]
        _, p   = ranksums(norm_v, fog_v)
        d      = cohens_d(fog_v, norm_v)
        raw_ps.append(p)
        ds.append(d)

    reject, adj_ps = bh_fdr(raw_ps)
    n_sig = int(np.sum(reject))

    patient_res = {}
    for i, fname in enumerate(feat_names):
        patient_res[fname] = dict(p=raw_ps[i], p_fdr=adj_ps[i],
                                  d=ds[i], reject=bool(reject[i]))
    all_results[pid] = patient_res

    # Print patient summary
    print(f"\nP{pid}  [Subtype {SUBTYPE[pid]}]  "
          f"Normal={n_norm} windows | FoG={n_fog} windows | "
          f"FDR-significant: {n_sig}/44")
    print(f"  {'Feature':<14}  {'Cohen d':>8}  {'p (raw)':>10}  {'p (FDR)':>10}  Result")
    print(f"  {'-'*60}")
    for band in BANDS:
        any_printed = False
        for ch in EEG_COLS:
            fname = f"{ch}_{band}"
            r     = patient_res[fname]
            # Print ALL results but highlight significant ones
            sig   = 'SIGNIFICANT *' if r['reject'] else 'ns'
            print(f"  {fname:<14}  {r['d']:>+8.3f}  {r['p']:>10.4f}  "
                  f"{r['p_fdr']:>10.4f}  {sig}")

# ── CROSS-PATIENT SUMMARY TABLE ──────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  CROSS-PATIENT SUMMARY  (how many patients survive FDR per feature)")
print("=" * 70)
print(f"  {'Feature':<14}  {'N FDR-sig':>9}  {'Mean d':>8}  {'Direction'}")
print(f"  {'-'*50}")

summary_rows = []
for band in BANDS:
    print(f"\n  [{band}]")
    for ch in EEG_COLS:
        fname    = f"{ch}_{band}"
        n_reject = sum(all_results[p][fname]['reject'] for p in all_results)
        mean_d   = np.mean([all_results[p][fname]['d'] for p in all_results])
        direction = 'suppressed' if mean_d < -0.15 else ('increased' if mean_d > 0.15 else 'mixed')
        summary_rows.append((fname, n_reject, mean_d, direction))
        print(f"  {fname:<14}  {n_reject:>4}/{len(all_results):<4}  {mean_d:>+8.3f}  {direction}")

# ── HEATMAP: FDR-CORRECTED COHEN'S D PER PATIENT ────────────────────────────
pids     = list(all_results.keys())
n_feat   = len(feat_names)
n_pat    = len(pids)

d_mat    = np.array([[all_results[p][f]['d']      for p in pids] for f in feat_names])
sig_mat  = np.array([[all_results[p][f]['reject']  for p in pids] for f in feat_names])

fig, ax = plt.subplots(figsize=(n_pat * 1.2 + 2, max(8, n_feat * 0.42 + 1.5)))
im = ax.imshow(d_mat, cmap='RdBu', vmin=-1.0, vmax=1.0, aspect='auto')
plt.colorbar(im, ax=ax, label="Cohen's d  (FoG - Normal)")

ax.set_xticks(range(n_pat))
ax.set_xticklabels(
    [f"P{p}\n[{SUBTYPE[p]}]" for p in pids], fontsize=9
)
ax.set_yticks(range(n_feat))
ax.set_yticklabels(feat_names, fontsize=9)
ax.set_title(
    "Per-Patient Band Power  |  Non-Overlapping Windows  |  BH-FDR Corrected\n"
    "Blue = suppression during FoG  |  Red = increase  |  * = FDR-significant",
    fontsize=11
)

for i in range(n_feat):
    for j in range(n_pat):
        if sig_mat[i, j]:
            ax.text(j, i, '*', ha='center', va='center',
                    fontsize=14, color='black', fontweight='bold')

plt.tight_layout()
save_path = r'C:\Users\arana\OneDrive - UIC\MATLAB\per_patient_fdr_nonoverlap.png'
plt.savefig(save_path, dpi=150)
plt.close()
print(f"\nHeatmap saved -> {save_path}")

# ── PER-PATIENT FDR SURVIVAL COUNT ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  PER-PATIENT FDR SURVIVAL SUMMARY")
print("=" * 70)
print(f"  {'Patient':<10}  {'Subtype':<10}  {'FDR-sig features':>16}  {'Delta direction'}")
print(f"  {'-'*55}")
for pid in pids:
    n_sig   = sum(all_results[pid][f]['reject'] for f in feat_names)
    delta_ds = [all_results[pid][f"{ch}_Delta"]['d'] for ch in EEG_COLS]
    delta_dir = 'suppressed' if np.mean(delta_ds) < 0 else 'increased'
    print(f"  P{pid:<9}  {SUBTYPE[pid]:<10}  {n_sig:>6}/44           {delta_dir}")
