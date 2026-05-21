#!/usr/bin/env python3
"""
Feature importance analysis — answers: "what is the RF actually learning?"
Trains within-patient 5-block temporal CV for all 10 patients, collects
feature importances per fold, and visualises a channels x bands heatmap.
If delta dominates, the model is extracting lower frequencies.
"""

import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os, warnings
warnings.filterwarnings('ignore')

# ── CONFIG (must match main pipeline) ────────────────────────────────────────
FILTERED_DIR = r'C:\Users\arana\OneDrive - UIC\MATLAB\r8gmbtv7w2-3\Filtered Data'
FS          = 500
WIN_SAMPLES = int(4.5 * FS)
STRIDE_SAMP = int(1.0 * FS)
FOG_THRESH  = 0.5
N_BLOCKS    = 5

EEG_COLS = {
    'F3': 4, 'F4': 5, 'Fz': 16,
    'C3': 6, 'C4': 7, 'Cz': 17,
    'P3': 8, 'P4': 9, 'Pz': 18,
    'O1': 10, 'O2': 11,
}
LABEL_COL = 60

BANDS = {
    'Delta': (1,  4),
    'Theta': (4,  8),
    'Alpha': (8,  13),
    'Beta':  (13, 30),
}

PATIENTS = {
    '001': [os.path.join(FILTERED_DIR, '001', f'task_{t}.txt') for t in range(1, 5)],
    '003': [os.path.join(FILTERED_DIR, '003', f'task_{t}.txt') for t in range(1, 5)],
    '004': [os.path.join(FILTERED_DIR, '004', f'task_{t}.txt') for t in range(1, 6)],
    '006': [os.path.join(FILTERED_DIR, '006', f'task_{t}.txt') for t in range(1, 5)],
    '007': [os.path.join(FILTERED_DIR, '007', f'task_{t}.txt') for t in range(1, 5)],
    '008': ([os.path.join(FILTERED_DIR, '008', 'OFF_1', f'task_{t}.txt') for t in range(1, 6)] +
            [os.path.join(FILTERED_DIR, '008', 'OFF_2', f'task_{t}.txt') for t in range(1, 5)]),
    # '009' excluded — severe class imbalance
    '010': [os.path.join(FILTERED_DIR, '010', f'task_{t}.txt') for t in range(1, 5)],
    '011': [os.path.join(FILTERED_DIR, '011', f'task_{t}.txt') for t in range(1, 5)],
    '012': [os.path.join(FILTERED_DIR, '012', f'task_{t}.txt') for t in range(1, 5)],
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
USE_COLS    = sorted(set(EEG_COLS.values()) | {LABEL_COL})
_COL_MAP    = {orig: i for i, orig in enumerate(USE_COLS)}
LBL_I       = _COL_MAP[LABEL_COL]
EEG_INDICES = [_COL_MAP[v] for v in EEG_COLS.values()]
CHAN_NAMES  = list(EEG_COLS.keys())
BAND_NAMES  = list(BANDS.keys())
FEAT_NAMES  = [f"{ch}_{band}" for ch in CHAN_NAMES for band in BAND_NAMES]


def band_power_db(sig, fs, fmin, fmax):
    f, pxx = scipy.signal.welch(sig, fs=fs, window='hann',
                                nperseg=fs, noverlap=fs // 2)
    mask = (f >= fmin) & (f <= fmax)
    return 10.0 * np.log10(np.mean(pxx[mask]) + 1e-12)


def extract_features(data):
    n = len(data)
    X, y = [], []
    for start in range(0, n - WIN_SAMPLES + 1, STRIDE_SAMP):
        chunk = data[start: start + WIN_SAMPLES]
        label = 1 if np.mean(chunk[:, LBL_I]) >= FOG_THRESH else 0
        feat = []
        for col_idx in EEG_INDICES:
            sig = chunk[:, col_idx]
            for flo, fhi in BANDS.values():
                feat.append(band_power_db(sig, FS, flo, fhi))
        X.append(feat)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_patient(file_list):
    chunks = []
    for fp in file_list:
        if not os.path.exists(fp):
            print(f"    WARNING: {fp} not found — skipping")
            continue
        df = pd.read_csv(fp, header=None, usecols=USE_COLS, on_bad_lines='skip')
        df = df[USE_COLS]
        chunks.append(df.values.astype(np.float64))
    return np.vstack(chunks) if chunks else None


# ── EXTRACT FEATURES ──────────────────────────────────────────────────────────
print("Extracting features ...")
all_X, all_y = {}, {}
for pid, files in PATIENTS.items():
    data = load_patient(files)
    if data is None:
        continue
    X, y = extract_features(data)
    all_X[pid] = X
    all_y[pid] = y
    print(f"  P{pid}: {len(y)} windows  FoG={np.sum(y==1)}  Normal={np.sum(y==0)}")

patient_ids = sorted(all_X.keys())

# ── WITHIN-PATIENT 5-BLOCK TEMPORAL CV ───────────────────────────────────────
print("\nTraining RF (within-patient block CV) ...")

# Collect: importances_per_patient[pid] = mean importance vector (44,)
importances_per_patient = {}

for pid in patient_ids:
    X_p, y_p = all_X[pid], all_y[pid]
    if np.sum(y_p == 1) < 5 or np.sum(y_p == 0) < 5:
        print(f"  P{pid}: too few samples — skipped")
        continue

    n_win   = len(X_p)
    indices = np.arange(n_win)
    fold_imps, fold_accs = [], []

    for fold in range(N_BLOCKS):
        t_start  = fold * (n_win // N_BLOCKS)
        t_end    = t_start + (n_win // N_BLOCKS) if fold < N_BLOCKS - 1 else n_win
        test_idx = indices[t_start:t_end]
        train_idx = np.concatenate([indices[:t_start], indices[t_end:]])
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=42, class_weight='balanced', n_jobs=-1,
        )
        clf.fit(X_p[train_idx], y_p[train_idx])
        fold_imps.append(clf.feature_importances_)
        fold_accs.append(accuracy_score(y_p[test_idx], clf.predict(X_p[test_idx])))

    importances_per_patient[pid] = np.mean(fold_imps, axis=0)
    print(f"  P{pid}: mean acc={np.mean(fold_accs):.3f}")

# ── RESHAPE TO CHANNELS x BANDS ───────────────────────────────────────────────
# Feature order: channel outer loop, band inner loop → (n_chan, n_band)
n_chan = len(CHAN_NAMES)
n_band = len(BAND_NAMES)

# Mean importance across all patients
mean_imp_all = np.mean(list(importances_per_patient.values()), axis=0)
imp_matrix   = mean_imp_all.reshape(n_chan, n_band)   # (11, 4)

# Per-patient matrices for std
per_patient_matrices = np.array(
    [v.reshape(n_chan, n_band) for v in importances_per_patient.values()]
)
std_matrix = np.std(per_patient_matrices, axis=0)

# Band-level totals (sum across channels) — shows which band dominates overall
band_totals = imp_matrix.sum(axis=0)   # (4,)

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("   BAND-LEVEL IMPORTANCE (summed across channels)")
print("=" * 55)
for band, total in zip(BAND_NAMES, band_totals):
    bar = '#' * int(total * 500)
    print(f"  {band:6s}: {total:.4f}  {bar}")

print("\n" + "=" * 55)
print("   TOP 10 INDIVIDUAL FEATURES (mean across patients)")
print("=" * 55)
order = np.argsort(mean_imp_all)[::-1]
for rank, idx in enumerate(order[:10]):
    bar = '#' * int(mean_imp_all[idx] * 300)
    print(f"  {rank+1:2d}. {FEAT_NAMES[idx]:14s} {mean_imp_all[idx]:.4f}  {bar}")

# ── FIGURES ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("RF Feature Importance — What Is the Model Learning?", fontsize=14, fontweight='bold')

# ── Panel 1: Heatmap channels × bands ────────────────────────────────────────
ax = axes[0]
im = ax.imshow(imp_matrix, cmap='YlOrRd', aspect='auto',
               vmin=0, vmax=imp_matrix.max())
ax.set_xticks(range(n_band))
ax.set_xticklabels(BAND_NAMES, fontsize=11)
ax.set_yticks(range(n_chan))
ax.set_yticklabels(CHAN_NAMES, fontsize=10)
ax.set_title('Mean Importance\n(channels × bands)', fontsize=11)
ax.set_xlabel('Frequency band')
ax.set_ylabel('EEG channel')

# Annotate cells
for i in range(n_chan):
    for j in range(n_band):
        val = imp_matrix[i, j]
        color = 'white' if val > imp_matrix.max() * 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=8, color=color)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── Panel 2: Band totals bar chart ───────────────────────────────────────────
ax = axes[1]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']  # delta=blue, theta=green, alpha=orange, beta=red
bars = ax.bar(BAND_NAMES, band_totals, color=colors, edgecolor='black', linewidth=0.8)
ax.set_title('Total Importance by Band\n(summed across all channels)', fontsize=11)
ax.set_ylabel('Summed feature importance')
ax.set_ylim(0, band_totals.max() * 1.2)
for bar, val in zip(bars, band_totals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Highlight dominant band
dominant_idx = np.argmax(band_totals)
bars[dominant_idx].set_edgecolor('black')
bars[dominant_idx].set_linewidth(2.5)
ax.text(dominant_idx, band_totals[dominant_idx] * 0.5,
        '★ dominant', ha='center', va='center', fontsize=9,
        color='white', fontweight='bold')

# ── Panel 3: Per-patient importance profile (band totals per patient) ─────────
ax = axes[2]
x = np.arange(n_band)
width = 0.08
pids = list(importances_per_patient.keys())
cmap = plt.cm.get_cmap('tab10', len(pids))

for i, pid in enumerate(pids):
    pat_matrix = importances_per_patient[pid].reshape(n_chan, n_band)
    pat_totals = pat_matrix.sum(axis=0)
    offset = (i - len(pids) / 2) * width
    ax.bar(x + offset, pat_totals, width, label=f'P{pid}',
           color=cmap(i), alpha=0.8, edgecolor='none')

ax.set_xticks(x)
ax.set_xticklabels(BAND_NAMES, fontsize=11)
ax.set_title('Band Importance per Patient\n(consistency check)', fontsize=11)
ax.set_ylabel('Summed importance')
ax.legend(fontsize=7, ncol=2, loc='upper right')

plt.tight_layout()
save_path = r'C:\Users\arana\OneDrive - UIC\MATLAB\feature_importance_heatmap.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved → {save_path}")
print("\nDone. If Delta dominates the heatmap and bar chart, the model is learning lower frequencies.")
