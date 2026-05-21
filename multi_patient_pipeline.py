#!/usr/bin/env python3
"""
Multi-patient LOPO FoG detection pipeline.
Reads authors' Filtered Data files directly (fast path).

Column layout (0-indexed): 0=idx, 1=time, 4=F3, 5=F4, 6=C3, 7=C4, 8=P3, 9=P4, 10=O1, 11=O2, 16=Fz, 17=Cz, 18=Pz, 60=Label
Sampling rate: 500 Hz
"""

import matplotlib
matplotlib.use('Agg')   # backend sin GUI, evita errores de tkinter en PowerShell
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
FILTERED_DIR = r'C:\Users\arana\OneDrive - UIC\MATLAB\r8gmbtv7w2-3\Filtered Data'
OUT_DIR      = r'C:\Users\arana\OneDrive - UIC\MATLAB'
FS           = 500
WIN_SAMPLES  = int(4.5 * FS)   # 2250 samples = 4.5 s
STRIDE_SAMP  = int(1.0 * FS)   # 500 samples  = 1.0 s
FOG_THRESH   = 0.5              # majority vote: ≥50% FoG samples → FoG window

EEG_COLS  = {
    'F3': 4,  'F4': 5,  'Fz': 16,   # Frontal
    'C3': 6,  'C4': 7,  'Cz': 17,   # Central (motor cortex)
    'P3': 8,  'P4': 9,  'Pz': 18,   # Parietal
    'O1': 10, 'O2': 11,              # Occipital
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
    # P009 excluded: severe class imbalance (dataset paper recommendation, Zhang et al. 2022) + specificity loss in multimodal (supervisor recommendation)
    '010': [os.path.join(FILTERED_DIR, '010', f'task_{t}.txt') for t in range(1, 5)],
    '011': [os.path.join(FILTERED_DIR, '011', f'task_{t}.txt') for t in range(1, 5)],
    '012': [os.path.join(FILTERED_DIR, '012', f'task_{t}.txt') for t in range(1, 5)],
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def band_power_db(sig, fs, fmin, fmax):
    f, pxx = scipy.signal.welch(sig, fs=fs, window='hann',
                                nperseg=fs, noverlap=fs // 2)
    mask = (f >= fmin) & (f <= fmax)
    return 10.0 * np.log10(np.mean(pxx[mask]) + 1e-12)


def extract_features(data):
    """Sliding-window feature extraction. Returns X (n_win, 44) and y (n_win,)."""
    n = len(data)
    X, y = [], []
    for start in range(0, n - WIN_SAMPLES + 1, STRIDE_SAMP):
        chunk = data[start: start + WIN_SAMPLES]
        fog_ratio = np.mean(chunk[:, LBL_I])
        label = 1 if fog_ratio >= FOG_THRESH else 0

        feat = []
        for col_idx in EEG_INDICES:
            sig = chunk[:, col_idx]
            for flo, fhi in BANDS.values():
                feat.append(band_power_db(sig, FS, flo, fhi))
        X.append(feat)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


USE_COLS    = sorted(set(EEG_COLS.values()) | {LABEL_COL})
_COL_MAP    = {orig: i for i, orig in enumerate(USE_COLS)}
LBL_I       = _COL_MAP[LABEL_COL]
EEG_INDICES = [_COL_MAP[v] for v in EEG_COLS.values()]


def load_patient(file_list):
    """Load only EEG + label columns, concatenate all task files."""
    chunks = []
    for fp in file_list:
        if not os.path.exists(fp):
            print(f"    WARNING: {fp} not found — skipping")
            continue
        df = pd.read_csv(fp, header=None, usecols=USE_COLS, on_bad_lines='skip')
        df = df[USE_COLS]
        chunks.append(df.values.astype(np.float64))
    return np.vstack(chunks) if chunks else None


# ── BUILD FEATURE MATRICES ───────────────────────────────────────────────────
print("Extracting features (this may take ~1-2 minutes) ...")
all_X, all_y = {}, {}

for pid, files in PATIENTS.items():
    data = load_patient(files)
    if data is None:
        print(f"  P{pid}: no data — skipped")
        continue
    X, y = extract_features(data)
    all_X[pid] = X
    all_y[pid] = y
    print(f"  P{pid}: {len(y):5d} windows  Normal={np.sum(y==0):4d}  FoG={np.sum(y==1):4d}")

patient_ids = sorted(all_X.keys())

# ── LOPO CROSS-VALIDATION ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("   LEAVE-ONE-PATIENT-OUT  (Random Forest, n=200)")
print("=" * 55)

all_y_true, all_y_pred = [], []
fold_importances        = []
per_patient             = {}

for test_pid in patient_ids:
    train_X = np.vstack([all_X[p] for p in patient_ids if p != test_pid])
    train_y = np.hstack([all_y[p] for p in patient_ids if p != test_pid])

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight='balanced', n_jobs=-1,
    )
    clf.fit(train_X, train_y)

    y_pred = clf.predict(all_X[test_pid])
    y_true = all_y[test_pid]

    acc  = accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp_, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp_) if (tn + fp_) > 0 else 0.0

    per_patient[test_pid] = dict(acc=acc, sens=sens, spec=spec, cm=cm)
    fold_importances.append(clf.feature_importances_)
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

    print(f"  P{test_pid}  Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  "
          f"[FoG {np.sum(y_true==1)}/{len(y_true)} windows]")

# Aggregate LOPO
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
agg_cm     = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
tn, fp_, fn, tp = agg_cm.ravel()

print("\n" + "=" * 55)
print("   AGGREGATE (all patients pooled)")
print("=" * 55)
print(f"Accuracy:    {accuracy_score(all_y_true, all_y_pred):.4f}")
print(f"Sensitivity: {tp/(tp+fn):.4f}   (FoG recall)")
print(f"Specificity: {tn/(tn+fp_):.4f}  (Normal recall)")
print("\nConfusion Matrix (rows=True, cols=Predicted):")
print(f"             Normal   FoG")
print(f"  Normal     {agg_cm[0,0]:6d}  {agg_cm[0,1]:6d}")
print(f"  FoG        {agg_cm[1,0]:6d}  {agg_cm[1,1]:6d}")
print()
print(classification_report(all_y_true, all_y_pred, target_names=['Normal', 'FoG']))

# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
feat_names = [f"{ch}_{band}" for ch in EEG_COLS for band in BANDS]
mean_imp   = np.mean(fold_importances, axis=0)
order      = np.argsort(mean_imp)[::-1]

print("=" * 55)
print("   MEAN FEATURE IMPORTANCE (averaged over LOPO folds)")
print("=" * 55)
for rank, idx in enumerate(order):
    bar = '#' * int(mean_imp[idx] * 200)
    print(f"  {rank+1:2d}. {feat_names[idx]:12s} {mean_imp[idx]:.4f}  {bar}")

# ── LOPO PLOTS ────────────────────────────────────────────────────────────────
_fig_h = max(5, len(feat_names) * 0.3)
fig, axes = plt.subplots(1, 3, figsize=(18, _fig_h))
fig.suptitle("Multi-Patient LOPO FoG Detection", fontsize=13)

ax = axes[0]
pids  = list(per_patient.keys())
accs  = [per_patient[p]['acc']  for p in pids]
senss = [per_patient[p]['sens'] for p in pids]
specs = [per_patient[p]['spec'] for p in pids]
x = np.arange(len(pids))
w = 0.28
ax.bar(x - w, accs,  w, label='Accuracy',    color='steelblue')
ax.bar(x,     senss, w, label='Sensitivity', color='tomato')
ax.bar(x + w, specs, w, label='Specificity', color='mediumseagreen')
ax.set_xticks(x); ax.set_xticklabels([f'P{p}' for p in pids])
ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
ax.set_title('Per-Patient LOPO Performance'); ax.legend(fontsize=8)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

ax = axes[1]
im = ax.imshow(agg_cm, cmap='Blues', interpolation='nearest')
ax.set_title('Aggregate LOPO Confusion Matrix')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'FoG']); ax.set_yticklabels(['Normal', 'FoG'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(agg_cm[i, j]), ha='center', va='center', fontsize=13,
                color='white' if agg_cm[i, j] > agg_cm.max() / 2 else 'black')
plt.colorbar(im, ax=ax)

ax = axes[2]
ax.barh(range(len(feat_names)), mean_imp[order[::-1]], color='steelblue')
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=9)
ax.set_xlabel('Mean importance')
ax.set_title('Feature Importance (mean over LOPO folds)')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'LOPO_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved -> {OUT_DIR}\\LOPO_results.png")

# ── WITHIN-PATIENT 5-BLOCK TEMPORAL CV ───────────────────────────────────────
print("\n" + "=" * 55)
print("   WITHIN-PATIENT 5-BLOCK TEMPORAL CV  (no leakage)")
print("=" * 55)

N_BLOCKS = 5
within_results     = {}
within_importances = []

SUBTYPES = {
    'A':          ['003', '008', '009', '011', '012'],
    'B':          ['001', '006', '010'],
    'Borderline': ['004', '007'],
}
SUBTYPE_COLORS = {'A': '#E8721C', 'B': '#2471A3', 'Borderline': '#7F8C8D'}

def get_subtype(pid):
    for st, lst in SUBTYPES.items():
        if pid in lst:
            return st
    return 'Borderline'

for pid in patient_ids:
    X_p = all_X[pid]
    y_p = all_y[pid]

    if np.sum(y_p == 1) < 5 or np.sum(y_p == 0) < 5:
        print(f"  P{pid}: too few samples — skipped")
        continue

    n_win   = len(X_p)
    indices = np.arange(n_win)
    fold_accs, fold_prec, fold_rec, fold_f1, fold_specs, fold_imps = [], [], [], [], [], []
    all_true_p, all_pred_p = [], []

    for fold in range(N_BLOCKS):
        t_start   = fold * (n_win // N_BLOCKS)
        t_end     = t_start + (n_win // N_BLOCKS) if fold < N_BLOCKS - 1 else n_win
        test_idx  = indices[t_start:t_end]
        train_idx = np.concatenate([indices[:t_start], indices[t_end:]])
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=42, class_weight='balanced', n_jobs=-1,
        )
        clf.fit(X_p[train_idx], y_p[train_idx])
        y_pred = clf.predict(X_p[test_idx])
        y_true = y_p[test_idx]

        all_true_p.extend(y_true)
        all_pred_p.extend(y_pred)

        cm_f = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp_, fn, tp = cm_f.ravel()
        fold_accs.append(accuracy_score(y_true, y_pred))
        fold_prec.append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        fold_rec.append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        fold_f1.append(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        fold_specs.append(tn / (tn + fp_) if (tn + fp_) > 0 else 0.0)
        fold_imps.append(clf.feature_importances_)

    all_true_p = np.array(all_true_p)
    all_pred_p = np.array(all_pred_p)
    cm_patient = confusion_matrix(all_true_p, all_pred_p, labels=[0, 1])

    within_results[pid] = dict(
        acc=np.mean(fold_accs),  acc_std=np.std(fold_accs),
        prec=np.mean(fold_prec), rec=np.mean(fold_rec),
        f1=np.mean(fold_f1),     spec=np.mean(fold_specs),
        imp=np.mean(fold_imps, axis=0),
        cm=cm_patient,
    )
    within_importances.append(np.mean(fold_imps, axis=0))

    subtype = get_subtype(pid)
    print(f"  P{pid} [{subtype}]  "
          f"Acc={np.mean(fold_accs):.3f}±{np.std(fold_accs):.3f}  "
          f"Recall={np.mean(fold_rec):.3f}  "
          f"Prec={np.mean(fold_prec):.3f}  "
          f"F1={np.mean(fold_f1):.3f}  "
          f"Spec={np.mean(fold_specs):.3f}")

    # Confusion matrix por paciente guardada como PNG
    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_patient, display_labels=['Normal', 'FoG'])
    disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title(
        f'P{pid}  ·  Subtipo {subtype}\n'
        f'Acc {np.mean(fold_accs)*100:.1f}%  |  '
        f'Recall FoG {np.mean(fold_rec)*100:.1f}%  |  '
        f'F1 {np.mean(fold_f1)*100:.1f}%',
        fontsize=10, fontweight='bold', color=SUBTYPE_COLORS[subtype]
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'confmat_within_P{pid}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ── TABLA RESUMEN ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"  {'Pat':>5}  {'Subtipo':>10}  {'Acc':>6}  {'Recall':>6}  {'Prec':>6}  {'F1':>6}  {'Spec':>6}  | LOPO")
print("  " + "-" * 73)
for pid in patient_ids:
    if pid not in within_results:
        continue
    r  = within_results[pid]
    l  = per_patient[pid]
    st = get_subtype(pid)
    print(f"  P{pid}  {st:>10}  "
          f"{r['acc']*100:>5.1f}%  "
          f"{r['rec']*100:>5.1f}%  "
          f"{r['prec']*100:>5.1f}%  "
          f"{r['f1']*100:>5.1f}%  "
          f"{r['spec']*100:>5.1f}%  | "
          f"{l['acc']*100:>5.1f}%")

mean_acc  = np.mean([within_results[p]['acc']  for p in within_results])
mean_rec  = np.mean([within_results[p]['rec']  for p in within_results])
mean_prec = np.mean([within_results[p]['prec'] for p in within_results])
mean_f1   = np.mean([within_results[p]['f1']   for p in within_results])
mean_lopo = np.mean([per_patient[p]['acc']     for p in patient_ids if p in within_results])
print("  " + "=" * 73)
print(f"  Media:  Acc={mean_acc*100:.1f}%  Recall={mean_rec*100:.1f}%  "
      f"Prec={mean_prec*100:.1f}%  F1={mean_f1*100:.1f}%  | LOPO={mean_lopo*100:.1f}%")

# ── WITHIN-PATIENT PLOT ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, _fig_h))
fig.suptitle("Within-Patient 5-Block Temporal CV  vs  LOPO", fontsize=13)

ax = axes[0]
pids_w = [p for p in patient_ids if p in within_results]
x = np.arange(len(pids_w))
w = 0.22
ax.bar(x - 1.5*w, [within_results[p]['acc'] for p in pids_w], w, label='Accuracy',   color='steelblue')
ax.bar(x - 0.5*w, [within_results[p]['rec'] for p in pids_w], w, label='Recall FoG', color='tomato')
ax.bar(x + 0.5*w, [within_results[p]['f1']  for p in pids_w], w, label='F1 FoG',    color='darkorange')
ax.bar(x + 1.5*w, [per_patient[p]['acc']    for p in pids_w], w, label='LOPO Acc',  color='gray', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([f'P{p}' for p in pids_w])
ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
ax.set_title('Within-Patient: Accuracy / Recall / F1 / LOPO')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Azar')
ax.legend(fontsize=8)

ax = axes[1]
if within_importances:
    wi_mean  = np.mean(within_importances, axis=0)
    wi_order = np.argsort(wi_mean)[::-1]
    ax.barh(range(len(feat_names)), wi_mean[wi_order[::-1]], color='steelblue')
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels([feat_names[i] for i in wi_order[::-1]], fontsize=9)
    ax.set_xlabel('Mean importance')
    ax.set_title('Feature Importance (within-patient folds)')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'within_patient_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlots guardados en: {OUT_DIR}")
print("Matrices de confusion por paciente: confmat_within_P***.png")
