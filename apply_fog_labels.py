#!/usr/bin/env python3
"""
Transfers FoG labels from the doctors' files into our 4-Labeled Data files,
aligned by timestamp.

Doctors' files (Filtered Data/010/task_N.txt):
  no header | col 0 = row index | col 1 = timestamp | last col = Label (1=FoG, 0=non-FoG)

Our files (4-Labeled Data/010/task_N.txt):
  no header | col 0 = row index | col 1 = TIME       | last col = Label (currently all 0)
"""

import pandas as pd
import os

DOCTORS_DIR = r"C:\Users\arana\OneDrive - UIC\MATLAB\r8gmbtv7w2-3\Filtered Data\010"
LABELED_DIR = r"C:\Users\arana\OneDrive - UIC\MATLAB\t8j8v4hnm4-1\Raw\010\4-Labeled Data\010"
TASKS       = 4

def normalize_ts(ts):
    ts = str(ts).strip()
    if '.' not in ts:
        ts += '.000'
    else:
        parts = ts.split('.')
        ts = parts[0] + '.' + parts[1][:3].ljust(3, '0')
    return ts

print('=' * 60)
total_fog = 0

for t in range(1, TASKS + 1):
    name     = f"task_{t}"
    doc_path = os.path.join(DOCTORS_DIR, f"{name}.txt")
    our_path = os.path.join(LABELED_DIR, f"{name}.txt")

    # --- Doctors' file ---
    doc      = pd.read_csv(doc_path, header=None)
    doc_ts   = doc.iloc[:, 1].astype(str).apply(normalize_ts)
    doc_lbl  = doc.iloc[:, -1].astype(int)
    label_map = dict(zip(doc_ts, doc_lbl))
    print(f"{name}: doctors → {len(doc)} rows | FoG = {doc_lbl.sum()} samples ({doc_lbl.sum()/500:.1f}s)")

    # --- Our file ---
    our    = pd.read_csv(our_path, header=None)
    our_ts = our.iloc[:, 1].astype(str).apply(normalize_ts)

    new_labels = our_ts.map(label_map)
    unmatched  = new_labels.isna().sum()
    if unmatched > 0:
        print(f"  note: {unmatched} timestamps past doctors' end → labelled 0 (expected)")

    new_labels = new_labels.fillna(0).astype(int)
    our.iloc[:, -1] = new_labels
    our.to_csv(our_path, header=False, index=False)

    fog_written = new_labels.sum()
    total_fog  += fog_written
    print(f"  our   → {len(our)} rows | FoG written = {fog_written} samples ({fog_written/500:.1f}s) ✓")

print('=' * 60)
print(f"Total FoG across all tasks: {total_fog} samples ({total_fog/500:.1f}s)")
