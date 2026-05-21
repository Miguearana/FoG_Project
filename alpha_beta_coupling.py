import numpy as np
from scipy.stats import pearsonr

patients = ['001','003','004','006','007','008','009','010','011','012']

# Mean Cohen's d across F3/F4/Fz (non-overlapping windows, n=10 patients)
alpha_d = [-0.226, +0.418, +0.071, -0.064, +0.032, +0.286, +0.351, -0.070, +0.160, +0.495]
beta_d  = [+0.152, +0.338, -0.224, -0.253, -0.146, +0.385, +0.664, -0.059, +0.110, +0.338]

r, p = pearsonr(alpha_d, beta_d)
print(f"Alpha-Beta coupling across patients:")
print(f"  Pearson r = {r:.3f},  p = {p:.4f}")
print()
print("Per-patient mean d (averaged F3/F4/Fz):")
print(f"  {'Patient':<8} {'Alpha d':>9} {'Beta d':>9}  Subtype")
for i, pid in enumerate(patients):
    sub = 'A (increase)' if alpha_d[i] > 0 else 'B (suppress)'
    print(f"  P{pid:<6} {alpha_d[i]:>+9.3f} {beta_d[i]:>+9.3f}  {sub}")

subtypeA = [pid for i, pid in enumerate(patients) if alpha_d[i] > 0]
subtypeB = [pid for i, pid in enumerate(patients) if alpha_d[i] < 0]
print()
print(f"Subtype A (alpha+beta increase): P{subtypeA}  n={len(subtypeA)}")
print(f"Subtype B (alpha+beta suppress): P{subtypeB}  n={len(subtypeB)}")
