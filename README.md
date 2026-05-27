# FoG_Project
EEG-Based Detection of Freezing of Gait in Parkinson’s Disease using multimodal machine learning (EEG, EMG, IMU), temporal cross-validation, and patient-specific cortical analysis.

# Multimodal Characterization and Detection of Freezing of Gait in Parkinson's Disease

**Author:** Miguel Arana Salas

A multimodal machine learning framework for the characterization and offline detection
of Freezing of Gait (FoG) episodes in Parkinson's disease, integrating cortical EEG,
electromyography (EMG), and inertial measurement unit (IMU) signals.

---

## Background

Freezing of Gait is one of the most disabling and least therapeutically addressable
motor complications of Parkinson's disease, affecting approximately 50% of all PD
patients and up to 80% of those in advanced stages. FoG episodes are the leading
cause of falls in PD, contribute significantly to loss of independence, and respond
inconsistently to levodopa and deep brain stimulation.

The pathophysiology is rooted in basal ganglia dysfunction: selective degeneration of
dopaminergic neurons in the substantia nigra disrupts the balance between direct and
indirect striatal pathways, leading to overinhibition of thalamocortical circuits and
a failure of motor initiation under conditions of dual-task demand or environmental
triggers.

Scalp EEG provides a non-invasive window into the thalamocortical consequences of
this dysfunction, while IMUs directly capture the kinematic expression of freezing at
the level of the limbs. Neither modality alone is sufficient — their combination is
the basis of this framework.

---

## Dataset

- **Source:** Zhang et al. (2022), *Scientific Data* — DOI: [10.17632/r8gmbtv7w2.3](https://doi.org/10.17632/r8gmbtv7w2.3)
- The only publicly available database combining simultaneous EEG, EMG, and IMU recordings in PD patients with FoG.
- 10 patients included in analysis; FoG episodes labeled by two qualified physicians.
- Modalities: EEG (25 channels, 500 Hz), EMG (5 channels), ACC/GYRO (lower limbs, trunk, arm).

---

## Methods

- **Feature extraction:** Band power (dB, Welch PSD) from 11 neuroanatomically justified
  EEG channels × 4 frequency bands (delta, theta, alpha, beta), combined with kinematic
  and electromyographic features.
- **Classifier:** Random Forest (n=200 estimators, max depth=8, balanced class weights).
- **Validation:** Leak-free five-block temporal cross-validation — training on earlier
  time blocks, predicting on later ones — to simulate prospective deployment and avoid
  the data leakage present in standard random cross-validation.
- **Generalizability:** Leave-one-patient-out (LOPO) evaluation for cross-patient
  performance assessment.

---

## Key Findings

1. **FoG is patient-specific.** Within-patient cross-validation achieved a mean AUC of
   **0.940** (multimodal), while LOPO AUC collapsed to **0.521** (EEG-only, near chance),
   confirming that generalized models are insufficient and individualized approaches
   are necessary.

2. **FoG is fundamentally biomechanical.** IMU sensors account for **73–91%** of
   classifier feature importance across patients; EEG contributes only **6–21%**,
   indicating that kinematic features dominate FoG detection.

3. **Two cortical subtypes exist.** Spectral analysis reveals inter-patient heterogeneity:
   - **Subtype A:** delta decrease + alpha and beta synchrony during FoG. 
   - **Subtype B:** delta decrease without cortical hypersynchrony, suggesting a
     distinct, potentially subcortical mechanism.

---

## Repository Structure

```text
├── multi_patient_pipeline.py       # Main ML pipeline: feature extraction, LOPO, within-patient CV
├── apply_fog_labels.py             # Transfers physician labels by timestamp alignment
├── group_stats.py                  # Wilcoxon rank-sum, Cohen's d, BH-FDR correction
├── per_patient_stats_nonoverlap.py # Non-overlapping window statistics per patient
├── signal_quality.py               # EEG quality metrics (HF/LF ratio, kurtosis)
├── alpha_beta_coupling.py          # Alpha–beta Pearson correlation, subtype analysis
├── feature_importance_analysis.py  # RF feature importance heatmap (channels × bands)
├── figures.py                      # Group PSD, boxplots, subtype figures 
└── narrative_figures.py            # Per-patient delta, LOPO vs within-patient, A–B scatter
```

---

## Requirements

```
Python 3.x
numpy
scipy
scikit-learn
matplotlib
pandas
```

---

## Citation

If you use this work, please cite the original dataset:

> Zhang, W. et al. (2022). A dataset of EEG, EMG, and inertial sensor data for
> Freezing of Gait in Parkinson's disease. *Scientific Data.*
> https://doi.org/10.17632/r8gmbtv7w2.3

---

## License

[MIT License](LICENSE)
