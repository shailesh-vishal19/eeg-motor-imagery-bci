# EEG Motor Imagery Classification (Brain-Computer Interface)

Decoding imagined left-hand vs. right-hand movement from EEG signals, comparing a classical
Common Spatial Patterns (CSP) + LDA baseline against a compact deep learning model (EEGNet).

## Motivation

Motor imagery — vividly imagining a movement without performing it — produces a measurable
neural signature: suppression of mu (8–12 Hz) and beta (13–30 Hz) rhythms over motor cortex,
concentrated around electrodes C3/C4. This is the basis for many real-world brain-computer
interfaces (e.g. controlling a prosthetic or cursor by imagined movement alone). This project
asks a simple question: can a lightweight, from-scratch pipeline decode *which hand* someone
is imagining moving, using only EEG?

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset** (public domain, no data-use agreement required).

- Subjects used: **[ADD YOUR FINAL SUBJECT COUNT — e.g. 40]**
- Runs: 4, 8, 12 (imagined left-fist / right-fist movement trials)
- Preprocessing: 7–30 Hz bandpass filter, epoched −0.5s to 3.5s around each cue
- Final dataset: **[ADD FINAL EPOCH COUNT AND CLASS BALANCE — e.g. 1800 epochs, 905 left / 895 right]**

## Methods

**Classical baseline — CSP + LDA.** CSP (6 components) learns spatial filters that maximize the
variance difference between the two classes; LDA classifies on the resulting log-variance
features. Evaluated two ways:
- *Pooled*: one CSP+LDA model fit across all subjects' data together.
- *Per-subject*: a separate CSP+LDA model fit and cross-validated within each subject, then
  averaged — the standard protocol in BCI research, since the scalp pattern of motor imagery
  varies somewhat person to person.

**Deep learning — EEGNet** (Lawhern et al., 2018). A compact CNN purpose-built for EEG: a
temporal convolution, followed by a depthwise spatial convolution (playing a similar role to
CSP), followed by a separable convolution and a linear classifier. Trained with Adam, 60 epochs,
on pooled multi-subject data.

## Results

| Model                              | Accuracy | Cohen's κ | Notes |
|-------------------------------------|----------|-----------|-------|
| CSP+LDA — pooled                    | [e.g. 0.529 ± 0.043] | — | Near chance; see discussion below |
| CSP+LDA — per-subject (mean)        | **[ADD YOUR PER-SUBJECT NUMBER]** | — | Standard BCI evaluation protocol |
| EEGNet — pooled, unscaled input     | 0.506 | 0.020 | Model collapsed to predicting one class — see *Lessons learned* |
| EEGNet — pooled, scaled to µV       | **[ADD YOUR FINAL 40-SUBJECT NUMBER]** | **[ADD κ]** | After fixing input amplitude scaling |

*(Fill in the pooled/per-subject numbers from your own run above — everything else in this table is already final.)*

## Key finding

A pooled, subject-generic classifier performs only modestly above chance on this task. This
matches a well-documented effect in BCI research: motor imagery signals differ enough
between people that models trained across many subjects at once struggle to find a single
shared pattern, and per-subject calibration recovers substantially more of the signal. The gap
between the pooled and per-subject numbers above **is** the finding, not a limitation to hide.

## Lessons learned (debugging notes)

Early runs of EEGNet on raw EEG (values on the order of 1e-6, in volts) produced a classifier
that collapsed to predicting a single class regardless of input (accuracy ~0.51, κ ~0.02) —
this is a known failure mode when neural network inputs are too small in magnitude for stable
gradients. Rescaling inputs to microvolts (×1e6) before training fixed this and produced a
genuinely discriminative (if still modest) model. This is included here deliberately, since
recognizing and diagnosing this kind of failure is itself part of the result.

## Interpretability

CSP spatial patterns (below) show which regions of the scalp the model relies on most. For a
genuinely motor-imagery-driven signal, this should emphasize central electrodes (around C3/C4),
consistent with known motor cortex topology.

`[Insert your topomap screenshot from Step 6 here — e.g. ![CSP patterns](csp_patterns.png)]`

## Limitations

- No cross-subject generalization test (train on some subjects, evaluate on entirely
  held-out subjects) — a natural next step, and a harder, more realistic test of a BCI system.
- Subject count is a fraction of the full 109-subject dataset; results may shift with more data.
- Hyperparameters (EEGNet architecture, training epochs) were not extensively tuned, given the
  time-boxed scope of this project.

## How to run

1. Open in Kaggle (or any Jupyter environment with GPU).
2. Enable GPU acceleration.
3. Run cells top to bottom — Step 1 downloads data automatically via MNE-Python (no manual
   download or credentialing needed).

## Citations

- Schalk, G., et al. (2004). *BCI2000: A General-Purpose Brain-Computer Interface (BCI)
  System.* IEEE Transactions on Biomedical Engineering.
- Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation.
- Lawhern, V. J., et al. (2018). *EEGNet: A Compact Convolutional Network for EEG-based
  Brain-Computer Interfaces.* Journal of Neural Engineering.
