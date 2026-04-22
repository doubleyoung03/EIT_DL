# Changelog

## 2026-04-17 — Match dataset preprocessing with inference (floor + dv clip)

### What changed
`src/dataset.py::EITDataset.__init__` now applies the SAME numerical
safeguards as `D:\Yang\EIT\Implementation\inference.py::preprocess` on the
differential-normalisation step:

1. Channels with `|V_ref| < _REF_ABS_FLOOR = 5e-5 V` (≈ the electrical
   neutral line) are treated as neutral — their `dv` row entries are set
   to `0.0` instead of the unstable `(v − v_ref) / |v_ref|`.
2. The resulting `dv` tensor is clipped to `±_DV_CLIP = 700.0` before
   being handed to `StandardScaler`.

Two module-level constants `_REF_ABS_FLOOR` and `_DV_CLIP` were added with
explicit documentation that they must stay in sync with
`Implementation/inference.py`.

### Why
Live validation with `pipeline_diagnostic.py` on the deployed
`transformer_decoder_large` checkpoint (epoch 193, val MSE 0.002645)
exposed a train↔inference mismatch:

- Training (old dataset.py): `dv = (v − v_ref) / |v_ref|` — **no floor,
  no clip**. Near-neutral channels ended up with dv values of 10^4–10^6
  magnitude, on which the `StandardScaler` fitted a degenerate statistics
  pair (tiny σ, large μ).
- Inference: floored & clipped — produced `dv ≈ 0` on those same
  channels, which the scaler then mapped to z-scores of −10^2..−10^4.

Symptoms observed in the GUI:
- Empty tank reconstructed as a saturated blue blob (NN output = −1.0).
- Moving the insulating bottle did not change the reconstruction —
  spatial information was drowned under the ~10^2× OOD offset.
- `pipeline_diagnostic` Stage 4 with floor = 5e-5 reported
  `|scaled|_max = 164`; with floor = 1e-12 it jumped to
  `|scaled|_max = 40495`, confirming the scaler was the fragile element.

Applying floor + clip in `dataset.py` refits the scaler on the SAME
distribution the model sees at inference, eliminating the OOD gap by
construction.

### Files affected
| File | Change |
|------|--------|
| `src/dataset.py` | Added `_REF_ABS_FLOOR`, `_DV_CLIP` constants; extended `EITDataset.__init__` to floor + zero neutral-channel dv and to clip dv before scaler fit. |
| `CHANGELOG.md` | This entry. |

### Retrain required
Yes. Existing checkpoints pre-date this change and are therefore coupled
to the unstable scaler; retrain from scratch against the 200-mm, 16 000-
sample dataset (`eit_dataset.csv`) before redeploying. Inference-side
band-aid (`_SCALED_CLIP = ±5` in `Implementation/inference.py`) keeps the
old checkpoint usable in the meantime.

### Validation
- Static: `ReadLints` clean.
- Required follow-up (Young): set `config.yaml::data_dir` to the folder
  containing the 16 000-sample CSV and `tank_radius: 100.0`, then
  `python src/train.py`. After training, `pipeline_diagnostic` Stage 4
  should report `|scaled|_max < 10` by construction.

---

## 2026-04-12 — Add Reference Voltage Differential Processing

### What changed
Introduced homogeneous-medium reference voltage subtraction and normalization
into the EIT data processing pipeline. Previously, raw absolute voltages were
masked and directly StandardScaler-normalized before being fed to the network.
Now the pipeline computes `(V_measured - V_ref) / |V_ref|` before scaling,
so the network sees relative voltage changes caused by the anomaly rather than
absolute voltage values.

### Why
In standard EIT practice the input to the reconstruction algorithm should be
the differential signal relative to a known homogeneous baseline. Using absolute
voltages conflates electrode-geometry effects with anomaly-induced changes and
degrades reconstruction quality.

### Files affected
| File | Change |
|------|--------|
| `config.yaml` | Added `reference_voltage_csv: reference_voltages.csv` key under the dataset section. |
| `src/dataset.py` | `get_dataloaders()` now loads the reference CSV, extracts and masks 224 reference values, and passes them to `EITDataset`. `EITDataset.__init__()` accepts a new `v_ref_224` parameter and applies `(V - V_ref) / |V_ref|` before `StandardScaler`. Module and class docstrings updated accordingly. |
| `src/train.py` | Checkpoint dict now includes `reference_voltage_csv` for reproducibility tracking. |
| `data/reference_voltages.csv` | New file — single-row CSV with 256 homogeneous-medium reference voltages (added by user). |

### Validation
- Linter: no errors introduced in any modified file.
- Input dimension remains 224; no model architecture change required.
- Existing checkpoints are incompatible with the new processing and require retraining.

## 2026-04-12 — Add Auto Threshold Calibration For Test Visualisation

### What changed
Improved test-time post-processing to reduce systematic anomaly over-segmentation.
`test.py` now supports automatic threshold calibration using the validation split:
it searches candidate `vis_class_threshold` values and selects the one that best
balances overlap quality (IoU) and predicted-vs-ground-truth area ratio.
Test logging now records the selected threshold and calibration statistics.
Visualization defaults were also tightened to reduce boundary dilation.

### Why
Predicted anomaly shapes were consistently larger than ground truth on testing.
The main contributors were low fixed thresholding and strong Gaussian smoothing.
Auto-calibration makes thresholding data-driven per model/checkpoint, reducing
manual trial-and-error and improving shape-size consistency.

### Files affected
| File | Change |
|------|--------|
| `src/test.py` | Added threshold-candidate parsing, validation-based auto-threshold calibration, and logging fields for selected threshold and calibration stats. |
| `config.yaml` | Updated visualisation defaults: lower smoothing (`vis_gaussian_sigma`), higher base threshold (`vis_class_threshold`), and enabled threshold auto-calibration with candidate list/sample count. |

### Validation
- Linter: no errors introduced in modified files.
- Syntax check: `python -m py_compile src/test.py src/train.py src/dataset.py` passed.
- Runtime note: `python src/test.py` currently cannot complete because no `.pth` checkpoint exists in `checkpoints/`.

## 2026-04-12 — Add Composite Training Loss And Area Error Metrics

### What changed
Added a configurable composite training objective to better control reconstruction
shape quality and over-segmentation:
`total = w_mse * MSE + w_dice * DiceForeground + w_area * AreaRatioPenalty`.
`train.py` now supports `loss_name: mse | composite`, logs both objective and MSE,
and stores selected loss settings in checkpoints.

`test.py` now computes and reports anomaly area error on visualised samples,
including absolute error (mm^2), relative error (%), and signed area bias (%).
Per-sample console output and prediction panel titles now include area error.
Area metrics are also appended into the JSONL test log.

### Why
The model tended to predict anomaly regions that were larger than ground truth.
MSE alone optimizes pixel intensity but does not strongly constrain shape overlap
or area. The new Dice + area-ratio terms provide direct supervision on footprint
consistency, while added test metrics make over/under-segmentation visible.

### Files affected
| File | Change |
|------|--------|
| `src/train.py` | Added `CompositeReconstructionLoss`, configurable loss builder, objective-aware training/validation logging, and checkpoint fields for loss settings. |
| `config.yaml` | Added `loss_name` and `loss` weights (`mse_weight`, `dice_weight`, `area_weight`, `eps`) with composite defaults. |
| `src/test.py` | Added area computation helper, per-sample area error reporting, aggregate area error statistics, and JSONL metric fields for area errors/bias. |

### Validation
- Linter: no errors introduced in modified files.
- Syntax check: `python -m py_compile src/train.py src/test.py src/dataset.py` passed.
- Runtime note: end-to-end `src/test.py` still requires an available checkpoint file.
