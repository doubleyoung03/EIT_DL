# Changelog

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
