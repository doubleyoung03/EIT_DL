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
