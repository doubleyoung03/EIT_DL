"""
Adjacent-pair differential utilities for 16-electrode EIT.
=========================================================

Converts the 256 single-ended electrode potentials produced by a 16-electrode
EIT system with adjacent current injection (src = inj+1, snk = inj+2 mod 16)
into the 208 canonical adjacent-pair differential voltages used by the
literature.

Layout of the input single-ended vector  V_256
----------------------------------------------
Row-major flattening of the (16, 16) matrix ``V[inj, ch]`` where

    V[inj, ch] = potential at electrode (ch+1) during injection inj,
    injection inj drives current from electrode (inj+1) → electrode (inj+2)
                                                       [indices mod 16]

This matches the CSV schema produced by ``generate_eit_dataset.py``:
``V_src01_snk02_ch01 … V_src16_snk01_ch16``.

Adjacent-pair differentiation
-----------------------------
For every injection ``inj`` and every electrode pair ``(k, k+1) mod 16``:

    dV[inj, k] = V[inj, (k+1) mod 16] − V[inj, k]

Pairs that contain either the source (electrode ``inj``) or the sink
(electrode ``(inj+1) mod 16``) are excluded because the measurement is
dominated by the contact-impedance drop at the injection electrodes:

    excluded k ∈ { (inj − 1) mod 16,  inj,  (inj + 1) mod 16 }

leaving  16 × 13 = **208** valid differential measurements per frame.

Shared across three repos
-------------------------
The exact same module is duplicated (byte-identical) in

    D:\\Yang\\EIT\\COMSOL4EIT\\adjacent_diff.py
    D:\\Yang\\EIT\\Network\\src\\adjacent_diff.py
    D:\\Yang\\EIT\\Implementation\\adjacent_diff.py

to keep every deployment independent (the realtime host doesn't need to
import from the training repo and vice-versa).  If you edit one, edit all
three and log the change in each CHANGELOG.
"""

from __future__ import annotations

import numpy as np

# ── Geometry ──────────────────────────────────────────────────────────────────
N_ELECTRODES: int = 16                                  # fixed by protocol
N_INJECTIONS: int = N_ELECTRODES                        # one adjacent drive per electrode
N_EXCLUDED_PER_INJ: int = 3                             # (inj-1, inj, inj+1)
N_VALID_PER_INJ: int = N_ELECTRODES - N_EXCLUDED_PER_INJ  # 13
N_VALID: int = N_INJECTIONS * N_VALID_PER_INJ           # 208
N_RAW_SE: int = N_ELECTRODES * N_ELECTRODES             # 256


# ──────────────────────────────────────────────────────────────────────────────
# Valid-measurement mask
# ──────────────────────────────────────────────────────────────────────────────

def _build_adj_valid_mask(n_el: int = N_ELECTRODES) -> np.ndarray:
    """Boolean (n_el, n_el) mask for the adjacent-diff vector.

    ``mask[inj, k] == True``  iff the pair ``(k, k+1) mod n_el`` contains
    neither the source (electrode ``inj``) nor the sink
    (electrode ``(inj+1) mod n_el``).  For n_el=16 this produces exactly
    208 True entries (13 per injection × 16 injections).
    """
    mask = np.ones((n_el, n_el), dtype=bool)
    for inj in range(n_el):
        # Invalid k: (inj-1), inj, (inj+1)  mod n_el
        for dk in (-1, 0, 1):
            mask[inj, (inj + dk) % n_el] = False
    return mask


ADJ_VALID_MASK_2D: np.ndarray = _build_adj_valid_mask()         # (16, 16) bool
ADJ_VALID_MASK_256: np.ndarray = ADJ_VALID_MASK_2D.ravel()      # (256,)  bool

assert int(ADJ_VALID_MASK_256.sum()) == N_VALID, (
    f"Adjacent valid mask produced {int(ADJ_VALID_MASK_256.sum())} valid entries, "
    f"expected {N_VALID}."
)


# ──────────────────────────────────────────────────────────────────────────────
# Core conversion  (single-ended 256  →  adjacent-diff 208)
# ──────────────────────────────────────────────────────────────────────────────

def singleend_to_adj_256(v_single_ended: np.ndarray) -> np.ndarray:
    """Compute the full (16, 16) adjacent-diff matrix before masking.

    Parameters
    ----------
    v_single_ended : array-like, shape ``(..., 256)``
        Row-major flattening of the single-ended ``V[inj, ch]`` matrix.

    Returns
    -------
    dv_256 : ndarray, shape ``(..., 256)``
        ``dV[inj, k] = V[inj, (k+1) % 16] − V[inj, k]``, still row-major.
    """
    v = np.asarray(v_single_ended, dtype=np.float64)
    if v.shape[-1] != N_RAW_SE:
        raise ValueError(
            f"singleend_to_adj_256 expects last dim = {N_RAW_SE}, got {v.shape[-1]}"
        )
    head = v.shape[:-1]
    v_2d = v.reshape(*head, N_ELECTRODES, N_ELECTRODES)
    # V[inj, (k+1)%16] − V[inj, k]: shift along the channel axis, wrap-around
    dv_2d = np.roll(v_2d, shift=-1, axis=-1) - v_2d
    return dv_2d.reshape(*head, N_RAW_SE)


def singleend_to_adj_208(v_single_ended: np.ndarray) -> np.ndarray:
    """Convert single-ended 256 → adjacent-diff 208 (valid pairs only).

    Parameters
    ----------
    v_single_ended : array-like, shape ``(..., 256)``.

    Returns
    -------
    dv_208 : ndarray, shape ``(..., 208)``.
    """
    dv_256 = singleend_to_adj_256(v_single_ended)
    return dv_256[..., ADJ_VALID_MASK_256]


# ──────────────────────────────────────────────────────────────────────────────
# Column naming  (mirrors the V_src / V_snk convention of the raw CSV)
# ──────────────────────────────────────────────────────────────────────────────

def adj_column_names(prefix: str = "dV") -> list[str]:
    """Return the 208 canonical column names for the adjacent-diff vector.

    Ordering is **(inj-major, k-minor)**, matching ``singleend_to_adj_208``
    after masking.  Example first & last names (prefix='dV')::

        dV_src01_snk02_p03p04   (inj=0, first valid pair k=2 → electrodes 3,4)
        dV_src16_snk01_p13p14   (inj=15, last valid pair k=12 → electrodes 13,14)

    Parameters
    ----------
    prefix : str
        Column-name prefix (default ``"dV"``).
    """
    names: list[str] = []
    for inj in range(N_ELECTRODES):
        src = inj + 1
        snk = (inj + 1) % N_ELECTRODES + 1
        for k in range(N_ELECTRODES):
            if not ADJ_VALID_MASK_2D[inj, k]:
                continue
            p1 = k + 1
            p2 = (k + 1) % N_ELECTRODES + 1
            names.append(f"{prefix}_src{src:02d}_snk{snk:02d}_p{p1:02d}p{p2:02d}")
    assert len(names) == N_VALID
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation floor  (adaptive, replaces legacy hard-coded _REF_ABS_FLOOR)
# ──────────────────────────────────────────────────────────────────────────────

def compute_adj_ref_floor(
    dv_ref_208: np.ndarray,
    percentile: float = 1.0,
    min_floor: float = 1e-9,
) -> float:
    """Adaptive floor for ``|dV_ref|`` in the differential-normalisation step.

    Channels whose reference magnitude falls below the floor are forced to
    zero in the normalised dv to avoid blowing up on electrically-neutral
    pairs.  The old single-ended floor (``5e-5 V``) was calibrated for
    absolute potentials; in adjacent-diff space, magnitudes span several
    decades and a fixed floor is unsafe — so we derive it from the
    reference frame itself.

    Parameters
    ----------
    dv_ref_208 : array-like, shape (208,) — the empty-tank adjacent-diff reference
                 (already the output of ``singleend_to_adj_208``).
    percentile : float in [0, 100], default 1.0.  Channels below this
                 percentile of ``|dV_ref|`` are zeroed as "neutral".
    min_floor  : absolute lower bound to protect against an exactly-zero
                 reference (e.g. perfect-symmetry simulations), volts.

    Returns
    -------
    floor : float.
    """
    ref = np.abs(np.asarray(dv_ref_208, dtype=np.float64).ravel())
    if ref.size != N_VALID:
        raise ValueError(
            f"compute_adj_ref_floor expects length {N_VALID}, got {ref.size}"
        )
    p = float(np.percentile(ref, percentile))
    return max(min_floor, p)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test  (run `python adjacent_diff.py`)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"N_ELECTRODES     = {N_ELECTRODES}")
    print(f"N_VALID_PER_INJ  = {N_VALID_PER_INJ}  (expect 13)")
    print(f"N_VALID (total)  = {N_VALID}          (expect 208)")
    print(f"mask sum (256)   = {int(ADJ_VALID_MASK_256.sum())}")

    # Unit test: injection-electrode pairs are excluded.
    for inj in range(N_ELECTRODES):
        for dk in (-1, 0, 1):
            k = (inj + dk) % N_ELECTRODES
            assert not ADJ_VALID_MASK_2D[inj, k], (
                f"mask failure at inj={inj}, k={k}"
            )

    # Round-trip sanity: random single-ended → diffs must sum to zero along ch axis.
    rng = np.random.default_rng(0)
    v_random = rng.standard_normal(N_RAW_SE)
    dv_256 = singleend_to_adj_256(v_random)
    dv_2d = dv_256.reshape(N_ELECTRODES, N_ELECTRODES)
    row_sums = dv_2d.sum(axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-12), (
        f"Adjacent diffs must sum to zero per row (telescoping); got {row_sums}"
    )
    print("Row-sum zero check: PASS")

    # Column-name sanity
    names = adj_column_names()
    assert len(names) == N_VALID
    # inj=0 (src=1, snk=2): invalid k ∈ {15, 0, 1} → first valid k=2 → pair (3,4)
    assert names[0] == "dV_src01_snk02_p03p04", names[0]
    # inj=15 (src=16, snk=1): invalid k ∈ {14, 15, 0} → last valid k=13 → pair (14,15)
    assert names[-1] == "dV_src16_snk01_p14p15", names[-1]
    print(f"First col : {names[0]}")
    print(f"Last col  : {names[-1]}")
    print("All checks passed.")
