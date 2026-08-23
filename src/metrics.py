"""
Single source of truth for gamma-space error and void-segmentation metrics.

Before this module the repository carried THREE mutually incompatible
definitions of "gamma MSE":

    A   0.5 * mean(...) over the PADDED (Nx+3, Ny+3) array
        -- U-Net / SegFormer / conventional FWI experiments
    B   0.5 * mean(...) over the INTERIOR (Nx+1, Ny+1)
        -- all eight INR experiments
    C   1.0 * mean(...) over the INTERIOR
        -- both pretraining validation loops

B and C differ by exactly 2x; A and B differ by the ghost ring. Because the ring
holds 1.0 in both prediction and target, the ring contributes nothing to the
numerator but inflates the denominator from 256*128 = 32768 to 258*130 = 33540,
so every padded-domain figure read 2.3% low.

The 0.5 is legitimate in exactly one place: the least-squares waveform misfit
0.5*||d_sim - d_obs||^2 in `adjoint.py`, which is the physical cost being
minimized. It was copy-pasted into the gamma-space diagnostic, where no such
derivation applies.

Convention adopted here, used everywhere:

    * the physical FWI cost keeps its 0.5 (it is not computed in this module)
    * every REPORTED gamma metric is a plain mean over the PHYSICAL INTERIOR
    * void masks are thresholded on gamma DIRECTLY, which is model-agnostic and
      requires no knowledge of the network's output head

Both torch tensors and numpy arrays are accepted; the arithmetic used here is
common to both.
"""

VOID_THRESHOLD = 0.5
_EPS = 1e-12


def strip_ghost(array, ghost=0):
    """
    Drop `ghost` cells from each spatial edge (the last two axes).

    `ghost=1` converts a padded (..., Nx+3, Ny+3) array to the physical
    (..., Nx+1, Ny+1) interior. `ghost=0` returns the array unchanged, for
    call sites that already hold interior arrays.
    """
    ghost = int(ghost)
    if ghost < 0:
        raise ValueError(f"ghost must be non-negative, got {ghost}")
    if ghost == 0:
        return array
    return array[..., ghost:-ghost, ghost:-ghost]


def gamma_mse(prediction, target, ghost=0):
    """
    Mean squared error between predicted and target gamma over the physical
    interior. No 0.5 factor -- see the module docstring.
    """
    prediction = strip_ghost(prediction, ghost)
    target = strip_ghost(target, ghost)
    if prediction.shape != target.shape:
        raise ValueError(
            f"gamma shape mismatch after stripping {ghost} ghost cell(s): "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    return float(((prediction - target) ** 2).mean())


def gamma_mae(prediction, target, ghost=0):
    """Mean absolute error over the physical interior."""
    prediction = strip_ghost(prediction, ghost)
    target = strip_ghost(target, ghost)
    return float((abs(prediction - target)).mean())


def void_mask(gamma, threshold=VOID_THRESHOLD):
    """
    Binary void indicator, thresholded on gamma directly.

    gamma = 1 is intact material and gamma = gamma0 (<< 1) is void, so a void is
    `gamma <= threshold`. Thresholding gamma rather than a network logit or a
    reconstructed voidness keeps this identical across the U-Net, the SegFormer
    and every INR ansatz.
    """
    return gamma <= float(threshold)


def segmentation_metrics(prediction, target, ghost=0, threshold=VOID_THRESHOLD):
    """Void-mask agreement: precision, recall, f1 (== Dice), IoU, accuracy."""
    predicted_void = void_mask(strip_ghost(prediction, ghost), threshold)
    target_void = void_mask(strip_ghost(target, ghost), threshold)

    true_positive = float((predicted_void & target_void).sum())
    false_positive = float((predicted_void & ~target_void).sum())
    false_negative = float((~predicted_void & target_void).sum())
    true_negative = float((~predicted_void & ~target_void).sum())

    precision = true_positive / max(true_positive + false_positive, _EPS)
    recall = true_positive / max(true_positive + false_negative, _EPS)
    total = true_positive + false_positive + false_negative + true_negative
    return {
        # f1 and Dice are the same number for a binary mask; only f1 is
        # reported, to avoid inflating the apparent metric count.
        "f1": 2.0 * precision * recall / max(precision + recall, _EPS),
        "iou": true_positive / max(true_positive + false_negative + false_positive, _EPS),
        "precision": precision,
        "recall": recall,
        "accuracy": (true_positive + true_negative) / max(total, _EPS),
        "void_fraction_pred": float(predicted_void.sum()) / max(total, _EPS),
        "void_fraction_target": float(target_void.sum()) / max(total, _EPS),
    }


def all_metrics(prediction, target, ghost=0, threshold=VOID_THRESHOLD):
    """
    Continuous AND binarized metrics side by side.

    The continuous ones (gamma_mse, gamma_mae) are the primary quantities; the
    binarized ones exist so that a regression-trained U-Net and a
    segmentation-trained SegFormer can be placed on a common footing.
    """
    metrics = {
        "gamma_mse": gamma_mse(prediction, target, ghost),
        "gamma_mae": gamma_mae(prediction, target, ghost),
        "void_threshold": float(threshold),
    }
    metrics.update(segmentation_metrics(prediction, target, ghost, threshold))
    return metrics
