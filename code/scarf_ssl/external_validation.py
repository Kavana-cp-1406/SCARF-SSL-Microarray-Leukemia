"""
External cohort validation (Section 2.8): evaluate models trained on
GSE13164 against the independent GSE9476 cohort without retuning.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .data import align_probes
from .training import predict

logger = logging.getLogger("scarf_ssl")


def evaluate_on_external_cohort(
    model: torch.nn.Module,
    source_scaler: StandardScaler,
    source_probe_ids: List[str],
    target_probe_ids: List[str],
    X_target_raw: np.ndarray,
    y_target: np.ndarray,
    target_class_names: np.ndarray,
    source_class_names: np.ndarray,
    device: torch.device,
    class_name_mapping: Dict[str, str],
) -> Dict[str, float]:
    """
    Aligns the external cohort's probes to the source cohort's feature
    space (zero-padding unmatched probes), scales with the SOURCE cohort's
    already-fitted scaler (never refit on external data), predicts with
    the trained model, and maps the model's fine-grained MILE-subtype
    predictions down to the external cohort's coarser class taxonomy via
    `class_name_mapping`.

    class_name_mapping is REQUIRED and dataset-specific — you must supply
    it yourself (e.g. {"AML with t(8;21)": "AML", "Non-leukaemia / healthy
    BM": "Normal", ...}) after inspecting both cohorts' logged class
    lists. This function does not fabricate a mapping — an unmapped
    source class raises rather than silently mis-scoring, because the
    manuscript's own description of GSE9476's class taxonomy is internally
    inconsistent (Methods says 7 classes, Results/Limitations say 5) and
    that needs to be resolved by you, not guessed at here.
    """
    X_aligned, n_matched, n_total = align_probes(source_probe_ids, target_probe_ids, X_target_raw)
    X_aligned_scaled = source_scaler.transform(X_aligned)

    y_pred_source_space = predict(model, X_aligned_scaled, device)
    pred_labels = [source_class_names[i] for i in y_pred_source_space]

    unmapped = set(pred_labels) - set(class_name_mapping.keys())
    if unmapped:
        raise ValueError(
            f"class_name_mapping is missing entries for predicted classes: {sorted(unmapped)}. "
            "Supply a mapping from every MILE-cohort class name to the corresponding "
            "external-cohort class name (or 'Unknown'/'Exclude') before scoring."
        )
    mapped_labels = [class_name_mapping[label] for label in pred_labels]

    target_name_lookup = {name: i for i, name in enumerate(target_class_names)}
    y_pred_target_space = np.array([target_name_lookup.get(lbl, -1) for lbl in mapped_labels])

    valid = y_pred_target_space != -1
    if not valid.all():
        logger.warning(
            "%d / %d predictions mapped to a label not present in the external cohort's "
            "class list and were excluded from scoring.",
            int((~valid).sum()), len(valid),
        )

    accuracy = float((y_pred_target_space[valid] == y_target[valid]).mean()) if valid.any() else float("nan")

    logger.info(
        "External validation: %d/%d probes matched (%.1f%%); accuracy = %.4f on %d scored samples.",
        n_matched, n_total, 100 * n_matched / n_total, accuracy, int(valid.sum()),
    )

    return {
        "accuracy": accuracy,
        "n_matched_probes": n_matched,
        "n_total_probes": n_total,
        "n_scored_samples": int(valid.sum()),
    }
