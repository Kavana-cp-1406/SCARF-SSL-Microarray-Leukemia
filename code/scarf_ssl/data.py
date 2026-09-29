"""
Data loading, preprocessing, and partitioning for the MILE cohort
(GSE13164) and the external validation cohort (GSE9476).

Raw values are returned unscaled; scaling/SMOTE happen downstream, fit only
on each fold's (sub)training partition, to avoid leakage.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import GEOparse

logger = logging.getLogger("scarf_ssl")


def _find_label_column(pheno_df: pd.DataFrame, keywords: List[str]) -> str:
    for col in pheno_df.columns:
        if "characteristics" in col or "title" in col or "source" in col:
            unique_vals = pheno_df[col].dropna().astype(str).tolist()
            if any(term in " ".join(unique_vals).lower() for term in keywords):
                return col
    return "title"


def _clean_label(label: str) -> str:
    """Regex cleanup of dataset-specific label prefixes (Section 2.1),
    e.g. stripping "MILES stage 2 data N2_0001: " down to the diagnosis."""
    label = re.sub(r"^(MILE[S]?\s*stage\s*\d+\s*data\s*)", "", label, flags=re.IGNORECASE)
    label = re.sub(r"^(N\d+_\d+\s*[:\-]?\s*)", "", label, flags=re.IGNORECASE)
    return label.strip()


def load_gse13164(
    dest_dir: str = "./geo_data",
    geo_accession: str = "GSE13164",
    missingness_threshold: float = 0.20,
    min_class_size: int = 5,
    expected_n_samples: Optional[int] = 1152,
    expected_n_classes: Optional[int] = 18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads GSE13164 (confirmed correct accession — n=1,152, GPL7473),
    excludes probes with >20% missingness (Section 2.1), cleans label
    strings, and excludes classes below min_class_size.

    Returns:
        X: (n_samples, n_features) float32 raw (unscaled) expression matrix.
        y: (n_samples,) int encoded labels.
        class_names: (n_classes,) original label strings.
        probe_ids: list of retained probe/feature identifiers, in the same
            column order as X (used for external-cohort probe alignment).
    """
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading / loading %s from NCBI GEO", geo_accession)
    gse = GEOparse.get_GEO(geo=geo_accession, destdir=dest_dir)

    try:
        logger.info("Resolved platform(s) for %s: %s", geo_accession, list(gse.gpls.keys()))
    except AttributeError:
        logger.warning("Could not introspect platform metadata for %s.", geo_accession)

    expr_df = gse.pivot_samples("VALUE")  # Probes x Samples
    X_df = expr_df.T                       # Samples x Probes

    # --- Exclude probes with >20% missingness (Section 2.1) ---
    missing_frac = X_df.isna().mean(axis=0)
    keep_probes = missing_frac[missing_frac <= missingness_threshold].index
    n_excluded = X_df.shape[1] - len(keep_probes)
    logger.info(
        "Excluding %d probes with >%.0f%% missingness (retaining %d).",
        n_excluded, missingness_threshold * 100, len(keep_probes),
    )
    X_df = X_df[keep_probes]

    # --- Fill remaining missing/inf values ---
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0.0)

    # --- Labels ---
    pheno_df = gse.phenotype_data
    target_col = _find_label_column(
        pheno_df, ["leukemia", "leukaemia", "all", "aml", "cll", "cml", "control", "normal"]
    )
    y_raw = pheno_df.loc[X_df.index, target_col].astype(str).map(_clean_label).values
    logger.info("Identified label column: '%s'", target_col)

    # --- Exclude classes below min_class_size ---
    unique, counts = np.unique(y_raw, return_counts=True)
    small_classes = unique[counts < min_class_size]
    if len(small_classes) > 0:
        logger.warning(
            "Excluding %d classes with <%d samples: %s",
            len(small_classes), min_class_size, list(small_classes),
        )
        keep_mask = ~np.isin(y_raw, small_classes)
        X_df = X_df.loc[keep_mask]
        y_raw = y_raw[keep_mask]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    n_samples, n_classes = X_df.shape[0], len(encoder.classes_)
    logger.info(
        "Dataset summary: %d samples, %d features, %d classes", n_samples, X_df.shape[1], n_classes
    )
    for i, c in enumerate(encoder.classes_):
        logger.info("  [%d] %s (n=%d)", i, c, int((y == i).sum()))

    if expected_n_samples is not None and n_samples != expected_n_samples:
        logger.warning(
            "SAMPLE COUNT MISMATCH: got %d, expected %d. Verify before trusting results.",
            n_samples, expected_n_samples,
        )
    if expected_n_classes is not None and n_classes != expected_n_classes:
        logger.warning(
            "CLASS COUNT MISMATCH: got %d, expected %d. Verify before trusting results.",
            n_classes, expected_n_classes,
        )

    return X_df.values.astype(np.float32), y, encoder.classes_, list(X_df.columns)


def load_gse9476(
    dest_dir: str = "./geo_data_external",
    geo_accession: str = "GSE9476",
    expected_n_samples: Optional[int] = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads the external validation cohort GSE9476 (GPL96; confirmed n=64:
    38 normal + 26 AML). The class taxonomy here is coarser than the
    18-class MILE cohort — inspect the logged class list yourself. The
    manuscript's Methods (2.8: "7 cell-type classes") and
    Results/Limitations (3.8/4.7: "5 AML/normal bone marrow classes")
    disagree with each other; reconcile against what's actually in this
    cohort's metadata before trusting Table 5.
    """
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading / loading %s from NCBI GEO (external validation cohort)", geo_accession)
    gse = GEOparse.get_GEO(geo=geo_accession, destdir=dest_dir)

    expr_df = gse.pivot_samples("VALUE")
    X_df = expr_df.T
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median())

    pheno_df = gse.phenotype_data
    target_col = _find_label_column(pheno_df, ["aml", "normal", "control", "leukemia", "healthy"])
    y_raw = pheno_df.loc[X_df.index, target_col].astype(str).map(_clean_label).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    n_samples, n_classes = X_df.shape[0], len(encoder.classes_)
    logger.info(
        "External cohort summary: %d samples, %d features, %d classes",
        n_samples, X_df.shape[1], n_classes,
    )
    for i, c in enumerate(encoder.classes_):
        logger.info("  [%d] %s (n=%d)", i, c, int((y == i).sum()))

    if expected_n_samples is not None and n_samples != expected_n_samples:
        logger.warning(
            "EXTERNAL SAMPLE COUNT MISMATCH: got %d, expected %d.", n_samples, expected_n_samples
        )

    return X_df.values.astype(np.float32), y, encoder.classes_, list(X_df.columns)


def align_probes(
    source_probe_ids: List[str], target_probe_ids: List[str], X_target: np.ndarray
) -> Tuple[np.ndarray, int, int]:
    """
    Aligns the external cohort's features to the source (training) cohort's
    probe ordering (Section 2.8): matched probes keep their values,
    unmatched source probes are zero-padded.

    Returns:
        X_aligned: (n_target_samples, n_source_probes) array.
        n_matched: number of probes that matched by identifier.
        n_total: total number of source probes (== X_aligned.shape[1]).
    """
    target_index = {pid: i for i, pid in enumerate(target_probe_ids)}
    n_total = len(source_probe_ids)
    X_aligned = np.zeros((X_target.shape[0], n_total), dtype=np.float32)

    n_matched = 0
    for j, pid in enumerate(source_probe_ids):
        if pid in target_index:
            X_aligned[:, j] = X_target[:, target_index[pid]]
            n_matched += 1

    logger.info(
        "Probe alignment: %d / %d source probes matched (%.1f%%).",
        n_matched, n_total, 100 * n_matched / n_total,
    )
    return X_aligned, n_matched, n_total


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.64,
    val_fraction: float = 0.16,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed 64:16:20 stratified partition (Section 2.2), independent of the
    5-fold CV protocol used for the main benchmark. Provided for any
    experiment that specifically calls for this fixed split rather than
    CV — see the audit notes on why the manuscript's two evaluation
    protocols (fixed split vs. 5-fold CV) are hard to run literally
    simultaneously, and how run_cv_benchmark() reconciles this instead."""
    assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-6
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_fraction + test_fraction), stratify=y, random_state=seed
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_subtrain_val_split(
    X_train: np.ndarray, y_train: np.ndarray, val_fraction: float = 0.20, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carves a validation subset out of a CV fold's training partition, for
    checkpoint selection / the ReduceLROnPlateau scheduler in two-stage
    fine-tuning (Section 2.4). This is the implementation choice that
    reconciles the manuscript's two different evaluation protocols (a
    fixed 64:16:20 split in Section 2.2, and 5-fold CV in Section 2.6.1)."""
    return train_test_split(X_train, y_train, test_size=val_fraction, stratify=y_train, random_state=seed)
