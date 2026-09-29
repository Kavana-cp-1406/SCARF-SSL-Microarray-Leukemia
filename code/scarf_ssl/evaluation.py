"""
Evaluation orchestration: 5-fold CV benchmark (Table 2), label-efficiency
sweep (Table 3), augmentation ablation (Table 4), and paired significance
testing (Section 2.6.1).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .augmentations import (
    CombinedAugmentation,
    GaussianNoiseAugmentation,
    GeneProbeSubsetting,
    MagnitudeWarping,
    RandomGeneMasking,
    SCARFCorruption,
)
from .baselines import train_random_forest, train_svm_rbf, train_xgboost
from .data import stratified_subtrain_val_split
from .stats import paired_significance
from .training import predict, pretrain_ssl, train_supervised_mlp, two_stage_finetune

logger = logging.getLogger("scarf_ssl")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _smote_fit_resample(X: np.ndarray, y: np.ndarray, seed: int):
    min_count = np.bincount(y).min()
    if min_count <= 1:
        logger.warning("  [SMOTE] Skipped — a class has only 1 sample.")
        return X, y
    k = max(1, min(5, min_count - 1))
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)


def run_cv_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    cfg,
    device: torch.device,
    label_fraction: float = 1.0,
    methods: Tuple[str, ...] = ("ssl", "supervised_mlp", "svm_rbf", "xgboost", "random_forest"),
) -> pd.DataFrame:
    """
    Runs stratified 5-fold CV for each requested method at a given labelled
    fraction, returning a long-format DataFrame: one row per (fold, method)
    with accuracy/f1/precision/recall. This is the core routine behind
    Table 2 (label_fraction=1.0, all methods) and Table 3 (multiple
    fractions, methods = ssl/supervised_mlp/xgboost/random_forest).

    SSL pre-training always uses the FULL unlabelled training fold
    regardless of label_fraction (Section 2.6.2) — only the fine-tuning
    label budget is reduced. Scaling and SMOTE are fit on the training
    fold (or its labelled subset) only, never on the held-out test fold.
    """
    skf = StratifiedKFold(n_splits=cfg.split.n_folds, shuffle=True, random_state=cfg.split.seed)
    rows: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        logger.info("=== Fold %d/%d (label_fraction=%.2f) ===", fold_idx, cfg.split.n_folds, label_fraction)
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Carve a validation subset out of the training fold, for
        # checkpoint selection / scheduler (neural methods only).
        X_subtrain, X_val, y_subtrain_full, y_val = stratified_subtrain_val_split(
            X_train_scaled, y_train_full, val_fraction=0.20, seed=cfg.split.seed
        )

        # Apply the requested label fraction to the subtrain labels only
        # (SSL pretraining below still uses the full X_train_scaled, unlabelled).
        if label_fraction < 1.0:
            n_classes_present = len(np.unique(y_subtrain_full))
            n_use = max(n_classes_present, int(len(X_subtrain) * label_fraction))
            n_use = min(n_use, len(X_subtrain))
            X_subtrain_lbl, _, y_subtrain_lbl, _ = train_test_split(
                X_subtrain, y_subtrain_full, train_size=n_use,
                stratify=y_subtrain_full, random_state=cfg.split.seed,
            )
        else:
            X_subtrain_lbl, y_subtrain_lbl = X_subtrain, y_subtrain_full

        X_subtrain_bal, y_subtrain_bal = _smote_fit_resample(X_subtrain_lbl, y_subtrain_lbl, cfg.split.seed)

        if "ssl" in methods:
            encoder = pretrain_ssl(
                X_train_scaled, SCARFCorruption(cfg.augmentation.scarf_corruption_rate), device,
                epochs=cfg.ssl.pretrain_epochs, batch_size=cfg.ssl.pretrain_batch_size,
                lr=cfg.ssl.pretrain_lr, weight_decay=cfg.ssl.pretrain_weight_decay,
                temperature=cfg.ssl.temperature, hidden_dim_1=cfg.ssl.hidden_dim_1,
                hidden_dim_2=cfg.ssl.hidden_dim_2, dropout=cfg.ssl.dropout,
                verbose=(fold_idx == 1),
            )
            ssl_model = two_stage_finetune(
                encoder, X_subtrain_bal, y_subtrain_bal, X_val, y_val, len(class_names), device,
                hidden_dim_2=cfg.ssl.hidden_dim_2,
                classifier_hidden_dim=cfg.finetune.classifier_hidden_dim,
                classifier_dropout=cfg.finetune.classifier_dropout,
                linear_probe_epochs=cfg.finetune.linear_probe_epochs,
                finetune_epochs=cfg.finetune.finetune_epochs,
                batch_size=cfg.finetune.finetune_batch_size, lr=cfg.finetune.lr,
                weight_decay=cfg.finetune.weight_decay,
                scheduler_patience=cfg.finetune.scheduler_patience,
                scheduler_factor=cfg.finetune.scheduler_factor, verbose=(fold_idx == 1),
            )
            y_pred = predict(ssl_model, X_test_scaled, device)
            rows.append({
                "fold": fold_idx, "method": "SSL pre-trained (ours)", "label_fraction": label_fraction,
                **_metrics(y_test, y_pred),
            })

        if "supervised_mlp" in methods:
            mlp_model = train_supervised_mlp(
                X_subtrain_bal, y_subtrain_bal, X_val, y_val, len(class_names), device,
                input_dim=X.shape[1], hidden_dim_1=cfg.ssl.hidden_dim_1, hidden_dim_2=cfg.ssl.hidden_dim_2,
                dropout=cfg.ssl.dropout, classifier_hidden_dim=cfg.finetune.classifier_hidden_dim,
                classifier_dropout=cfg.finetune.classifier_dropout,
                epochs=cfg.finetune.linear_probe_epochs + cfg.finetune.finetune_epochs,
                batch_size=cfg.finetune.finetune_batch_size, lr=cfg.finetune.lr,
                weight_decay=cfg.finetune.weight_decay, verbose=(fold_idx == 1),
            )
            y_pred = predict(mlp_model, X_test_scaled, device)
            rows.append({
                "fold": fold_idx, "method": "Supervised MLP", "label_fraction": label_fraction,
                **_metrics(y_test, y_pred),
            })

        if "random_forest" in methods:
            rf = train_random_forest(
                X_subtrain_bal, y_subtrain_bal, cfg.baseline.rf_n_estimators,
                cfg.baseline.rf_max_depth, cfg.split.seed,
            )
            y_pred = rf.predict(X_test_scaled)
            rows.append({
                "fold": fold_idx, "method": "Random Forest", "label_fraction": label_fraction,
                **_metrics(y_test, y_pred),
            })

        if "svm_rbf" in methods:
            svm = train_svm_rbf(
                X_subtrain_bal, y_subtrain_bal, cfg.baseline.svm_C, cfg.baseline.svm_gamma, cfg.split.seed
            )
            y_pred = svm.predict(X_test_scaled)
            rows.append({
                "fold": fold_idx, "method": "SVM-RBF", "label_fraction": label_fraction,
                **_metrics(y_test, y_pred),
            })

        if "xgboost" in methods:
            xgb = train_xgboost(
                X_subtrain_bal, y_subtrain_bal, cfg.baseline.xgb_n_estimators,
                cfg.baseline.xgb_max_depth, cfg.baseline.xgb_learning_rate, cfg.split.seed,
            )
            y_pred = xgb.predict(X_test_scaled)
            rows.append({
                "fold": fold_idx, "method": "XGBoost", "label_fraction": label_fraction,
                **_metrics(y_test, y_pred),
            })

    return pd.DataFrame(rows)


def summarize_cv_results(df: pd.DataFrame) -> pd.DataFrame:
    """Fold-wise mean +/- SD per method per label_fraction (Table 2/3 shape)."""
    return df.groupby(["method", "label_fraction"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1_weighted", "mean"), f1_std=("f1_weighted", "std"),
        precision_mean=("precision_weighted", "mean"), precision_std=("precision_weighted", "std"),
        recall_mean=("recall_weighted", "mean"), recall_std=("recall_weighted", "std"),
    ).reset_index()


def run_label_efficiency_sweep(
    X: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    cfg,
    device: torch.device,
    methods: Tuple[str, ...] = ("ssl", "supervised_mlp", "xgboost", "random_forest"),
) -> pd.DataFrame:
    """Section 2.6.2 / Table 3: reruns run_cv_benchmark at each labelled
    fraction and concatenates the results."""
    all_results = []
    for fraction in cfg.label_efficiency.fractions:
        logger.info("### Label-efficiency sweep: fraction = %.2f ###", fraction)
        df = run_cv_benchmark(X, y, class_names, cfg, device, label_fraction=fraction, methods=methods)
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True)


def run_significance_tests(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Section 2.6.1: paired t-test + Wilcoxon between Supervised MLP and
    each comparator present in the results, at each label_fraction."""
    rows = []
    for fraction in sorted(cv_results["label_fraction"].unique()):
        subset = cv_results[cv_results["label_fraction"] == fraction]
        mlp = subset[subset["method"] == "Supervised MLP"].sort_values("fold")["accuracy"].tolist()
        if not mlp:
            continue
        for comparator in ["SSL pre-trained (ours)", "XGBoost", "Random Forest", "SVM-RBF"]:
            comp_acc = subset[subset["method"] == comparator].sort_values("fold")["accuracy"].tolist()
            if len(comp_acc) != len(mlp) or len(comp_acc) == 0:
                continue
            result = paired_significance(mlp, comp_acc)
            rows.append({"label_fraction": fraction, "comparison": f"Supervised MLP vs {comparator}", **result})
    return pd.DataFrame(rows)


def run_augmentation_ablation(
    X: np.ndarray, y: np.ndarray, class_names: np.ndarray, cfg, device: torch.device, seed: int = 42
) -> pd.DataFrame:
    """
    Table 4: each of the four biologically-motivated augmentations (plus
    their combination) evaluated in isolation via a SINGLE stratified
    train/test split at 100% label availability — matching the
    manuscript's "single-split performance" framing (Section 3.6). This is
    NOT the same evaluation protocol as the 5-fold CV used for Table 2/3 —
    don't directly compare these numbers to Table 2's SSL row without
    noting that difference (flagged in the audit).
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=cfg.split.test_fraction, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_subtrain, X_val, y_subtrain, y_val = stratified_subtrain_val_split(X_train, y_train, 0.20, seed)
    X_subtrain_bal, y_subtrain_bal = _smote_fit_resample(X_subtrain, y_subtrain, seed)

    a = cfg.augmentation
    strategies = {
        "Gene Probe Subsetting (80% retention)": GeneProbeSubsetting(a.probe_subset_retain),
        "Random Gene Masking (15% mask ratio)": RandomGeneMasking(a.gene_mask_ratio),
        "Gaussian Noise Addition (sigma=0.1)": GaussianNoiseAugmentation(a.gaussian_noise_sigma),
        "Magnitude Warping (log-normal sigma=0.1)": MagnitudeWarping(sigma=a.magnitude_warp_sigma),
        "Full Combined (all four)": CombinedAugmentation(
            a.gaussian_noise_sigma, a.gene_mask_ratio, a.probe_subset_retain, a.magnitude_warp_sigma
        ),
    }

    rows = []
    for name, augmenter in strategies.items():
        logger.info("### Augmentation ablation: %s ###", name)
        encoder = pretrain_ssl(
            X_train, augmenter, device, epochs=cfg.ssl.pretrain_epochs,
            batch_size=cfg.ssl.pretrain_batch_size, lr=cfg.ssl.pretrain_lr,
            weight_decay=cfg.ssl.pretrain_weight_decay, temperature=cfg.ssl.temperature,
            hidden_dim_1=cfg.ssl.hidden_dim_1, hidden_dim_2=cfg.ssl.hidden_dim_2,
            dropout=cfg.ssl.dropout, verbose=True,
        )
        model = two_stage_finetune(
            encoder, X_subtrain_bal, y_subtrain_bal, X_val, y_val, len(class_names), device,
            hidden_dim_2=cfg.ssl.hidden_dim_2, classifier_hidden_dim=cfg.finetune.classifier_hidden_dim,
            classifier_dropout=cfg.finetune.classifier_dropout,
            linear_probe_epochs=cfg.finetune.linear_probe_epochs,
            finetune_epochs=cfg.finetune.finetune_epochs, batch_size=cfg.finetune.finetune_batch_size,
            lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay,
            scheduler_patience=cfg.finetune.scheduler_patience,
            scheduler_factor=cfg.finetune.scheduler_factor, verbose=True,
        )
        y_pred = predict(model, X_test, device)
        rows.append({"augmentation": name, **_metrics(y_test, y_pred)})

    return pd.DataFrame(rows)
