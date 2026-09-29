#!/usr/bin/env python3
"""
End-to-end orchestrator for the full SCARF-SSL benchmarking study,
matching the manuscript's Methods (Section 2) end to end:

  1. Load GSE13164, preprocess (Section 2.1)
  2. 5-fold CV benchmark: SSL, Supervised MLP, RF, SVM-RBF, XGBoost (Table 2)
  3. Label-efficiency sweep across 5/10/20/100% (Table 3)
  4. Paired significance testing: t-test + Wilcoxon (Section 2.6.1)
  5. Augmentation ablation: 4 strategies + combined, single split (Table 4)
  6. SHAP (PCA + logistic-regression surrogate) + pathway enrichment (2.7)
  7. External validation on GSE9476 (Table 5) — requires a manual
     class-name mapping; see scarf_ssl/external_validation.py

Run: python run_pipeline.py --help

NOTE ON RUNTIME: this runs SSL pre-training + two-stage fine-tuning
repeatedly — 5 folds x (1 [Table 2] + 4 fractions [Table 3]) + 5
[ablation] + 1 [final model] = ~30 full training runs, plus baselines.
On CPU this can take a long time on 1,457-feature / 1,152-sample data.
Use --skip-* flags liberally while iterating, and only run the full thing
once the pipeline is verified end to end on a subset.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from scarf_ssl.augmentations import SCARFCorruption
from scarf_ssl.config import Config
from scarf_ssl.data import load_gse13164, stratified_subtrain_val_split
from scarf_ssl.evaluation import (
    run_augmentation_ablation,
    run_cv_benchmark,
    run_label_efficiency_sweep,
    run_significance_tests,
    summarize_cv_results,
)
from scarf_ssl.interpretability import (
    back_project_to_genes,
    compute_shap_pca_surrogate,
    get_combined_linear_weight,
    run_pathway_enrichment,
)
from scarf_ssl.training import pretrain_ssl, two_stage_finetune

logger = logging.getLogger("scarf_ssl")


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full SCARF-SSL benchmarking pipeline (matches manuscript Section 2)."
    )
    p.add_argument("--results-dir", type=str, default="./results")
    p.add_argument("--skip-baselines", action="store_true",
                    help="Skip RF/SVM-RBF/XGBoost in Table 2 (SSL + Supervised MLP only)")
    p.add_argument("--skip-label-efficiency", action="store_true",
                    help="Skip the 5/10/20/100%% sweep (Table 3)")
    p.add_argument("--skip-ablation", action="store_true",
                    help="Skip the augmentation ablation (Table 4)")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-enrichment", action="store_true",
                    help="Skip Enrichr pathway enrichment (needs internet + real gene symbols)")
    p.add_argument("--skip-external", action="store_true",
                    help="Skip GSE9476 external validation (needs a manual class mapping — "
                         "see scarf_ssl/external_validation.py)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=not args.quiet)
    cfg = Config()
    cfg.results_dir = args.results_dir
    os.makedirs(cfg.results_dir, exist_ok=True)

    torch.manual_seed(cfg.split.seed)
    np.random.seed(cfg.split.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.split.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- 1. Load data ---
    X, y, class_names, probe_ids = load_gse13164(
        dest_dir=cfg.data.geo_dest_dir, geo_accession=cfg.data.geo_accession,
        missingness_threshold=cfg.data.missingness_threshold,
        min_class_size=cfg.data.min_class_size_for_cv,
        expected_n_samples=cfg.data.expected_n_samples,
        expected_n_classes=cfg.data.expected_n_classes,
    )

    # --- 2. Table 2: full 5-fold CV benchmark at 100% labels ---
    methods = ("ssl", "supervised_mlp") if args.skip_baselines else \
              ("ssl", "supervised_mlp", "svm_rbf", "xgboost", "random_forest")
    table2_raw = run_cv_benchmark(X, y, class_names, cfg, device, label_fraction=1.0, methods=methods)
    table2_raw.to_csv(os.path.join(cfg.results_dir, "table2_cv_raw.csv"), index=False)
    table2_summary = summarize_cv_results(table2_raw)
    table2_summary.to_csv(os.path.join(cfg.results_dir, "table2_cv_summary.csv"), index=False)
    logger.info("Table 2 (5-fold CV benchmark) saved.")

    # --- 3. Table 3: label-efficiency sweep, and 4. significance testing ---
    if not args.skip_label_efficiency:
        sweep_methods = ("ssl", "supervised_mlp", "xgboost", "random_forest")
        table3_raw = run_label_efficiency_sweep(X, y, class_names, cfg, device, methods=sweep_methods)
        table3_raw.to_csv(os.path.join(cfg.results_dir, "table3_label_efficiency_raw.csv"), index=False)
        table3_summary = summarize_cv_results(table3_raw)
        table3_summary.to_csv(os.path.join(cfg.results_dir, "table3_label_efficiency_summary.csv"), index=False)
        logger.info("Table 3 (label-efficiency sweep) saved.")

        sig_results = run_significance_tests(table3_raw)
        sig_results.to_csv(os.path.join(cfg.results_dir, "significance_tests.csv"), index=False)
    else:
        sig_results = run_significance_tests(table2_raw)
        sig_results.to_csv(os.path.join(cfg.results_dir, "significance_tests_100pct_only.csv"), index=False)
    logger.info("Significance tests saved.")

    # --- 5. Table 4: augmentation ablation ---
    if not args.skip_ablation:
        table4 = run_augmentation_ablation(X, y, class_names, cfg, device, seed=cfg.split.seed)
        table4.to_csv(os.path.join(cfg.results_dir, "table4_augmentation_ablation.csv"), index=False)
        logger.info("Table 4 (augmentation ablation) saved.")

    # --- 6. Final SSL model on full data, for SHAP / enrichment / external validation ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    encoder = pretrain_ssl(
        X_scaled, SCARFCorruption(cfg.augmentation.scarf_corruption_rate), device,
        epochs=cfg.ssl.pretrain_epochs, batch_size=cfg.ssl.pretrain_batch_size,
        lr=cfg.ssl.pretrain_lr, weight_decay=cfg.ssl.pretrain_weight_decay,
        temperature=cfg.ssl.temperature, hidden_dim_1=cfg.ssl.hidden_dim_1,
        hidden_dim_2=cfg.ssl.hidden_dim_2, dropout=cfg.ssl.dropout, verbose=True,
    )
    X_subtrain, X_val, y_subtrain, y_val = stratified_subtrain_val_split(X_scaled, y, 0.20, cfg.split.seed)
    min_count = np.bincount(y_subtrain).min()
    if min_count > 1:
        smote = SMOTE(random_state=cfg.split.seed, k_neighbors=max(1, min(5, min_count - 1)))
        X_subtrain, y_subtrain = smote.fit_resample(X_subtrain, y_subtrain)
    final_model = two_stage_finetune(
        encoder, X_subtrain, y_subtrain, X_val, y_val, len(class_names), device,
        hidden_dim_2=cfg.ssl.hidden_dim_2, classifier_hidden_dim=cfg.finetune.classifier_hidden_dim,
        classifier_dropout=cfg.finetune.classifier_dropout,
        linear_probe_epochs=cfg.finetune.linear_probe_epochs,
        finetune_epochs=cfg.finetune.finetune_epochs, batch_size=cfg.finetune.finetune_batch_size,
        lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay,
        scheduler_patience=cfg.finetune.scheduler_patience,
        scheduler_factor=cfg.finetune.scheduler_factor, verbose=True,
    )
    torch.save(final_model.state_dict(), os.path.join(cfg.results_dir, "final_model.pt"))
    import joblib
    joblib.dump(scaler, os.path.join(cfg.results_dir, "scaler.pkl"))
    logger.info("Final model and scaler saved to %s/", cfg.results_dir)

    # --- 7. SHAP (PCA surrogate) ---
    if not args.skip_shap:
        shap_values, pca, surrogate = compute_shap_pca_surrogate(
            final_model, X_scaled, y, device, cfg.results_dir, seed=cfg.split.seed,
            n_pca_components=cfg.interpretability.pca_n_components,
            min_explained_variance=cfg.interpretability.pca_min_explained_variance,
            background_size=cfg.interpretability.shap_background_size,
            sample_size=cfg.interpretability.shap_sample_size,
        )
        combined_weight = get_combined_linear_weight(final_model.encoder)
        gene_importance_df = back_project_to_genes(
            shap_values, pca, combined_weight, probe_ids, cfg.results_dir,
            top_n_genes=cfg.interpretability.top_n_genes,
        )

        # --- 8. Pathway enrichment ---
        if not args.skip_enrichment:
            logger.warning(
                "Pathway enrichment needs GENE SYMBOLS, not probe IDs. gene_importance_df "
                "currently holds probe/feature identifiers from GSE13164 (GPL7473) — map "
                "these to HGNC symbols using the platform's annotation table before this step "
                "produces meaningful results. See interpretability.py's run_pathway_enrichment "
                "docstring."
            )
            try:
                run_pathway_enrichment(
                    gene_importance_df["probe_id"].tolist(), cfg.results_dir,
                    gene_sets=cfg.interpretability.enrichr_gene_sets,
                    fdr_threshold=cfg.interpretability.enrichr_fdr_threshold,
                )
            except Exception as e:
                logger.error(
                    "Pathway enrichment failed (likely needs internet access and/or a "
                    "probe-to-gene-symbol mapping first): %s", e,
                )

    # --- 9. External validation (requires manual class mapping) ---
    if not args.skip_external:
        logger.warning(
            "External validation (Table 5) requires a manual class_name_mapping from your "
            "18 MILE classes to GSE9476's class taxonomy — this is not automated (see "
            "scarf_ssl/external_validation.py). Wire up your mapping and call "
            "evaluate_on_external_cohort() directly, or extend this script, once you've "
            "resolved the 7-vs-5-class discrepancy flagged in the manuscript audit."
        )

    logger.info("Pipeline complete. All outputs saved to: %s/", cfg.results_dir)


if __name__ == "__main__":
    main()
