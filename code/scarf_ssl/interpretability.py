"""
Interpretability (Section 2.7): SHAP via a PCA + logistic-regression
surrogate (matching the manuscript's stated method — distinct from running
DeepExplainer directly on the raw model), plus Enrichr pathway enrichment.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shap  # noqa: E402

logger = logging.getLogger("scarf_ssl")


def compute_shap_pca_surrogate(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    results_dir: str,
    seed: int = 42,
    n_pca_components: int = 50,
    min_explained_variance: float = 0.99,
    background_size: int = 100,
    sample_size: int = 100,
) -> Tuple[np.ndarray, PCA, LogisticRegression]:
    """
    Section 2.7 method: extract encoder embeddings -> PCA to
    n_pca_components -> fit a logistic-regression surrogate on the reduced
    embeddings -> compute SHAP values via KernelExplainer on the
    surrogate. Returns raw SHAP values in PCA-component space (not yet
    back-projected to genes — see back_project_to_genes below).
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        embeddings = model.embed(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()

    pca = PCA(n_components=n_pca_components, random_state=seed)
    embeddings_pca = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA: %d components explain %.2f%% variance (target >= %.0f%%)",
        n_pca_components, explained * 100, min_explained_variance * 100,
    )
    if explained < min_explained_variance:
        logger.warning(
            "PCA explained variance (%.2f%%) is below the manuscript's target (%.0f%%). "
            "Consider increasing n_pca_components.",
            explained * 100, min_explained_variance * 100,
        )

    surrogate = LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=seed)
    surrogate.fit(embeddings_pca, y)
    train_acc = surrogate.score(embeddings_pca, y)
    logger.info(
        "Logistic-regression surrogate training accuracy: %.4f. This should be reasonably "
        "high for SHAP-on-the-surrogate to be a faithful proxy for the real model's behaviour "
        "— if it's low, the attributions below describe the surrogate, not the SSL classifier.",
        train_acc,
    )

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(embeddings_pca), size=min(background_size, len(embeddings_pca)), replace=False)
    sample_idx = rng.choice(len(embeddings_pca), size=min(sample_size, len(embeddings_pca)), replace=False)
    background = embeddings_pca[bg_idx]
    samples = embeddings_pca[sample_idx]

    explainer = shap.KernelExplainer(surrogate.predict_proba, background)
    shap_values = explainer.shap_values(samples)

    np.save(os.path.join(results_dir, "shap_values_pca_space.npy"), np.array(shap_values, dtype=object))
    return np.array(shap_values), pca, surrogate


def get_combined_linear_weight(encoder: torch.nn.Module) -> np.ndarray:
    """Extracts and combines the encoder's two Linear layers into a single
    (output_dim, input_dim) matrix — a linear approximation of the encoder,
    used only for the SHAP back-projection heuristic below (ignores
    BatchNorm/ReLU/Dropout, which don't have a closed-form linear inverse;
    the manuscript's stated back-projection method doesn't address this
    either)."""
    linears = [m for m in encoder.net if isinstance(m, torch.nn.Linear)]
    if len(linears) != 2:
        raise ValueError(f"Expected exactly 2 Linear layers in encoder.net, found {len(linears)}.")
    W1 = linears[0].weight.detach().cpu().numpy()  # (hidden_dim_1, input_dim)
    W2 = linears[1].weight.detach().cpu().numpy()  # (hidden_dim_2, hidden_dim_1)
    return np.abs(W2) @ np.abs(W1)  # (hidden_dim_2, input_dim)


def back_project_to_genes(
    shap_values_pca: np.ndarray,
    pca: PCA,
    encoder_combined_weight: np.ndarray,
    probe_ids: List[str],
    results_dir: str,
    top_n_genes: int = 150,
) -> pd.DataFrame:
    """
    Back-projects SHAP values from PCA-component space through the PCA
    loadings and the encoder's (linearly-approximated) weights to
    approximate per-gene importance scores (Section 2.7).

    NOTE on approximation: "back-projected through the PCA loadings matrix
    and the first encoder weight layer" is, taken literally, dimensionally
    inconsistent (the encoder's actual first layer maps input_dim -> 512,
    not to the 256-dim space PCA was fit on). This implementation uses
    absolute-value loadings/weights as an importance-propagation
    heuristic through BOTH encoder linear layers combined (see
    get_combined_linear_weight) — a standard simplification for this kind
    of gene-attribution back-projection, but an approximation, not an
    exact attribution. State this explicitly as a limitation in the
    manuscript if it isn't already.
    """
    os.makedirs(results_dir, exist_ok=True)

    shap_arr = np.array(shap_values_pca)
    if shap_arr.ndim == 3:  # (n_classes, n_samples, n_components)
        mean_abs_per_component = np.mean(np.abs(shap_arr), axis=(0, 1))
    else:  # (n_samples, n_components)
        mean_abs_per_component = np.mean(np.abs(shap_arr), axis=0)

    loadings = np.abs(pca.components_)                      # (n_components, hidden_dim_2)
    latent_importance = mean_abs_per_component @ loadings   # (hidden_dim_2,)
    gene_importance = latent_importance @ np.abs(encoder_combined_weight)  # (input_dim,)

    n_genes = min(top_n_genes, len(probe_ids))
    top_idx = np.argsort(gene_importance)[::-1][:n_genes]
    result = pd.DataFrame({
        "probe_id": [probe_ids[i] for i in top_idx],
        "importance": gene_importance[top_idx],
    })
    result.to_csv(os.path.join(results_dir, "shap_top_genes.csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(range(min(20, n_genes)), result["importance"].values[:20][::-1])
    plt.yticks(range(min(20, n_genes)), result["probe_id"].values[:20][::-1])
    plt.xlabel("Back-projected importance (a.u.)")
    plt.title(f"Top {min(20, n_genes)} genes by back-projected SHAP importance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "shap_top_genes.png"), dpi=150)
    plt.close()

    logger.info("Saved top-%d gene importance table and plot to %s/", n_genes, results_dir)
    return result


def run_pathway_enrichment(
    gene_list: List[str],
    results_dir: str,
    gene_sets: Optional[List[str]] = None,
    fdr_threshold: float = 0.05,
):
    """
    Section 2.7: submits a gene list to Enrichr (via gseapy) against the
    given gene-set libraries, filters to Benjamini-Hochberg adjusted p <
    fdr_threshold, and saves a results table + top-10 bar plot.

    IMPORTANT: `gene_list` must be actual gene SYMBOLS (e.g. "CD34",
    "MPO"), not probe/feature IDs. probe_ids returned by load_gse13164 are
    Roche AmpliChip probe identifiers, not gene symbols — you need a
    probe-to-symbol mapping step before this will produce meaningful
    results. Inspect GPL7473's platform annotation table
    (`gse.gpls['GPL7473'].table`) for a usable gene-symbol column, or
    supply your own mapping; this function does not fabricate one.

    Requires internet access to Enrichr (https://maayanlab.cloud) — this
    will not run in a network-restricted sandbox.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.error("gseapy is not installed — run `pip install gseapy` to use pathway enrichment.")
        raise

    os.makedirs(results_dir, exist_ok=True)
    gene_sets = gene_sets or ["GO_Biological_Process_2021", "KEGG_2021_Human"]

    logger.info("Submitting %d genes to Enrichr against %s", len(gene_list), gene_sets)
    enr = gp.enrichr(
        gene_list=gene_list, gene_sets=gene_sets, organism="human",
        outdir=os.path.join(results_dir, "enrichr"), no_plot=True,
    )

    results = enr.results
    significant = results[results["Adjusted P-value"] < fdr_threshold].sort_values("Adjusted P-value")
    significant.to_csv(os.path.join(results_dir, "pathway_enrichment_significant.csv"), index=False)
    logger.info("Found %d significant pathways (adj. p < %.2f).", len(significant), fdr_threshold)

    if len(significant) > 0:
        top10 = significant.head(10).copy()
        top10["neglog10_padj"] = -np.log10(top10["Adjusted P-value"])
        plt.figure(figsize=(8, 5))
        plt.barh(top10["Term"][::-1], top10["neglog10_padj"][::-1])
        plt.axvline(-np.log10(fdr_threshold), color="red", linestyle="--", label=f"p={fdr_threshold}")
        plt.xlabel("-log10(adjusted p-value)")
        plt.title("Top enriched pathways")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "pathway_enrichment_top10.png"), dpi=150)
        plt.close()

    return significant
