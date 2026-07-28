"""
SCARF-SSL: Self-Supervised Contrastive Learning for Multi-Class
Leukaemia Subtype Classification from Microarray Gene Expression Data

Version: 1.0.0

Repository:
    https://github.com/<username>/SCARF-SSL-Microarray-Leukemia

Dataset:
    NCBI GEO GSE13164 (MILE Study; 1,152 samples; 18 leukaemia subtypes)

License:
    GNU GPL v3.0

Citation:
    Please cite the associated publication if this repository contributes
    to your research.

Pipeline:
    1. Automated NCBI GEO data fetching & preprocessing (GSE13164)
    2. Stratified k-fold cross-validation with per-fold scaling (no leakage)
    3. SMOTE class balancing applied to training folds only
    4. Tabular SCARF augmentation (marginal feature corruption)
    5. NT-Xent contrastive loss for unsupervised pre-training
    6. Supervised fine-tuning for leukaemia subtype classification
    7. SHAP feature-importance analysis on the final model
    8. t-SNE visualisation of learned embeddings

Design notes:
    - StandardScaler and SMOTE are fit on the TRAINING fold only, then
      applied to the held-out fold — no global fit-before-split leakage.
    - Reported accuracy/F1 come from stratified k-fold CV (fold-wise mean
      +/- std, plus an aggregated out-of-fold classification report), not
      from a single train/test split.
    - A separate "final" model is trained on the full dataset purely for
      SHAP/t-SNE/deployment artefacts — it is never used to report
      accuracy, so it does not affect the CV numbers above.
    - All randomness (numpy, random, torch CPU + CUDA) is seeded.
    - Detected class labels are logged explicitly so they can be verified
      against the expected 18 MILE subtypes before trusting results.
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import copy
import logging
import os
import random
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE

import joblib

import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots to disk
import matplotlib.pyplot as plt

import shap
import GEOparse

warnings.filterwarnings("ignore")

logger = logging.getLogger("scarf_ssl")

# ============================================================================
# 1. DEFAULT CONFIGURATION (overridable via CLI — see parse_args())
# ============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GEO_ACCESSION = "GSE13164"
GEO_DEST_DIR = "./geo_data"
RESULTS_DIR = "./results"

CORRUPTION_RATE = 0.3
TEMPERATURE = 0.3
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 30
LEARNING_RATE = 1e-3
N_FOLDS = 5
LABEL_FRACTION = 1.0  # fraction of each training fold used for fine-tuning
                       # (set < 1.0 only for deliberate semi-supervised-label experiments)


def setup_logging(verbose: bool = True) -> None:
    """Configure root logging with timestamps and levels."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_all_seeds(seed: int = SEED) -> None:
    """Seed numpy, random, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# 2. DATA LOADING & PREPROCESSING FOR GSE13164
# ============================================================================
def load_gse13164(
    dest_dir: str = GEO_DEST_DIR,
    geo_accession: str = GEO_ACCESSION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downloads a GEO series and extracts the raw (unscaled) expression matrix
    and clinical labels. Scaling is intentionally NOT done here — it must
    happen per-fold, inside cross-validation, to avoid leakage.

    Returns:
        X: (n_samples, n_features) float32 array of raw expression values.
        y: (n_samples,) int array of encoded class labels.
        class_names: (n_classes,) array of original class label strings.
    """
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading / loading %s from NCBI GEO", geo_accession)

    gse = GEOparse.get_GEO(geo=geo_accession, destdir=dest_dir)

    try:
        platforms = list(gse.gpls.keys())
        logger.info("Resolved platform(s) for %s: %s", geo_accession, platforms)
    except AttributeError:
        logger.warning("Could not introspect platform metadata — verify manually via GEO.")

    logger.info("Extracting gene expression matrix...")
    expr_df = gse.pivot_samples("VALUE")   # Probes x Samples
    X_df = expr_df.T                        # Samples x Probes
    X_df = X_df.fillna(X_df.median())

    logger.info("Parsing clinical metadata for leukaemia subtypes...")
    pheno_df = gse.phenotype_data

    target_col: Optional[str] = None
    for col in pheno_df.columns:
        if "characteristics" in col or "title" in col or "source" in col:
            unique_vals = pheno_df[col].dropna().astype(str).tolist()
            if any(
                term in " ".join(unique_vals).lower()
                for term in ["leukemia", "leukaemia", "all", "aml", "cll", "cml", "control", "normal"]
            ):
                target_col = col
                break
    if target_col is None:
        target_col = "title"

    y_raw = pheno_df.loc[X_df.index, target_col].values
    logger.info("Identified label column: '%s'", target_col)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    logger.info("Dataset summary: %d samples, %d features (probes)", X_df.shape[0], X_df.shape[1])
    logger.info("Classes detected (%d):", len(encoder.classes_))
    for i, c in enumerate(encoder.classes_):
        n = int((y == i).sum())
        logger.info("  [%d] %s  (n=%d)", i, c, n)
    logger.info(
        "VERIFY this matches the expected 18 MILE subtypes before trusting "
        "downstream metrics. If the count or labels look wrong, inspect "
        "`pheno_df.columns` and set `target_col` manually."
    )

    return X_df.values.astype(np.float32), y, encoder.classes_

# ============================================================================
# 3. TABULAR AUGMENTATION (SCARF)
# ============================================================================
class TabularSCARFAugmenter:
    """
    Marginal feature corruption for tabular data. Swaps a fraction of genes
    with values from other patients in the same batch (preserving the
    marginal distribution) rather than zero-masking, plus small Gaussian
    technical noise.
    """

    def __init__(self, corruption_rate: float = 0.3, noise_std: float = 0.05) -> None:
        self.corruption_rate = corruption_rate
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = x.shape

        aug1 = x.clone()
        mask1 = torch.rand_like(aug1) < self.corruption_rate
        idx1 = torch.randperm(batch_size, device=x.device)
        aug1[mask1] = x[idx1][mask1]

        aug2 = x.clone()
        mask2 = torch.rand_like(aug2) < self.corruption_rate
        idx2 = torch.randperm(batch_size, device=x.device)
        aug2[mask2] = x[idx2][mask2]

        aug1 += torch.randn_like(aug1) * self.noise_std
        aug2 += torch.randn_like(aug2) * self.noise_std

        return aug1, aug2

# ============================================================================
# 4. CONTRASTIVE LOSS
# ============================================================================
class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss."""

    def __init__(self, temperature: float = 0.3) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, p=2, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim[~mask].view(2 * batch_size, -1)

        positives = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)]
        ).to(z.device)

        return F.cross_entropy(sim, positives)

# ============================================================================
# 5. NEURAL NETWORK ARCHITECTURES
# ============================================================================
class GeneExpressionEncoder(nn.Module):
    """MLP backbone extracting latent representations from gene expression vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class LeukemiaClassifier(nn.Module):
    """Classification head attached to the pre-trained encoder."""

    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

# ============================================================================
# 6. TRAINING ROUTINES
# ============================================================================
def pretrain_encoder_scarf(
    X_unlabeled: np.ndarray,
    epochs: int = PRETRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    corruption_rate: float = CORRUPTION_RATE,
    temperature: float = TEMPERATURE,
    verbose: bool = True,
) -> nn.Module:
    """Phase 1: unsupervised contrastive pre-training (no labels used)."""
    model = GeneExpressionEncoder(input_dim=X_unlabeled.shape[1]).to(DEVICE)
    dataset = TensorDataset(torch.FloatTensor(X_unlabeled))
    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(X_unlabeled)), shuffle=True, drop_last=True
    )

    augmenter = TabularSCARFAugmenter(corruption_rate=corruption_rate)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE)
            x_aug1, x_aug2 = augmenter(x_batch)
            z1, z2 = model(x_aug1), model(x_aug2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 10 == 0:
            avg = epoch_loss / max(len(loader), 1)
            logger.info("  [Pretrain] Epoch %d/%d - Contrastive Loss: %.4f", epoch + 1, epochs, avg)

    return model

def finetune_classifier(
    pretrained_encoder: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = 16,
    lr: float = LEARNING_RATE,
    verbose: bool = True,
) -> nn.Module:
    """Phase 2: supervised fine-tuning on labelled data."""
    encoder_clone = copy.deepcopy(pretrained_encoder)
    model = LeukemiaClassifier(encoder_clone, num_classes).to(DEVICE)

    dataset = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    if verbose:
        logger.info("  [Finetune] Complete.")
    return model

def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X).to(DEVICE))
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

# ============================================================================
# 7. STRATIFIED K-FOLD CROSS-VALIDATION (leakage-safe)
# ============================================================================
def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    corruption_rate: float = CORRUPTION_RATE,
    temperature: float = TEMPERATURE,
    batch_size: int = BATCH_SIZE,
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    label_fraction: float = LABEL_FRACTION,
    results_dir: str = RESULTS_DIR,
) -> Tuple[List[float], List[float], np.ndarray, np.ndarray]:
    """Runs leakage-safe stratified k-fold CV and saves metrics/artefacts."""
    os.makedirs(results_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accuracies: List[float] = []
    fold_f1s: List[float] = []
    oof_true = np.zeros_like(y)
    oof_pred = np.zeros_like(y)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        logger.info("=== Fold %d/%d ===", fold_idx, n_folds)
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Scale: fit on TRAIN fold only, transform test fold with it ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # --- SMOTE: fit/resample on TRAIN fold only ---
        min_class_count = np.bincount(y_train).min()
        k_neighbors = max(1, min(5, min_class_count - 1))
        if min_class_count > 1:
            smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        else:
            logger.warning("  [SMOTE] Skipped — a class has only 1 sample in this fold.")
            X_train_bal, y_train_bal = X_train, y_train

        # --- Phase 1: unsupervised pre-training on the (unbalanced) train fold ---
        encoder = pretrain_encoder_scarf(
            X_train,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            corruption_rate=corruption_rate,
            temperature=temperature,
            verbose=(fold_idx == 1),
        )

        # --- Phase 2: supervised fine-tuning on SMOTE-balanced train fold ---
        if label_fraction < 1.0:
            n_use = int(len(X_train_bal) * label_fraction)
            X_ft, _, y_ft, _ = train_test_split(
                X_train_bal, y_train_bal, train_size=n_use, stratify=y_train_bal, random_state=seed
            )
        else:
            X_ft, y_ft = X_train_bal, y_train_bal

        model = finetune_classifier(
            encoder, X_ft, y_ft, len(class_names),
            epochs=finetune_epochs, lr=learning_rate, verbose=(fold_idx == 1),
        )

        # --- Evaluate on held-out fold (never scaled/resampled with test info) ---
        y_pred = predict(model, X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        oof_true[test_idx] = y_test
        oof_pred[test_idx] = y_pred

        logger.info("  Fold %d — Accuracy: %.2f%%, Weighted F1: %.2f%%", fold_idx, acc * 100, f1 * 100)

    logger.info(
        "Cross-validated Accuracy: %.2f%% +/- %.2f%%",
        np.mean(fold_accuracies) * 100, np.std(fold_accuracies) * 100,
    )
    logger.info(
        "Cross-validated Weighted F1: %.2f%% +/- %.2f%%",
        np.mean(fold_f1s) * 100, np.std(fold_f1s) * 100,
    )

    report = classification_report(
        oof_true, oof_pred, target_names=[str(c) for c in class_names], zero_division=0
    )
    logger.info("Aggregated Out-of-Fold Classification Report:\n%s", report)

    with open(os.path.join(results_dir, "cv_classification_report.txt"), "w") as f:
        f.write(report)

    pd.DataFrame({
        "fold": range(1, n_folds + 1),
        "accuracy": fold_accuracies,
        "weighted_f1": fold_f1s,
    }).to_csv(os.path.join(results_dir, "cv_fold_metrics.csv"), index=False)

    cm = confusion_matrix(oof_true, oof_pred)
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)

    return fold_accuracies, fold_f1s, oof_true, oof_pred

# ============================================================================
# 8. FINAL MODEL (trained on all data — for SHAP / t-SNE / deployment only,
#    NOT used to report accuracy — accuracy comes from cross-validation above)
# ============================================================================
def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    seed: int = SEED,
    corruption_rate: float = CORRUPTION_RATE,
    temperature: float = TEMPERATURE,
    batch_size: int = BATCH_SIZE,
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    results_dir: str = RESULTS_DIR,
) -> Tuple[nn.Module, StandardScaler, np.ndarray]:
    """Trains a model on the full dataset and persists both model and scaler."""
    logger.info("Training final model on the full dataset (for SHAP / t-SNE / deployment)")
    os.makedirs(results_dir, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=seed, k_neighbors=min(5, np.bincount(y).min() - 1))
    X_bal, y_bal = smote.fit_resample(X_scaled, y)

    encoder = pretrain_encoder_scarf(
        X_scaled, epochs=pretrain_epochs, batch_size=batch_size, lr=learning_rate,
        corruption_rate=corruption_rate, temperature=temperature, verbose=True,
    )
    model = finetune_classifier(
        encoder, X_bal, y_bal, len(class_names),
        epochs=finetune_epochs, lr=learning_rate, verbose=True,
    )

    model_path = os.path.join(results_dir, "final_model.pt")
    scaler_path = os.path.join(results_dir, "scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved final model to %s", model_path)
    logger.info("Saved fitted scaler to %s (required for inference on new samples)", scaler_path)

    return model, scaler, X_scaled

# ============================================================================
# 9. SHAP FEATURE IMPORTANCE
# ============================================================================
def run_shap_analysis(
    model: nn.Module,
    X_scaled: np.ndarray,
    results_dir: str = RESULTS_DIR,
    seed: int = SEED,
    background_size: int = 100,
    sample_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Running SHAP analysis")
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_scaled), size=min(background_size, len(X_scaled)), replace=False)
    sample_idx = rng.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)

    background = torch.FloatTensor(X_scaled[bg_idx]).to(DEVICE)
    samples = torch.FloatTensor(X_scaled[sample_idx]).to(DEVICE)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(samples)

    np.save(os.path.join(results_dir, "shap_values.npy"), np.array(shap_values))

    mean_abs_shap = np.mean(np.abs(np.array(shap_values)), axis=(0, 1))
    top_features = np.argsort(mean_abs_shap)[::-1][:20]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_features)), mean_abs_shap[top_features][::-1])
    plt.yticks(range(len(top_features)), [f"probe_{i}" for i in top_features[::-1]])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 20 Most Important Probes")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "shap_top_features.png"), dpi=150)
    plt.close()

    logger.info("Saved SHAP values and top-feature plot to %s/", results_dir)
    return shap_values, top_features

# ============================================================================
# 10. t-SNE VISUALISATION
# ============================================================================
def run_tsne_visualization(
    model: nn.Module,
    X_scaled: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    results_dir: str = RESULTS_DIR,
    seed: int = SEED,
) -> None:
    logger.info("Running t-SNE visualisation of learned embeddings")
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(torch.FloatTensor(X_scaled).to(DEVICE)).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X_scaled) - 1))
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, cmap="tab20", s=15, alpha=0.8)
    plt.legend(
        handles=scatter.legend_elements(num=len(class_names))[0],
        labels=[str(c) for c in class_names],
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
    )
    plt.title("t-SNE of SCARF-SSL Learned Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tsne_embeddings.png"), dpi=150)
    plt.close()

    logger.info("Saved t-SNE plot to %s/tsne_embeddings.png", results_dir)

# ============================================================================
# 11. CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCARF-SSL: self-supervised contrastive learning for leukaemia "
            "subtype classification from GEO microarray data."
        )
    )
    parser.add_argument("--geo-accession", type=str, default=GEO_ACCESSION,
                         help="NCBI GEO accession to download (default: %(default)s)")
    parser.add_argument("--dest-dir", type=str, default=GEO_DEST_DIR,
                         help="Directory to cache downloaded GEO data (default: %(default)s)")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                         help="Directory to save results, plots, and checkpoints (default: %(default)s)")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS,
                         help="Number of stratified CV folds (default: %(default)s)")
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS,
                         help="Unsupervised SCARF pre-training epochs (default: %(default)s)")
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS,
                         help="Supervised fine-tuning epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help="Pre-training batch size (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                         help="Learning rate for both training phases (default: %(default)s)")
    parser.add_argument("--corruption-rate", type=float, default=CORRUPTION_RATE,
                         help="SCARF feature corruption rate (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                         help="NT-Xent temperature (default: %(default)s)")
    parser.add_argument("--label-fraction", type=float, default=LABEL_FRACTION,
                         help="Fraction of each training fold's labels used for fine-tuning "
                              "(default: %(default)s; set <1.0 for semi-supervised experiments)")
    parser.add_argument("--seed", type=int, default=SEED,
                         help="Random seed for reproducibility (default: %(default)s)")
    parser.add_argument("--skip-shap", action="store_true",
                         help="Skip SHAP analysis (can be slow on high-dimensional data)")
    parser.add_argument("--skip-tsne", action="store_true",
                         help="Skip t-SNE visualisation")
    parser.add_argument("--quiet", action="store_true",
                         help="Reduce logging verbosity to warnings only")
    return parser.parse_args()

# ============================================================================
# 12. MAIN PIPELINE
# ============================================================================
def main() -> None:
    args = parse_args()
    setup_logging(verbose=not args.quiet)
    set_all_seeds(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    logger.info("SCARF-SSL v%s", __version__)
    logger.info("Using compute device: %s", DEVICE)

    X, y, class_names = load_gse13164(dest_dir=args.dest_dir, geo_accession=args.geo_accession)

    # --- Cross-validated evaluation (this is what reported metrics should cite) ---
    run_cross_validation(
        X, y, class_names,
        n_folds=args.n_folds,
        seed=args.seed,
        corruption_rate=args.corruption_rate,
        temperature=args.temperature,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        learning_rate=args.learning_rate,
        label_fraction=args.label_fraction,
        results_dir=args.results_dir,
    )

    # --- Final model on all data, for interpretability artefacts only ---
    final_model, final_scaler, X_scaled_full = train_final_model(
        X, y, class_names,
        seed=args.seed,
        corruption_rate=args.corruption_rate,
        temperature=args.temperature,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        learning_rate=args.learning_rate,
        results_dir=args.results_dir,
    )

    if not args.skip_shap:
        run_shap_analysis(final_model, X_scaled_full, results_dir=args.results_dir, seed=args.seed)
    if not args.skip_tsne:
        run_tsne_visualization(final_model, X_scaled_full, y, class_names,
                                results_dir=args.results_dir, seed=args.seed)

    logger.info("All results, plots, and the final model checkpoint saved to: %s/", args.results_dir)


if __name__ == "__main__":
    main()
