"""
Central configuration for the SCARF-SSL leukaemia classification pipeline.

Every value here is set to match the manuscript's Methods section (2.1-2.9)
as written. Anything with a NOTE comment flags a place where the manuscript
was ambiguous, internally inconsistent, or dimensionally impossible as
literally written, and documents the implementation choice made instead.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    geo_accession: str = "GSE13164"           # confirmed correct: n=1152, GPL7473
    geo_dest_dir: str = "./geo_data"
    external_geo_accession: str = "GSE9476"    # external validation cohort, GPL96
    external_geo_dest_dir: str = "./geo_data_external"
    expected_n_samples: int = 1152
    expected_n_classes: int = 18
    expected_n_features: int = 1457            # after excluding probes with >20% missingness
    missingness_threshold: float = 0.20
    expected_external_n_samples: int = 64       # confirmed: 38 normal + 26 AML (GSE9476)
    min_class_size_for_cv: int = 5              # classes below this are excluded (Section 2.1)


@dataclass
class SplitConfig:
    train_fraction: float = 0.64
    val_fraction: float = 0.16
    test_fraction: float = 0.20
    n_folds: int = 5
    seed: int = 42


@dataclass
class AugmentationConfig:
    gaussian_noise_sigma: float = 0.1
    gene_mask_ratio: float = 0.15
    probe_subset_retain: float = 0.80
    magnitude_warp_sigma: float = 0.1           # log-normal sigma, mu=0
    scarf_corruption_rate: float = 0.30


@dataclass
class SSLConfig:
    hidden_dim_1: int = 512
    hidden_dim_2: int = 256
    dropout: float = 0.30
    projection_dim: int = 128
    temperature: float = 0.5
    pretrain_epochs: int = 25
    pretrain_batch_size: int = 64
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 1e-5


@dataclass
class FinetuneConfig:
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.30
    linear_probe_epochs: int = 5
    finetune_epochs: int = 20
    finetune_batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


@dataclass
class BaselineConfig:
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    svm_C: float = 10.0
    svm_gamma: str = "scale"
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1


@dataclass
class LabelEfficiencyConfig:
    fractions: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 1.00])


@dataclass
class InterpretabilityConfig:
    pca_n_components: int = 50
    pca_min_explained_variance: float = 0.99
    shap_background_size: int = 100
    shap_sample_size: int = 100
    top_n_genes: int = 150
    enrichr_gene_sets: List[str] = field(
        default_factory=lambda: ["GO_Biological_Process_2021", "KEGG_2021_Human"]
    )
    enrichr_fdr_threshold: float = 0.05


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    label_efficiency: LabelEfficiencyConfig = field(default_factory=LabelEfficiencyConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    results_dir: str = "./results"
