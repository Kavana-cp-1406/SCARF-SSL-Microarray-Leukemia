"""
Neural network architectures for SCARF-SSL: encoder, projection head
(pre-training only, discarded afterward), and classification head.
Matches manuscript Sections 2.3.1 and 2.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeneExpressionEncoder(nn.Module):
    """Two-block MLP backbone (Section 2.3.1): each block is
    Linear -> BatchNorm1d -> ReLU -> Dropout, with hidden dimensions
    512 and 256."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 256,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """Non-linear projection head used only during contrastive pre-training
    (SimCLR-style design) and discarded afterward.

    NOTE: the manuscript (Section 2.3.1) describes this as "512 -> 128 ->
    128", but the encoder's own output is 256-dim, not 512-dim — the 512
    figure appears to describe the encoder's *first* hidden layer, not
    what actually feeds the projection head. This implementation takes the
    encoder's real 256-dim output and projects 256 -> 128 -> 128, the only
    architecture that is dimensionally consistent with the rest of the
    pipeline. Worth fixing in the manuscript text (flagged separately in
    the audit).
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    """Supervised classification head (Section 2.4): 256 -> 128 ->
    num_classes, with ReLU and Dropout between the two linear layers."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 18,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSLPretrainModel(nn.Module):
    """Encoder + projection head, used only for contrastive pre-training."""

    def __init__(self, encoder: GeneExpressionEncoder, projection_head: ProjectionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.encoder(x))


class LeukemiaClassifier(nn.Module):
    """Encoder + classification head, used for supervised fine-tuning and
    inference. The projection head is intentionally not part of this
    model — per Section 2.3.1 it is discarded after pre-training."""

    def __init__(self, encoder: GeneExpressionEncoder, classifier_head: ClassifierHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(self.encoder(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder-only embeddings (used by SHAP / t-SNE)."""
        return self.encoder(x)
