"""
Contrastive loss for SCARF-SSL pre-training (Section 2.3.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR-style).

    NOTE on the "theoretical lower bound" language in the manuscript
    (Section 3.1): log(N-1) is the EXPECTED loss under random
    initialisation (uniform similarity), not a true lower bound — the
    loss can and should fall below it as training converges. If the
    manuscript still calls it a "lower bound", reword to "expected value
    under random initialisation" (flagged in the audit).
    """

    def __init__(self, temperature: float = 0.5) -> None:
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
