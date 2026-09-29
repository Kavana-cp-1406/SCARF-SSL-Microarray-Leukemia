"""
Tabular augmentation strategies for SCARF-SSL contrastive pre-training.

Implements the four biologically-motivated augmentations from Section 2.3.2
of the manuscript, the SCARF marginal-feature-corruption strategy
(Section 2.3.2, Bahri et al.), and a "Full Combined" augmenter that applies
all four biologically-motivated strategies together (matching Table 4's
"Full Combined" ablation row).

Every augmenter follows the same interface: __call__(x) -> (aug1, aug2),
two independently augmented views of the same mini-batch, matching Section
2.3.2 ("two independently augmented views were generated per mini-batch
iteration").
"""

from __future__ import annotations
from typing import Tuple

import torch


class BaseAugmenter:
    """Common interface for all augmentation strategies."""

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GaussianNoiseAugmentation(BaseAugmenter):
    """Additive Gaussian noise simulating hybridisation/measurement variability."""

    def __init__(self, sigma: float = 0.1) -> None:
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aug1 = x + torch.randn_like(x) * self.sigma
        aug2 = x + torch.randn_like(x) * self.sigma
        return aug1, aug2


class RandomGeneMasking(BaseAugmenter):
    """Stochastic binary masking (set to zero) simulating probe hybridisation
    failures or signal saturation."""

    def __init__(self, mask_ratio: float = 0.15) -> None:
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask1 = torch.rand_like(x) < self.mask_ratio
        mask2 = torch.rand_like(x) < self.mask_ratio
        aug1 = x.clone()
        aug2 = x.clone()
        aug1[mask1] = 0.0
        aug2[mask2] = 0.0
        return aug1, aug2


class GeneProbeSubsetting(BaseAugmenter):
    """Retains a random `retain_ratio` fraction of probes per view; the
    remaining probes are zeroed, forcing the encoder to reconstruct global
    signatures from partial, partially-overlapping gene sets."""

    def __init__(self, retain_ratio: float = 0.80) -> None:
        self.retain_ratio = retain_ratio

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keep1 = (torch.rand_like(x) < self.retain_ratio).float()
        keep2 = (torch.rand_like(x) < self.retain_ratio).float()
        return x * keep1, x * keep2


class MagnitudeWarping(BaseAugmenter):
    """Element-wise multiplication by a log-normal scaling factor (mu, sigma)
    simulating between-sample normalisation variability."""

    def __init__(self, mu: float = 0.0, sigma: float = 0.1) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        factor1 = torch.exp(torch.randn_like(x) * self.sigma + self.mu)
        factor2 = torch.exp(torch.randn_like(x) * self.sigma + self.mu)
        return x * factor1, x * factor2


class SCARFCorruption(BaseAugmenter):
    """Marginal feature corruption (Bahri et al., SCARF): swaps a fraction
    of features with values drawn from another sample in the same
    mini-batch, preserving each feature's marginal distribution. This is
    the main pre-training augmentation used for Tables 2, 3, and 5 (the
    paper is titled "SCARF-SSL"); Table 4 separately ablates the four
    biologically-motivated augmentations above instead."""

    def __init__(self, corruption_rate: float = 0.30, noise_std: float = 0.0) -> None:
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

        if self.noise_std > 0:
            aug1 = aug1 + torch.randn_like(aug1) * self.noise_std
            aug2 = aug2 + torch.randn_like(aug2) * self.noise_std

        return aug1, aug2


class CombinedAugmentation(BaseAugmenter):
    """Applies Gaussian noise, gene masking, probe subsetting, and magnitude
    warping in sequence to each view independently — the "Full Combined"
    strategy in Table 4."""

    def __init__(
        self,
        gaussian_sigma: float = 0.1,
        mask_ratio: float = 0.15,
        subset_retain: float = 0.80,
        warp_sigma: float = 0.1,
    ) -> None:
        self.gaussian = GaussianNoiseAugmentation(gaussian_sigma)
        self.masking = RandomGeneMasking(mask_ratio)
        self.subsetting = GeneProbeSubsetting(subset_retain)
        self.warping = MagnitudeWarping(sigma=warp_sigma)

    def _apply_one_view(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gaussian(x)
        out, _ = self.masking(out)
        out, _ = self.subsetting(out)
        out, _ = self.warping(out)
        return out

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._apply_one_view(x), self._apply_one_view(x)
