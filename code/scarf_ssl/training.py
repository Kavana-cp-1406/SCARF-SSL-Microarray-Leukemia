"""
Training routines: SSL contrastive pre-training (Section 2.3.3), two-stage
supervised fine-tuning (Section 2.4), and a from-scratch supervised MLP
baseline sharing the same architecture but no pre-training (Section 2.5).
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .augmentations import BaseAugmenter
from .losses import NTXentLoss
from .models import (
    ClassifierHead,
    GeneExpressionEncoder,
    LeukemiaClassifier,
    ProjectionHead,
    SSLPretrainModel,
)

logger = logging.getLogger("scarf_ssl")


def pretrain_ssl(
    X_unlabeled: np.ndarray,
    augmenter: BaseAugmenter,
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float = 0.5,
    hidden_dim_1: int = 512,
    hidden_dim_2: int = 256,
    dropout: float = 0.30,
    verbose: bool = True,
) -> GeneExpressionEncoder:
    """Phase 1 (Section 2.3.3): contrastive pre-training. Returns the
    trained encoder only — the projection head is discarded, matching the
    manuscript's stated design."""
    encoder = GeneExpressionEncoder(
        input_dim=X_unlabeled.shape[1],
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
    ).to(device)
    projection_head = ProjectionHead(input_dim=hidden_dim_2).to(device)
    ssl_model = SSLPretrainModel(encoder, projection_head).to(device)

    dataset = TensorDataset(torch.FloatTensor(X_unlabeled))
    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(X_unlabeled)), shuffle=True, drop_last=True
    )

    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=weight_decay)

    ssl_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_aug1, x_aug2 = augmenter(x_batch)
            z1, z2 = ssl_model(x_aug1), ssl_model(x_aug2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 5 == 0:
            avg = epoch_loss / max(len(loader), 1)
            logger.info("  [Pretrain] Epoch %d/%d - NT-Xent Loss: %.4f", epoch + 1, epochs, avg)

    return encoder  # projection head discarded here, per Section 2.3.1


def _run_epoch(
    model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def _accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X).to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()
    return float((preds == y).mean())


def two_stage_finetune(
    pretrained_encoder: GeneExpressionEncoder,
    X_subtrain: np.ndarray,
    y_subtrain: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    hidden_dim_2: int = 256,
    classifier_hidden_dim: int = 128,
    classifier_dropout: float = 0.30,
    linear_probe_epochs: int = 5,
    finetune_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    verbose: bool = True,
) -> LeukemiaClassifier:
    """Phase 2 (Section 2.4): (i) linear probing — encoder frozen, only the
    classification head trained; (ii) end-to-end fine-tuning with
    ReduceLROnPlateau (monitoring validation accuracy), retaining the
    checkpoint with the highest validation accuracy seen."""
    encoder_clone = copy.deepcopy(pretrained_encoder).to(device)
    classifier_head = ClassifierHead(
        input_dim=hidden_dim_2,
        hidden_dim=classifier_hidden_dim,
        num_classes=num_classes,
        dropout=classifier_dropout,
    ).to(device)
    model = LeukemiaClassifier(encoder_clone, classifier_head).to(device)

    train_dataset = TensorDataset(torch.FloatTensor(X_subtrain), torch.LongTensor(y_subtrain))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # --- Stage (i): linear probing, encoder frozen ---
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    for _ in range(linear_probe_epochs):
        _run_epoch(model, train_loader, criterion, optimizer, device)
    if verbose:
        logger.info("  [Finetune] Linear probing complete (%d epochs)", linear_probe_epochs)

    # --- Stage (ii): end-to-end fine-tuning ---
    for p in model.encoder.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=scheduler_patience, factor=scheduler_factor
    )

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(finetune_epochs):
        _run_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = _accuracy(model, X_val, y_val, device)
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if verbose:
        logger.info("  [Finetune] End-to-end complete. Best val accuracy: %.4f", best_val_acc)
    return model


def train_supervised_mlp(
    X_subtrain: np.ndarray,
    y_subtrain: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    input_dim: int,
    hidden_dim_1: int = 512,
    hidden_dim_2: int = 256,
    dropout: float = 0.30,
    classifier_hidden_dim: int = 128,
    classifier_dropout: float = 0.30,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    verbose: bool = True,
) -> LeukemiaClassifier:
    """Section 2.5 baseline: identical architecture to the SSL encoder,
    randomly initialised, trained end-to-end on labelled data without any
    contrastive pre-training. No linear-probe stage (there's no pretrained
    encoder to protect); trained for a total epoch budget matching SSL's
    combined 5+20=25 epochs for a fair comparison, with the best-val-
    accuracy checkpoint retained."""
    encoder = GeneExpressionEncoder(
        input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, dropout=dropout
    ).to(device)
    classifier_head = ClassifierHead(
        input_dim=hidden_dim_2,
        hidden_dim=classifier_hidden_dim,
        num_classes=num_classes,
        dropout=classifier_dropout,
    ).to(device)
    model = LeukemiaClassifier(encoder, classifier_head).to(device)

    train_dataset = TensorDataset(torch.FloatTensor(X_subtrain), torch.LongTensor(y_subtrain))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        _run_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = _accuracy(model, X_val, y_val, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if verbose:
        logger.info("  [Supervised MLP] Complete. Best val accuracy: %.4f", best_val_acc)
    return model


def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X).to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()
    return preds
