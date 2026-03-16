from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    '''Training and validation loop with checkpoint saving.

    Args:
        model:           Model to train (instance of Model).
        optimizer:       PyTorch optimizer (e.g. Adam, SGD).
        loss_fn:         Loss function (e.g. nn.CrossEntropyLoss).
        device:          Compute device ('cuda' or 'cpu').
        checkpoints_dir: Directory where checkpoints will be saved. Defaults to 'checkpoints'.
    '''

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
        checkpoints_dir: str = 'checkpoints',
    ):
        self.model           = model.to(device)
        self.optimizer       = optimizer
        self.loss_fn         = loss_fn
        self.device          = device
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_acc = 0.0

    # ------------------------------------------------------------------
    # Training and validation steps
    # ------------------------------------------------------------------

    def _training_step(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss   = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct    += (output.argmax(dim=1) == y).sum().item()
            total      += x.size(0)

        return total_loss / total, correct / total

    def _validation_step(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for x, y in loader:
                x, y   = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss   = self.loss_fn(output, y)

                total_loss += loss.item() * x.size(0)
                correct    += (output.argmax(dim=1) == y).sum().item()
                total      += x.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> None:
        '''Run the full training loop.

        Saves a checkpoint at the end of each epoch and keeps the best
        model based on validation accuracy.

        Args:
            train_loader: DataLoader for the training set.
            val_loader:   DataLoader for the validation set.
            epochs:       Number of epochs to train.
        '''
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._training_step(train_loader)
            val_loss,   val_acc   = self._validation_step(val_loader)

            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
                f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}"
            )

            self._save_checkpoint(epoch, val_acc)

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        '''Save a checkpoint for the current epoch and update best model if improved.'''
        state = {
            'epoch':           epoch,
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_acc':         val_acc,
        }

        path = self.checkpoints_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(state, path)

        if val_acc > self._best_val_acc:
            self._best_val_acc = val_acc
            torch.save(state, self.checkpoints_dir / 'best_model.pth')
            print(f"  → New best model saved (val acc: {val_acc:.4f})")

    def load_checkpoint(self, path: str) -> int:
        '''Load a checkpoint and restore model and optimizer state.

        Args:
            path: Path to the checkpoint .pth file.

        Returns:
            Epoch at which the checkpoint was saved.
        '''
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self._best_val_acc = checkpoint.get('val_acc', 0.0)

        print(f"Checkpoint loaded: epoch {checkpoint['epoch']}, val acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
        return checkpoint['epoch']
