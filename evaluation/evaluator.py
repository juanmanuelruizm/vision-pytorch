import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import MetricsTracker, format_report


class Evaluator:
    '''Evaluation loop over a test set with full metrics reporting.

    Computes loss, accuracy, top-k accuracy, per-class precision/recall/F1
    (with macro and weighted averages) and the confusion matrix. The report
    is printed to console and the raw metrics can be saved as JSON.

    Args:
        model:   Trained model to evaluate.
        device:  Compute device ('cuda' or 'cpu').
        loss_fn: Loss function for the reported test loss.
                 Defaults to nn.CrossEntropyLoss.
        topk:    k values for top-k accuracy. Defaults to (1, 5).
    '''

    def __init__(
        self,
        model: nn.Module,
        device: str,
        loss_fn: nn.Module | None = None,
        topk: tuple[int, ...] = (1, 5),
    ):
        self.model   = model.to(device)
        self.device  = device
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.topk    = topk

    def evaluate(self, loader: DataLoader, class_names: list[str] | None = None) -> dict:
        '''Run evaluation over the loader and return the metrics dictionary.

        Args:
            loader:      DataLoader over the test set.
            class_names: Class names in label order, used in the printed report.
        '''
        self.model.eval()
        tracker    = MetricsTracker(topk=self.topk)
        total_loss = 0.0
        total      = 0

        with torch.no_grad():
            for x, y in loader:
                x, y   = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss   = self.loss_fn(output, y)

                tracker.update(output, y)
                total_loss += loss.item() * x.size(0)
                total      += x.size(0)

        metrics = tracker.compute()
        metrics['loss'] = total_loss / total
        if class_names is not None:
            metrics['class_names'] = class_names

        print(f"\n{'=' * 60}\nTest evaluation — {total} images\n{'=' * 60}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(format_report(metrics, class_names))

        return metrics

    @staticmethod
    def save_metrics(metrics: dict, path: str) -> None:
        '''Save the metrics dictionary as a JSON file.'''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {path}")
