import torch


class MetricsTracker:
    '''Accumulates predictions across batches and computes classification metrics.

    Metrics are derived from an internal confusion matrix, so memory usage is
    independent of dataset size. Usage:

        tracker = MetricsTracker(topk=(1, 5))
        for x, y in loader:
            logits = model(x)
            tracker.update(logits, y)
        results = tracker.compute()

    Args:
        topk: Tuple of k values for top-k accuracy. Defaults to (1,).
    '''

    def __init__(self, topk: tuple[int, ...] = (1,)):
        self.topk          = tuple(sorted(set(topk)))
        self._cm           = None
        self._topk_correct = {k: 0 for k in self.topk}
        self._total        = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        '''Accumulate a batch of predictions.

        Args:
            logits:  Model outputs, shape [B, num_classes].
            targets: Ground-truth labels, shape [B].
        '''
        logits  = logits.detach()
        targets = targets.detach().cpu()
        num_classes = logits.size(1)

        if self._cm is None:
            self._cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

        preds = logits.argmax(dim=1).cpu()
        flat  = targets * num_classes + preds
        self._cm += torch.bincount(flat, minlength=num_classes ** 2).reshape(num_classes, num_classes)

        maxk = min(max(self.topk), num_classes)
        topk_preds = logits.topk(maxk, dim=1).indices.cpu()
        for k in self.topk:
            kk = min(k, num_classes)
            self._topk_correct[k] += (topk_preds[:, :kk] == targets.unsqueeze(1)).any(dim=1).sum().item()

        self._total += targets.size(0)

    def compute(self) -> dict:
        '''Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary with:
              - accuracy, top{k}_accuracy (for each k)
              - macro_precision, macro_recall, macro_f1
              - weighted_precision, weighted_recall, weighted_f1
              - per_class: {precision, recall, f1, support} as lists
              - confusion_matrix: rows = true class, cols = predicted class
        '''
        if self._cm is None or self._total == 0:
            raise RuntimeError("No predictions accumulated. Call update() first.")

        cm        = self._cm.float()
        tp        = cm.diag()
        support   = cm.sum(dim=1)               # true instances per class
        predicted = cm.sum(dim=0)               # predicted instances per class

        precision = tp / predicted.clamp(min=1)
        recall    = tp / support.clamp(min=1)
        f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

        weights = support / support.sum()

        results = {
            'accuracy':           (tp.sum() / self._total).item(),
            'macro_precision':    precision.mean().item(),
            'macro_recall':       recall.mean().item(),
            'macro_f1':           f1.mean().item(),
            'weighted_precision': (precision * weights).sum().item(),
            'weighted_recall':    (recall * weights).sum().item(),
            'weighted_f1':        (f1 * weights).sum().item(),
            'per_class': {
                'precision': precision.tolist(),
                'recall':    recall.tolist(),
                'f1':        f1.tolist(),
                'support':   support.long().tolist(),
            },
            'confusion_matrix': self._cm.tolist(),
        }

        for k in self.topk:
            results[f'top{k}_accuracy'] = self._topk_correct[k] / self._total

        return results

    def reset(self) -> None:
        '''Clear all accumulated state to reuse the tracker.'''
        self._cm           = None
        self._topk_correct = {k: 0 for k in self.topk}
        self._total        = 0


def format_report(metrics: dict, class_names: list[str] | None = None) -> str:
    '''Format a metrics dictionary as a human-readable evaluation report.

    Args:
        metrics:     Output of MetricsTracker.compute().
        class_names: Class names in label order. Defaults to class indices.
    '''
    per_class   = metrics['per_class']
    num_classes = len(per_class['support'])
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    name_width = max(len(n) for n in class_names + ['weighted avg'])

    lines = []
    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    for key in sorted(metrics):
        if key.startswith('top') and key.endswith('_accuracy'):
            lines.append(f"{key.replace('_', ' ').capitalize()}: {metrics[key]:.4f}")
    lines.append("")

    header = f"{'':<{name_width}}  {'precision':>9}  {'recall':>9}  {'f1':>9}  {'support':>8}"
    lines.append(header)
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<{name_width}}  {per_class['precision'][i]:>9.4f}  "
            f"{per_class['recall'][i]:>9.4f}  {per_class['f1'][i]:>9.4f}  "
            f"{per_class['support'][i]:>8}"
        )
    lines.append("")

    total = sum(per_class['support'])
    for avg in ('macro', 'weighted'):
        lines.append(
            f"{avg + ' avg':<{name_width}}  {metrics[f'{avg}_precision']:>9.4f}  "
            f"{metrics[f'{avg}_recall']:>9.4f}  {metrics[f'{avg}_f1']:>9.4f}  {total:>8}"
        )

    return "\n".join(lines)
