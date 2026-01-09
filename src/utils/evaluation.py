"""Evaluation metrics for patent classification."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """Evaluate classification model performance."""

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of class labels
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        if verbose:
            self.print_metrics(metrics)

        return metrics

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Pretty print evaluation metrics.

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*50)
        print("Evaluation Metrics")
        print("="*50)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Precision (wtd):   {metrics['precision_weighted']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"Recall (wtd):      {metrics['recall_weighted']:.4f}")
        print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
        print(f"F1 (weighted):     {metrics['f1_weighted']:.4f}")
        print("="*50 + "\n")

    def classification_report_detailed(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes

        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0
        )
        return report

    def confusion_matrix_display(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of class labels
            normalize: Whether to normalize the confusion matrix

        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return cm

    def top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        k: int = 3
    ) -> float:
        """
        Calculate top-k accuracy.

        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities (n_samples, n_classes)
            k: Number of top predictions to consider

        Returns:
            Top-k accuracy score
        """
        # Get indices of top k predictions for each sample
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]

        # Check if true label is in top k predictions
        correct = 0
        for i, true_label_idx in enumerate(y_true):
            if true_label_idx in top_k_preds[i]:
                correct += 1

        return correct / len(y_true)

    def get_misclassified_samples(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        indices: Optional[np.ndarray] = None,
        max_samples: int = 10
    ) -> pd.DataFrame:
        """
        Get misclassified samples for error analysis.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            indices: Original indices of samples
            max_samples: Maximum number of samples to return

        Returns:
            DataFrame with misclassified samples
        """
        if indices is None:
            indices = np.arange(len(y_true))

        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = indices[misclassified_mask]
        misclassified_true = y_true[misclassified_mask]
        misclassified_pred = y_pred[misclassified_mask]

        # Create DataFrame
        df = pd.DataFrame({
            'index': misclassified_indices[:max_samples],
            'true_label': misclassified_true[:max_samples],
            'predicted_label': misclassified_pred[:max_samples]
        })

        return df


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        verbose: Whether to print results

    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate(y_true, y_pred, verbose=verbose)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> None:
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
    """
    evaluator = ModelEvaluator()
    report = evaluator.classification_report_detailed(
        y_true,
        y_pred,
        target_names=target_names
    )
    print("\nDetailed Classification Report:")
    print("="*50)
    print(report)
