"""Tests for the evaluation module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluate import compute_weighted_score


class TestWeightedScore:
    """Tests for the weighted score computation."""

    def test_basic_weighted_score(self):
        """Weighted score should combine metrics correctly."""
        metrics = {"accuracy": 0.9, "f1_score": 0.85, "roc_auc": 0.92}
        weights = {"accuracy": 0.3, "f1_score": 0.4, "roc_auc": 0.3}
        score = compute_weighted_score(metrics, weights)
        expected = 0.9 * 0.3 + 0.85 * 0.4 + 0.92 * 0.3
        assert abs(score - expected) < 1e-6

    def test_missing_metric_ignored(self):
        """Missing metrics should be skipped (contribute 0)."""
        metrics = {"accuracy": 0.9, "f1_score": 0.85}
        weights = {"accuracy": 0.3, "f1_score": 0.4, "roc_auc": 0.3}
        score = compute_weighted_score(metrics, weights)
        expected = 0.9 * 0.3 + 0.85 * 0.4
        assert abs(score - expected) < 1e-6

    def test_zero_weights(self):
        """Zero weights should produce zero score."""
        metrics = {"accuracy": 0.9, "f1_score": 0.85}
        weights = {"accuracy": 0.0, "f1_score": 0.0}
        score = compute_weighted_score(metrics, weights)
        assert score == 0.0

    def test_perfect_scores(self):
        """Perfect metrics with equal weights should give 1.0."""
        metrics = {"accuracy": 1.0, "f1_score": 1.0, "roc_auc": 1.0}
        weights = {"accuracy": 0.33, "f1_score": 0.34, "roc_auc": 0.33}
        score = compute_weighted_score(metrics, weights)
        assert abs(score - 1.0) < 0.01
