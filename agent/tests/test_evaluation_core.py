"""Unit tests for huginn/evaluation/core.py."""

from __future__ import annotations

import numpy as np
import pytest

from huginn.evaluation.core import (
    compute_weights,
    evaluate,
    method_grey,
    method_promethee,
    method_rsr,
    method_todim,
    method_topsis,
    method_vikor,
    normalize_matrix,
    sensitivity_random_weights,
    vector_normalize,
    weight_ahp,
    weight_critic,
    weight_cv,
    weight_entropy,
    weight_pca,
)


class TestNormalizeMatrix:
    def test_default_max_direction(self):
        X = np.array([[1, 2], [3, 4]], dtype=float)
        Xn = normalize_matrix(X)
        np.testing.assert_array_almost_equal(
            Xn, np.array([[0.0, 0.0], [1.0, 1.0]])
        )

    def test_min_direction(self):
        X = np.array([[1, 10], [3, 5]], dtype=float)
        Xn = normalize_matrix(X, directions=["max", "min"])
        np.testing.assert_array_almost_equal(
            Xn, np.array([[0.0, 0.0], [1.0, 1.0]])
        )

    def test_constant_column(self):
        X = np.array([[5, 1], [5, 2]], dtype=float)
        Xn = normalize_matrix(X)
        np.testing.assert_array_almost_equal(Xn[:, 0], np.array([1.0, 1.0]))
        np.testing.assert_array_almost_equal(Xn[:, 1], np.array([0.0, 1.0]))


class TestVectorNormalize:
    def test_basic(self):
        X = np.array([[3, 4], [0, 4]], dtype=float)
        Xn = vector_normalize(X)
        np.testing.assert_array_almost_equal(
            Xn, np.array([[1.0, 1.0 / 2**0.5], [0.0, 1.0 / 2**0.5]])
        )

    def test_zero_norm(self):
        X = np.array([[0, 1], [0, 2]], dtype=float)
        Xn = vector_normalize(X)
        np.testing.assert_array_almost_equal(Xn[:, 0], np.array([0.0, 0.0]))


class TestWeightMethods:
    def test_weight_entropy(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        w = weight_entropy(X)
        assert w.shape == (2,)
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_weight_cv(self):
        X = np.array([[1, 100], [2, 101], [3, 100]], dtype=float)
        w = weight_cv(X)
        assert w.shape == (2,)
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_weight_critic(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        w = weight_critic(X)
        assert w.shape == (3,)
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_weight_ahp(self):
        pairwise = np.array([[1, 3, 0.5], [1 / 3, 1, 0.2], [2, 5, 1]])
        w = weight_ahp(pairwise)
        assert w.shape == (3,)
        np.testing.assert_almost_equal(w.sum(), 1.0)
        assert all(w >= 0)

    def test_weight_pca(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        w = weight_pca(X)
        assert w.shape == (3,)
        np.testing.assert_almost_equal(w.sum(), 1.0)


class TestComputeWeights:
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    @pytest.mark.parametrize(
        "method",
        ["entropy", "cv", "critic", "pca", "equal"],
    )
    def test_compute_weights_methods(self, method):
        w = compute_weights(self.X, method)
        assert w.shape == (2,)
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_compute_weights_ahp(self):
        pairwise = np.array([[1, 3], [1 / 3, 1]])
        w = compute_weights(self.X, "ahp", ahp_matrix=pairwise)
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_compute_weights_ahp_missing_matrix(self):
        with pytest.raises(ValueError, match="ahp_matrix required"):
            compute_weights(self.X, "ahp")

    def test_compute_weights_unknown(self):
        with pytest.raises(ValueError, match="Unknown weight method"):
            compute_weights(self.X, "magic")


class TestMCDAMethods:
    X = np.array(
        [[3, 1, 4], [2, 5, 1], [4, 2, 5], [1, 4, 2]], dtype=float
    )
    weights = np.array([0.4, 0.35, 0.25])
    alternatives = ["A", "B", "C", "D"]

    def test_method_topsis(self):
        scores = method_topsis(self.X, self.weights)
        assert scores.shape == (4,)
        assert np.all(scores >= 0)

    def test_method_vikor(self):
        scores = method_vikor(self.X, self.weights)
        assert scores.shape == (4,)

    def test_method_todim(self):
        scores = method_todim(self.X, self.weights)
        assert scores.shape == (4,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_method_promethee(self):
        scores = method_promethee(self.X, self.weights)
        assert scores.shape == (4,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_method_rsr(self):
        scores = method_rsr(self.X, self.weights)
        assert scores.shape == (4,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_method_grey(self):
        scores = method_grey(self.X, self.weights)
        assert scores.shape == (4,)
        assert np.all(scores >= 0)


class TestEvaluate:
    alternatives = ["A", "B", "C"]
    criteria = ["c1", "c2"]
    matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    @pytest.mark.parametrize(
        "weight_method, eval_method",
        [
            ("equal", "topsis"),
            ("entropy", "vikor"),
            ("cv", "todim"),
            ("critic", "promethee"),
            ("pca", "rsr"),
            ("equal", "grey"),
        ],
    )
    def test_evaluate_combinations(self, weight_method, eval_method):
        result = evaluate(
            self.alternatives,
            self.criteria,
            self.matrix,
            weight_method=weight_method,
            eval_method=eval_method,
        )
        assert result.method
        assert len(result.ranking) == 3
        assert set(result.scores.keys()) == set(self.alternatives)
        assert set(result.weights.keys()) == set(self.criteria)

    def test_evaluate_with_ahp(self):
        pairwise = np.array([[1, 3], [1 / 3, 1]])
        result = evaluate(
            self.alternatives,
            self.criteria,
            self.matrix,
            weight_method="ahp",
            ahp_matrix=pairwise,
        )
        assert len(result.ranking) == 3

    def test_evaluate_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Matrix dimensions"):
            evaluate(
                ["A", "B"],
                self.criteria,
                self.matrix,
            )

    def test_evaluate_eval_kwargs(self):
        result = evaluate(
            self.alternatives,
            self.criteria,
            self.matrix,
            eval_method="vikor",
            eval_kwargs={"v": 0.4},
        )
        assert len(result.ranking) == 3


class TestSensitivity:
    def test_sensitivity_random_weights(self):
        alternatives = ["A", "B", "C"]
        criteria = ["c1", "c2"]
        matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        result = sensitivity_random_weights(
            alternatives,
            criteria,
            matrix,
            eval_method="topsis",
            n_trials=50,
            perturbation=0.3,
        )
        assert "original_ranking" in result
        assert "stability_score" in result
        assert result["n_trials"] == 50
        assert set(result["top_frequency"].keys()) == set(alternatives)
