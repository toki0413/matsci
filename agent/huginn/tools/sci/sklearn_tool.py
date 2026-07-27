"""General-purpose scikit-learn wrapper for classify / regress / cluster / evaluate.

Why this exists: the sci/ tools were getting filtered out by _step2_filter, so
bench/DL tasks that just needed a quick RandomForest or KMeans had no path
through the agent. This wraps the standard sklearn fit/predict/score interface
so a caller can hand it CSV/JSON + a target column + a model name and get back
holdout metrics + feature importances in one shot.

No hyperparameter search (GridSearchCV / RandomizedSearchCV) — too heavy for the
autoloop budget, and the agent can pass model_params itself once it sees the
baseline metrics. ponytail: ceiling = no auto-tuning; upgrade path = add a
"tune" action that calls RandomizedSearchCV with an explicit n_iter cap.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


# Flat model registries — caller passes "random_forest" and we resolve to the
# sklearn class without a chain of if/elif. Keys are lowercased so the LLM's
# "RandomForest" / "random forest" / "random_forest" all hit the same entry.
_CLASSIFIERS = {
    "logistic_regression": "LogisticRegression",
    "random_forest": "RandomForestClassifier",
    "gradient_boosting": "GradientBoostingClassifier",
    "svm": "SVC",
}
_REGRESSORS = {
    "linear_regression": "LinearRegression",
    "random_forest": "RandomForestRegressor",
    "gradient_boosting": "GradientBoostingRegressor",
    "ridge": "Ridge",
}
_CLUSTERERS = {
    "kmeans": "KMeans",
    "dbscan": "DBSCAN",
    "agglomerative": "AgglomerativeClustering",
}

# Estimators that accept random_state; we seed them by default for reproducible
# autoloop runs. Deterministic estimators (LinearRegression, Ridge, SVC with the
# default lbfgs-ish solvers, DBSCAN, AgglomerativeClustering) are left alone.
_STOCHASTIC = {
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "KMeans",
}


class SklearnInput(BaseModel):
    action: Literal["classify", "regress", "cluster", "evaluate"] = Field(
        ...,
        description=(
            "classify: supervised classification; "
            "regress: supervised regression; "
            "cluster: unsupervised clustering; "
            "evaluate: cross_val_score / classification_report / confusion_matrix"
        ),
    )
    model_type: str = Field(
        default="random_forest",
        description=(
            "Model key. classify: logistic_regression|random_forest|gradient_boosting|svm. "
            "regress: linear_regression|random_forest|gradient_boosting|ridge. "
            "cluster: kmeans|dbscan|agglomerative. evaluate: any supervised key."
        ),
    )
    data_json: dict[str, list[Any]] | None = Field(
        default=None,
        description="Inline data {column: [values]}; target_column picks y.",
    )
    data_file: str | None = Field(
        default=None, description="CSV path with a header row."
    )
    target_column: str = Field(
        default="",
        description="Name of the y column (required for classify/regress/evaluate).",
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="Feature names; auto = all columns except target.",
    )
    model_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs passed straight to the sklearn estimator constructor.",
    )
    # evaluate-only
    eval_metric: Literal[
        "cross_val_score", "classification_report", "confusion_matrix"
    ] = Field(default="cross_val_score")
    cv: int = Field(default=5, ge=2, le=20, description="CV folds for cross_val_score.")
    scoring: str = Field(
        default="",
        description="sklearn scoring string; default depends on the inferred task.",
    )
    # cluster-only
    n_clusters: int = Field(
        default=3, ge=2, description="k for KMeans / AgglomerativeClustering."
    )
    random_state: int = Field(default=0)
    test_size: float = Field(
        default=0.25, ge=0.05, le=0.5,
        description="Holdout fraction for classify/regress/evaluate metrics.",
    )


# ── data loading ────────────────────────────────────────────────

def _coerce(v: str) -> Any:
    """Try float, fall back to the raw string — LabelEncoder handles the rest."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _load_xy(args: SklearnInput) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names). y is 1-D. Raises on bad input."""
    if args.data_json is not None:
        d = args.data_json
        if not args.target_column:
            raise ValueError("target_column is required with data_json")
        if args.target_column not in d:
            raise ValueError(
                f"target column '{args.target_column}' not in data_json keys {list(d)}"
            )
        y = np.asarray(d[args.target_column])
        feats = args.feature_columns or [c for c in d if c != args.target_column]
        if not feats:
            raise ValueError("no feature columns (only target present)")
        X = np.column_stack([np.asarray(d[c], dtype=float) for c in feats])
        return X, y, list(feats)

    if not args.data_file:
        raise ValueError("either data_json or data_file must be provided")
    path = Path(args.data_file)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    fields = reader.fieldnames or []
    tkey = args.target_column or fields[-1]
    if tkey not in fields:
        raise ValueError(f"target column '{tkey}' not in CSV header {fields}")
    feats = args.feature_columns or [c for c in fields if c != tkey]
    if not feats:
        raise ValueError("no feature columns (only target present)")
    y = np.array([_coerce(r[tkey]) for r in rows])
    X = np.array([[float(r[c]) for c in feats] for r in rows], dtype=float)
    return X, y, list(feats)


def _load_x_only(args: SklearnInput) -> tuple[np.ndarray, list[str]]:
    """Load features with no target — for cluster action."""
    if args.data_json is not None:
        d = args.data_json
        feats = args.feature_columns or list(d.keys())
        if not feats:
            raise ValueError("no feature columns in data_json")
        X = np.column_stack([np.asarray(d[c], dtype=float) for c in feats])
        return X, list(feats)
    if not args.data_file:
        raise ValueError("cluster requires data_json or data_file")
    path = Path(args.data_file)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fields = reader.fieldnames or []
    feats = args.feature_columns or fields
    if not feats:
        raise ValueError("no feature columns in CSV")
    X = np.array([[float(r[c]) for c in feats] for r in rows], dtype=float)
    return X, list(feats)


# ── estimator resolution ───────────────────────────────────────

def _resolve_estimator(action: str, model_type: str, args: SklearnInput):
    """Lazily import the sklearn class and instantiate with model_params."""
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.svm import SVC

    pool = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "SVC": SVC,
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "Ridge": Ridge,
        "KMeans": KMeans,
        "DBSCAN": DBSCAN,
        "AgglomerativeClustering": AgglomerativeClustering,
    }

    table = {
        "classify": _CLASSIFIERS,
        "regress": _REGRESSORS,
        "cluster": _CLUSTERERS,
    }[action]

    key = model_type.lower().replace(" ", "_").replace("-", "_")
    if key not in table:
        raise ValueError(
            f"unknown model_type '{model_type}' for action '{action}'. "
            f"valid: {sorted(table)}"
        )
    cls_name = table[key]
    cls = pool[cls_name]

    params = dict(args.model_params)
    if cls_name in _STOCHASTIC and "random_state" not in params:
        params["random_state"] = args.random_state
    if cls_name in {"KMeans", "AgglomerativeClustering"} and "n_clusters" not in params:
        params["n_clusters"] = args.n_clusters
    return cls(**params)


def _feature_importances(estimator, feature_names: list[str]) -> dict[str, float] | None:
    """Pull feature_importances_ or coef_ out of a fitted estimator, if present."""
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        return {n: float(v) for n, v in zip(feature_names, np.asarray(importances).ravel())}
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        arr = np.asarray(coef).ravel()
        if arr.shape == (len(feature_names),):
            return {n: float(v) for n, v in zip(feature_names, arr)}
    return None


class SklearnTool(HuginnTool):
    """sklearn wrapper — quick classify / regress / cluster / evaluate on tabular data."""

    name = "sklearn_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "General-purpose scikit-learn wrapper. Fits classify/regress/cluster models "
        "on CSV or inline JSON data and returns holdout metrics + feature importances; "
        "evaluate action runs cross_val_score / classification_report / confusion_matrix. "
        "No GridSearchCV — pass model_params to tune."
    )
    input_schema = SklearnInput

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            return ToolResult(
                data=None, success=False,
                error="scikit-learn is not installed in this environment",
            )

        try:
            input_data = SklearnInput(**args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"bad input: {e}")

        try:
            if input_data.action == "classify":
                return self._supervised(input_data, "classify")
            if input_data.action == "regress":
                return self._supervised(input_data, "regress")
            if input_data.action == "cluster":
                return self._cluster(input_data)
            if input_data.action == "evaluate":
                return self._evaluate(input_data)
            return ToolResult(
                data=None, success=False,
                error=f"unknown action: {input_data.action}",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"sklearn_tool failed: {e}")

    # ── supervised: classify / regress share the same fit+holdout path ──

    def _supervised(
        self, args: SklearnInput, task: Literal["classify", "regress"]
    ) -> ToolResult:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        X, y, feats = _load_xy(args)
        # String targets → label-encode so sklearn estimators accept them.
        if y.dtype.kind in {"U", "S", "O"}:
            y = LabelEncoder().fit_transform(y)

        stratify = y if task == "classify" and len(set(y)) > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state,
            stratify=stratify,
        )

        est = _resolve_estimator(task, args.model_type, args)
        est.fit(X_tr, y_tr)
        preds = est.predict(X_te)

        data: dict[str, Any] = {
            "action": task,
            "model_type": args.model_type,
            "model_class": type(est).__name__,
            "model_params": est.get_params(),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": feats,
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "predictions_sample": np.asarray(preds[:10]).tolist(),
        }

        if task == "classify":
            data["metrics"] = {
                "accuracy": float(accuracy_score(y_te, preds)),
                "f1_macro": float(f1_score(y_te, preds, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_te, preds, average="weighted", zero_division=0)),
            }
        else:
            mse = float(mean_squared_error(y_te, preds))
            data["metrics"] = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_te, preds)),
                "r2": float(r2_score(y_te, preds)),
            }

        imp = _feature_importances(est, feats)
        if imp is not None:
            data["feature_importances"] = imp

        data["message"] = f"{type(est).__name__} fit on {X.shape[0]} samples ({task})."
        return ToolResult(data=data, success=True)

    # ── unsupervised: cluster ────────────────────────────────────

    def _cluster(self, args: SklearnInput) -> ToolResult:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        X, feats = _load_x_only(args)
        # Standardize so distance-based thresholds (DBSCAN eps, KMeans scale) behave.
        X = StandardScaler().fit_transform(X)

        est = _resolve_estimator("cluster", args.model_type, args)
        labels = est.fit_predict(X)
        labels = np.asarray(labels)

        # KMeans / AgglomerativeClustering expose n_clusters_; DBSCAN doesn't.
        n_found = (
            int(getattr(est, "n_clusters_", len(set(labels)) - (1 if -1 in labels else 0)))
        )
        n_noise = int(np.sum(labels == -1))

        sil = float("nan")
        # silhouette needs >=2 clusters and at least as many points as clusters.
        unique = set(labels.tolist()) - {-1}
        if len(unique) > 1 and len(labels) > len(unique):
            try:
                sil = float(silhouette_score(X, labels))
            except Exception:
                sil = float("nan")

        return ToolResult(
            data={
                "action": "cluster",
                "model_type": args.model_type,
                "model_class": type(est).__name__,
                "model_params": est.get_params(),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "feature_names": list(feats),
                "n_clusters_found": n_found,
                "n_noise": n_noise,
                "labels_sample": labels[:10].tolist(),
                "silhouette": sil,
                "message": f"{type(est).__name__} found {n_found} clusters on {X.shape[0]} samples.",
            },
            success=True,
        )

    # ── evaluate ─────────────────────────────────────────────────

    def _evaluate(self, args: SklearnInput) -> ToolResult:
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import LabelEncoder

        X, y, feats = _load_xy(args)
        if y.dtype.kind in {"U", "S", "O"}:
            y = LabelEncoder().fit_transform(y)

        # Infer task: small integer cardinality → classify, else regress.
        # ponytail: simple heuristic, ceiling = mislabels ordinal regression as
        # classify; upgrade = let caller pass task explicitly via a new field.
        inferred_task = (
            "classify" if len(set(y.tolist())) <= 20 and y.dtype.kind in {"i", "u"} else "regress"
        )
        est = _resolve_estimator(inferred_task, args.model_type, args)

        if args.eval_metric == "cross_val_score":
            scoring = args.scoring or (None if inferred_task == "regress" else "f1_macro")
            scores = cross_val_score(est, X, y, cv=args.cv, scoring=scoring)
            label = scoring or ("r2" if inferred_task == "regress" else "f1_macro")
            return ToolResult(
                data={
                    "action": "evaluate",
                    "eval_metric": "cross_val_score",
                    "model_class": type(est).__name__,
                    "task": inferred_task,
                    "scoring": label,
                    "cv": args.cv,
                    "n_samples": int(X.shape[0]),
                    "scores": scores.tolist(),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "message": (
                        f"{args.cv}-fold CV ({label}): "
                        f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
                    ),
                },
                success=True,
            )

        # classification_report / confusion_matrix need a fitted model on a holdout.
        stratify = y if len(set(y.tolist())) > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state,
            stratify=stratify,
        )
        est.fit(X_tr, y_tr)
        preds = est.predict(X_te)

        if args.eval_metric == "classification_report":
            report = classification_report(y_te, preds, output_dict=True, zero_division=0)
            return ToolResult(
                data={
                    "action": "evaluate",
                    "eval_metric": "classification_report",
                    "model_class": type(est).__name__,
                    "n_test": int(X_te.shape[0]),
                    "report": report,
                    "message": "classification_report generated on holdout split.",
                },
                success=True,
            )

        # confusion_matrix
        cm = confusion_matrix(y_te, preds).tolist()
        return ToolResult(
            data={
                "action": "evaluate",
                "eval_metric": "confusion_matrix",
                "model_class": type(est).__name__,
                "n_test": int(X_te.shape[0]),
                "confusion_matrix": cm,
                "message": "confusion_matrix generated on holdout split.",
            },
            success=True,
        )


# ── self-check ─────────────────────────────────────────────────
# Smallest thing that fails if fit/predict/score breaks. Run directly:
#   python -m huginn.tools.sci.sklearn_tool

def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 60
    # Two well-separated blobs for classification.
    X_clf = np.vstack([
        rng.normal(loc=[-2, -2], size=(n // 2, 2)),
        rng.normal(loc=[ 2,  2], size=(n // 2, 2)),
    ])
    y_clf = np.array([0] * (n // 2) + [1] * (n // 2))
    # Linear regression: y = 3*x0 - 2*x1 + noise.
    X_reg = rng.normal(size=(n, 2))
    y_reg = 3.0 * X_reg[:, 0] - 2.0 * X_reg[:, 1] + 0.1 * rng.normal(size=n)

    inline_clf = {
        "x0": X_clf[:, 0].tolist(), "x1": X_clf[:, 1].tolist(), "y": y_clf.tolist(),
    }
    inline_reg = {
        "x0": X_reg[:, 0].tolist(), "x1": X_reg[:, 1].tolist(), "y": y_reg.tolist(),
    }

    tool = SklearnTool()

    print("== classify (random_forest) ==")
    r = tool.call({
        "action": "classify", "model_type": "random_forest",
        "data_json": inline_clf, "target_column": "y",
    })
    assert r.success, r.error
    acc = r.data["metrics"]["accuracy"]
    print(f"  acc={acc:.3f}  f1_macro={r.data['metrics']['f1_macro']:.3f}")
    assert acc > 0.85, f"acc too low: {acc}"
    assert "feature_importances" in r.data

    print("== regress (gradient_boosting) ==")
    r = tool.call({
        "action": "regress", "model_type": "gradient_boosting",
        "data_json": inline_reg, "target_column": "y",
    })
    assert r.success, r.error
    r2 = r.data["metrics"]["r2"]
    print(f"  r2={r2:.3f}  rmse={r.data['metrics']['rmse']:.3f}")
    assert r2 > 0.7, f"r2 too low: {r2}"
    assert "feature_importances" in r.data

    print("== cluster (kmeans) ==")
    r = tool.call({
        "action": "cluster", "model_type": "kmeans",
        "data_json": inline_clf, "n_clusters": 2,
    })
    assert r.success, r.error
    print(f"  n_clusters_found={r.data['n_clusters_found']}  silhouette={r.data['silhouette']:.3f}")
    assert r.data["n_clusters_found"] == 2

    print("== evaluate (cross_val_score) ==")
    r = tool.call({
        "action": "evaluate", "data_json": inline_clf, "target_column": "y",
        "eval_metric": "cross_val_score", "cv": 3,
    })
    assert r.success, r.error
    print(f"  task={r.data['task']}  mean={r.data['mean']:.3f}  std={r.data['std']:.3f}")
    assert r.data["mean"] > 0.7

    print("== evaluate (classification_report) ==")
    r = tool.call({
        "action": "evaluate", "data_json": inline_clf, "target_column": "y",
        "eval_metric": "classification_report",
    })
    assert r.success, r.error
    print(f"  report keys={list(r.data['report'].keys())}")

    print("== evaluate (confusion_matrix) ==")
    r = tool.call({
        "action": "evaluate", "data_json": inline_clf, "target_column": "y",
        "eval_metric": "confusion_matrix",
    })
    assert r.success, r.error
    cm = r.data["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2
    print(f"  cm={cm}")

    print("\nALL OK")


if __name__ == "__main__":
    _self_check()
