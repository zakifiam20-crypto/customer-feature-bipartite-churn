"""Reviewer-comment-5 experiments for the telecom churn study.

The script compares the same 55-column quartile-indicator matrix, its
32-dimensional truncated-SVD representation, graph embeddings, and feature
fusion. Model selection and threshold selection use training folds only.
Graph representations are transductive and use predictors, never labels, for
all customers. Outputs are saved separately from the original analysis.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Telco_customer_churn.xlsx"
OOF_PATH = ROOT / "final_node2vec_metapath2vec_oof_predictions.csv"
OUTPUT_STEM = "comment5"

SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
INDICATOR_DIM = 55
SVD_DIM = 32
EMBEDDING_DIM = 32
WALKS_PER_CUSTOMER = 10
WALK_LENGTH = 80
CONTEXT_WINDOW = 4
NEGATIVE_SAMPLES = 3
SGNS_EPOCHS = 8
SGNS_BATCH_SIZE = 2048
MAX_POSITIVE_PAIRS = 1_500_000
EMBEDDING_SEEDS = [42, 43, 44, 45, 46]

ID_COLUMN = "CustomerID"
TARGET_COLUMN = "Churn Label"
NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
METHODS = ["Node2Vec", "MetaPath2Vec"]


def load_data():
    if not DATA_PATH.is_file() or not OOF_PATH.is_file():
        raise FileNotFoundError("Dataset atau OOF proyek tidak ditemukan.")
    raw = pd.read_excel(DATA_PATH)
    needed = {ID_COLUMN, TARGET_COLUMN, *FEATURES}
    missing = needed.difference(raw.columns)
    if missing:
        raise KeyError(f"Kolom dataset tidak tersedia: {sorted(missing)}")
    data = raw.drop_duplicates().copy()
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
    missing_charge = data["Total Charges"].isna()
    if (missing_charge & ~data["Tenure Months"].eq(0)).any():
        raise ValueError("Total Charges kosong pada pelanggan dengan tenure bukan nol.")
    data.loc[missing_charge, "Total Charges"] = 0.0
    labels = data[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if labels.isna().any():
        raise ValueError("Label churn memiliki nilai di luar No/Yes.")
    data[ID_COLUMN] = data[ID_COLUMN].astype(str)
    labels = labels.astype(np.int64).to_numpy()

    reference = pd.read_csv(OOF_PATH)
    ids = reference[ID_COLUMN].astype(str).reset_index(drop=True)
    expected_labels = reference["Actual"].to_numpy(dtype=np.int64)
    train_rows = data.set_index(ID_COLUMN).reindex(ids)
    if train_rows[FEATURES].isna().any().any():
        raise ValueError("ID training pada OOF tidak cocok dengan dataset.")
    aligned_y = train_rows[TARGET_COLUMN].map({"No": 0, "Yes": 1}).to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(aligned_y, expected_labels):
        raise ValueError("Label training tidak sejajar dengan OOF sebelumnya.")
    return data.reset_index(drop=True), ids, expected_labels


def build_indicator_matrix(data: pd.DataFrame):
    features = pd.DataFrame(index=data.index)
    for column in CATEGORICAL_FEATURES:
        features[column] = (
            data[column].astype("string").fillna("<MISSING>").astype(str)
        )
    for column in NUMERIC_FEATURES:
        codes, _ = pd.qcut(
            data[column], q=4, labels=False, retbins=True, duplicates="drop"
        )
        if codes.isna().any():
            raise ValueError(f"Gagal melakukan binning kuartil: {column}")
        features[column] = codes.astype(int).map(lambda value: f"Q{value + 1}")

    blocks = []
    names = []
    for column in FEATURES:
        values = features[column]
        categories = sorted(values.astype(str).unique())
        block = np.column_stack(
            [(values.astype(str).to_numpy() == category) for category in categories]
        ).astype(np.float32)
        blocks.append(block)
        names.extend(f"{column}={category}" for category in categories)
    matrix = np.concatenate(blocks, axis=1)
    if matrix.shape[1] != INDICATOR_DIM:
        raise ValueError(
            f"Indikator semestinya {INDICATOR_DIM} kolom, ditemukan {matrix.shape[1]}."
        )
    if not np.all(matrix.sum(axis=1) == len(FEATURES)):
        raise ValueError("Setiap pelanggan harus memiliki tepat 19 indikator aktif.")
    return matrix, names


def make_adjacency(indicators: np.ndarray):
    customer_count, feature_count = indicators.shape
    adjacency = [[] for _ in range(customer_count + feature_count)]
    for customer_index, row in enumerate(indicators):
        for feature_index in np.flatnonzero(row):
            feature_node = customer_count + int(feature_index)
            adjacency[customer_index].append(feature_node)
            adjacency[feature_node].append(customer_index)
    return [np.asarray(items, dtype=np.int32) for items in adjacency]


def random_walk(start: int, adjacency, rng, method: str, length: int, p=1.0, q=2.0):
    walk = [start]
    while len(walk) < length:
        current = walk[-1]
        neighbors = adjacency[current]
        if len(neighbors) == 0:
            break
        if method == "Node2Vec" and len(walk) > 1:
            previous = walk[-2]
            weights = np.full(len(neighbors), 1.0 / q, dtype=np.float64)
            weights[neighbors == previous] = 1.0 / p
            weights /= weights.sum()
            next_node = int(rng.choice(neighbors, p=weights))
        else:
            next_node = int(rng.choice(neighbors))
        walk.append(next_node)
    return np.asarray(walk, dtype=np.int32)


def sample_context_pairs(walks, window, max_pairs, seed):
    rng = np.random.default_rng(seed)
    def pair_count(length):
        return sum(
            min(length, position + window + 1)
            - max(0, position - window)
            - 1
            for position in range(length)
        )

    total = sum(pair_count(len(walk)) for walk in walks)
    if total > max_pairs:
        budget_per_walk = max_pairs // len(walks)
        extra = max_pairs % len(walks)
    else:
        budget_per_walk = max_pairs
        extra = 0

    sampled_centers = []
    sampled_contexts = []
    for index, walk in enumerate(walks):
        local_centers = []
        local_contexts = []
        for position, center in enumerate(walk):
            left = max(0, position - window)
            right = min(len(walk), position + window + 1)
            for context_position in range(left, right):
                if context_position != position:
                    local_centers.append(center)
                    local_contexts.append(walk[context_position])
        local_centers = np.asarray(local_centers, dtype=np.int32)
        local_contexts = np.asarray(local_contexts, dtype=np.int32)
        budget = min(
            len(local_centers),
            budget_per_walk + (index < extra if total > max_pairs else 0),
        )
        if budget < len(local_centers):
            selected = rng.choice(len(local_centers), size=budget, replace=False)
            local_centers = local_centers[selected]
            local_contexts = local_contexts[selected]
        sampled_centers.append(local_centers)
        sampled_contexts.append(local_contexts)
    return np.concatenate(sampled_centers), np.concatenate(sampled_contexts)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -15.0, 15.0)))


def train_embedding(
    centers,
    contexts,
    adjacency,
    customer_count,
    dimensions,
    method,
    seed,
):
    rng = np.random.default_rng(seed)
    node_count = len(adjacency)
    input_vectors = rng.uniform(
        -0.5 / dimensions, 0.5 / dimensions, size=(node_count, dimensions)
    ).astype(np.float32)
    output_vectors = np.zeros_like(input_vectors)
    degrees = np.asarray([max(len(neighbors), 1) for neighbors in adjacency])
    weights = degrees.astype(np.float64) ** 0.75
    global_probabilities = weights / weights.sum()
    customer_nodes = np.arange(customer_count, dtype=np.int32)
    feature_nodes = np.arange(customer_count, node_count, dtype=np.int32)
    typed = {
        "customer": (customer_nodes, weights[customer_nodes] / weights[customer_nodes].sum()),
        "feature": (feature_nodes, weights[feature_nodes] / weights[feature_nodes].sum()),
    }

    order = np.arange(len(centers))
    for epoch in range(SGNS_EPOCHS):
        rng.shuffle(order)
        for start in range(0, len(order), SGNS_BATCH_SIZE):
            rows = order[start : start + SGNS_BATCH_SIZE]
            batch_centers = centers[rows]
            batch_contexts = contexts[rows]
            count = len(rows)
            negatives = np.empty((count, NEGATIVE_SAMPLES), dtype=np.int32)
            if method == "MetaPath2Vec":
                context_is_customer = batch_contexts < customer_count
                for is_customer, (vocabulary, probabilities) in typed.items():
                    mask = context_is_customer if is_customer == "customer" else ~context_is_customer
                    positions = np.flatnonzero(mask)
                    if len(positions):
                        negatives[positions] = rng.choice(
                            vocabulary,
                            size=(len(positions), NEGATIVE_SAMPLES),
                            replace=True,
                            p=probabilities,
                        )
            else:
                negatives[:] = rng.choice(
                    node_count,
                    size=(count, NEGATIVE_SAMPLES),
                    replace=True,
                    p=global_probabilities,
                )

            center_vectors = input_vectors[batch_centers].copy()
            positive_vectors = output_vectors[batch_contexts].copy()
            negative_vectors = output_vectors[negatives].copy()
            positive_scores = np.sum(center_vectors * positive_vectors, axis=1)
            negative_scores = np.einsum(
                "bd,bkd->bk", center_vectors, negative_vectors, optimize=True
            )
            positive_gradients = sigmoid(positive_scores) - 1.0
            negative_gradients = sigmoid(negative_scores)
            center_gradients = positive_gradients[:, None] * positive_vectors
            center_gradients += np.einsum(
                "bk,bkd->bd", negative_gradients, negative_vectors, optimize=True
            )
            progress = (epoch + start / max(len(order), 1)) / SGNS_EPOCHS
            learning_rate = 0.025 * (1.0 - 0.9 * progress)
            np.add.at(input_vectors, batch_centers, -learning_rate * center_gradients)
            np.add.at(
                output_vectors,
                batch_contexts,
                -learning_rate * positive_gradients[:, None] * center_vectors,
            )
            for negative_index in range(NEGATIVE_SAMPLES):
                np.add.at(
                    output_vectors,
                    negatives[:, negative_index],
                    -learning_rate
                    * negative_gradients[:, negative_index, None]
                    * center_vectors,
                )

    result = input_vectors[:customer_count]
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class EmbeddingConfig:
    walks: int
    length: int
    dimensions: int


def generate_embedding(indicators, adjacency, method, config, seed):
    customer_count = indicators.shape[0]
    rng = np.random.default_rng(seed)
    walks = []
    starts = np.arange(customer_count, dtype=np.int32)
    for _ in range(config.walks):
        for start in rng.permutation(starts):
            walks.append(
                random_walk(
                    int(start),
                    adjacency,
                    rng,
                    method,
                    config.length,
                )
            )
    centers, contexts = sample_context_pairs(
        walks,
        CONTEXT_WINDOW,
        MAX_POSITIVE_PAIRS,
        seed + 1,
    )
    print(
        f"Embedding {method}, seed={seed}, config={config}: "
        f"{len(walks):,} walks, {len(centers):,} pairs"
    )
    return train_embedding(
        centers,
        contexts,
        adjacency,
        customer_count,
        config.dimensions,
        method,
        seed + 2,
    )


class FoldWeightedXGBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y):
        labels = np.asarray(y, dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        if (counts == 0).any():
            raise ValueError("Kedua kelas harus ada pada setiap fold fit.")
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=counts[0] / counts[1],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model_.fit(X, labels)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def parameter_candidates(model_name):
    if model_name == "Random Forest":
        return [
            {
                "n_estimators": trees,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": features,
            }
            for trees, depth, leaf, features in [
                (200, None, 1, "sqrt"),
                (200, 15, 2, 0.5),
                (400, None, 2, "sqrt"),
                (400, 15, 1, 0.5),
                (200, None, 2, 0.5),
                (200, 15, 1, "sqrt"),
                (400, None, 1, 0.5),
                (400, 15, 2, "sqrt"),
            ]
        ]
    return [
        {
            "n_estimators": trees,
            "max_depth": depth,
            "learning_rate": rate,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "min_child_weight": child,
            "reg_lambda": reg,
        }
        for trees, depth, rate, subsample, colsample, child, reg in [
            (150, 3, 0.05, 0.8, 0.8, 1, 1.0),
            (150, 5, 0.05, 1.0, 1.0, 5, 5.0),
            (250, 3, 0.10, 1.0, 0.8, 5, 1.0),
            (250, 5, 0.10, 0.8, 1.0, 1, 5.0),
            (250, 3, 0.05, 0.8, 1.0, 1, 5.0),
            (150, 5, 0.10, 1.0, 0.8, 1, 1.0),
            (150, 3, 0.10, 0.8, 0.8, 5, 5.0),
            (250, 5, 0.05, 1.0, 1.0, 5, 1.0),
        ]
    ]


def make_estimator(model_name, params, representation):
    if model_name == "Random Forest":
        classifier = RandomForestClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
            **params,
        )
    else:
        classifier = FoldWeightedXGBClassifier(random_state=SEED, **params)
    steps = []
    if representation == "SVD-32":
        steps.append(("svd", TruncatedSVD(n_components=SVD_DIM, random_state=SEED)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def best_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best = np.flatnonzero(scores == scores.max())
    index = best[-1]
    return float(thresholds[index]), float(scores[index])


def nested_threshold_evaluation(
    representation,
    model_name,
    X,
    y,
    ids,
    run_grid=True,
    fixed_params=None,
):
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    fold_predictions = np.full(len(y), np.nan)
    fold_probabilities = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=np.int32)
    rows = []
    selected_by_fold = []
    candidates = parameter_candidates(model_name) if run_grid else [fixed_params]

    for fold, (train_idx, valid_idx) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(
            n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold
        )
        best_score = -np.inf
        best_params = None
        best_threshold = 0.5
        for candidate in candidates:
            estimator = make_estimator(model_name, candidate, representation)
            inner_probabilities = cross_val_predict(
                estimator,
                X[train_idx],
                y[train_idx],
                cv=inner,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            threshold, score = best_f1_threshold(y[train_idx], inner_probabilities)
            if score > best_score:
                best_score = score
                best_params = candidate
                best_threshold = threshold

        final_estimator = make_estimator(model_name, best_params, representation)
        final_estimator.fit(X[train_idx], y[train_idx])
        probabilities = final_estimator.predict_proba(X[valid_idx])[:, 1]
        predictions = (probabilities >= best_threshold).astype(np.int64)
        fold_probabilities[valid_idx] = probabilities
        fold_predictions[valid_idx] = predictions
        fold_ids[valid_idx] = fold
        fold_f1 = f1_score(y[valid_idx], predictions, zero_division=0)
        rows.append(
            {
                "Representasi": representation,
                "Model": model_name,
                "Outer fold": fold,
                "Inner OOF best F1": best_score,
                "Threshold inner OOF": best_threshold,
                "Outer F1 tuned": fold_f1,
                "Outer F1 threshold 0.5": f1_score(
                    y[valid_idx], probabilities >= 0.5, zero_division=0
                ),
                "Best parameters": repr(best_params),
            }
        )
        selected_by_fold.append(best_params)
        print(
            f"{representation} | {model_name} | fold {fold}: "
            f"threshold={best_threshold:.2f}, outer F1={fold_f1:.4f}"
        )

    if not np.isfinite(fold_probabilities).all() or (fold_ids < 1).any():
        raise RuntimeError(f"OOF tidak lengkap: {representation}, {model_name}")
    return {
        "fold_rows": rows,
        "predictions": fold_predictions,
        "probabilities": fold_probabilities,
        "fold_ids": fold_ids,
        "selected_params": selected_by_fold,
    }


def summarize_oof(representation, model_name, y, result):
    pred = result["predictions"]
    prob = result["probabilities"]
    from sklearn.metrics import roc_auc_score

    return {
        "Representasi": representation,
        "Model": model_name,
        "F1 OOF threshold 0.5": f1_score(y, prob >= 0.5, zero_division=0),
        "F1 OOF threshold inner": f1_score(y, pred, zero_division=0),
        "Mean outer F1 threshold inner": float(
            np.mean([row["Outer F1 tuned"] for row in result["fold_rows"]])
        ),
        "SD outer F1 threshold inner": float(
            np.std([row["Outer F1 tuned"] for row in result["fold_rows"]], ddof=1)
        ),
        "ROC-AUC OOF": float(roc_auc_score(y, prob)),
    }


def modal_parameters(parameters):
    encoded = [json.dumps(params, sort_keys=True) for params in parameters]
    return json.loads(Counter(encoded).most_common(1)[0][0])


def run_seed_and_sensitivity(
    indicators,
    adjacency,
    train_positions,
    train_ids,
    y,
    core_results,
    seed_embeddings,
):
    configs = {
        "baseline_walks_10_length_80_dim_32": EmbeddingConfig(10, 80, 32),
        "sensitivity_walks_2_length_80_dim_32": EmbeddingConfig(2, 80, 32),
        "sensitivity_walks_10_length_12_dim_32": EmbeddingConfig(10, 12, 32),
        "sensitivity_walks_10_length_80_dim_16": EmbeddingConfig(10, 80, 16),
        "sensitivity_walks_10_length_80_dim_64": EmbeddingConfig(10, 80, 64),
    }
    summary_rows = []
    fold_rows = []
    for config_name, config in configs.items():
        config_seeds = EMBEDDING_SEEDS if config_name.startswith("baseline_") else [42]
        for method in METHODS:
            for embedding_seed in config_seeds:
                if (
                    config_name.startswith("baseline_")
                    and embedding_seed == EMBEDDING_SEEDS[0]
                ):
                    embedding = seed_embeddings[method]
                else:
                    embedding = generate_embedding(
                        indicators,
                        adjacency,
                        method,
                        config,
                        embedding_seed,
                    )
                X = embedding[train_positions]
                for model_name in ["Random Forest", "XGBoost"]:
                    core_key = (method, model_name)
                    fixed_params = modal_parameters(
                        core_results[core_key]["selected_params"]
                    )
                    result = nested_threshold_evaluation(
                        method,
                        model_name,
                        X,
                        y,
                        train_ids,
                        run_grid=False,
                        fixed_params=fixed_params,
                    )
                    for row in result["fold_rows"]:
                        fold_rows.append(
                            {
                                **row,
                                "Eksperimen": config_name,
                                "Seed embedding": embedding_seed,
                            }
                        )
                    summary_rows.append(
                        {
                            "Eksperimen": config_name,
                            "Embedding": method,
                            "Seed embedding": embedding_seed,
                            "Walks/customer": config.walks,
                            "Panjang walk": config.length,
                            "Dimensi": config.dimensions,
                            "Model": model_name,
                            **summarize_oof(method, model_name, y, result),
                            "Parameter model (mode dari CV utama)": repr(fixed_params),
                        }
                    )
                    print(
                        f"Sensitivity {config_name} | {method} | seed "
                        f"{embedding_seed} | {model_name}: "
                        f"F1={summary_rows[-1]['F1 OOF threshold inner']:.4f}"
                    )
    results = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    results.to_csv(ROOT / f"{OUTPUT_STEM}_seed_sensitivity_metrics.csv", index=False)
    folds.to_csv(ROOT / f"{OUTPUT_STEM}_seed_sensitivity_outer_folds.csv", index=False)
    return results, folds


def main():
    started = perf_counter()
    data, train_ids, y = load_data()
    indicators, indicator_names = build_indicator_matrix(data)
    id_to_index = pd.Series(np.arange(len(data)), index=data[ID_COLUMN])
    train_positions = id_to_index.reindex(train_ids).to_numpy(dtype=np.int64)
    if len(train_positions) != len(y):
        raise ValueError("Posisi train tidak sejajar dengan OOF.")

    adjacency = make_adjacency(indicators)
    config = EmbeddingConfig(WALKS_PER_CUSTOMER, WALK_LENGTH, EMBEDDING_DIM)
    graph_embeddings = {}
    for method in METHODS:
        graph_embeddings[method] = generate_embedding(
            indicators, adjacency, method, config, EMBEDDING_SEEDS[0]
        )

    all_representations = {
        "OneHot-55": ("OneHot-55", indicators),
        "SVD-32": ("SVD-32", indicators),
    }
    for method in METHODS:
        all_representations[method] = (
            method,
            graph_embeddings[method],
        )
        all_representations[f"Fusion-{method}"] = (
            f"Fusion-{method}",
            np.concatenate([indicators, graph_embeddings[method]], axis=1),
        )

    summaries = []
    folds = []
    prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "Actual": y,
            "Outer fold": np.full(len(y), -1, dtype=np.int32),
        }
    )
    core_results = {}
    seed_embeddings = graph_embeddings.copy()
    for representation_name, (pipeline_representation, full_X) in all_representations.items():
        X = full_X[train_positions]
        for model_name in ["Random Forest", "XGBoost"]:
            print(f"\n=== Nested CV: {representation_name} | {model_name} ===")
            result = nested_threshold_evaluation(
                pipeline_representation,
                model_name,
                X,
                y,
                train_ids,
            )
            core_results[(representation_name, model_name)] = result
            if representation_name in METHODS:
                core_results[(representation_name, model_name)] = result
            summaries.append(
                summarize_oof(representation_name, model_name, y, result)
            )
            folds.extend(result["fold_rows"])
            prediction_frame[f"{representation_name}|{model_name}|probability"] = (
                result["probabilities"]
            )
            prediction_frame[f"{representation_name}|{model_name}|prediction"] = (
                result["predictions"]
            )
            prediction_frame["Outer fold"] = result["fold_ids"]

    pd.DataFrame(summaries).to_csv(
        ROOT / f"{OUTPUT_STEM}_representation_metrics.csv", index=False
    )
    pd.DataFrame(folds).to_csv(
        ROOT / f"{OUTPUT_STEM}_representation_outer_folds.csv", index=False
    )
    prediction_frame.to_csv(
        ROOT / f"{OUTPUT_STEM}_representation_oof_predictions.csv", index=False
    )
    pd.DataFrame({"Indicator": indicator_names}).to_csv(
        ROOT / f"{OUTPUT_STEM}_55_indicator_columns.csv", index=False
    )
    print("\nCore representation experiments complete.")
    print(pd.DataFrame(summaries).to_string(index=False))

    seed_metrics, _ = run_seed_and_sensitivity(
        indicators,
        adjacency,
        train_positions,
        train_ids,
        y,
        core_results,
        seed_embeddings,
    )
    print("\n=== Seed variability and embedding sensitivity ===")
    print(
        seed_metrics.groupby(
            ["Eksperimen", "Embedding", "Model", "Walks/customer", "Panjang walk", "Dimensi"],
            dropna=False,
        )["F1 OOF threshold inner"]
        .agg(["mean", "std", "count"])
        .to_string()
    )
    print(f"Elapsed: {(perf_counter() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
