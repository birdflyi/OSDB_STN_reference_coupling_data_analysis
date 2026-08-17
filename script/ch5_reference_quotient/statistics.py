"""Small deterministic statistical helpers for the frozen RefQ run."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return np.array([], dtype=bool), values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted <= alpha, adjusted


def describe_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": column,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "max": values.max(),
                "skew": values.skew(),
                "kurtosis": values.kurtosis(),
            }
        )
    return pd.DataFrame(rows)


def split_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [label.strip() for label in str(value).replace(";", ",").replace("|", ",").split(",") if label.strip()]


def category_long_frame(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for record in frame.to_dict("records"):
        labels = split_labels(record.get("category_label"))
        if mode == "exclude_mixed_or_multilabel" and len(labels) != 1:
            continue
        for label in labels:
            item = dict(record)
            item["category"] = label
            rows.append(item)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=[*frame.columns, "category"])


def kruskal_fdr(
    frame: pd.DataFrame,
    features: Iterable[str],
    mode: str,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = category_long_frame(frame, mode)
    tests = []
    descriptions = []
    for feature in features:
        groups = []
        group_names = []
        for category, group in long.groupby("category"):
            values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy()
            if len(values) >= min_group_size:
                groups.append(values)
                group_names.append(str(category))
                descriptions.append(
                    {
                        "label_mode": mode,
                        "feature": feature,
                        "category": category,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    }
                )
        if len(groups) < 2:
            continue
        all_values = np.concatenate(groups)
        if np.ptp(all_values) == 0:
            statistic, p_value = 0.0, 1.0
            test_status = "all_values_identical"
        else:
            from scipy import stats

            statistic, p_value = stats.kruskal(*groups)
            test_status = "computed"
        n = sum(len(values) for values in groups)
        k = len(groups)
        epsilon_squared = max(0.0, float((statistic - k + 1) / (n - k))) if n > k else 0.0
        tests.append(
            {
                "label_mode": mode,
                "feature": feature,
                "groups": k,
                "n_with_replacement": n,
                "categories": "|".join(group_names),
                "kruskal_h": float(statistic),
                "p_value": float(p_value),
                "epsilon_squared": epsilon_squared,
                "test_status": test_status,
            }
        )
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty:
        reject, adjusted = benjamini_hochberg(test_frame["p_value"])
        test_frame["fdr_bh_p_value"] = adjusted
        test_frame["fdr_bh_reject_0_05"] = reject
    return test_frame, pd.DataFrame(descriptions)
