"""Phase 1 dataset audit for the Borg traces Kaggle CSV.

Reads data/raw/borg_traces_data.csv (single ~314 MB file, ~406k logical
records after correct CSV parsing) and produces:

  - docs/dataset_audit.md   (human-readable audit report)
  - docs/dataset_summary.json (machine-readable stats backing the report)

This script is read-only with respect to data/raw/. It never modifies or
duplicates the raw file. It loads the raw CSV into memory in full for
analysis (the parsed dataset is ~600 MB in memory, well within the
memory budget observed on this machine at run time), rather than
chunking, because a preliminary chunked pandas read showed the entire
dataset comfortably fits in memory. This choice is recorded explicitly
in the audit report rather than left implicit.

Run with:
    .venv/bin/python src/data/inspect_borg.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "borg_traces_data.csv"
DOCS_DIR = REPO_ROOT / "docs"
SUMMARY_JSON = DOCS_DIR / "dataset_summary.json"


def physical_line_count(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def safe_literal_eval(value):
    if not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return "PARSE_ERROR"


def analyze_dict_column(series: pd.Series, n_sample: int = 5000) -> dict:
    non_null = series.dropna()
    sample = non_null.sample(min(n_sample, len(non_null)), random_state=0) if len(non_null) else non_null
    parsed = sample.map(safe_literal_eval)
    n_parse_errors = int((parsed == "PARSE_ERROR").sum()) if len(parsed) else 0
    keys = Counter()
    key_none_counts = Counter()
    n_ok = 0
    for v in parsed:
        if isinstance(v, dict):
            n_ok += 1
            for k, val in v.items():
                keys[k] += 1
                if val is None:
                    key_none_counts[k] += 1
    return {
        "n_sampled": int(len(sample)),
        "n_parsed_ok": n_ok,
        "n_parse_errors": n_parse_errors,
        "keys_seen": dict(keys),
        "none_value_counts_per_key": dict(key_none_counts),
    }


def analyze_array_string_column(series: pd.Series, n_sample: int = 5) -> dict:
    non_null = series.dropna()
    examples = non_null.head(n_sample).tolist()
    contains_embedded_newline = non_null.str.contains("\n", regex=False).sum() if len(non_null) else 0
    contains_commas = non_null.str.contains(",", regex=False).sum() if len(non_null) else 0
    n_values_extracted = []
    float_re = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
    for v in non_null.head(200):
        n_values_extracted.append(len(float_re.findall(v)))
    return {
        "n_non_null": int(len(non_null)),
        "n_with_embedded_newline": int(contains_embedded_newline),
        "n_with_commas": int(contains_commas),
        "example_raw_values": examples,
        "extracted_value_count_stats_first_200": {
            "min": int(np.min(n_values_extracted)) if n_values_extracted else None,
            "max": int(np.max(n_values_extracted)) if n_values_extracted else None,
            "mean": float(np.mean(n_values_extracted)) if n_values_extracted else None,
        },
    }


def col_stats(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n = len(s)
    n_null = int(s.isna().sum())
    stat = {
        "dtype": str(s.dtype),
        "n_null": n_null,
        "pct_null": round(100 * n_null / n, 4) if n else None,
        "n_unique": int(s.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(s):
        non_null = s.dropna()
        if len(non_null):
            stat.update(
                {
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "median": float(non_null.median()),
                    "pct_zero": round(100 * float((non_null == 0).sum()) / len(non_null), 4),
                    "p01": float(non_null.quantile(0.01)),
                    "p99": float(non_null.quantile(0.99)),
                }
            )
    else:
        vc = s.value_counts(dropna=True).head(10)
        stat["top_values"] = {str(k): int(v) for k, v in vc.items()}
        examples = s.dropna().unique()[:5]
        stat["example_values"] = [str(x)[:120] for x in examples]
    return stat


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    file_size_bytes = RAW_PATH.stat().st_size
    phys_lines = physical_line_count(RAW_PATH)

    df = pd.read_csv(RAW_PATH, low_memory=False)
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1e6

    columns = list(df.columns)

    # --- Exact full-row duplicate check (hash of the tuple of all values per row) ---
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    n_exact_duplicate_rows = int(n_rows - row_hashes.nunique())

    # --- Duplicate check excluding the leading unnamed index column ---
    cols_no_idx = [c for c in columns if c != "Unnamed: 0"]
    row_hashes_no_idx = pd.util.hash_pandas_object(df[cols_no_idx], index=False)
    n_exact_duplicate_rows_no_idx = int(n_rows - row_hashes_no_idx.nunique())

    # --- Per-column stats ---
    per_column = {c: col_stats(df, c) for c in columns}

    # --- Null-pattern co-occurrence analysis (union-of-tables vs join hypothesis) ---
    pattern_cols = [
        "instance_events_type",
        "collections_events_type",
        "average_usage",
        "start_time",
        "cpu_usage_distribution",
        "event",
        "resource_request",
        "machine_id",
    ]
    pattern_cols = [c for c in pattern_cols if c in columns]
    null_mask = df[pattern_cols].isna()
    patterns = list(null_mask.itertuples(index=False, name=None))
    pattern_counts = Counter(patterns)
    null_pattern_report = [
        {
            "pattern_is_null": dict(zip(pattern_cols, pat)),
            "count": cnt,
            "pct": round(100 * cnt / n_rows, 4),
        }
        for pat, cnt in pattern_counts.most_common(20)
    ]

    # --- Timestamp analysis ---
    timestamp_cols = [c for c in ["time", "start_time", "end_time"] if c in columns]
    timestamp_report = {}
    for c in timestamp_cols:
        vals = df[c].dropna()
        uniq_sorted = np.sort(vals.unique())
        diffs = np.diff(uniq_sorted)
        diff_counts = Counter(diffs.tolist())
        top_diffs = diff_counts.most_common(10)
        timestamp_report[c] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_unique_values": int(len(uniq_sorted)),
            "most_common_gaps_between_consecutive_unique_values": [
                {"gap": g, "count": cnt} for g, cnt in top_diffs
            ],
            "min_gap": float(diffs.min()) if len(diffs) else None,
            "max_gap": float(diffs.max()) if len(diffs) else None,
        }
    # cross-check start_time <= end_time
    if "start_time" in columns and "end_time" in columns:
        both = df[["start_time", "end_time"]].dropna()
        n_end_before_start = int((both["end_time"] < both["start_time"]).sum())
        n_equal = int((both["end_time"] == both["start_time"]).sum())
        timestamp_report["start_end_consistency"] = {
            "n_rows_with_both": int(len(both)),
            "n_end_before_start": n_end_before_start,
            "n_end_equal_start": n_equal,
        }

    # --- Dict-like usage columns ---
    dict_cols = [c for c in ["average_usage", "maximum_usage", "random_sample_usage", "resource_request"] if c in columns]
    dict_col_report = {c: analyze_dict_column(df[c]) for c in dict_cols}

    # --- Array-string distribution columns ---
    array_cols = [c for c in ["cpu_usage_distribution", "tail_cpu_usage_distribution"] if c in columns]
    array_col_report = {c: analyze_array_string_column(df[c]) for c in array_cols}

    # --- Categorical / semantic columns of interest ---
    categorical_summary = {}
    for c in [
        "instance_events_type",
        "collections_events_type",
        "collection_type",
        "scheduling_class",
        "priority",
        "vertical_scaling",
        "scheduler",
        "cluster",
        "event",
        "failed",
    ]:
        if c in columns:
            vc = df[c].value_counts(dropna=False)
            categorical_summary[c] = {str(k): int(v) for k, v in vc.items()}

    # --- Entity cardinalities ---
    entity_cardinality = {}
    for c in ["machine_id", "collection_id", "alloc_collection_id", "user", "collection_name", "collection_logical_name", "instance_index"]:
        if c in columns:
            entity_cardinality[c] = int(df[c].nunique(dropna=True))

    # --- Rows per (collection_id, instance_index) to understand granularity ---
    if "collection_id" in columns and "instance_index" in columns:
        grp = df.groupby(["collection_id", "instance_index"]).size()
        rows_per_instance = {
            "n_distinct_collection_instance_pairs": int(len(grp)),
            "mean_rows_per_instance": float(grp.mean()),
            "median_rows_per_instance": float(grp.median()),
            "max_rows_per_instance": int(grp.max()),
        }
    else:
        rows_per_instance = None

    summary = {
        "file": {
            "path": str(RAW_PATH.relative_to(REPO_ROOT)),
            "size_bytes": file_size_bytes,
            "size_mb": round(file_size_bytes / 1e6, 2),
            "physical_line_count_including_header": phys_lines,
            "logical_csv_record_count": n_rows,
            "note_on_line_count_discrepancy": (
                "physical_line_count_including_header counts raw newline-terminated "
                "lines in the file. logical_csv_record_count counts actual CSV records "
                "as parsed by pandas' CSV parser. These differ because several columns "
                "(cpu_usage_distribution, tail_cpu_usage_distribution) contain the string "
                "representation of a numpy array printed with numpy's default repr, which "
                "inserts literal newline characters inside a quoted CSV field when the "
                "array is long. A naive line-count-based row estimate is therefore wrong "
                "by roughly 3.3x; only a proper CSV parser gives the correct record count."
            ),
        },
        "shape": {"n_rows": n_rows, "n_cols": n_cols, "in_memory_mb_deep": round(mem_mb, 1)},
        "columns": columns,
        "duplicates": {
            "n_exact_duplicate_rows_including_index_col": n_exact_duplicate_rows,
            "n_exact_duplicate_rows_excluding_index_col": n_exact_duplicate_rows_no_idx,
        },
        "per_column": per_column,
        "null_pattern_analysis": {
            "columns_used": pattern_cols,
            "top_patterns": null_pattern_report,
            "n_distinct_patterns": len(pattern_counts),
        },
        "timestamps": timestamp_report,
        "dict_like_columns": dict_col_report,
        "array_string_columns": array_col_report,
        "categorical_summary": categorical_summary,
        "entity_cardinality": entity_cardinality,
        "rows_per_collection_instance": rows_per_instance,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {SUMMARY_JSON}")
    print(json.dumps({k: v for k, v in summary.items() if k in ("file", "shape", "duplicates")}, indent=2))


if __name__ == "__main__":
    main()
