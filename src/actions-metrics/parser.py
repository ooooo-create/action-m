import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

try:
	import polars as pl
except Exception as e:  # pragma: no cover - environment may not have polars
	raise ImportError(
		"polars is required for this module. Install with `pip install polars`"
	) from e


def _clean_colname(name: str) -> str:
	"""Clean weird quoted CSV column names and convert to snake_case.

	Examples of raw headers in the provided CSVs look like: "'Job",
	so this will strip surrounding quotes/apostrophes and replace
	non-alphanumerics with underscores.
	"""
	if name is None:
		return ""
	# remove surrounding repeated quotes and apostrophes
	cleaned = name.strip()
	cleaned = re.sub(r"^[\'\"]+|[\'\"]+$", "", cleaned)
	# normalize spaces and slashes etc to underscores, lowercase
	cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned).lower()
	cleaned = re.sub(r"__+", "_", cleaned).strip("_")
	return cleaned


def _normalize_df_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Trim and strip surrounding quotes from all string columns."""
    exprs = []
    for c, t in df.schema.items():
        if t == pl.Utf8:
            exprs.append(
                pl.col(c)
                .str.replace_all(r"^[\'\"]+|[\'\"]+$", "")
                .str.strip_chars()
                .alias(c)
            )
    if exprs:
        df = df.with_columns(exprs)
    return df
def load_and_clean_csv(path: Union[str, Path]) -> pl.DataFrame:
	"""Load a CSV with polars and normalize column names and string values.

	- Strips strange quoting in headers and values.
	- Converts column names to snake_case.
	- Attempts to cast well-known numeric columns to Int64.
	"""
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")

	df = pl.read_csv(str(path), ignore_errors=True)

	# rename columns to cleaned names
	mapping = {orig: _clean_colname(orig) for orig in df.columns}
	df = df.rename(mapping)

	# clean up string columns (remove surrounding quotes, trim)
	df = _normalize_df_strings(df)

	# cast commonly numeric columns if present
	for numeric in ("total_minutes", "job_runs"):
		if numeric in df.columns:
			try:
				df = df.with_columns(pl.col(numeric).cast(pl.Int64))
			except Exception:
				# best effort: try to coerce via str -> int where possible
				df = df.with_columns(
					pl.col(numeric).cast(pl.Utf8).str.replace_all(r"[^0-9-]+", "").cast(pl.Int64)
				)

	return df


def merge_usage_performance(
	usage: pl.DataFrame,
	performance: pl.DataFrame,
	on: Optional[Iterable[str]] = ("job", "workflow"),
	how: str = "outer",
) -> pl.DataFrame:
	"""Merge two cleaned tables (usage and performance).

	Strategy:
	- Perform a join on the provided keys (default: `job`, `workflow`).
	- For overlapping non-key columns from `performance` we create a
	  coalesced column that prefers values from `performance` when present,
	  falling back to `usage` otherwise.
	- Keeps a compact, modern schema (snake_case names).
	"""
	keys = list(on) if on is not None else []

	# ensure keys exist in both frames
	for k in keys:
		if k not in usage.columns and k not in performance.columns:
			raise KeyError(f"Join key '{k}' not found in either table")

	# avoid clobbering: rename non-key cols in performance with a suffix
	suffix = "_perf"
	perf_renames = {}
	for c in performance.columns:
		if c not in keys and c in usage.columns:
			perf_renames[c] = f"{c}{suffix}"
	performance = performance.rename(perf_renames)

	# perform join
	merged = usage.join(performance, on=keys, how=how, coalesce=True)

    # coalesce duplicated columns (e.g. total_minutes and total_minutes_perf)
	to_drop: List[str] = []
	for c in usage.columns:
		if c in keys:
			continue
		perf_c = f"{c}{suffix}"
		if perf_c in merged.columns:
			merged = merged.with_columns(
				pl.coalesce(pl.col(perf_c), pl.col(c)).alias(c)
			)
			to_drop.append(perf_c)

	# drop unwanted columns
	to_drop.extend(["runner_type", "runner_labels"])

	if to_drop:
		merged = merged.drop(to_drop)

	return merged


def filter_jobs(df: pl.DataFrame) -> pl.DataFrame:
	"""Filter out unwanted jobs and workflows.

	- Removes rows where 'job' contains 'Cancel' or 'Check bypass'.
	- Removes rows where 'workflow' contains 'copilot' or 'fleety'.
	"""
	if "job" in df.columns:
		df = df.filter(~pl.col("job").str.contains(r"(?i)Cancel|Check bypass"))

	if "workflow" in df.columns:
		df = df.filter(~pl.col("workflow").str.contains(r"(?i)copilot|fleety"))

	return df


def sort_by_workflow_and_job(df: pl.DataFrame) -> pl.DataFrame:
	"""Sort the DataFrame by workflow and then by job."""
	if "workflow" in df.columns and "job" in df.columns:
		return df.sort(["workflow", "job"])
	return df


def prepare_excel_data(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
	"""Prepare data for Excel export to support native sorting and formatting.

	Converts time units to Excel serial date format (days) and percentages to 0-1 decimals.
	Returns the transformed DataFrame and a dictionary of column formats.
	"""
	exprs = []
	formats = {}

	# total_minutes (int) -> days (float)
	if "total_minutes" in df.columns:
		exprs.append((pl.col("total_minutes") / (24 * 60)).alias("total_minutes"))
		formats["total_minutes"] = '[h]"h"mm"m"'

	# failure_rate (0-100) -> (0-1)
	if "failure_rate" in df.columns:
		exprs.append((pl.col("failure_rate") / 100).alias("failure_rate"))
		formats["failure_rate"] = "0.00%"

	# avg_run_time, avg_queue_time (ms) -> days (float)
	for col in ("avg_run_time", "avg_queue_time"):
		if col in df.columns:
			exprs.append((pl.col(col) / (24 * 60 * 60 * 1000)).alias(col))
			formats[col] = '[h]"h"mm"m"ss"s"'

	if exprs:
		df = df.with_columns(exprs)
	
	return df, formats


def load_examples_and_merge(example_dir: Union[str, Path] = "example") -> pl.DataFrame:
	"""Convenience loader for the repo example CSVs.

	Returns the merged DataFrame for quick inspection.
	"""
	d = Path(example_dir)
	usage_path = d / "usage.csv"
	perf_path = d / "performance.csv"

	usage_df = load_and_clean_csv(usage_path)
	perf_df = load_and_clean_csv(perf_path)

	merged = merge_usage_performance(usage_df, perf_df)
	return merged


__all__ = [
	"load_and_clean_csv",
	"merge_usage_performance",
	"load_examples_and_merge",
	"filter_jobs",
	"sort_by_workflow_and_job",
	"prepare_excel_data",
]

