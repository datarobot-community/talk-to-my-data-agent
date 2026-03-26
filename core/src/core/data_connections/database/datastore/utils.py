# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from datetime import date, datetime
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Sequence,
    cast,
)

import numpy as np
import pandas as pd

from core.data_connections.database.datastore.preview import (
    DatasetPreview,
    DatasetPreviewColumn,
)


@dataclass(frozen=True)
class _NormalizeHandler:
    predicate: Callable[[Any], bool]
    normalize: Callable[[Any], Any]


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_sample_value(value: Any) -> Any:
    if value is None or _is_missing_scalar(value):
        return None

    handlers: List[_NormalizeHandler] = [
        _NormalizeHandler(
            lambda v: isinstance(v, str) and v == "NaT",
            lambda v: None,
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, pd.Timestamp),
            lambda v: None if pd.isna(v) else v.isoformat(),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, np.datetime64),
            lambda v: None if pd.isna(v) else pd.Timestamp(v).isoformat(),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, pd.Timedelta),
            lambda v: None if pd.isna(v) else v.isoformat(),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, np.timedelta64),
            lambda v: pd.Timedelta(v).isoformat(),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, pd.Period),
            lambda v: str(v),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, np.bool_),
            lambda v: bool(v),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, np.integer),
            lambda v: int(v),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, np.floating),
            lambda v: None if pd.isna(v) else float(v),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, np.ndarray),
            lambda v: v.tolist(),
        ),
        _NormalizeHandler(
            lambda v: isinstance(
                v, (pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex)
            ),
            lambda v: v.astype(str).tolist(),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, (pd.Series, pd.Index)),
            lambda v: v.astype(object).tolist(),
        ),
        _NormalizeHandler(
            lambda v: isinstance(v, (datetime, date)),
            lambda v: v.isoformat(),
        ),
    ]

    for handler in handlers:
        if handler.predicate(value):
            return handler.normalize(value)

    return value


def _normalize_rows(
    sample_rows: Sequence[Mapping[str, Any]] | pd.DataFrame | None, sample_size: int
) -> list[dict[str, Any]]:
    if sample_rows is None:
        return []

    rows: list[Mapping[str, Any]]
    if isinstance(sample_rows, pd.DataFrame):
        rows = cast(
            list[Mapping[str, Any]],
            sample_rows.head(sample_size).to_dict("records"),
        )
    else:
        rows = list(sample_rows)[:sample_size]

    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {str(key): _normalize_sample_value(value) for key, value in row.items()}
        )
    return normalized


def _columns_from_mapping(
    column_types: Mapping[str, Any] | None,
) -> list[DatasetPreviewColumn]:
    if not column_types:
        return []
    return [
        DatasetPreviewColumn(name=name, data_type=str(column_types[name]))
        for name in column_types
    ]


def _human_summary(
    *,
    title: str | None,
    source_kind: str,
    rows_previewed: int,
    rows_total: int | None,
    is_complete: bool,
) -> str:
    base = (
        f"Showing {rows_previewed} row(s)"
        if rows_total is None
        else f"Showing {rows_previewed} of {rows_total} row(s)"
    )
    suffix = "complete result" if is_complete else "sampled result"
    prefix = f"{title} – " if title else ""
    return f"{prefix}{base} ({suffix}; source={source_kind})"


def _columns_to_markdown(columns: Sequence["DatasetPreviewColumn"]) -> str:
    if not columns:
        return "(no column metadata)"
    try:
        data = [
            {
                "name": col.name,
                "type": col.data_type or "unknown",
                "nullable": col.nullable,
            }
            for col in columns
        ]
        return pd.DataFrame(data).to_markdown(index=False)
    except Exception:
        lines = ["name | type | nullable"]
        for col in columns:
            lines.append(f"{col.name} | {col.data_type or 'unknown'} | {col.nullable}")
        return "\n".join(lines)


def build_dataset_preview(
    *,
    source_kind: str,
    source_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    sample_rows: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None,
    rows_total: int | None,
    sample_size: int,
    column_types: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> DatasetPreview:
    rows = _normalize_rows(sample_rows, sample_size)
    rows_previewed = len(rows)
    is_complete = rows_total is not None and rows_previewed >= rows_total

    columns = _columns_from_mapping(column_types)
    if not columns and rows:
        inferred = {key: None for key in rows[0].keys()}
        columns = _columns_from_mapping(inferred)

    preview = DatasetPreview(
        source_kind=source_kind,
        source_id=source_id,
        title=title,
        description=description,
        rows_previewed=rows_previewed,
        rows_total=rows_total,
        is_complete=is_complete,
        sample_rows=rows,
        columns=columns,
        column_markdown=_columns_to_markdown(columns),
        human_summary=_human_summary(
            title=title,
            source_kind=source_kind,
            rows_previewed=rows_previewed,
            rows_total=rows_total,
            is_complete=is_complete,
        ),
        notes=list(notes) if notes else [],
        extra=dict(extra) if extra else None,
    )
    return preview
