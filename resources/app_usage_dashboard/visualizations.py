import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from i18n_setup import _
from streamlit.logger import get_logger
from wordcloud import WordCloud

# For Japanese tokenization, use janome
try:
    from janome.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

# For TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = get_logger("streamlit")

# --- Chart Generators ---


# Example: Active User Trend (Line Chart)
def get_active_user_trend_data(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> pd.DataFrame:
    """
    Get the data for active user trend chart.
    """
    if "date" not in df.columns or "user_email" not in df.columns:
        return pd.DataFrame()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    trend = filtered.groupby("period")["user_email"].nunique().reset_index()
    trend.columns = ["period", "active_users"]
    return trend


def plot_active_user_trend(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot the trend of active users over time.
    granularity: 'D' (daily), 'W' (weekly), 'M' (monthly)
    """
    trend = get_active_user_trend_data(df, timeframe, granularity)
    if trend.empty:
        return go.Figure()
    fig = px.line(
        trend,
        x="period",
        y="active_users",
        markers=True,
        labels={
            "period": _(f"filters.granularity.{granularity.lower()}"),
            "active_users": _("kpi_labels.total_users"),
        },
        title=_("charts.active_user_trend"),
    )
    return fig


# Number of Chats Trend (Line Chart)
def get_number_of_chats_trend_data(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> pd.DataFrame:
    """
    Get the data for number of chats trend chart.
    """
    if "date" not in df.columns:
        return pd.DataFrame()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    trend = filtered.groupby("period").size().reset_index(name="num_chats")
    return trend


def plot_number_of_chats_trend(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot the trend of number of chats (user messages) over time.
    """
    trend = get_number_of_chats_trend_data(df, timeframe, granularity)
    if trend.empty:
        return go.Figure()
    fig = px.line(
        trend,
        x="period",
        y="num_chats",
        markers=True,
        labels={
            "period": _(f"filters.granularity.{granularity.lower()}"),
            "num_chats": _("charts.number_of_chats"),
        },
        title=_("charts.number_of_chats_trend"),
    )
    return fig


# User Activity Heatmap
def get_user_activity_heatmap_data(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> pd.DataFrame:
    """
    Generate the pivot table data for user activity heatmap.
    Returns a DataFrame with users as index, periods as columns, and chat counts as values.
    """
    if "date" not in df.columns or "user_email" not in df.columns:
        return pd.DataFrame()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    pivot = filtered.pivot_table(
        index="user_email",
        columns="period",
        values="userMsg",
        aggfunc="count",
        fill_value=0,
    )
    return pivot


def plot_user_activity_heatmap(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot a heatmap of user activity (number of chats per user per period).
    Dynamically adjusts height and margins for large datasets.
    """
    pivot = get_user_activity_heatmap_data(df, timeframe, granularity)
    if pivot.empty:
        return go.Figure()
    # Dynamic height calculation
    base_height = 300
    row_height = 22
    min_height = 300
    max_height = 1200
    n_rows = pivot.shape[0]
    height = min(max_height, max(base_height + n_rows * row_height, min_height))
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            text=pivot.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="Blues",
        )
    )
    fig.update_layout(
        title=_("charts.user_activity_heatmap"),
        xaxis_title=_(f"filters.granularity.{granularity.lower()}"),
        yaxis_title=_("kpi_labels.user_email"),
        coloraxis_colorbar_title=_("charts.num_chats"),
        height=height,
        yaxis=dict(automargin=True),
        xaxis=dict(automargin=True),
    )
    return fig


# LLM Call Count Trend (Grouped Line Chart)
def plot_llm_call_count_trend(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot grouped line chart for LLM call counts by query_no type (02, 03, 04, 05).
    """
    count_cols = [
        col
        for col in df.columns
        if col.endswith("_count") and col[:2] in {"02", "03", "04", "05"}
    ]
    if not count_cols or "date" not in df.columns:
        return go.Figure()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    data = filtered.groupby("period")[count_cols].sum().reset_index()
    fig = go.Figure()
    for col in count_cols:
        fig.add_trace(
            go.Scatter(
                x=data["period"],
                y=data[col],
                mode="lines+markers",
                name=col,
            )
        )
    fig.update_layout(
        title=_("charts.llm_call_count_trend"),
        xaxis_title=_(f"filters.granularity.{granularity.lower()}"),
        yaxis_title=_("charts.llm_call_count"),
    )
    return fig


# LLM Error Count Trend (Grouped Line Chart)
def plot_llm_error_avg_trend(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot grouped line chart for LLM error average counts by query_no type (02, 03, 04, 05).
    """
    error_cols = [
        col
        for col in df.columns
        if col.endswith("_error") and col[:2] in {"02", "03", "04", "05"}
    ]
    if not error_cols or "date" not in df.columns:
        return go.Figure()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    data = filtered.groupby("period")[error_cols].mean().reset_index()
    fig = go.Figure()
    for col in error_cols:
        fig.add_trace(
            go.Scatter(
                x=data["period"],
                y=data[col],
                mode="lines+markers",
                name=col,
            )
        )
    fig.update_layout(
        title=_("charts.llm_error_avg_trend"),
        xaxis_title=_(f"filters.granularity.{granularity.lower()}"),
        yaxis_title=_("charts.llm_error_avg"),
    )
    return fig


# Average LLM Call Process Time Trend (Multi-Line Chart)
def plot_llm_avg_process_time_trend(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot multi-line chart for average LLM call process time by query_no type (02, 03, 04, 05).
    """
    time_cols = [
        col
        for col in df.columns
        if col.endswith("_time") and col[:2] in {"02", "03", "04", "05"}
    ]
    count_cols = [
        col
        for col in df.columns
        if col.endswith("_count") and col[:2] in {"02", "03", "04", "05"}
    ]
    if not time_cols or not count_cols or "date" not in df.columns:
        return go.Figure()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    time_data = filtered.groupby("period")[time_cols].sum().reset_index()
    count_data = filtered.groupby("period")[count_cols].sum().reset_index()
    fig = go.Figure()
    for t_col, c_col in zip(time_cols, count_cols):
        avg_time = time_data[t_col] / count_data[c_col].replace(0, np.nan)
        fig.add_trace(
            go.Scatter(
                x=time_data["period"],
                y=avg_time,
                mode="lines+markers",
                name=t_col.replace("_time", ""),
            )
        )
    fig.update_layout(
        title=_("charts.llm_avg_process_time_trend"),
        xaxis_title=_(f"filters.granularity.{granularity.lower()}"),
        yaxis_title=_("charts.llm_avg_process_time"),
    )
    return fig


# Data Source Usage Trend (Stacked Bar Chart)
def plot_data_source_usage_trend(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot stacked bar chart for data source usage over time.
    """
    if "date" not in df.columns or "dataSource" not in df.columns:
        return go.Figure()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    data = filtered.groupby(["period", "dataSource"]).size().reset_index(name="count")
    fig = px.bar(
        data,
        x="period",
        y="count",
        color="dataSource",
        labels={
            "period": _(f"filters.granularity.{granularity.lower()}"),
            "count": _("charts.num_chats"),
            "dataSource": _("charts.data_source"),
        },
        title=_("charts.data_source_usage_trend"),
    )
    fig.update_layout(barmode="stack")
    return fig


# Unexpected Finish Trend (Line Chart)
def plot_unexpected_finish_trend(
    df: pd.DataFrame, timeframe: tuple[pd.Timestamp, pd.Timestamp], granularity: str
) -> go.Figure:
    """
    Plot line chart for count of chats where stopUnexpected is True.
    """
    if "date" not in df.columns or "stopUnexpected" not in df.columns:
        return go.Figure()
    mask = (df["date"] >= timeframe[0]) & (df["date"] <= timeframe[1])
    filtered = df.loc[mask].copy()
    filtered["period"] = (
        pd.to_datetime(filtered["date"]).dt.to_period(granularity).dt.to_timestamp()
    )
    data = (
        filtered[filtered["stopUnexpected"] == True]
        .groupby("period")
        .size()
        .reset_index(name="unexpected_count")
    )
    fig = px.line(
        data,
        x="period",
        y="unexpected_count",
        markers=True,
        labels={
            "period": _(f"filters.granularity.{granularity.lower()}"),
            "unexpected_count": _("charts.unexpected_finish_count"),
        },
        title=_("charts.unexpected_finish_trend"),
    )
    return fig


# --- Word Cloud Utilities ---


def generate_user_wordcloud(text_data, font_path, current_language, _):
    if not text_data:
        st.warning(_("warnings.no_data_for_wordcloud"))
        return
    processed_text = text_data
    if current_language == "ja":
        if JANOME_AVAILABLE:
            try:
                words = [token.surface for token in tokenizer.tokenize(text_data)]
                processed_text = " ".join(words)
            except Exception as e:
                st.error(_("errors.janome_tokenize_error", error=str(e)))
                return
        else:
            st.error(_("errors.janome_init_error"))
            return
    # Count word frequencies
    from collections import Counter

    word_freq = Counter(processed_text.split())
    try:
        wordcloud = WordCloud(
            font_path=font_path, width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_freq)
        st.image(wordcloud.to_array())
    except RuntimeError as e:
        st.error(_("errors.wordcloud_font_error", font_path=font_path, e=str(e)))
    except Exception as e:
        st.error(_("errors.wordcloud_unexpected_error", e=str(e)))


def generate_error_wordcloud(text_list, font_path, current_language, _):
    if not text_list or not any(text_list):
        st.warning(_("warnings.no_data_for_wordcloud"))
        return
    if current_language == "ja":
        if JANOME_AVAILABLE:
            try:
                # Tokenize each error message
                text_list = [
                    " ".join([token.surface for token in tokenizer.tokenize(t)])
                    for t in text_list
                ]
            except Exception as e:
                st.error(_("errors.janome_tokenize_error", error=str(e)))
                logger.error(_("errors.janome_tokenize_error", error=str(e)))
                logger.error(traceback.format_exc())
                return
        else:
            st.error(_("errors.janome_init_error"))
            return
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for TF-IDF word cloud.")
        return
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        tfidf_matrix = vectorizer.fit_transform(text_list)
        scores = tfidf_matrix.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        word_scores = dict(zip(words, scores))
        wordcloud = WordCloud(
            font_path=font_path, width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_scores)
        st.image(wordcloud.to_array())
    except RuntimeError as e:
        st.error(_("errors.wordcloud_font_error", font_path=font_path, e=str(e)))
    except Exception as e:
        st.error(_("errors.wordcloud_unexpected_error", e=str(e)))
