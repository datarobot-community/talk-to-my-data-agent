import datetime
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from i18n_setup import _, setup_i18n
from modules import (
    chart_generators,
    data_utils,
    kpi_calculations,
    ui_components,
    wordcloud_utils,
)

st.set_page_config(
    layout="wide",
    page_title="Usage Dashboard",
    page_icon=":material/dashboard:",
)

if "language" not in st.session_state:
    st.session_state.language = "en"

setup_i18n()

# Hide the Deploy button in the top right corner
st.markdown(
    r"""
    <style>
    .stAppDeployButton {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(_("titles.admin_dashboard"))

# Load data
with st.spinner("Loading data..."):
    trace_chat_df, trace_raw_df = data_utils.get_or_generate_data(datetime.date.today())

# if trace_raw_df is only 1 row, show a warning and stop the app
if trace_raw_df.shape[0] == 1:
    st.warning("No usage data found. Please run the data pipeline to generate data.")
    st.stop()

# Get all user emails
user_emails = (
    sorted(trace_chat_df["user_email"].dropna().unique().tolist())
    if not trace_chat_df.empty
    else []
)
if "input_date_range" not in st.session_state:
    st.session_state["input_date_range"] = (
        trace_chat_df["date"].min(),
        trace_chat_df["date"].max(),
    )

# Status indicators and refresh button in sidebar
with st.sidebar:
    st.markdown("## Data Status")
    chat_path = Path("data/trace_chat.parquet")
    raw_path = Path("data/trace_raw.parquet")

    def file_status(path):
        if path.exists():
            stat = path.stat()
            return f"✅ {path.name} | {stat.st_size // 1024} KB | Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))}"
        else:
            return f"❌ {path.name} | Not found"

    st.write(file_status(chat_path))
    st.write(file_status(raw_path))
    # if st.button("🔄 Refresh Data from DataRobot"):
    #     data_utils.refresh_data()
    #     st.rerun()

# Render global filters (sidebar includes language selector)
filters = ui_components.render_global_filters(user_emails)
# filters: (timeframe_key, date_range, user_email, granularity)

timeframe_key, date_range, user_email, granularity = filters

# Helper: Convert timeframe_key/date_range to start/end datetime

now = pd.Timestamp.now()
if timeframe_key == "last_7_days":
    start = now - pd.Timedelta(days=6)
    end = now
elif timeframe_key == "last_30_days":
    start = now - pd.Timedelta(days=29)
    end = now
elif timeframe_key == "today":
    start = now.normalize()
    end = now
elif timeframe_key == "yesterday":
    start = (now - pd.Timedelta(days=1)).normalize()
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
elif timeframe_key == "this_month":
    start = now.replace(day=1).normalize()
    end = now
elif timeframe_key == "last_month":
    first_this_month = now.replace(day=1).normalize()
    last_month_end = first_this_month - pd.Timedelta(days=1)
    start = last_month_end.replace(day=1).normalize()
    end = last_month_end
elif timeframe_key == "custom" and date_range and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1])
else:
    start = now - pd.Timedelta(days=6)
    end = now

# Convert start and end to datetime.date for type consistency
start = start.date()
end = end.date()

# Filter by user if not ALL
filtered_df = trace_chat_df.copy()
if user_email != "ALL":
    filtered_df = filtered_df[filtered_df["user_email"] == user_email]

# Calculate KPIs
# For retention, calculate previous period
period_days = (end - start).days + 1
prev_end = start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days - 1)

kpis = kpi_calculations.calculate_all_kpis(
    filtered_df,
    (start, end),
    (prev_start, prev_end),
)

# Display KPIs in columns
st.subheader(_("section_titles.kpis"))
kpi_labels = [
    ("total_users", _("kpi_labels.total_users")),
    ("recent_active_users", _("kpi_labels.recent_active_users")),
    ("new_users", _("kpi_labels.new_users")),
    ("retention_rate", _("kpi_labels.user_retention_rate")),
    ("total_chats", _("kpi_labels.total_chats")),
    ("avg_chats_per_user", _("kpi_labels.avg_chats_per_user")),
    ("recent_total_chats", _("kpi_labels.recent_total_chats")),
    ("recent_avg_chats_per_user", _("kpi_labels.recent_avg_chats_per_user")),
]
cols = st.columns(len(kpi_labels))
for i, (k, label) in enumerate(kpi_labels):
    value = kpis.get(k)
    if value is None:
        display = "N/A"
    elif k == "retention_rate":
        display = f"{value:.1f}%" if value is not None else "N/A"
    elif isinstance(value, float):
        display = f"{value:.2f}"
    else:
        display = str(value)
    cols[i].metric(label, display)

# Display Active User Trend chart
st.subheader(_("section_titles.active_user_trend"))
fig = chart_generators.plot_active_user_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig, use_container_width=True)

# Display Number of Chats Trend chart
st.subheader(_("section_titles.number_of_chats_trend"))
fig2 = chart_generators.plot_number_of_chats_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig2, use_container_width=True)

# Display User Activity Heatmap
st.subheader(_("section_titles.user_activity_heatmap"))
fig3 = chart_generators.plot_user_activity_heatmap(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig3, use_container_width=True)

# Display LLM Call Count Trend chart
st.subheader(_("section_titles.llm_call_count_trend"))
fig4 = chart_generators.plot_llm_call_count_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig4, use_container_width=True)

# Display LLM Error Count Trend chart
st.subheader(_("section_titles.llm_error_avg_trend"))
fig5 = chart_generators.plot_llm_error_avg_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig5, use_container_width=True)

# Display Average LLM Call Process Time Trend chart
st.subheader(_("section_titles.llm_avg_process_time_trend"))
fig6 = chart_generators.plot_llm_avg_process_time_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig6, use_container_width=True)

# Display Data Source Usage Trend chart
st.subheader(_("section_titles.data_source_usage_trend"))
fig7 = chart_generators.plot_data_source_usage_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig7, use_container_width=True)

# Display Unexpected Finish Trend chart
st.subheader(_("section_titles.unexpected_finish_trend"))
fig8 = chart_generators.plot_unexpected_finish_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig8, use_container_width=True)

# Display User Message Word Cloud
st.subheader(_("section_titles.user_message_wordcloud"))
font_path = "./font/NotoSansJP-VariableFont_wght.ttf"
if not os.path.exists(font_path):
    st.warning(
        "Word cloud font file not found. Please set font_path to a valid Japanese font."
    )
else:
    user_text = (
        " ".join(filtered_df["userMsg"].dropna().astype(str))
        if not filtered_df.empty and "userMsg" in filtered_df.columns
        else ""
    )
    wordcloud_utils.generate_user_wordcloud(
        user_text, font_path, st.session_state.language, _
    )

# Display Error Message Word Cloud
st.subheader(_("section_titles.error_message_wordcloud"))
if not os.path.exists(font_path):
    st.warning(
        "Word cloud font file not found. Please set font_path to a valid Japanese font."
    )
else:
    error_text_list = (
        trace_raw_df["error_message"].dropna().astype(str).tolist()
        if not trace_raw_df.empty and "error_message" in trace_raw_df.columns
        else []
    )
    wordcloud_utils.generate_error_wordcloud(
        error_text_list, font_path, st.session_state.language, _
    )

# Display Detailed Chat Log Table
st.subheader(_("section_titles.detailed_chat_log"))
if filtered_df.empty:
    st.info("No chat log data for the selected filters.")
else:
    display_cols = [
        "user_email",
        "chat_id",
        "chat_seq",
        "userMsg",
        "dataSource",
        "datasetNames",
        "stopUnexpected",
        "startTimestamp",
        "errorCount",
    ]
    col_headers = [_(f"dataframe_headers.{col}") for col in display_cols]
    table_df = filtered_df[display_cols].copy()
    table_df.columns = col_headers
    st.dataframe(table_df, use_container_width=True)
