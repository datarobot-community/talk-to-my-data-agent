import datetime
import os
import time
from pathlib import Path

import data_manager
import kpi_calculations
import pandas as pd
import streamlit as st
import visualizations
from i18n_setup import _, setup_i18n

st.set_page_config(
    layout="wide",
    page_title="Usage Dashboard",
    page_icon=":material/dashboard:",
)

if "language" not in st.session_state:
    st.session_state.language = "en"
if "trace_chat" not in st.session_state:
    st.session_state.trace_chat = pd.DataFrame()
if "trace_raw" not in st.session_state:
    st.session_state.trace_raw = pd.DataFrame()

setup_i18n()

# Hide the Deploy button in the top right corner and style download buttons
st.markdown(
    r"""
    <style>
    .stAppDeployButton {
            visibility: hidden;
        }
    .stDownloadButton > button {
        width: 100%;
        margin-top: 1rem;
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.375rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 500;
        text-align: center;
    }
    .stDownloadButton > button:hover {
        background-color: #0052a3;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(_("titles.admin_dashboard"))

# Load data
with st.spinner("Loading data..."):
    # today_str = datetime.date.today().strftime("%Y-%m-%d")
    today_str = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:00"
    )  # floor to minutes
    trace_chat_df, trace_raw_df = data_manager.get_or_generate_data(today_str)
    st.session_state.trace_chat = trace_chat_df
    st.session_state.trace_raw = trace_raw_df

# if trace_raw_df is only 1 row, show a warning and stop the app
if st.session_state.trace_raw.shape[0] == 1:
    st.warning("No usage data found. Please run the data pipeline to generate data.")
    st.stop()

# Get all user emails
user_emails = (
    sorted(st.session_state.trace_chat["user_email"].dropna().unique().tolist())
    if not st.session_state.trace_chat.empty
    else []
)
if "input_date_range" not in st.session_state:
    st.session_state["input_date_range"] = (
        st.session_state.trace_chat["date"].min(),
        st.session_state.trace_chat["date"].max(),
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

    # Add refresh button, skipped for now
    # if st.button("🔄 Refresh Data"):
    #     with st.spinner("Refreshing data..."):
    #         try:
    #             data_manager.run_pipeline()
    #             st.success("Data refreshed successfully!")
    #             # Reload data after refresh
    #             trace_chat_df, trace_raw_df = data_manager.get_or_generate_data(
    #                 today_str
    #             )
    #         except Exception as e:
    #             st.error(f"Failed to refresh data: {str(e)}")
    #             st.error("Please check the logs for more details.")
    #             # Try to load existing data if refresh failed
    #             if chat_path.exists() and raw_path.exists():
    #                 st.warning("Using existing data files despite refresh failure.")
    #             else:
    #                 st.error("No existing data files found. Cannot proceed.")
    #                 # Return empty dataframes
    #                 trace_chat_df, trace_raw_df = pd.DataFrame(), pd.DataFrame()
    #         st.session_state.trace_chat = trace_chat_df
    #         st.session_state.trace_raw = trace_raw_df

    # Add diagnostic button, skipped for now
    # if st.button("🔍 Environment Diagnostic"):
    # data_manager.diagnose_environment()


# UI Components functions (integrated from ui_components.py)
def update_language():
    import i18n

    i18n.set("locale", st.session_state.language)


def render_global_filters(user_emails):
    # Language selector with on_change
    st.sidebar.radio(
        _("filters.language"),
        options=["en", "ja"],
        index=["en", "ja"].index(st.session_state.get("language", "en")),
        key="language",
        on_change=update_language,
    )

    # Set default values if not present
    if "timeframe_key" not in st.session_state:
        st.session_state["timeframe_key"] = "last_7_days"
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = "ALL"
    if "granularity" not in st.session_state:
        st.session_state["granularity"] = "daily"
    if "date_range" not in st.session_state:
        st.session_state["date_range"] = None

    timeframe_options = [
        ("last_7_days", _("filters.timeframe.last_7_days")),
        ("last_30_days", _("filters.timeframe.last_30_days")),
        ("today", _("filters.timeframe.today")),
        ("yesterday", _("filters.timeframe.yesterday")),
        ("this_month", _("filters.timeframe.this_month")),
        ("last_month", _("filters.timeframe.last_month")),
        ("custom", _("filters.timeframe.custom")),
    ]
    timeframe_keys = [x[0] for x in timeframe_options]
    st.session_state["timeframe_key"] = st.sidebar.selectbox(
        label=_("filters.timeframe.label"),
        options=timeframe_keys,
        format_func=lambda x: dict(timeframe_options)[x],
        key="timeframe_select",
        index=timeframe_keys.index(st.session_state["timeframe_key"]),
    )
    if st.session_state["timeframe_key"] == "custom":
        # Ensure date_range is valid
        if (
            "date_range" not in st.session_state
            or st.session_state["date_range"] is None
            or st.session_state["date_range"][0] is None
            or st.session_state["date_range"][1] is None
        ):
            st.session_state["date_range"] = (
                st.session_state["input_date_range"][0],
                st.session_state["input_date_range"][1],
            )
        st.session_state["date_range"] = st.sidebar.slider(
            _("filters.timeframe.custom"),
            key="custom_date_range",
            min_value=st.session_state["input_date_range"][0],
            max_value=st.session_state["input_date_range"][1],
            value=(
                st.session_state["date_range"][0],
                st.session_state["date_range"][1],
            ),
        )
    else:
        st.session_state["date_range"] = (
            st.session_state["input_date_range"][0],
            st.session_state["input_date_range"][1],
        )

    user_email_options = ["ALL"] + user_emails
    st.session_state["user_email"] = st.sidebar.selectbox(
        label=_("filters.user_email.label"),
        options=user_email_options,
        format_func=lambda x: _("filters.user_email.all") if x == "ALL" else x,
        key="user_email_select",
        index=user_email_options.index(st.session_state["user_email"]),
    )

    granularity_options = ["daily", "weekly", "monthly"]
    st.session_state["granularity"] = st.sidebar.selectbox(
        label=_("filters.granularity.label"),
        options=granularity_options,
        format_func=lambda x: _(f"filters.granularity.{x}"),
        key="granularity_select",
        index=granularity_options.index(st.session_state["granularity"]),
    )

    return (
        st.session_state["timeframe_key"],
        st.session_state["date_range"],
        st.session_state["user_email"],
        st.session_state["granularity"],
    )


# Render global filters (sidebar includes language selector)
filters = render_global_filters(user_emails)
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
filtered_df = st.session_state.trace_chat.copy()
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
    ("recent_active_users", _("kpi_labels.recent_active_users")),
    ("total_users", _("kpi_labels.total_users")),
    ("new_users", _("kpi_labels.new_users")),
    ("retention_rate", _("kpi_labels.user_retention_rate")),
    ("recent_total_chats", _("kpi_labels.recent_total_chats")),
    ("total_chats", _("kpi_labels.total_chats")),
    ("recent_avg_chats_per_user", _("kpi_labels.recent_avg_chats_per_user")),
    ("avg_chats_per_user", _("kpi_labels.avg_chats_per_user")),
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
fig = visualizations.plot_active_user_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig, use_container_width=True, key="active_user_trend_chart")

# Add download button below the chart
trend_data = visualizations.get_active_user_trend_data(
    filtered_df, (start, end), granularity[0].upper()
)

if not trend_data.empty:
    csv_data = trend_data.to_csv(index=False)
    st.download_button(
        label="📥 Download",
        data=csv_data,
        file_name=f"active_user_trend_{start}_{end}.csv",
        mime="text/csv",
        help="Download the active user trend data as CSV",
    )

# Display Number of Chats Trend chart
st.subheader(_("section_titles.number_of_chats_trend"))
fig2 = visualizations.plot_number_of_chats_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig2, use_container_width=True, key="number_of_chats_trend_chart")

# Add download button below the chart
chats_trend_data = visualizations.get_number_of_chats_trend_data(
    filtered_df, (start, end), granularity[0].upper()
)

if not chats_trend_data.empty:
    csv_data = chats_trend_data.to_csv(index=False)
    st.download_button(
        label="📥 Download",
        data=csv_data,
        file_name=f"number_of_chats_trend_{start}_{end}.csv",
        mime="text/csv",
        help="Download the number of chats trend data as CSV",
    )

# Display User Activity Heatmap
st.subheader(_("section_titles.user_activity_heatmap"))
fig3 = visualizations.plot_user_activity_heatmap(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig3, use_container_width=True, key="user_activity_heatmap_chart")

# Add download button below the chart
heatmap_data = visualizations.get_user_activity_heatmap_data(
    filtered_df, (start, end), granularity[0].upper()
)

if not heatmap_data.empty:
    csv_data = heatmap_data.to_csv()
    st.download_button(
        label="📥 Download",
        data=csv_data,
        file_name=f"user_activity_heatmap_{start}_{end}.csv",
        mime="text/csv",
        help="Download the user activity heatmap data as CSV",
    )

# Display LLM Call Count Trend chart
st.subheader(_("section_titles.llm_call_count_trend"))
fig4 = visualizations.plot_llm_call_count_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig4, use_container_width=True, key="llm_call_count_trend_chart")

# Display LLM Error Count Trend chart
st.subheader(_("section_titles.llm_error_avg_trend"))
fig5 = visualizations.plot_llm_error_avg_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig5, use_container_width=True, key="llm_error_avg_trend_chart")

# Display Average LLM Call Process Time Trend chart
st.subheader(_("section_titles.llm_avg_process_time_trend"))
fig6 = visualizations.plot_llm_avg_process_time_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig6, use_container_width=True, key="llm_avg_process_time_trend_chart")

# Display Data Source Usage Trend chart
st.subheader(_("section_titles.data_source_usage_trend"))
fig7 = visualizations.plot_data_source_usage_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig7, use_container_width=True, key="data_source_usage_trend_chart")

# Display Unexpected Finish Trend chart
st.subheader(_("section_titles.unexpected_finish_trend"))
fig8 = visualizations.plot_unexpected_finish_trend(
    filtered_df, (start, end), granularity[0].upper()
)
st.plotly_chart(fig8, use_container_width=True, key="unexpected_finish_trend_chart")

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
    visualizations.generate_user_wordcloud(
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
    visualizations.generate_error_wordcloud(
        error_text_list, font_path, st.session_state.language, _
    )

# Display Detailed Chat Log Table, st.dataframe has download button built-in
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
