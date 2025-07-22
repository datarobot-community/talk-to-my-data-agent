# main.py

import os  # Added import for os.path.exists

# Import necessary datetime components for timezone handling
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS, WordCloud


# ---------------------------
# Helper functions for caching
# ---------------------------
@st.cache_data
# Add an unused argument that changes daily to invalidate the cache
def load_csv_data(file_path, cache_key):
    """Loads a CSV file into pandas DataFrame."""
    # cache_key is not used inside, only for cache invalidation
    # Check if file exists before attempting to load
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Error: File not found at {file_path}")
        # Return empty dataframe with expected columns if possible, or just empty
        # Adjust based on expected columns in your actual files
        return pd.DataFrame()  # Return empty DataFrame


# ---------------------------
# Load Data Files
# ---------------------------
# Define base path relative to the script location might be safer
# CWD is project root, script is in notebooks/, output is in notebooks/output/
OUTPUT_DIR = "notebooks/output"


def load_all_data():
    data = {}
    # Define GMT+9 timezone
    gmt_plus_9 = timezone(timedelta(hours=9))
    # Get current date in GMT+9 timezone as string
    today_str = datetime.now(tz=gmt_plus_9).date().isoformat()

    # Load the consolidated daily summary metrics file
    data["daily_summary"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "daily_summary_metrics.csv"), cache_key=today_str
    )

    # Keep loading other non-consolidated files
    data["daily_query_type_metrics"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "daily_query_type_metrics.csv"), cache_key=today_str
    )
    data["chat_flows"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "chat_flows.csv"), cache_key=today_str
    )
    data["daily_feature_usage"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "daily_feature_usage.csv"), cache_key=today_str
    )
    data["daily_error_metrics"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "daily_error_metrics.csv"), cache_key=today_str
    )
    data["daily_dataset_usage"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "daily_dataset_usage.csv"), cache_key=today_str
    )
    data["daily_user_governance"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "daily_user_governance.csv"), cache_key=today_str
    )
    data["daily_word_cloud_data"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "daily_word_cloud_data.csv"), cache_key=today_str
    )
    # Load sessions separately as it's not just a daily count
    data["sessions"] = load_csv_data(
        os.path.join(OUTPUT_DIR, "sessions.csv"), cache_key=today_str
    )
    return data


data = load_all_data()


# ---------------------------
# Convert 'date' columns to datetime if needed
# ---------------------------
def convert_date_column(df, date_col="date"):
    # Check if the date column exists and is not already datetime type
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df[date_col]
    ):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            st.warning(f"Could not convert column '{date_col}' to datetime: {e}")
    return df


# Apply conversion where necessary, checking if dataframes are not empty first
# Convert date in the new daily_summary file
if not data["daily_summary"].empty:
    data["daily_summary"] = convert_date_column(data["daily_summary"])
# Keep conversion for other relevant files if they exist and have date columns
# Example: if not data["daily_query_type_metrics"].empty:
#    data["daily_query_type_metrics"] = convert_date_column(data["daily_query_type_metrics"])
# Convert sessions start/end times if needed
if not data["sessions"].empty:
    data["sessions"] = convert_date_column(data["sessions"], date_col="session_start")
    data["sessions"] = convert_date_column(data["sessions"], date_col="session_end")


# ---------------------------
# Sidebar: Global Filters and Trend Granularity Selector
# ---------------------------
st.sidebar.header("Global Filters")
# Date filter: now we use it only for context – overall time window might be extended later.
# Provide a default date that's likely present in data if possible, else today
# Check if dataframes have 'date' column and are not empty before accessing .min()/.max()
# We use the date range to inform the user, but the filter itself isn't applied yet.
available_dates = []
# Collect all available dates from the consolidated summary dataframe
if (
    not data["daily_summary"].empty
    and "date" in data["daily_summary"].columns
    and pd.api.types.is_datetime64_any_dtype(data["daily_summary"]["date"])
):
    available_dates.extend(data["daily_summary"]["date"].dropna().tolist())


if available_dates:
    min_date = min(available_dates)
    max_date = max(available_dates)
    # Ensure default_date is within the range if possible
    default_date_today = pd.to_datetime("today")
    default_date = max(min(default_date_today, max_date), min_date)

else:
    # Fallback if no dates found
    min_date = pd.to_datetime("today") - pd.Timedelta(days=7)
    max_date = pd.to_datetime("today")
    default_date = max_date

# Use the date input as a reference point or start date
selected_date = st.sidebar.date_input(
    "Select Starting Date", value=default_date, min_value=min_date, max_value=max_date
)
trend_granularity = st.sidebar.selectbox(
    "Select Trend Granularity", ["Daily", "Weekly", "Monthly"]
)


# ---------------------------
# Utility: Function to aggregate daily data based on granularity
# ---------------------------
def aggregate_trends(df, value_column="value", date_col="date", granularity="Daily"):
    # Ensure date column is datetime and exists
    if date_col not in df.columns:
        st.warning(f"Date column '{date_col}' not found in dataframe for aggregation.")
        return pd.DataFrame(
            {date_col: [], value_column: []}
        )  # Return empty DF with expected columns

    # Create a copy to avoid modifying the original DataFrame passed from cache
    df_copy = df.copy()

    # Convert if not already datetime
    df_copy = convert_date_column(df_copy, date_col)

    # Ensure date column is now datetime type after conversion attempt
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        st.warning(
            f"Date column '{date_col}' could not be converted to datetime for aggregation."
        )
        return pd.DataFrame({date_col: [], value_column: []})

    # Check for numeric value column
    if value_column not in df_copy.columns or not pd.api.types.is_numeric_dtype(
        df_copy[value_column]
    ):
        st.warning(
            f"Value column '{value_column}' not found or not numeric for aggregation."
        )
        return pd.DataFrame({date_col: [], value_column: []})

    try:
        if granularity == "Weekly":
            # Group by week. Use 'W-MON' or similar if specific start day needed. Reset index.
            agg_df = (
                df_copy.groupby(pd.Grouper(key=date_col, freq="W"))
                .sum(numeric_only=True)
                .reset_index()
            )
        elif granularity == "Monthly":
            # Update freq='M' to freq='ME' to avoid FutureWarning
            agg_df = (
                df_copy.groupby(pd.Grouper(key=date_col, freq="ME"))
                .sum(numeric_only=True)
                .reset_index()
            )
        else:  # Daily
            agg_df = df_copy[[date_col, value_column]].copy()  # Use the copy
        # Ensure the value column exists in the aggregated df
        if value_column not in agg_df.columns:
            # If sum() dropped the column (e.g., all NaNs), recreate it with 0
            agg_df[value_column] = 0

        return agg_df[
            [date_col, value_column]
        ]  # Return only date and the specified value column
    except Exception as e:
        st.error(f"Error during data aggregation: {e}")
        return pd.DataFrame({date_col: [], value_column: []})


###########################
## Navigation Tabs
###########################
tabs = st.tabs(
    [
        "Overview",
        "User Engagement & Query Flow",
        "Quality & Governance",
        "Natural Language Insights",
    ]
)

##############################
## Tab 1: Overview Dashboard
##############################
with tabs[0]:
    st.header("Overview Dashboard")

    # KPI Cards in a Row using st.columns
    col1, col2, col3, col4 = st.columns(4)
    # Ensure dataframes are not empty and columns exist before accessing .iloc[-1]
    # Source KPI data from the consolidated 'daily_summary' DataFrame
    summary_df = data["daily_summary"]
    unique_sessions_value_col = "unique_sessions"
    unique_sessions_value = (
        summary_df[unique_sessions_value_col].iloc[-1]
        if not summary_df.empty and unique_sessions_value_col in summary_df.columns
        else 0
    )
    active_users_value_col = "active_users"
    active_users_value = (
        summary_df[active_users_value_col].iloc[-1]
        if not summary_df.empty and active_users_value_col in summary_df.columns
        else 0
    )
    llm_value_col = "total_llm_calls"
    llm_calls_value = (
        summary_df[llm_value_col].iloc[-1]
        if not summary_df.empty and llm_value_col in summary_df.columns
        else 0
    )
    threads_value_col = "chat_threads"
    chat_threads_value = (
        summary_df[threads_value_col].iloc[-1]
        if not summary_df.empty and threads_value_col in summary_df.columns
        else 0
    )

    with col1:
        st.metric("Active Users", active_users_value)
    with col2:
        st.metric("Unique Sessions", unique_sessions_value)
    with col3:
        st.metric("Chat Threads", chat_threads_value)
    with col4:
        st.metric("LLM Calls", llm_calls_value)

    # --- LLM Calls Trend Chart ---
    st.subheader("LLM Calls Trend")
    # Source data from 'daily_summary'
    if (
        not summary_df.empty
        and "date" in summary_df.columns
        and llm_value_col in summary_df.columns
    ):
        llm_df_agg = aggregate_trends(
            summary_df,  # Use summary_df
            value_column=llm_value_col,
            granularity=trend_granularity,
        )
        if not llm_df_agg.empty:
            fig_llm = px.line(
                llm_df_agg,
                x="date",
                y=llm_value_col,
                title=f"LLM Calls Trend ({trend_granularity})",
            )
            st.plotly_chart(fig_llm, use_container_width=True)
        else:
            st.warning("Could not generate LLM Calls trend chart after aggregation.")
    else:
        st.warning("LLM Calls data is missing or incomplete for the chart.")

    # --- Unique Sessions Trend Chart ---
    st.subheader("Unique Sessions Trend")
    # Source data from 'daily_summary'
    if (
        not summary_df.empty
        and "date" in summary_df.columns
        and unique_sessions_value_col in summary_df.columns
    ):
        sessions_df_agg = aggregate_trends(
            summary_df,  # Use summary_df
            value_column=unique_sessions_value_col,
            granularity=trend_granularity,
        )
        if not sessions_df_agg.empty:
            fig_sessions = px.line(
                sessions_df_agg,
                x="date",
                y=unique_sessions_value_col,
                title=f"Unique Sessions Trend ({trend_granularity})",
            )
            st.plotly_chart(fig_sessions, use_container_width=True)
        else:
            st.warning(
                "Could not generate Unique Sessions trend chart after aggregation."
            )
    else:
        st.warning("Unique Sessions data is missing or incomplete for the chart.")

    # --- Active Users Trend Chart ---
    st.subheader("Active Users Trend")
    # Source data from 'daily_summary'
    if (
        not summary_df.empty
        and "date" in summary_df.columns
        and active_users_value_col in summary_df.columns
    ):
        users_df_agg = aggregate_trends(
            summary_df,  # Use summary_df
            value_column=active_users_value_col,
            granularity=trend_granularity,
        )
        if not users_df_agg.empty:
            fig_users = px.line(
                users_df_agg,
                x="date",
                y=active_users_value_col,
                title=f"Active Users Trend ({trend_granularity})",
            )
            st.plotly_chart(fig_users, use_container_width=True)
        else:
            st.warning("Could not generate Active Users trend chart after aggregation.")
    else:
        st.warning("Active Users data is missing or incomplete for the chart.")

    # --- Chat Threads Trend Chart ---
    st.subheader("Chat Threads Trend")
    # Source data from 'daily_summary'
    if (
        not summary_df.empty
        and "date" in summary_df.columns
        and threads_value_col in summary_df.columns
    ):
        threads_df_agg = aggregate_trends(
            summary_df,  # Use summary_df
            value_column=threads_value_col,
            granularity=trend_granularity,
        )
        if not threads_df_agg.empty:
            fig_threads = px.line(
                threads_df_agg,
                x="date",
                y=threads_value_col,
                title=f"Chat Threads Trend ({trend_granularity})",
            )
            st.plotly_chart(fig_threads, use_container_width=True)
        else:
            st.warning("Could not generate Chat Threads trend chart after aggregation.")
    else:
        st.warning("Chat Threads data is missing or incomplete for the chart.")


###########################################
## Tab 2: User Engagement & Query Flow
###########################################
with tabs[1]:
    st.header("User Engagement & Query Flow")

    # Funnel Chart / Sequential Display of Query Types (using bar chart if needed)
    st.subheader("Query Type Distribution")
    if (
        not data["daily_query_type_metrics"].empty
        and "query_type" in data["daily_query_type_metrics"].columns
        and "count" in data["daily_query_type_metrics"].columns
    ):
        fig2 = px.bar(
            data["daily_query_type_metrics"],
            x="query_type",
            y="count",
            title="Counts by Query Type",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Query Type Metrics data is missing or incomplete for the chart.")

    # Chat Flow & Retry Analysis from chat_flows.csv
    st.subheader("Chat Flow & Retry Analysis")
    if not data["chat_flows"].empty:
        # Display basic statistics – adjust column names based on your CSV structure.
        st.write(
            "Total Chats:",
            (
                data["chat_flows"]["chat_id"].nunique()
                if "chat_id" in data["chat_flows"].columns
                else "N/A"
            ),
        )
        # Assuming 'retry_count_code_gen' might be the intended column based on pipeline doc
        retry_col = "retry_count_code_gen"
        if retry_col in data["chat_flows"].columns:
            # Calculate mean only if column is numeric and has non-NA values
            if (
                pd.api.types.is_numeric_dtype(data["chat_flows"][retry_col])
                and data["chat_flows"][retry_col].notna().any()
            ):
                avg_retry = data["chat_flows"][retry_col].mean()
                st.metric("Average Code Generation Retry", f"{avg_retry:.2f}")
            else:
                st.info(f"'{retry_col}' data not available or not numeric.")
        else:
            # Check for the old 'retry_count' as a fallback maybe? Or just state it's missing.
            st.info(f"'{retry_col}' column not found.")
        st.dataframe(data["chat_flows"].head(10))
    else:
        st.warning("Chat Flows data is missing or incomplete.")


########################################
## Tab 3: Quality & Governance
########################################
with tabs[2]:
    st.header("Quality, Errors & Governance")

    # Error Metrics Visualization
    st.subheader("Error Metrics")
    if (
        not data["daily_error_metrics"].empty
        and "error_category" in data["daily_error_metrics"].columns
        and "count" in data["daily_error_metrics"].columns
    ):
        fig3 = px.bar(
            data["daily_error_metrics"],
            x="error_category",
            y="count",
            title="Error Occurrences by Category",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Error Metrics data is missing or incomplete for the chart.")

    # Dataset Usage Panel
    st.subheader("Dataset Usage")
    if (
        not data["daily_dataset_usage"].empty
        and "dataset_name" in data["daily_dataset_usage"].columns
        and "selection_count" in data["daily_dataset_usage"].columns
    ):
        fig4 = px.bar(
            data["daily_dataset_usage"],
            x="dataset_name",
            y="selection_count",
            title="Dataset Selection Frequency",
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Dataset Usage data is missing or incomplete for the chart.")

    # User & Domain Governance Metrics
    st.subheader("User Domain & Role Breakdown")
    if (
        not data["daily_user_governance"].empty
        and "domain" in data["daily_user_governance"].columns
        and "count" in data["daily_user_governance"].columns
        and "userType" in data["daily_user_governance"].columns
    ):
        fig5 = px.bar(
            data["daily_user_governance"],
            x="domain",
            y="count",
            color="userType",
            title="User Domains and Roles",
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("User Governance data is missing or incomplete for the chart.")


#########################################
## Tab 4: Natural Language Insights
#########################################
with tabs[3]:
    st.header("Natural Language Insights")

    st.subheader("Japanese Word Cloud from User Queries")
    # Assuming daily_word_cloud_data.csv has columns: token, frequency.
    # Check if data exists and is not empty
    if (
        not data["daily_word_cloud_data"].empty
        and "token" in data["daily_word_cloud_data"].columns
        and "frequency" in data["daily_word_cloud_data"].columns
    ):
        # Ensure 'token' is string and 'frequency' is numeric, drop NaNs
        # Create a copy before modification to avoid SettingWithCopyWarning
        wc_data = (
            data["daily_word_cloud_data"].dropna(subset=["token", "frequency"]).copy()
        )
        # Use .loc to safely modify the copy
        wc_data.loc[:, "token"] = wc_data["token"].astype(str)
        # Attempt to convert frequency to numeric, coercing errors to NaN, then fillna with 0 and cast to int
        wc_data.loc[:, "frequency"] = (
            pd.to_numeric(wc_data["frequency"], errors="coerce").fillna(0).astype(int)
        )

        # Filter out zero frequencies if necessary (this operation is safe as it returns a new DataFrame)
        wc_data = wc_data[wc_data["frequency"] > 0]

        # Final check to ensure only string tokens are used
        word_freq = {
            str(row["token"]): int(row["frequency"])
            for idx, row in wc_data.iterrows()
            if isinstance(row["token"], str)
            and str(row["token"]).strip() != ""  # Ensure it's a non-empty string
        }

        # Final safety filter: Ensure all keys are strings before passing to WordCloud
        word_freq = {k: v for k, v in word_freq.items() if isinstance(k, str)}

        if word_freq:
            # --- Word Cloud Code Block ---
            # Using the user provided path again, but still checking existence
            FONT_PATH = "notebooks/font/NotoSansJP-VariableFont_wght.ttf"
            if not os.path.exists(FONT_PATH):
                st.warning(
                    f"Font file not found at {FONT_PATH}. Word cloud may not display correctly."
                )
                # Optionally fall back to default font or skip word cloud
                FONT_PATH = None  # Let WordCloud use its default if path invalid

            # Check if FONT_PATH is valid before proceeding with wordcloud generation that requires it
            if FONT_PATH and os.path.exists(FONT_PATH):
                try:
                    final_word_freq = {}
                    for token, freq in word_freq.items():
                        if (
                            isinstance(token, str)
                            and token.strip() != ""
                            and isinstance(freq, (int, float))
                            and int(freq) > 0
                        ):
                            final_word_freq[token.strip()] = int(freq)

                    if not final_word_freq:
                        st.write("No valid word frequency data for word cloud.")
                    else:
                        string_stopwords = set(map(str, STOPWORDS))
                        wordcloud = WordCloud(
                            background_color="white",
                            width=800,
                            height=400,
                            stopwords=string_stopwords,
                            font_path=FONT_PATH,  # Use the validated path
                        ).generate_from_frequencies(final_word_freq)
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                        ax_wc.imshow(wordcloud, interpolation="bilinear")
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                except Exception as e:
                    st.error(f"Error generating word cloud: {e}")
            elif not FONT_PATH:
                st.warning(
                    "Could not generate word cloud as a valid font path was not found/provided."
                )
            # --- End of Code Block ---
        else:
            st.write("No word frequency data available for the word cloud.")
    else:
        st.write("Word cloud data file is missing or incomplete.")

    # Optionally, display a sample table of user questions (using the word cloud data for now)
    st.subheader("Example User Query Tokens & Frequencies")
    if not data["daily_word_cloud_data"].empty:
        # Display relevant columns, ensure they exist
        display_cols = [
            col
            for col in ["token", "frequency"]
            if col in data["daily_word_cloud_data"].columns
        ]
        if display_cols:
            st.dataframe(data["daily_word_cloud_data"][display_cols].head(10))
        else:
            st.warning(
                "Relevant columns ('token', 'frequency') not found in word cloud data."
            )
    else:
        st.warning("Word Cloud data (used for examples) is missing.")
