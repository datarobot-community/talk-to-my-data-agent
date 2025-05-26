import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from .data_pipeline import run_pipeline


def get_today():
    return datetime.date.today()


@st.cache_data
def get_or_generate_data(_today):
    chat_path = Path("data/trace_chat.parquet")
    raw_path = Path("data/trace_raw.parquet")

    run_pipeline()
    try:
        df_chat = pd.read_parquet(chat_path)
    except Exception as e:
        st.error(f"Failed to load trace_chat.parquet: {e}")
        df_chat = pd.DataFrame()
    try:
        df_raw = pd.read_parquet(raw_path)
    except Exception as e:
        st.error(f"Failed to load trace_raw.parquet: {e}")
        df_raw = pd.DataFrame()
    return df_chat, df_raw


def load_trace_chat_data(_today):
    df_chat, _ = get_or_generate_data(_today)
    return df_chat


def load_trace_raw_data(_today):
    _, df_raw = get_or_generate_data(_today)
    return df_raw


# Optional: Expose a manual refresh function for the Streamlit app


def refresh_data():
    st.info("Refreshing data from DataRobot and regenerating Parquet files...")
    run_pipeline()
    st.success("Data refreshed!")
