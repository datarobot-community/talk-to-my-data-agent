# --- Utility functions ported from notebooks ---
import datetime
import json
import logging
import os
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict
from pydantic_settings.sources import parse_env_vars
from streamlit import logger as st_logger
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

logger = st_logger.get_logger(__name__)


# --- Configuration ---
class PulumiSettingsSource(EnvSettingsSource):
    """Pulumi stack outputs as a pydantic settings source."""

    _PULUMI_OUTPUTS = None

    def __init__(self, *args, **kwargs):
        self.read_pulumi_outputs()
        super().__init__(*args, **kwargs)

    def read_pulumi_outputs(self):
        try:
            raw_outputs = json.loads(
                subprocess.check_output(
                    ["pulumi", "stack", "output", "-j"], text=True
                ).strip()
            )
            self._PULUMI_OUTPUTS = {
                k: v if isinstance(v, str) else json.dumps(v)
                for k, v in raw_outputs.items()
            }
        except Exception:
            self._PULUMI_OUTPUTS = {}

    def _load_env_vars(self):
        return parse_env_vars(
            self._PULUMI_OUTPUTS,
            self.case_sensitive,
            self.env_ignore_empty,
            self.env_parse_none_str,
        )


class DynamicSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            PulumiSettingsSource(settings_cls),
            env_settings,
        )


# --- Dataset Helper ---
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(3) + wait_random(0, 10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def download_dataset(endpoint: str, token: str, dataset_id: str) -> pd.DataFrame:
    """Download a dataset from DataRobot."""
    logger.info(f"Downloading dataset {dataset_id} from {endpoint}")
    url = f"{endpoint}/datasets/{dataset_id}/file/"
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "*/*",
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    csv_content = response.content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_content))
    logger.info(f"Downloaded dataset {dataset_id} with {len(df)} rows.")
    return df


# --- Data Pipeline ---
def parse_json(x):
    try:
        return json.loads(x) if pd.notnull(x) else {}
    except Exception:
        return {}


def extract_error_message_regex(text):
    # Placeholder: implement your regex extraction logic here
    return text if pd.notnull(text) else None


def load_and_normalize_data(df_trace: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes raw log data DataFrame, parses timestamps, normalizes data types, and consolidates user identity.
    """
    logger.info(f"Starting data normalization for {len(df_trace)} records.")

    # --- Error Message Extraction ---
    df_trace["error_message"] = df_trace["promptText"].apply(
        extract_error_message_regex
    )
    df_trace["error_type"] = df_trace["error_message"].str.split(":").str[0]
    actual_values_dicts = df_trace["actual_value"].apply(parse_json)
    actual_values_normalized = pd.json_normalize(actual_values_dicts.tolist())
    df_trace = pd.concat(
        [
            df_trace.reset_index(drop=True),
            actual_values_normalized.reset_index(drop=True),
        ],
        axis=1,
    )
    df_trace["query_no"] = df_trace["query_type"].str[:2]
    df_trace["timestamp"] = pd.to_datetime(
        df_trace["timestamp"].str[:19]
    ) + pd.Timedelta(hours=9)
    df_trace["startTimestamp"] = pd.to_datetime(
        df_trace["startTimestamp"].str[:19]
    ) + pd.Timedelta(hours=9)
    df_trace["endTimestamp"] = pd.to_datetime(
        df_trace["endTimestamp"].str[:19]
    ) + pd.Timedelta(hours=9)
    df_trace["date"] = df_trace["startTimestamp"].dt.date
    df_trace["chat_seq"] = df_trace["chat_seq"].astype("Int64")
    df_trace.sort_values("startTimestamp", inplace=True)
    df_trace.sort_values(
        ["user_email", "chat_id", "chat_seq", "startTimestamp"], inplace=True
    )
    col_list = [
        "user_email",
        "date",
        "startTimestamp",
        "endTimestamp",
        "association_id",
        "enable_chart_generation",
        "enable_business_insights",
        "chat_id",
        "chat_seq",
        "query_type",
        "query_no",
        "data_source",
        "datasets_names",
        "user_msg",
        "error_message",
        "error_type",
        "promptText",
        "DR_RESERVED_PREDICTION_VALUE",
    ]
    return df_trace[col_list].reset_index(drop=True).copy()


def infer_chat_sessions(df_trace: pd.DataFrame) -> pd.DataFrame:
    """
    Infers user sessions from normalized log data.
    """
    import hashlib

    logger.info(f"Starting session inference for {len(df_trace)} log entries.")
    df_trace_chat = (
        df_trace.groupby(by=["user_email", "chat_id", "chat_seq"])
        .agg(
            date=("date", "first"),
            startTimestamp=("startTimestamp", "first"),
            endTimestamp=("endTimestamp", "last"),
            userMsg=("user_msg", "first"),
            datasetCount=(
                "datasets_names",
                lambda x: len(list(set([item for sublist in x for item in sublist]))),
            ),
            datasetNames=("datasets_names", "first"),
            dataSource=("data_source", "first"),
            chartGen=("enable_chart_generation", "first"),
            businessInsights=("enable_business_insights", "first"),
            errorCount=("error_message", lambda x: x.notna().sum()),
            callCount=("association_id", "count"),
        )
        .reset_index()
    )
    df_trace_chat["idealCount"] = (
        (df_trace_chat["chartGen"].astype(str).str.lower() == "true").astype(int)
        + (df_trace_chat["businessInsights"].astype(str).str.lower() == "true").astype(
            int
        )
        + 2
    )
    df_trace_chat["stopUnexpected"] = (
        df_trace_chat["callCount"] - df_trace_chat["errorCount"]
    ) < df_trace_chat["idealCount"]
    df_trace_chat["time"] = (
        (df_trace_chat["endTimestamp"] - df_trace_chat["startTimestamp"])
        .dt.total_seconds()
        .astype("Int64")
    )
    df_trace_chat.insert(
        0,
        "id",
        df_trace_chat["user_email"].map(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:8]
        )
        + "-"
        + df_trace_chat["chat_id"]
        + "-"
        + df_trace_chat["chat_seq"].astype(str),
    )
    df_trace_req = (
        df_trace.groupby(by=["user_email", "chat_id", "chat_seq", "query_no"])
        .agg(
            count=("association_id", "count"),
            startTimestamp=("startTimestamp", "first"),
            endTimestamp=("endTimestamp", "last"),
            error=("error_message", lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    df_trace_req["time"] = (
        (df_trace_req["endTimestamp"] - df_trace_req["startTimestamp"])
        .dt.total_seconds()
        .astype("Int64")
    )
    df_trace_req = df_trace_req.pivot(
        index=["user_email", "chat_id", "chat_seq"],
        columns="query_no",
        values=["count", "time", "error"],
    ).reset_index()
    df_trace_req.columns = [
        "_".join(col[::-1]).strip() if col[1] else col[0]
        for col in df_trace_req.columns.values
    ]
    df_trace_chat = df_trace_chat.merge(
        df_trace_req,
        on=["user_email", "chat_id", "chat_seq"],
        how="left",
    )
    logger.info(f"Session inference complete. Generated {len(df_trace_chat)} sessions.")
    return df_trace_chat


# Helper for loading DATASET_TRACE_ID like LLMDeployment
class TraceDatasetId(DynamicSettings):
    id: str = Field(
        validation_alias=AliasChoices(
            "MLOPS_RUNTIME_PARAM_DATASET_TRACE_ID",
            "DATASET_TRACE_ID",
        )
    )


def run_pipeline() -> list[pd.DataFrame, pd.DataFrame]:
    """
    Downloads, normalizes, sessionizes, and saves trace data as Parquet files.

    Returns:
        list[pd.DataFrame, pd.DataFrame]: Tuple containing the sessionized chat
        DataFrame and the raw trace DataFrame.
    """
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Check required environment variables
        endpoint = os.getenv("DATAROBOT_ENDPOINT")
        token = os.getenv("DATAROBOT_API_TOKEN")

        if not endpoint:
            raise ValueError("DATAROBOT_ENDPOINT environment variable is not set")
        if not token:
            raise ValueError("DATAROBOT_API_TOKEN environment variable is not set")

        # Get dataset ID
        try:
            trace_dataset_id = TraceDatasetId().id
            logger.info(f"Downloading dataset with ID: {trace_dataset_id}")
        except Exception as e:
            raise ValueError(f"Failed to get DATASET_TRACE_ID: {str(e)}")

        # Download raw CSV from DataRobot
        try:
            df_raw_csv = download_dataset(endpoint, token, trace_dataset_id)
            logger.info(f"Raw data downloaded with {len(df_raw_csv)} records.")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {str(e)}")

        # Normalize data
        try:
            df_raw = load_and_normalize_data(df_raw_csv)
            logger.info(
                f"Data normalization complete. Processed {len(df_raw)} records."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to normalize data: {str(e)}")

        # Infer sessions
        try:
            df_trace = infer_chat_sessions(df_raw)
            logger.info(
                f"Session inference complete. Generated {len(df_trace)} sessions."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to infer chat sessions: {str(e)}")

        logger.info("Pipeline complete: saving DataFrames.")
        # Save DataFrames to Parquet files
        df_raw.to_parquet(data_dir / "trace_raw.parquet", index=False)
        df_trace.to_parquet(data_dir / "trace_chat.parquet", index=False)

        logger.info("Pipeline complete: returning DataFrames.")
        return df_trace, df_raw

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


# --- Data Utils ---
@st.cache_data(ttl=28800)  # Cache for 8 hours (28800 seconds)
def get_or_generate_data(_today_str):
    # Convert string back to datetime object
    logger.info(f"Using data for {_today_str}...")
    _now = datetime.datetime.strptime(_today_str, "%Y-%m-%d %H:%M:%S")

    chat_path = Path("data/trace_chat.parquet")
    raw_path = Path("data/trace_raw.parquet")

    # Check if files exist and were modified within the last minute
    should_refresh = False
    if not chat_path.exists() or not raw_path.exists():
        should_refresh = True
    else:
        # Check if files were last modified more than 1 minute ago
        chat_mtime = datetime.datetime.fromtimestamp(chat_path.stat().st_mtime)
        raw_mtime = datetime.datetime.fromtimestamp(raw_path.stat().st_mtime)

        # Calculate time difference in seconds
        chat_age_seconds = (_now - chat_mtime).total_seconds()
        raw_age_seconds = (_now - raw_mtime).total_seconds()

        # Refresh if either file is older than 60 seconds (1 minute)
        if chat_age_seconds > 60 or raw_age_seconds > 60:
            should_refresh = True
            logger.info(
                f"Data is stale (chat: {chat_age_seconds:.0f}s, raw: {raw_age_seconds:.0f}s old). Refreshing..."
            )

    if should_refresh:
        try:
            df_chat, df_raw = run_pipeline()
            logger.info("Data refreshed successfully!")
        except Exception as e:
            logger.error(f"Failed to refresh data: {str(e)}")
            logger.error("Please check the logs for more details.")
            # Try to load existing data if refresh failed
            if chat_path.exists() and raw_path.exists():
                logger.warning("Using existing data files despite refresh failure.")
            else:
                logger.error("No existing data files found. Cannot proceed.")
                # Return empty dataframes
                return pd.DataFrame(), pd.DataFrame()
    else:
        logger.info("Loading existing data...")
        try:
            df_chat = pd.read_parquet(chat_path)
            df_raw = pd.read_parquet(raw_path)
            logger.info("Data loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load existing data: {str(e)}")
            # Return empty dataframes if loading fails
            return pd.DataFrame(), pd.DataFrame()
    return df_chat, df_raw


def diagnose_environment():
    """Diagnostic function to check environment configuration"""
    import os

    st.subheader("Environment Diagnostic")

    # Check environment variables
    env_vars = {
        "DATAROBOT_ENDPOINT": os.getenv("DATAROBOT_ENDPOINT"),
        "DATAROBOT_API_TOKEN": os.getenv("DATAROBOT_API_TOKEN"),
        "DATASET_TRACE_ID": os.getenv("DATASET_TRACE_ID"),
        "MLOPS_RUNTIME_PARAM_DATASET_TRACE_ID": os.getenv(
            "MLOPS_RUNTIME_PARAM_DATASET_TRACE_ID"
        ),
    }

    for var_name, value in env_vars.items():
        if value:
            # Mask sensitive data
            if "TOKEN" in var_name:
                masked_value = (
                    f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                )
                st.success(f"✅ {var_name}: {masked_value}")
            else:
                st.success(f"✅ {var_name}: {value}")
        else:
            st.error(f"❌ {var_name}: Not set")

    # Check dataset ID resolution
    try:
        dataset_id = TraceDatasetId().id
        st.success(f"✅ Dataset ID resolved: {dataset_id}")
    except Exception as e:
        st.error(f"❌ Failed to resolve dataset ID: {str(e)}")

    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        st.success(f"✅ Data directory exists: {data_dir.absolute()}")
        files = list(data_dir.glob("*.parquet"))
        if files:
            st.info(f"📁 Found {len(files)} parquet files:")
            for file in files:
                stat = file.stat()
                import time

                mtime = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
                )
                st.text(
                    f"  • {file.name} ({stat.st_size // 1024} KB, modified: {mtime})"
                )
        else:
            st.warning("⚠️ No parquet files found in data directory")
    else:
        st.error(f"❌ Data directory does not exist: {data_dir.absolute()}")

    # Check if we're in a cloud environment (Pulumi likely not available)
    try:
        import json
        import subprocess

        result = subprocess.run(
            ["pulumi", "stack", "output", "-j"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            outputs = json.loads(result.stdout)
            st.success(f"✅ Pulumi stack outputs available: {len(outputs)} items")
        else:
            st.info("ℹ️ Pulumi not available (likely in cloud environment)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        st.info("ℹ️ Pulumi not available (likely in cloud environment)")
    except Exception as e:
        st.warning(f"⚠️ Pulumi check failed: {str(e)}")
