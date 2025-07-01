import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# All KPI calculation functions for the dashboard
# Each function returns a value or None if not applicable
# All user-facing text should be handled via i18n in the main app, not here


def calculate_total_unique_users(df: pd.DataFrame) -> int:
    """Return the count of unique users in the DataFrame."""
    return df["user_email"].nunique() if "user_email" in df.columns else 0


def calculate_recent_active_users(
    df: pd.DataFrame, start: datetime.date, end: datetime.date
) -> int:
    """Return the count of unique users active in the selected timeframe."""
    if "user_email" not in df.columns or "date" not in df.columns:
        return 0
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, "user_email"].nunique()


def calculate_new_users(
    df: pd.DataFrame, start: datetime.date, end: datetime.date
) -> int:
    """Return the count of users whose first chat is within the selected timeframe."""
    if "user_email" not in df.columns or "startTimestamp" not in df.columns:
        return 0
    first_chat = df.groupby("user_email")["startTimestamp"].min().reset_index()
    mask = (first_chat["startTimestamp"].dt.date >= start) & (
        first_chat["startTimestamp"].dt.date <= end
    )
    return mask.sum()


def calculate_user_retention_rate(
    df: pd.DataFrame,
    p1: Tuple[datetime.date, datetime.date],
    p2: Tuple[datetime.date, datetime.date],
) -> Optional[float]:
    """
    Calculate user retention rate: percentage of users active in P1 who were also active in P2.
    Returns None if not applicable (e.g., A_P1 == 0).
    """
    if "user_email" not in df.columns or "date" not in df.columns:
        return None
    p1_mask = (df["date"] >= p1[0]) & (df["date"] <= p1[1])
    p2_mask = (df["date"] >= p2[0]) & (df["date"] <= p2[1])
    users_p1 = set(df.loc[p1_mask, "user_email"].unique())
    users_p2 = set(df.loc[p2_mask, "user_email"].unique())
    if not users_p1:
        return None
    retained = users_p1 & users_p2
    return (len(retained) / len(users_p1)) * 100


def calculate_total_chats(df: pd.DataFrame) -> int:
    """Return the total number of user messages (rows)."""
    return len(df)


def calculate_average_chats_per_user(
    total_chats: int, total_users: int
) -> Optional[float]:
    """Return the average number of chats per user."""
    if total_users == 0:
        return None
    return total_chats / total_users


def calculate_recent_total_chats(
    df: pd.DataFrame, start: datetime.date, end: datetime.date
) -> int:
    """Return the number of user messages in the selected timeframe."""
    if "date" not in df.columns:
        return 0
    mask = (df["date"] >= start) & (df["date"] <= end)
    return mask.sum()


def calculate_recent_average_chats_per_user(
    recent_total_chats: int, recent_active_users: int
) -> Optional[float]:
    """Return the average number of chats per active user in the selected timeframe."""
    if recent_active_users == 0:
        return None
    return recent_total_chats / recent_active_users


def calculate_all_kpis(
    df: pd.DataFrame,
    timeframe: Tuple[datetime.date, datetime.date],
    previous_timeframe: Optional[Tuple[datetime.date, datetime.date]] = None,
) -> Dict[str, Any]:
    """
    Calculate all KPIs and return as a dictionary.
    timeframe: (start, end) for current period
    previous_timeframe: (start, end) for retention calculation
    """
    total_users_all_time = calculate_total_unique_users(df)
    recent_active_users = calculate_recent_active_users(df, *timeframe)
    new_users = calculate_new_users(df, *timeframe)
    retention_rate = None
    if previous_timeframe:
        retention_rate = calculate_user_retention_rate(
            df, previous_timeframe, timeframe
        )
    total_chats = calculate_total_chats(df)
    avg_chats_per_user = calculate_average_chats_per_user(
        total_chats, total_users_all_time
    )
    recent_total_chats = calculate_recent_total_chats(df, *timeframe)
    recent_avg_chats_per_user = calculate_recent_average_chats_per_user(
        recent_total_chats, recent_active_users
    )

    return {
        "total_users": total_users_all_time,
        "total_users_all_time": total_users_all_time,  # Keep both for backward compatibility
        "recent_active_users": recent_active_users,
        "new_users": new_users,
        "retention_rate": retention_rate,
        "total_chats": total_chats,
        "avg_chats_per_user": avg_chats_per_user,
        "recent_total_chats": recent_total_chats,
        "recent_avg_chats_per_user": recent_avg_chats_per_user,
    }
