# Copyright 2024 DataRobot, Inc.
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

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from utils.schema import DatasetType, DataSourceType, UserRoleType

class Base(DeclarativeBase):
    """Base class for all models."""
    pass

class DatasetMetadata(Base):
    """Model for dataset metadata."""
    __tablename__ = "dataset_metadata"

    table_name: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_type: Mapped[str] = mapped_column(Enum(DatasetType))
    original_name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    columns: Mapped[list[str]] = mapped_column(JSON)
    row_count: Mapped[int] = mapped_column(Integer)
    data_source: Mapped[str] = mapped_column(Enum(DataSourceType))
    file_size: Mapped[int] = mapped_column(Integer, default=0)

class CleansingReport(Base):
    """Model for cleansing reports."""
    __tablename__ = "cleansing_reports"

    dataset_name: Mapped[str] = mapped_column(String, ForeignKey("dataset_metadata.table_name"), primary_key=True)
    report: Mapped[dict[str, Any]] = mapped_column(JSON)

class ChatHistory(Base):
    """Model for chat history."""
    __tablename__ = "chat_history"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    chat_name: Mapped[str] = mapped_column(String, nullable=False)
    data_source: Mapped[str] = mapped_column(String, default="catalog")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="chat", cascade="all, delete-orphan")

class ChatMessage(Base):
    """Model for chat messages."""
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    chat_id: Mapped[str] = mapped_column(String, ForeignKey("chat_history.id"), nullable=False)
    role: Mapped[str] = mapped_column(Enum(UserRoleType))
    content: Mapped[str] = mapped_column(Text)
    components: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    in_progress: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chat: Mapped["ChatHistory"] = relationship(back_populates="messages")