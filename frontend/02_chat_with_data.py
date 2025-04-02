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

import asyncio
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import streamlit as st
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from streamlit.delta_generator import DeltaGenerator

sys.path.append("..")
# Import FastAPI functions directly
from app_settings import (
    apply_custom_css,
    display_page_logo,
)
from datarobot_connect import DataRobotTokenManager
from helpers import log_error_details, state_init

from utils.analyst_db import AnalystDB, DataSourceType
from utils.api import (
    AnalysisGenerationError,
    run_complete_analysis,
)
from utils.database_helpers import app_infra
from utils.logging_helper import get_logger
from utils.schema import (
    AnalysisError,
    AnalystChatMessage,
    ChatRequest,
    EnhancedQuestionGeneration,
    GetBusinessAnalysisResult,
    RunAnalysisResult,
    RunChartsResult,
    RunDatabaseAnalysisResult,
)

warnings.filterwarnings("ignore")
logger = get_logger("DataAnalystFrontend")

# Custom CSS
apply_custom_css()


def clear_chat() -> None:
    st.session_state.chat_messages = []
    st.session_state.chat_input_key += 1
    st.session_state.current_chat_name = (
        f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.session_state.current_chat_id = None


@dataclass
class RenderContainers:
    """Containers for UI elements"""

    rephrase: DeltaGenerator
    bottom_line: DeltaGenerator
    analysis: DeltaGenerator
    charts: DeltaGenerator
    insights: DeltaGenerator
    followup: DeltaGenerator


class UnifiedRenderer:
    """Handles rendering for both historical and live messages"""

    def __init__(self, is_live: bool = False):
        self.is_live = is_live
        self._containers: RenderContainers | None = None

    def set_containers(self, containers: RenderContainers) -> None:
        """Set containers for live rendering"""
        self._containers = containers

    @property
    def containers(self) -> RenderContainers:
        if self._containers is None:
            raise ValueError("Containers not initialized")
        return self._containers

    def render_message(
        self,
        message: AnalystChatMessage,
        within_chat_context: bool = False,
    ) -> None:
        """
        Render a single message with all its components
        within_chat_context: If True, assumes we're already inside a chat_message context
        """

        if not within_chat_context:
            with st.chat_message(
                message.role,
                avatar="bot.jpg" if message.role == "assistant" else "you.jpg",
            ):
                self._render_message_content(message)
        else:
            self._render_message_content(message)

    def _render_message_content(self, message: AnalystChatMessage) -> None:
        """Internal method to render message content and components"""
        # Render main content
        if message.role == "user":
            # For user messages, just render the content
            st.markdown(message.content)
        else:
            # For assistant messages, only render the main content if there's no EnhancedQuestionGeneration
            has_enhanced = any(
                isinstance(comp, EnhancedQuestionGeneration)
                for comp in message.components
            )
            if not has_enhanced:
                st.markdown(message.content)
        components = message.components if not self.is_live else message.components[-1:]
        if message.role == "assistant":
            # Sort components by type for consistent rendering order
            enhanced_q = None
            analysis_result = None
            charts_result = None
            business_result = None
            exception = None

            for component in components:
                if isinstance(component, EnhancedQuestionGeneration):
                    enhanced_q = component
                elif isinstance(
                    component, (RunAnalysisResult, RunDatabaseAnalysisResult)
                ):
                    analysis_result = component
                elif isinstance(component, RunChartsResult):
                    charts_result = component
                elif isinstance(component, GetBusinessAnalysisResult):
                    business_result = component
                elif isinstance(component, AnalysisError):
                    exception = component

            # Render components in order
            if enhanced_q:
                with self.containers.rephrase:
                    st.markdown(enhanced_q.enhanced_user_message)

            if analysis_result:
                self.render_analysis_results(
                    analysis_result,
                    isinstance(analysis_result, RunDatabaseAnalysisResult),
                )

            if charts_result:
                self.render_charts(charts_result)

            if business_result:
                self.render_business_results(business_result)
            if exception:
                self.render_exception(exception)

    def render_analysis_results(
        self, result: RunAnalysisResult | RunDatabaseAnalysisResult, is_database: bool
    ) -> None:
        """Render analysis results and code"""

        with self.containers.analysis:
            if result.status == "error":
                self.render_exception(result.metadata.exception)
                return
            if result.code:
                with st.expander("Analysis Code", expanded=False):
                    language = "sql" if is_database else "python"
                    st.code(result.code, language=language)
            if result.dataset:
                with st.expander("Analysis Results", expanded=True):
                    st.dataframe(result.dataset.to_df(), use_container_width=True)

    def render_charts(self, result: RunChartsResult) -> None:
        """Render charts"""
        with self.containers.charts:
            if result.status == "error":
                self.render_exception(result.metadata.exception)

            index = uuid.uuid4()
            if result.fig1:
                st.plotly_chart(
                    result.fig1,
                    use_container_width=True,
                    key=f"message_{index}_fig1",
                )
            if result.fig2:
                st.plotly_chart(
                    result.fig2,
                    use_container_width=True,
                    key=f"message_{index}_fig2",
                )

    def render_business_results(self, result: GetBusinessAnalysisResult) -> None:
        """Render business analysis results"""
        if result.status == "error":
            with self.containers.bottom_line:
                if result.metadata is not None and result.metadata.exception_str:
                    st.error(
                        f"Error running business analysis\n{result.metadata.exception_str}"
                    )
                else:
                    st.error("Error running business analysis")
        with self.containers.bottom_line:
            with st.expander("Bottom Line", expanded=True):
                st.markdown((result.bottom_line or "").replace("$", r"\$"))

        with self.containers.insights:
            if result.additional_insights:
                with st.expander("Additional Insights", expanded=True):
                    st.markdown(result.additional_insights.replace("$", r"\$"))

        with self.containers.followup:
            if result.follow_up_questions:
                with st.expander("Follow-up Questions", expanded=True):
                    for q in result.follow_up_questions:
                        st.markdown(f"- {q}".replace("$", r"\$"))

    def render_exception(self, exception: AnalysisError | None) -> None:
        if (
            exception is None
            or exception.exception_history is None
            or len(exception.exception_history) == 0
        ):
            st.error("An error occurred during analysis. Please retry")
            return
        last_exception = exception.exception_history[-1]
        st.error(f"Error: {last_exception.exception_str}")
        if last_exception.code is not None:
            with st.expander("Last Executed Code"):
                st.code(last_exception.code)


# Usage for historical messages
def render_conversation_history(messages: list[AnalystChatMessage]) -> None:
    renderer = UnifiedRenderer(is_live=False)
    for message in messages:
        renderer.render_message(message)


async def run_complete_analysis_st(
    chat_request: ChatRequest, error_context: dict[str, Any]
) -> None:
    """Run the complete analysis pipeline"""
    renderer = UnifiedRenderer(is_live=True)

    logger.info("start analysis")
    logger.info(f"Current Chat name: {st.session_state.current_chat_name}")
    with st.chat_message("assistant", avatar="bot.jpg"):
        containers = RenderContainers(
            rephrase=st.container(),
            bottom_line=st.container(),
            analysis=st.container(),
            charts=st.container(),
            insights=st.container(),
            followup=st.container(),
        )
        renderer.set_containers(containers)

        try:
            selected_datasets = [
                dataset_name
                for dataset_name in st.session_state.datasets_names
                if st.session_state[f"dataset_{dataset_name}"]
            ]
            run_analysis_iterator = run_complete_analysis(
                chat_request=chat_request,
                data_source=st.session_state.data_source,
                datasets_names=selected_datasets,
                analyst_db=st.session_state.analyst_db,
                chat_id=st.session_state.current_chat_id,
                enable_chart_generation=st.session_state.enable_chart_generation,
                enable_business_insights=st.session_state.enable_business_insights,
            )
            with st.spinner("Analysing question..."):
                enhanced_message = await anext(run_analysis_iterator)

            assistant_message = AnalystChatMessage(
                role="assistant",
                content=enhanced_message,
                components=[
                    EnhancedQuestionGeneration(enhanced_user_message=enhanced_message)
                ],
            )
            renderer.render_message(assistant_message, within_chat_context=True)

            with st.spinner("Generating insights..."):
                async for message in run_analysis_iterator:
                    if isinstance(message, AnalysisGenerationError):
                        st.error(message.message)
                        break
                    else:
                        assistant_message.components.append(message)
                    renderer.render_message(assistant_message, within_chat_context=True)

            st.session_state.chat_messages.append(assistant_message)
        except Exception as e:
            error_context["component"] = "main_process"
            log_error_details(e, error_context)
            st.error(f"Error processing chat and analysis: {str(e)}")


# Initialize session state variables
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}
if "current_chat_name" not in st.session_state:
    st.session_state.current_chat_name = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


async def main() -> None:
    await state_init()
    # Main page content (Chat Interface)
    display_page_logo()
    if "analyst_db" not in st.session_state:
        time.sleep(1)
        st.rerun()

    analyst_db: AnalystDB = st.session_state.analyst_db
    # Sidebar UI
    all_chats = await analyst_db.get_chat_list()
    if not st.session_state.data_source:
        all_datasets = []
    elif st.session_state.data_source == DataSourceType.DATABASE:
        all_datasets = await analyst_db.list_analyst_datasets(
            data_source=DataSourceType(st.session_state.data_source)
        )
    else:
        all_datasets = await analyst_db.list_analyst_datasets(
            data_source=DataSourceType.FILE
        ) + await analyst_db.list_analyst_datasets(data_source=DataSourceType.CATALOG)

    st.session_state.datasets_names = all_datasets
    if (
        "current_chat_id" not in st.session_state
        or st.session_state.current_chat_id is None
    ):
        clear_chat()
    else:
        try:
            st.session_state.chat_messages = await analyst_db.get_chat_messages(
                chat_id=st.session_state.current_chat_id
            )
        except Exception as e:
            logger.error(f"Error retrieving chat: {e}")
            clear_chat()
    # Sidebar with New Chat button only
    with st.sidebar:
        st.title("Chat Controls")

        if app_infra.database != "no_database":

            def set_database_mode() -> None:
                if st.session_state.database_mode == "Local":
                    st.session_state.data_source = DataSourceType.FILE
                else:
                    st.session_state.data_source = DataSourceType.DATABASE

            st.radio(
                "Database Mode",
                options=["Local", app_infra.database.title()],
                key="database_mode",
                horizontal=True,
                on_change=set_database_mode,
                index=1
                if st.session_state.data_source == DataSourceType.DATABASE
                else 0,
            )

        # Chat History in expander
        if len(all_datasets) > 0:
            with st.expander("Available Datasets", expanded=True):
                for dataset_name in all_datasets:
                    st.checkbox(dataset_name, key=f"dataset_{dataset_name}", value=True)

            st.divider()
        st.checkbox(
            "Generate charts in conversation",
            value=True,
            key="enable_chart_generation",
        )
        st.checkbox(
            "Enable business insights and follow up questions in conversation",
            value=True,
            key="enable_business_insights",
        )

        # Quick Actions - Always visible
        col1, col2 = st.columns(2)

        status_container = st.container()
        st.divider()

        with col1:
            st.button(
                "New Chat",
                on_click=clear_chat,
                use_container_width=True,
                type="primary",
            )
        with col2:
            if st.button(
                "Save Chat",
                use_container_width=True,
                type="secondary",
            ):
                with status_container:
                    if not st.session_state.current_chat_id:
                        st.session_state.current_chat_id = await analyst_db.create_chat(
                            chat_name=st.session_state.current_chat_name
                        )
                    await analyst_db.chat_handler.update_chat(
                        chat_id=st.session_state.current_chat_id,
                        messages=st.session_state.chat_messages,
                    )
                    st.rerun()

        # List all saved chats
        if len(all_chats) == 0:
            st.write("No saved chats available.")
        else:
            st.subheader("Saved Chats")
            for chat in all_chats:
                chat_id = chat["id"]
                chat_name = chat["name"]
                with st.container():
                    # Create columns for name, load button, and delete
                    col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
                    with col2:
                        if st.button(
                            "✎",  # Folder icon for load
                            key=f"edit_{chat_id}",
                            use_container_width=True,
                        ):
                            st.session_state[
                                f"name_{chat_id}_edit"
                            ] = not st.session_state[f"name_{chat_id}_edit"]
                    with col1:
                        if f"name_{chat_id}_edit" not in st.session_state:
                            st.session_state[f"name_{chat_id}_edit"] = True

                        new_name = st.text_input(
                            "name",
                            value=chat["name"],  # Current name as default value
                            key=f"name_{chat_id}",
                            label_visibility="collapsed",
                            disabled=st.session_state.get(f"name_{chat_id}_edit", True),
                            # Make it look like text until clicked
                            placeholder=chat["name"],
                        )
                        if new_name and new_name != chat["name"]:
                            with status_container:
                                await analyst_db.rename_chat(chat["id"], new_name)
                                st.session_state.current_chat_name = new_name
                                st.rerun()
                            st.session_state[f"name_{chat_id}_edit"] = True

                    with col3:
                        if st.button(
                            "➡️",  # Folder icon for load
                            key=f"load_{chat_id}",
                            use_container_width=True,
                        ):
                            with status_container:
                                st.session_state.chat_messages = (
                                    await analyst_db.get_chat_messages(chat_id=chat_id)
                                )
                                st.session_state.current_chat_name = chat_name
                                st.session_state.current_chat_id = chat_id
                                st.rerun()

                    with col4:
                        if st.button(
                            "🗑️",
                            key=f"delete_{chat_id}",
                            use_container_width=True,
                        ):
                            with status_container:
                                await analyst_db.delete_chat(chat_id=chat_id)
                                st.rerun()

        # Current chat info at the bottom
        if st.session_state.current_chat_name:
            st.divider()
            st.caption(f"Current chat: {st.session_state.current_chat_name}")
        st.divider()

    st.session_state.chat_messages = cast(
        list[AnalystChatMessage], st.session_state.chat_messages
    )
    if not st.session_state.datasets_names and not st.session_state.chat_messages:
        st.info(
            "Please upload and process data using the sidebar before starting the chat"
        )
    else:
        # Render existing chat history
        renderer = UnifiedRenderer(is_live=False)
        for message in st.session_state.chat_messages:
            with st.chat_message(
                message.role,
                avatar="bot.jpg" if message.role == "assistant" else "you.jpg",
            ):
                containers = RenderContainers(
                    rephrase=st.container(),
                    bottom_line=st.container(),
                    analysis=st.container(),
                    charts=st.container(),
                    insights=st.container(),
                    followup=st.container(),
                )
                renderer.set_containers(containers)
                renderer.render_message(message, within_chat_context=True)
        # Handle new chat input
        if question := st.chat_input(
            "Ask a question about your data",
        ):
            # Create and add user message
            user_message = AnalystChatMessage(
                role="user", content=question, components=[]
            )
            if not st.session_state.current_chat_id:
                st.session_state.current_chat_id = await analyst_db.create_chat(
                    chat_name=st.session_state.current_chat_name
                )

            await analyst_db.update_chat(
                chat_id=st.session_state.current_chat_id,
                chat_message=user_message,
                mode="append",
            )
            st.session_state.chat_messages.append(user_message)

            # Display user's original message
            with st.chat_message("user", avatar="you.jpg"):
                st.markdown(question)

            # Prepare chat messages
            valid_messages: list[ChatCompletionMessageParam] = [
                msg.to_openai_message_param()
                for msg in st.session_state.chat_messages
                if msg.content.strip()
            ]
            valid_messages.append(
                ChatCompletionUserMessageParam(role="user", content=question)
            )
            # Create chat request and run analysis
            chat_request = ChatRequest(messages=valid_messages)
            error_context = {
                "question": question,
                "chat_history_length": len(valid_messages),
            }

            await run_complete_analysis_st(chat_request, error_context)
            st.rerun()


if __name__ == "__main__":
    datarobot_connect = DataRobotTokenManager()
    st.session_state.datarobot_connect = datarobot_connect

    asyncio.run(main())
else:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
