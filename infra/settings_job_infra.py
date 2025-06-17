import hashlib
import os
import textwrap
from pathlib import Path
from typing import Sequence, Tuple

import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.exec_envs import RuntimeEnvironment
from datarobot_pulumi_utils.schema.guardrails import (
    Condition,
    GuardConditionComparator,
    ModerationAction,
    Stage,
)

from utils.custom_job_helper import (
    create_or_update_custom_job_schedule,
    poll_custom_job_run_status,
    run_custom_job,
)

from .settings_main import PROJECT_ROOT

# scheduler configuration
SCHEDULER_HOUR = int(os.environ.get("SCHEDULER_HOUR", "21"))

# guardrails

prompt_tokens = datarobot.CustomModelGuardConfigurationArgs(
    name="Prompt Tokens",
    template_name="Prompt Tokens",
    stages=[Stage.PROMPT],
    intervention=datarobot.CustomModelGuardConfigurationInterventionArgs(
        action=ModerationAction.REPORT,
        condition=Condition(
            comparand="4096",
            comparator=GuardConditionComparator.GREATER_THAN,
        ).model_dump_json(),
    ),
)
response_tokens = datarobot.CustomModelGuardConfigurationArgs(
    name="Response Tokens",
    template_name="Response Tokens",
    stages=[Stage.RESPONSE],
    intervention=datarobot.CustomModelGuardConfigurationInterventionArgs(
        action=ModerationAction.REPORT,
        condition=Condition(
            comparand="4096",
            comparator=GuardConditionComparator.GREATER_THAN,
        ).model_dump_json(),
    ),
)
guardrails = [prompt_tokens, response_tokens]

# environment id
base_environment_id = RuntimeEnvironment(
    name="[DataRobot] Python 3.11 Custom Metrics"
).id

# setup paths
dataset_trace_path = str(
    PROJECT_ROOT / "resources" / "job_telemetry_exporter" / "sample_trace.csv"
)
dataset_access_log_path = str(
    PROJECT_ROOT / "resources" / "job_telemetry_exporter" / "sample_access_log.csv"
)
job_path = PROJECT_ROOT / "resources" / "job_telemetry_exporter"

# set the source bundle
resource_bundle_id = "cpu.medium"

# resource names
dataset_trace_name = f"Dataset Trace [{PROJECT_NAME}]"
dataset_access_log_name = f"Dataset Access Log [{PROJECT_NAME}]"
job_resource_name: str = f"Usage Export Job [{PROJECT_NAME}]"


def _prep_metadata_yaml(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs
        | datarobot.CustomModelRuntimeParameterValueArgs
    ],
) -> None:
    from jinja2 import BaseLoader, Environment

    llm_runtime_parameter_specs = "\n".join(
        [
            textwrap.dedent(
                f"""\
            - fieldName: {param.key}
              type: {param.type}
        """
            )
            for param in runtime_parameter_values
        ]
    )
    with open(job_path / "metadata.yaml.jinja") as f:
        template = Environment(loader=BaseLoader()).from_string(f.read())
    (job_path / "metadata.yaml").write_text(
        template.render(
            additional_params=llm_runtime_parameter_specs,
        )
    )


def get_job_files(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs
        | datarobot.CustomModelRuntimeParameterValueArgs,
    ],
) -> Tuple[list[tuple[str, str]], str]:
    _prep_metadata_yaml(runtime_parameter_values)
    # Get all files from job path, excluding specific patterns
    files_to_include: list[Path] = []
    for f in job_path.glob("**/*"):
        if (
            f.is_file()
            and not f.name.endswith(".yaml")
            and "__pycache__" not in f.parts
            and not (f.name.endswith(".pyc") or f.name.endswith(".pyo"))
            and f.name != ".DS_Store"
            and f.name != "run_local.sh"
        ):
            files_to_include.append(f)

    # Add the generated metadata.yaml
    metadata_file_path = job_path / "metadata.yaml"
    files_to_include.append(metadata_file_path)

    # Calculate hash based on file contents
    hasher = hashlib.sha256()
    # Sort files by path to ensure consistent hash order
    files_to_include.sort()
    for file_path in files_to_include:
        try:
            with open(file_path, "rb") as file_content:
                while True:
                    chunk = file_content.read(4096)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except FileNotFoundError:
            # metadata.yaml might not exist on the very first run before _prep_metadata_yaml
            # This is okay, the hash will change once it's created.
            pass

    content_hash = hasher.hexdigest()

    # Prepare the list of tuples for Pulumi
    source_files_tuples = [
        (f.as_posix(), f.relative_to(job_path).as_posix()) for f in files_to_include
    ]

    return source_files_tuples, content_hash


def create_job_schedule(
    custom_job_id: str,
) -> str:
    """
    Create a schedule for a DataRobot custom job.

    Run at the configured hour on weekdays.

    Args:
        custom_job_id (str): The ID of the custom job.

    Returns:
        str: The ID of the created schedule.
    """
    return create_or_update_custom_job_schedule(
        custom_job_id,
        minute=[0],
        hour=[SCHEDULER_HOUR],
        day_of_month=["*"],
        month=["*"],
        day_of_week=[0, 1, 2, 3, 4],  # Sunday–Thursday UTC
    )


def run_job_once(custom_job_id: str) -> str:
    import pulumi

    pulumi.log.info(f"[run_job_once] Called with custom_job_id: {custom_job_id}")
    runtime_parameters = [
        {"fieldName": "MODE", "value": "overwrite", "type": "string"},
    ]
    try:
        custom_run_id = run_custom_job(custom_job_id, runtime_parameters)
        pulumi.log.info(
            f"[run_job_once] run_custom_job returned run_id: {custom_run_id}"
        )
        poll_custom_job_run_status(custom_job_id, custom_run_id)
        return {"success": True, "run_id": custom_run_id}
    except Exception as e:
        pulumi.log.warn(f"[run_job_once] Error running custom job: {e}")
        return {"success": False, "error": str(e)}
