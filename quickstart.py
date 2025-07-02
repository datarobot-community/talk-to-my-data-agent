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

# type: ignore
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info[0] < 3 or (sys.version_info[0] >= 3 and sys.version_info[1] < 9):
    print("Must be using Python version 3.9 or higher")
    exit(1)

work_dir = Path(os.path.dirname(__file__))
dot_env_file = Path(work_dir / ".env")
venv_dir = work_dir / ".venv"


def is_datarobot_codespace():
    return os.getenv("DATAROBOT_NOTEBOOK_IMAGE") is not None


def check_pulumi_installed():
    try:
        subprocess.check_call(
            ["pulumi"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError:
        print(
            "Is pulumi installed? If not, please go to `https://www.pulumi.com/docs/iac/download-install/`"
        )
        exit(1)


def check_pulumi_login():
    try:
        subprocess.check_call(
            ["pulumi", "whoami", "-j", "--non-interactive"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        print(
            "Please login to pulumi and rerun. Use `pulumi login --local` to log in locally or `pulumi login` to login to pulumi cloud with an API token"
        )
        exit(1)


def check_dotenv_exists():
    if not dot_env_file.exists():
        print(
            "Could not find `.env`. Please rename the file `.env.template` and fill in your details"
        )
        exit(1)


def is_windows():
    return os.name == "nt"


def get_activate_command():
    if is_datarobot_codespace():
        return []
    if is_conda_environment():
        if is_windows():
            activate_cmd = ["conda", "activate", f"{venv_dir}", "&&"]
        else:
            # see https://github.com/conda/conda/issues/7980
            activate_cmd = [
                "eval",
                '"$(conda shell.bash hook)"',
                "&&",
                "conda",
                "activate",
                f"{str(venv_dir)}",
                "&&",
            ]

    else:
        # Regular venv activation
        if is_windows():
            activate_script = str(venv_dir / "Scripts" / "activate.bat")
            activate_cmd = ["call", f"{activate_script}", "&&"]
        else:
            activate_script = str(venv_dir / "bin" / "activate")
            activate_cmd = [
                "source",
                f"{activate_script}",
                "&&",
            ]
    return activate_cmd


def is_conda_environment():
    return os.environ.get("CONDA_DEFAULT_ENV") is not None


def parse_args():
    parser = argparse.ArgumentParser(description="Infrastructure management script")
    parser.add_argument("stack_name", help="Stack name to use")
    parser.add_argument(
        "--action",
        choices=["up", "destroy"],
        default="up",
        required=False,
        help="Action to perform (up or destroy)",
    )
    return parser.parse_args()


def get_python_executable():
    if is_conda_environment():
        return shutil.which("python")
    return sys.executable


def run_subprocess_in_venv(command: list[str]):
    if is_windows():
        full_cmd = get_activate_command() + command
        # shell = True, otherwise CMD complains it can't find the file
        result = subprocess.run(
            " ".join(full_cmd),
            check=True,
            cwd=work_dir,
            shell=True,
            capture_output=True,
            text=True,
        )
    else:
        full_cmd = ["bash", "-c", " ".join(get_activate_command() + command)]
        print(full_cmd)
        result = subprocess.run(
            full_cmd,
            check=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )


def create_virtual_environment() -> None:
    if not venv_dir.exists():
        if is_conda_environment():
            print("Creating conda environment...")
            subprocess.run(
                [
                    "conda",
                    "create",
                    "--prefix",
                    str(venv_dir),
                    "python=3.11",
                    "pip",
                    "-y",
                ],
                check=True,
                cwd=work_dir,
            )
        else:
            # Regular venv creation
            python_executable = get_python_executable()
            subprocess.run(
                [python_executable, "-m", "venv", ".venv"],
                check=True,
                cwd=work_dir,
            )


def setup_virtual_environment() -> None:
    """Create and configure a virtual environment in a cross-platform manner."""
    print("Preparing virtual environment...")

    try:
        # Handle activation and package installation

        try:
            run_subprocess_in_venv(["pip", "install", "-U", "uv"])
            if is_datarobot_codespace():
                os.system("uv pip install $(pip freeze | grep ipykernel)")
            # Install requirements using uv
            run_subprocess_in_venv(["uv", "pip", "install", "-r", "requirements.txt"])
        except Exception as e:
            print(f"UV installation/usage failed: {e}")
            print("Falling back to pip")

            run_subprocess_in_venv(["pip", "install", "-r", "requirements.txt"])

    except subprocess.CalledProcessError as e:
        print(f"Error during virtual environment setup: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during virtual environment setup: {e}")
        raise


def load_dotenv():
    with open(".env") as f:
        content = f.read()

    pattern = r"""
        (?:^|\n)         # Must start at beginning of string or after newline
        (?![\s#])        # Negative lookahead: next char cannot be whitespace or #
        ([A-Za-z_]\w*)   # Key: start with letter/underscore, then word chars
        =\s*             # Equals with optional whitespace after
        (?:
            '(.*?)'      # Single quoted (group 2)
            |"(.*?)"     # Double quoted (group 3)
            |([^\n]+)    # Unquoted: everything until newline (group 4)
        )
    """

    env_vars = {}
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL | re.VERBOSE):
        key = match.group(1).strip()

        if match.group(2) is not None:  # Single quoted
            value = match.group(2).strip()
        elif match.group(3) is not None:  # Double quoted
            value = match.group(3).strip()
        else:  # Unquoted
            value = match.group(4)
            if " #" in value:  # Only strip comments after space
                value = value.split(" #", 1)[0]
            value = value.strip()

        env_vars[key] = value

    os.environ.update(env_vars)
    return env_vars


def run_pulumi_command(command: list, work_dir: Path, env_vars: dict):
    """Run a Pulumi command using shell activation with PTY support."""
    cmd_str = " ".join(command)
    try:
        if is_windows():
            os.system(f"{' '.join(get_activate_command())}{cmd_str}")
        else:
            os.system(f"bash -c '{' '.join(get_activate_command())}{cmd_str}'")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def setup_pulumi_config(work_dir: Path, stack_name: str, env_vars: dict):
    stack_select = ["pulumi", "stack", "select", stack_name, "--create"]

    # Run commands
    run_pulumi_command(stack_select, work_dir, env_vars)


def print_app_url():
    try:
        pulumi_output = subprocess.check_output(["pulumi", "stack", "output", "-j"])
        pulumi_output_dict = json.loads(pulumi_output)
        application_id = pulumi_output_dict["DATAROBOT_APPLICATION_ID"]
        datarobot_endpoint = os.environ["DATAROBOT_ENDPOINT"]
        url = f"{datarobot_endpoint.rstrip('/').replace('api/v2', '')}custom_applications/{application_id}/"
        print("\n\n")
        print("=" * 80)
        print(f"\n    Your app is ready! Application URL:\n\n    {url}\n")
        print("=" * 80)
    except Exception as e:
        print(e)


def main():
    args = parse_args()
    if args.stack_name == "YOUR_PROJECT_NAME":
        print("Please use a different project name")
        sys.exit(0)
    check_dotenv_exists()
    # Load environment variables
    env_vars = load_dotenv()

    check_pulumi_installed()
    check_pulumi_login()

    # Skip venv setup in Codespaces or if explicitly requested
    if not is_datarobot_codespace():
        create_virtual_environment()
    setup_virtual_environment()
    # Setup Pulumi configuration
    setup_pulumi_config(work_dir, args.stack_name, env_vars)

    # Refresh the stack
    print("\nRefreshing stack...")
    run_pulumi_command(
        ["pulumi", "refresh", "--yes", "--skip-pending-creates"], work_dir, env_vars
    )

    if args.action == "destroy":
        print("\nDestroying stack...")
        run_pulumi_command(["pulumi", "destroy", "--yes"], work_dir, env_vars)
        print("Stack destroy complete")
    else:
        print("\nCreate/update stack...")
        run_pulumi_command(["pulumi", "up", "--yes"], work_dir, env_vars)
        print("Stack update complete")

        print_app_url()


if __name__ == "__main__":
    main()
