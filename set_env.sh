#!/bin/bash

PULUMI_MANAGED_VENV="runtime:  name: python  options:    toolchain: pip    virtualenv: .venv"

MANUAL_DEP_INSTRUCTIONS="Please comment out the following three lines 'Pulumi.yaml' to enable manual dependency management with Pulumi:
  # options:
  #   toolchain: pip
  #   virtualenv: .venv

Then install dependencies manually (make sure to address any conflicts pip identifies):
  pip install -r requirements.txt"

# Function to check if script is sourced
check_if_sourced() {
    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        echo "Please run this script using 'source' or '.' like this:"
        echo "  source ${0}"
        echo "  . ${0}"
        return 1
    fi
}

# Function to check for DataRobot Codespaces environment
check_datarobot_environment() {
    if [[ -n "${DATAROBOT_NOTEBOOK_IMAGE}" ]]; then
        if tr -d '\n' < Pulumi.yaml | grep -Fq "$PULUMI_MANAGED_VENV"; then
            echo "DR Codespaces requires manual management of dependencies."
            echo "$MANUAL_DEP_INSTRUCTIONS"
            return 1
        fi
    fi
}

# Function to check for active conda environment
check_active_conda_env() {
    if [[ -n "${CONDA_DEFAULT_ENV}" ]]; then
        if tr -d '\n' < Pulumi.yaml | grep -Fq "$PULUMI_MANAGED_VENV"; then
            echo "Using Pulumi with conda requires manual management of dependencies."
            echo "$MANUAL_DEP_INSTRUCTIONS"
            return 1
        fi
    fi
}


# Function to load .env file
load_env_file() {
    if [ ! -f .env ]; then
        echo "Error: .env file not found."
        echo "Please create a .env file with VAR_NAME=value pairs."
        return 1
    fi
    set -a
    source .env
    set +a
    echo "Environment variables from .env have been set."
}

# Function to activate the virtual environment if it exists
activate_virtual_environment() {
    if [ -f .venv/bin/activate ]; then
        echo "Activated virtual environment found at .venv/bin/activate"
        source .venv/bin/activate
    fi
}

# Main execution
if check_if_sourced ; then
    load_env_file
    activate_virtual_environment
    check_datarobot_environment
    check_active_conda_env
fi
