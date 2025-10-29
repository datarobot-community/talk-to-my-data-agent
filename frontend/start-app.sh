#!/usr/bin/env bash
#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#

# Disable telemetry for local streamlit runs (in the production deployment, the platform will launch the app without this script)
export DISABLE_TELEMETRY=true

echo "Starting App"
streamlit run 'app.py' --server.maxUploadSize 200
