# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.15] - 2025-10-03

### Added

- Token usage estimate tracking for LLM calls.
- Refactored message storage (analyst dataset) to prevent oversized chat API responses and failures.
- Integration with DataRobot Python client 3.9.1 `CustomApplication` and `CustomApplicationSource` entities.
- Dynamic resource fetching from CustomApplicationSource when creating CustomApplications.
- Fixed `RuntimeError: asyncio.run() cannot be called from a running event loop` in Streamlit frontend by applying `nest_asyncio` patch.

### Improvements

- Performance improvements in app's database.
  + Replace expensive + blocking check for updates with separate read/write connections.
  + Replaced synchronous persistence methods with asynchronous methods.

## Documentation

- Updated README to describe connecting to DataRobot data stores.

## [0.3.14] - 2025-10-03

### Fixes

- Reset logging level to INFO (accidentally set to DEBUG in prior release).
- Filled out previous changelog entry.

## [0.3.13] - 2025-10-03

### Features

- Add connection to DataRobot data stores (currently Postgres and Redshift).

### Improvements

- Move some expensive calls into background thread to avoid blocking main event loop.
- Updated chat UX to progressively load components of chat response.
- Only show warning when both TEXTGEN_REGISTERED_MODEL_ID and TEXTGEN_DEPLOYMENT_ID are set when using a deployed LLM.

## [0.3.12] - 2025-09-25

### Improvements

- Dataset search UX improvements. Search term highlighting, updated texts, reset when switching views
- Bump pulumi-datarobot to version 0.10.20

## [0.3.11] - 2025-09-22

### Improvements

- Add setup prerequisites to README
- Update localization assets

### Bug Fixes

- Fix streamlit local dev path issues, add new `make run-local-streamlit` helper
- Fix issues while reloading app in browser
- Fix streamlit e2e tests
- Restore e2e test cleanup
- Fixes and improvements for spark recipes

## [0.3.10] - 2025-09-12

### Bug Fixes

- Register remote datasets before data is fetched so they appear in UI in loading.
- Fixed an issue with spark recipes being created with invalid datasets.
- Corrected API version needed for spark recipes.
- Remove the remote data registry UI in cases where the DR version is too old.

## [0.3.9] - 2025-09-10

### Features

- Add functionality to analyze large datasets with DataRobot's data wrangling platform.

### Improvements

- Filter Data Registry download to datasets eligible to be downloaded.

## [0.3.8] - 2025-08-21

### Docs

- Add missing changelog notes for version 0.3.7

### Improvements

- Add frontend install and build commands to the makefile

## [0.3.7] - 2025-08-21

### Improvements

- Increased in-progress message polling speed for better user experience.
- Localized app chat responses to match user's language preference.
- Frontend assets are now automatically built during `pulumi up`, simplifying the overall deployment process.

### Bug Fixes

- Fixed empty screen issue that could occur during message loading.
- Fixed the issue with fetching the API token on STS and on-prem.

## [0.3.6] - 2025-08-12

### Features

- Allow adding BOM when exporting and fix dictionary exports containing Japanese characters in name.
- Performance improvements: Cache DataRobot client and Deployment ID.

### Bug Fixes

- Message deletion fixes.

### Improvements

- Hide welcome modal on close click.
- Updated README with local React development instructions.

## [0.3.5] - 2025-08-07

### Added

- Export individual chat message (question-answer) functionality. That includes underlying data, charts, summary and insights.
- Integration of LLM Gateway with Model Deployments

### Changed

- Updated welcome modal
- Disabled clicking follow-up suggestions while answering is in progress; replaced icon with button

### Fixed

- Improved overall exporting experience
- Better visual feedback when something fails
- UX improvements for chat messages: removed excessive auto-scrolling and enhanced the Send button

## [0.3.4] - 2025-08-01

### Added

- Allow use of DataRobot LLM Gateway instead of DataRobot-hosted pre-built LLM (https://docs.datarobot.com/en/docs/gen-ai/genai-code/dr-llm-gateway.html)

## [0.3.3] - 2025-07-29

### Changed

- Fix Snowflake connector issue by upgrading pulumi-datarobot to 0.10.13

## [0.3.2] - 2025-07-24

### Added

- Implemented search control for each dataset dictionary

### Fixed

- Updates translation files after fixing automation

## [0.3.1] - 2025-07-11

### Changed

- Fix prompt column issue by upgrading pulumi-datarobot to 0.10.8

## [0.3.0] - 2025-07-08

### Changed

- Add chat and dataset deletion safeguards
- Add chat time in name and add confirm modal for deletion

## [0.2.2] - 2025-07-01

### Fixed

- Fix chat conversation context for React frontend apps

## [0.2.1] - 2025-06-25

### Fixed

- Set LLM Gateway Inference runtime parameter to False always for user-provided credentials

## [0.2.00] - 2025-06-16

### Fixed

- Support separate google creds for vertexAI and BQ
- New google model name in credentials check
- Data dictionary generation timeout
- Data dictionary generation didn't return partial results
- Analyst dataset incorrectly inferred schema at read time
- Fix rare DR Catalog ingest issue

### Changed

- Renamed "Save chat" to "Export chat"

### Added

- Persistent storage functionality

## [0.1.14] - 2025-05-30

### Fixed

- Data cleansing now automatically removes leading and trailing whitespace from string columns
- Fixed chat endpoint when DATAROBOT_ENDPOINT has a trailing slash
- Fixed the raw data preview for SAP (react)

### Added

- Ability to download a specific chat history (including charts) in the React version.

### Changed

- React-based Frontend as the default for the Application
- Improved error handling when prompting (react)
- Change react frontend `deploy` app to use AF fastapi template
- Change react frontend to use AF react template, removes `frontend_react` in favor of `app_frontend`

## [0.1.13] - 2025-05-06

### Fixed

- Improved Registry / File / Database toggle functionality (react)
- React App now accepts local xlsx files (react)
- Fixed missing @/lib/utils import (react)

### Added

- Ability to delete individual chat messages (streamlit)

### Changed

- Improved logging (react)

## [0.1.12] - 2025-04-30

### Changed

- Pinned streamlit to version 1.44.1 to improve stability

## [0.1.11] - 2025-04-29

### Fixed

- Fixed AI Catalog dropdown for long dataset names
- Fixed horizontal scrolling in the raw rows view
- Fixed issue with certain CSV file uploads and improved error messaging for failed uploads
- Fixed scroll area for collapsible panel with Code preview
- Fixed issue with composing message for complex character sets (Japanese/Chinese/Korean)
- Fixed issue with constantly creating new db file for each request in some cases (rest_api)

## [0.1.10] - 2025-04-17

### Fixed

- Fixed a bug where snowflake ssh keyfile was not working correctly
- Fixed the incorrect max_completion_length validation in the LLMSettings schema

### Added

- Alternative React frontend
- Allow users to use the app without an API token when it is externally shared

### Changed

- Changed the LLMSettings schema for LLM proxy settings instead of Pulumi class

## [0.1.9] - 2025-04-07

### Fixed

- Code generation inflection logic didn't quote the last but first error at retry
- Python generation prompt referred to SQL
- Fixed error when user doesn't have a last name

### Added

- Added test suite for each supported Python version

### Changed

- Installed [the datarobot-pulumi-utils library](https://github.com/datarobot-oss/datarobot-pulumi-utils) to incorporate majority of reused logic in the `infra.*` subpackages.
- Snowflake prompt more robust to lower case table and column names
- More robust code generation

## [0.1.8] - 2025-03-27

### Added

- Support for NIMs
- Support for existing TextGen deployments
- SAP Datasphere support

### Fixed

- AI Catalog and Database caching
- Fix StreamlitDuplicateElementKey error

### Changed

- Disabled session affinity for application
- Made REST API endpoints OpenAPI compliant
- Better DR token handling
- Changed AI Catalog to Data Registry

## [0.1.7] - 2025-03-07

### Added

- Shared app will use the user's API key if available to query the data catalog
- Polars added for faster big data processing
- Duck Db integration
- Datasets will be remembered as long as the session is active (the app did not restart)
- Chat sessions will be remembered as long as the session is active (the app did not restart)
- Added a button to clear the chat history
- Added a button to clear the data
- Added the ability to pick datasets used during the analysis step
- radio button to switch between snowflake mode and python mode

### Fixed

- Memory usage cut by ~50%
- Some JSON encoding errors during the analysis steps
- Snowflake bug when table name included non-uppercase characters
- pandas to polars conversion error when pandas.period is involved
- data dictionary generation was confusing the LLM on snowflake

### Changed

- More consistent logging
- use st.navigation

## [0.1.6] - 2025-02-18

### Fixed

- remove information about tools from prompt if there are none
- tools-related error fixed
- remove hard-coded environment ID from LLM deployment

## [0.1.5] - 2025-02-12

### Added

- LLM tool use support
- Checkboxes allow changing conversation
- DATABASE_CONNECTION_TYPE can be set from environment

### Fixed

- Fix issue where plotly charts reuse the same key
- Fix [Clear Data] button
- Fix logo rendering on first load
- Fix Data Dictionary editing

## [0.1.4] - 2025-02-03

### Changed

- Better cleansing report, showing more information
- Better memory usage, reducing memory footprint by up to 80%
- LLM is set to GPT 4o (4o mini can struggle with code generation)

## [0.1.3] - 2025-01-30

### Added

- Errors are displayed consistently in the app
- Invalid generated code is displayed on error

### Changed

- Additional modules provided to the code execution function
- Improved date parsing
- Default to GPT 4o mini to be compatible with the trial

## [0.1.2] - 2025-01-29

### Changed

- asyncio based frontend
- general clean-up of the interface
- pandas based analysis dataset
- additional tests
- unified renderer for analysis frontend

## [0.1.1] - 2025-01-24

### Added

- Initial functioning version of Pulumi template for data analyst
- Changelog file to keep track of changes in the project.
- pytest for api functions
