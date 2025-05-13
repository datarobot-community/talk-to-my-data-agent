# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v0.1.7-release.11.0.1a] - 2025-05-13

### Fixed

- Use urljoin to ensure robust endpoints

## [v0.1.7-release.11.0.1] - 2025-05-06

### Fixed 

- Fixed error when user doesn't have a last name

### Changed

- Pinned streamlit to version 1.44.1 to improve stability

## [v0.1.7-release.11.0] - 2025-04-02

### Added
- Support for NIMs
- Support for existing TextGen deployments
- SAP Datasphere support

### Fixed
- Less aggressive caching for DB and AI Catalog
- Page refresh errors causing application state loss

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

