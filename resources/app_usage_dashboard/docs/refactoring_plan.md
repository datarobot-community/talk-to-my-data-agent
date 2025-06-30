
# Codebase Refactoring and Simplification Plan

## 1. Overview

This document outlines a plan to refactor the application by restructuring and simplifying the codebase. The goal is to improve maintainability and readability by consolidating related logic into fewer, more focused modules, while keeping all existing functionality intact.

This plan is designed to be implemented by developers of all levels.

## 2. Core Principles

*   **Consolidate by Function:** Group related code into single files based on their core function (e.g., data management, visualization).
*   **Improve Readability:** Reduce the number of files and directories to make the project easier to navigate.
*   **Maintain Functionality:** Ensure that all existing features of the dashboard remain unchanged.
*   **Keep File Size Manageable:** Avoid creating overly large files. Each file should ideally be under 500 lines of code.

## 3. "As Is" File Structure

The current file structure is highly modular, with logic fragmented across many small files in the `modules/` directory:

```
.
├── app.py
├── modules/
│   ├── chart_generators.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── data_utils.py
│   ├── dataset_helper.py
│   ├── kpi_calculations.py
│   ├── log_helper.py
│   ├── ui_components.py
│   └── wordcloud_utils.py
├── ... (other files)
```

## 4. "To Be" File Structure

The refactoring will result in a flatter, more streamlined file structure:

```
.
├── app.py
├── data_manager.py
├── visualizations.py
├── kpi_calculations.py
├── config.py
├── i18n_setup.py
├── ... (other files)
```

## 5. Implementation Steps

### Step 1: Consolidate Data Management Logic

1.  **Create `data_manager.py`:** Create a new file named `data_manager.py` in the root directory.
2.  **Merge Data-Related Files:** Move and merge the contents of the following files into `data_manager.py`:
    *   `modules/data_pipeline.py`
    *   `modules/data_utils.py`
    *   `modules/dataset_helper.py`
3.  **Refactor and Clean Up:**
    *   Review the merged code in `data_manager.py` to identify and remove any redundant imports or functions.
    *   Ensure that all data-related logic (downloading, processing, caching) is self-contained within this file.
    *   Update any imports in other files that previously pointed to the old data modules.

### Step 2: Consolidate Visualization Logic

1.  **Create `visualizations.py`:** Create a new file named `visualizations.py` in the root directory.
2.  **Merge Visualization-Related Files:** Move and merge the contents of the following files into `visualizations.py`:
    *   `modules/chart_generators.py`
    *   `modules/wordcloud_utils.py`
3.  **Refactor and Clean Up:**
    *   Review the merged code in `visualizations.py` to ensure consistency.
    *   Update any imports in other files that previously pointed to the old visualization modules.

### Step 3: Consolidate KPI Calculations

1.  **Move `kpi_calculations.py`:** Move the `modules/kpi_calculations.py` file to the root directory.
2.  **Update Imports:** Update any imports that reference this file to reflect its new location.

### Step 4: Simplify UI Components

1.  **Move UI Logic to `app.py`:** Move the contents of `modules/ui_components.py` directly into `app.py`.
2.  **Integrate and Refactor:**
    *   Integrate the UI filter logic seamlessly within the main application flow in `app.py`.
    *   Remove any unnecessary functions or abstractions that were required for the separate module.

### Step 5: Clean Up

1.  **Remove the `modules/` Directory:** Once all the logic has been moved, delete the now-empty `modules/` directory.
2.  **Remove Unused Files:** Delete the `modules/log_helper.py` file, as it is not being used.
3.  **Final Review:**
    *   Run the application and test all functionality to ensure that the refactoring has not introduced any regressions.
    *   Review the updated file structure to confirm that it aligns with the plan.

## 6. Estimated Lines of Code

This refactoring plan is designed to keep the resulting files at a manageable size:

*   **`data_manager.py`**: ~460 lines
*   **`app.py`**: ~460 lines
*   **`visualizations.py`**: ~430 lines
*   **`kpi_calculations.py`**: ~130 lines

These estimates are based on the current line counts and may be reduced after removing redundant code during the merge process.
