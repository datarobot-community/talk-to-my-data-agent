# Base component

The base component is required for every App Framework application template. It provides two things: the baseline Infrastructure as Code (IaC) scaffolding powered by [Pulumi](https://www.pulumi.com/) that all components depend on, and an optional `core` Python library for code you want to share across multiple components in the same app framework monorepo project.

## Overview

| Sub-component | Optional | Description |
|---|---|---|
| **Infrastructure** | No | Pulumi project, task runner, and CI/CD scaffolding. |
| **Core library** | Yes | Shared Python package included when your recipe has more than one component that needs common utilities. |

## Infrastructure (IaC)

The base component lays down the Pulumi project that every other component extends. When you apply additional components (FastAPI backend, agent, etc.), they each register their own Pulumi resources into the same stack. It also creates the DataRobot Use Case for all other components to use for organization and scaffolds out the feature flag verification system.

### Infra folder structure

```
infra/
├── __main__.py          # Entrypoint — auto-discovers all infra modules
├── infra/               # Per-component resource modules
│   └── ...
├── configurations/      # Swappable configuration modules (feature flags, etc.)
│   └── ...
├── feature_flags/       # Feature flag definitions
└── Pulumi.yaml
```

The `__main__.py` entrypoint auto-discovers and loads all modules inside `infra/infra/`. You add resources to the stack by placing Python modules in that directory&mdash;no manual registration required.

### Configuration toggling

The `configurations/` folder holds swappable modules that you activate at deploy time using environment variables:

```
INFRA_ENABLE_<FOLDER>=<filename>
```

For example, to activate `configurations/llm/external_llm.py` as the active LLM configuration:

```
INFRA_ENABLE_LLM=external_llm.py
```

### Common tasks

Run infrastructure tasks from the `infra/` directory using [Task](https://taskfile.dev/):

| Task | Description |
|---|---|
| `task infra:deploy` | Deploy the full Pulumi stack. |
| `task infra:destroy` | Tear down all provisioned resources. |
| `task infra:auth-check` | Validate DataRobot credentials before deploying. |

## Core library

The `core` library is an optional Python package that is included when you answer `yes` to the `include_core` prompt during template application. Use it to house utilities and business logic that are shared across multiple components in the same recipe&mdash;for example, between a `fastapi-backend` and an `agent` component.

The library lives at `core/` in the root of the generated repository and is installed as a local editable dependency in any component that needs it.

### When to include core

Include `core` if your recipe has two or more components that share:

- Database access or persistent state.
- Auth helpers or SDK client setup.
- Domain models or business logic.
- Any utility you'd otherwise copy-paste between components.

If your recipe has only one component, you can skip `core` and put shared code directly in that component.

### Persistent file system

DataRobot Applications are stateless by default. The `core.persistent_fs` module provides a persistent file system API backed by the DataRobot Key-Value store, giving your app durable storage without an external database.

**`DRFileSystem`**&mdash;an [fsspec](https://filesystem-spec.readthedocs.io/)-compatible file system that reads and writes files to the DataRobot file storage API. It transparently syncs a local cache with remote storage, so you can use it with any library that accepts an fsspec file system.

```python
from core.persistent_fs.dr_file_system import DRFileSystem

fs = DRFileSystem()
with fs.open("my-data/config.json", "w") as f:
    f.write('{"key": "value"}')
```

When `APPLICATION_ID` is not set in the environment (i.e., running locally), `DRFileSystem` skips remote sync so local development works without credentials.

### Persistent SQLite

`core.persistent_fs.sqlite_extension` provides a drop-in replacement for `aiosqlite.connect` that automatically persists your SQLite database file to the DataRobot file storage API on open and close.

```python
from core.persistent_fs.sqlite_extension import connect_dr_fs

async with connect_dr_fs("app.db") as db:
    await db.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY)")
    await db.commit()
```

On open, the file is pulled from remote storage if it exists. On close, the file is pushed back only if its checksum changed&mdash;avoiding unnecessary uploads on read-only sessions.

### Persistent DuckDB

`core.persistent_fs.duckdb_extension` provides the same pattern for [DuckDB](https://duckdb.org/):

```python
from core.persistent_fs.duckdb_extension import connect_dr_fs

with connect_dr_fs("analytics.duckdb") as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS events AS SELECT 1 AS id")
```

DuckDB connections also register the `DRFileSystem` as a native fsspec filesystem, so you can query remote files directly using the `dr://` protocol prefix.

### Read/write lock

`core.utils.rw_lock` provides thread-safe and async-safe read/write lock implementations for protecting shared state in FastAPI apps that mix background tasks with request handlers.

```python
from core.utils.rw_lock import ThreadReadWriteLock

lock = ThreadReadWriteLock()

# Multiple concurrent readers are allowed.
with lock.read_lock():
    data = read_shared_state()

# Writers get exclusive access.
with lock.write_lock():
    update_shared_state()
```

Both `ThreadReadWriteLock` (for `threading`-based concurrency) and an async variant are available. Writers block new readers from acquiring the lock while waiting, preventing write starvation.

### Adding shared code

Add new modules directly under `core/src/core/`. Any component in the recipe can import from `core` once it declares `core` as a dependency in its own `pyproject.toml`.

```
core/src/core/
├── auth/           # DataRobot auth helpers
├── persistent_fs/  # Persistent file system, SQLite, and DuckDB drivers
└── utils/          # Read/write lock and other utilities
```

## Configuration

The base component records your answers in `.datarobot/answers/base.yml`. These answers drive future automated updates from the component system.

| Answer key | Description |
|---|---|
| `template_name` | Human-readable name for the application template. |
| `template_code_name` | Code-friendly slug derived from `template_name`. |
| `template_description` | Markdown-compatible description used in generated documentation. |
| `copyright_year` | Year embedded in license headers. |
| `include_core` | Whether to include the `core` shared library. |

## Prerequisites

- [Pulumi CLI](https://www.pulumi.com/docs/install/) installed and authenticated.
- [Task](https://taskfile.dev/#/installation) installed.
- [uv](https://docs.astral.sh/uv/) installed (used for Python dependency management).
- A DataRobot account with an API token set in `DATAROBOT_API_TOKEN`.

## Best practices

- **Put only shared code in `core`**&mdash;Code that is only used by one component belongs in that component. `core` is for genuinely shared utilities.
- **Use the persistent drivers in production**&mdash;`DRFileSystem`, `connect_dr_fs` (SQLite), and `connect_dr_fs` (DuckDB) all fall back gracefully to local-only mode when `APPLICATION_ID` is absent, so you can develop locally without any changes.
- **Protect shared state with the RW lock**&mdash;If multiple FastAPI background tasks or request handlers access the same in-memory state, use `ThreadReadWriteLock` to avoid race conditions.
- **Keep `configurations/` clean**&mdash;Only place swappable infrastructure modules in `configurations/`. Fixed resources that are always deployed belong directly in `infra/infra/`.
