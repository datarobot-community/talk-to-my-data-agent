# Airgapped Execution Environment

To run the application in an environment without access to download and install dependencies (an airgapped environment without a package repository), bundle those dependencies into an execution environment.

The environment is built with the [`dr-environment`](https://github.com/datarobot-oss/dr-environment) plugin for the [DataRobot CLI](https://github.com/datarobot-oss/cli). It reads this repository as an App Framework recipe (root `Taskfile.yaml` includes, `.datarobot/cli/versions.yaml`, and each component's lockfile) and produces a Docker context for a Wolfi-based image with pre-warmed `uv`, `npm` and Go caches. The image runs strictly offline (`UV_OFFLINE=1`, `NPM_CONFIG_OFFLINE=true`, `GOPROXY=off`), so `task install` and `uv sync` resolve from the baked caches.

## Build

Run these steps in an environment with an internet connection (or at least access to a Docker registry and Python/npm repositories). Prerequisites: Docker, [uv](https://docs.astral.sh/uv/getting-started/installation/), and the `dr` CLI.

1. Install the plugin: `uv tool install dr-environment`.
2. From the repository root, generate the Docker context. All lockfiles must be up to date (`uv lock` / `npm install`), or generation fails:

   ```bash
   dr environment recipe --recipe-path .
   ```

   This writes `docker_context/` and `docker_context.tar.gz` (both git-ignored). The image's Python version (`python.version`) and the baked `pulumi-datarobot` and `pulumi-command` provider versions come from `.datarobot/cli/versions.yaml`. The `Versions Lock Check` workflow fails if a provider version doesn't match `infra/uv.lock`, or if the Python version doesn't satisfy every `uv.lock`. If a provider were missing from the image, `task deploy` would try to download it from GitHub.

3. Build the image for **linux/amd64** (required, including on Apple Silicon) and export it:

   ```bash
   cd docker_context
   docker build --platform linux/amd64 -t talk-to-my-data-exec-env .
   docker image save talk-to-my-data-exec-env -o image.tar
   ```

4. Copy `image.tar` into the airgapped environment and upload it as a custom execution environment in DataRobot (Registry > Environments).

## Use

Create a Codespace or custom application using the uploaded environment, then set the following in `.env`:

```
APPLICATION_EXECUTION_ENVIRONMENT_VERSION_ID=
APPLICATION_EXECUTION_ENVIRONMENT_ID=
```

The image contains only dependency caches, not application code. Install dependencies in the Codespace with `task install`; they resolve offline from the baked caches. Because the caches are keyed to the lockfiles at build time, rebuild the environment whenever a lockfile changes.
