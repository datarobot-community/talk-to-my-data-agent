## Project Deployment

Run the following shell commands to deploy the project:

```shell
dr task run infra:up-yes
```

In case the deployment process fails, you can try deleting it by running the following command:

```shell
dr task run infra:down-yes
```

## Local Development

One-time setup (deploys backing infrastructure):

```shell
dr start
```

## Pre-deploy Checklist

Before running `dr task run infra:up-yes`, ask the user to ensure their environment is configured. The following variables must be set:

- `DATAROBOT_API_TOKEN` and `DATAROBOT_ENDPOINT`
- `PULUMI_CONFIG_PASSPHRASE`
- `SESSION_SECRET_KEY` (required if using the FastAPI component)
- `DATAROBOT_DEFAULT_EXECUTION_ENVIRONMENT` (required if your agent uses a custom execution environment)

If the user hasn't set up their environment yet, ask them to run:

```shell
dr start
```

or

```shell
dr dotenv setup
```

## Troubleshooting

**Deploy exits with code 1 but a Custom Application URL appears in the output**
The app is live — the non-zero exit came from a post-deploy cleanup step, not the app itself. Run the following to reconcile Pulumi state:

```shell
dr task run infra:refresh -- -y
```

Do not re-deploy.

**422 error when deleting ApplicationSource**
The source is still attached to a live application. Reconcile Pulumi state and retry:

```shell
dr task run infra:refresh -- -y
```

**Docker context error on first deploy**
Set `DATAROBOT_DEFAULT_EXECUTION_ENVIRONMENT` in your `.env` file to point to an existing execution environment.

**Container fails to install dependencies at startup**
If your app depends on a local package (e.g. `core/`), ensure it is included in the application bundle before deploying.

## Expected Deploy Times

| Operation | Duration |
|---|---|
| First deploy | ~10–15 min |
| Re-deploy | ~5–10 min |
| `task refresh` | ~1–2 min |
