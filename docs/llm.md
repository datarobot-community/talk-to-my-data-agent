# LLM component

The LLM component provides the language model integration for your application. It supports multiple ways to connect an LLM.

During project setup (`dr start` or `dr dotenv setup`), the CLI prompts you to choose one of six LLM integration options. Each option creates different DataRobot resources and requires different configuration.

| Option | Best for | Deploys resources? | Requires credentials? |
|---|---|---|---|
| [LLM Gateway](#llm-gateway) | Getting started quickly | No | No |
| [DataRobot Deployed LLM](#datarobot-deployed-llm) | Using an existing deployment | Yes (Playground only) | No |
| [DataRobot NIM Deployed LLM](#datarobot-nim-deployed-llm) | Using an existing NVIDIA NIM deployment | Yes (Playground only) | No |
| [External LLM](#external-llm) | Bringing your own provider (Azure, Bedrock, etc.) | Yes | Yes |
| [LLM Blueprint with LLM Gateway](#llm-blueprint-with-llm-gateway) | Most production controls, multiple LLMs via one deployment | Yes | No |
| [LLM from a Registered Model](#llm-from-a-registered-model) | Deploying a registered model (e.g. NVIDIA NIM) | Yes | No |

## LLM Gateway

The simplest setup. Uses DataRobot's managed LLM Gateway directly with no custom model deployment.

### Resources created

This option deploys no DataRobot resources.

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `<LLM>_DEFAULT_MODEL` | Yes | `datarobot/azure/gpt-5-mini-2025-08-07` | Model ID from the LLM Gateway catalog |

To list available models:

```sh
dr get-llms
```

## DataRobot Deployed LLM

Use this option when you already have a custom model deployed LLM and a deployment ID. The component pulls it into the playground and use case.

### Resources created

| Resource | Type | Description |
|---|---|---|
| LLM Playground | `datarobot.Playground` | Playground linked to the use case |

The component references the existing deployment and its prediction environment.

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `<LLM>_DEPLOYMENT_ID` | Yes | -- | Deployment ID of the existing LLM (e.g. `6510c7b7c4f3f9407e24a849`) |
| `<LLM>_DEFAULT_MODEL` | No | `datarobot/datarobot-deployed-llm` | Model label. The deployment routes by its ID, so this is inert; set it to the real model name if you want datarobot-genai to match provider-specific reasoning parameters. |

**Note:** The deployment ID variable was formerly named `TEXTGEN_DEPLOYMENT_ID`. Use `<LLM>_DEPLOYMENT_ID` in current templates.

### Stack outputs

Surfaced by `task infra:info` or `pulumi stack output`:

| Output | Description |
|---|---|
| `Deployment ID [LLM_APP_NAME]` | ID of the referenced deployment |

## DataRobot NIM Deployed LLM

Use this option when you already have an NVIDIA NIM model deployed on DataRobot. The component pulls the deployment into the playground and use case. You must provision and keep the NIM deployment running yourself.

### Resources created

| Resource | Type | Description |
|---|---|---|
| LLM Playground | `datarobot.Playground` | Playground linked to the use case |

The component references the existing NIM deployment and its prediction environment.

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NIM_DEPLOYMENT_ID` (or `<LLM>_NIM_DEPLOYMENT_ID`) | Yes | -- | Deployment ID of the existing NIM LLM. Exported as both `NIM_DEPLOYMENT_ID` and `<LLM>_NIM_DEPLOYMENT_ID` so it resolves for datarobot-genai and for `Config` alike. |
| `<LLM>_DEFAULT_MODEL` | Yes | `datarobot/datarobot-deployed-llm` | Model your NIM serves (e.g. `meta-llama/Llama-3.1-8B`); stored `datarobot/`-prefixed. The placeholder is only a last-resort fallback. |

### Stack outputs

Surfaced by `task infra:info` or `pulumi stack output`:

| Output | Description |
|---|---|
| `Deployment ID [LLM_APP_NAME]` | ID of the referenced NIM deployment |
| `NIM_DEPLOYMENT_ID` | Same deployment ID (the name datarobot-genai reads to route to NIM) |
| `<LLM>_NIM_DEPLOYMENT_ID` | Same deployment ID, app-prefixed (the name `Config` reads) |
| `USE_DATAROBOT_LLM_GATEWAY` | `0` |

## External LLM

Use this option when you already have an LLM from Azure, Bedrock, Anthropic, Vertex, Cohere, or TogetherAI. You can monitor and scale your LLM with the added benefits of the DataRobot platform such as governance, guard models, controlled API access, and monitoring.

### Resources created

| Resource | Type | Description |
|---|---|---|
| LLM Playground | `datarobot.Playground` | Playground linked to the use case |
| LLM Blueprint | `datarobot.LlmBlueprint` | Blueprint configured with the external LLM |
| LLM Custom Model | `datarobot.CustomModel` | Text generation custom model sourced from the blueprint |
| Prediction Environment | `datarobot.PredictionEnvironment` | Serverless prediction environment (or existing if `DATAROBOT_DEFAULT_PREDICTION_ENVIRONMENT` is set) |
| Registered Model | `datarobot.RegisteredModel` | Registered model version for deployment |
| LLM Deployment | `datarobot.Deployment` | Serverless deployment with monitoring and data collection enabled |

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `<LLM>_DEFAULT_MODEL` | No | `datarobot/azure/gpt-5-mini` | External LLM model in `provider/model` form; stored `datarobot/`-prefixed |
| `<LLM>_DEFAULT_LLM_ID` | No | `azure-openai-gpt-5-mini` | DataRobot Playground LLM ID (a DataRobot identifier, not a LiteLLM model string) |
| `<LLM>_DEFAULT_LLM_NAME` | No | `Azure OpenAI GPT-5 Mini` | Friendly name shown in the UI |

You must also configure credentials for your chosen provider:

#### Azure OpenAI

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key |
| `OPENAI_API_BASE` | Base URL (e.g. `https://ENDPOINT.openai.azure.com`) |
| `OPENAI_API_DEPLOYMENT_ID` | Deployment ID (e.g. `gpt-5-mini`) |
| `OPENAI_API_VERSION` | API version (e.g. `2024-08-01-preview`) |

#### OpenAI-compatible (Nebius, Groq, xAI, DeepSeek, self-hosted vLLM, ...)

First, provide the model prefix for the preferred provider that LiteLLM can route (DataRobot supports all LiteLLM inferencing providers). Set `<LLM>_DEFAULT_MODEL` with that prefix (e.g., `nebius/moonshotai/Kimi-K2.7-Code`). **Do not** set `OPENAI_API_DEPLOYMENT_ID` — it is Azure-only and forces an Azure-style URL that non-Azure providers reject.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Provider API key |
| `OPENAI_API_BASE` | Provider base URL (e.g., `https://api.tokenfactory.us-central1.nebius.com/v1`) |

#### AWS Bedrock

| Variable | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `AWS_REGION_NAME` | AWS region (e.g. `us-east-1`) |
| `AWS_SESSION_TOKEN` | Optional. Session token for temporary AWS credentials |

#### Google VertexAI

| Variable | Description |
|---|---|
| `VERTEXAI_APPLICATION_CREDENTIALS` | Path to credentials JSON file |
| `VERTEXAI_SERVICE_ACCOUNT` | Google service account email |
| `GOOGLE_REGION` | Optional. Region for Vertex AI calls; defaults to `us-west1` |

#### Anthropic

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | API key |

#### Cohere

| Variable | Description |
|---|---|
| `COHERE_API_KEY` | API key |

#### TogetherAI

| Variable | Description |
|---|---|
| `TOGETHERAI_API_KEY` | API key |

**Note**: `blueprint_with_external_llm.py` smoke-tests the provider directly by stripping the `datarobot/` prefix from `<LLM>_DEFAULT_MODEL` (e.g., `azure/gpt-5-mini`, `nebius/...`, `bedrock/...`). For Azure, it addresses the model by its deployment name via `OPENAI_API_DEPLOYMENT_ID`; every other OpenAI-compatible provider is addressed by the model string as-is (no deployment ID). Set `<LLM>_DEFAULT_MODEL` to match your provider. See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for the exact model string each provider expects.

### Stack outputs

Surfaced by `task infra:info` or `pulumi stack output`:

| Output | Description |
|---|---|
| `Deployment ID [LLM_APP_NAME]` | ID of the deployed LLM |
| `Deployment Console [LLM_APP_NAME]` | URL to the Deployment Console page |
| `RAG Playground URL [LLM_APP_NAME]` | URL to the Playground comparison chat |

## LLM Blueprint with LLM Gateway

The most flexible option with the most production controls. Uses the LLM Blueprint and LLM Gateway options to enable multiple LLMs through a single deployment with all of the DataRobot governance and monitoring.

### Resources created

| Resource | Type | Description |
|---|---|---|
| LLM Playground | `datarobot.Playground` | Playground linked to the use case |
| LLM Blueprint | `datarobot.LlmBlueprint` | Blueprint configured with the LLM Gateway model |
| LLM Custom Model | `datarobot.CustomModel` | Text generation custom model sourced from the blueprint |
| Prediction Environment | `datarobot.PredictionEnvironment` | Serverless prediction environment (or existing if `DATAROBOT_DEFAULT_PREDICTION_ENVIRONMENT` is set) |
| Registered Model | `datarobot.RegisteredModel` | Registered model version for deployment |
| LLM Deployment | `datarobot.Deployment` | Serverless deployment with monitoring and data collection enabled |

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `<LLM>_DEFAULT_MODEL` | Yes | `datarobot/azure/gpt-5-mini-2025-08-07` | Model ID from the LLM Gateway catalog |
| `<LLM>_DEFAULT_LLM_ID` | No | `azure-openai-gpt-5-mini` | LLM ID used in the Playground |

### Stack outputs

Surfaced by `task infra:info` or `pulumi stack output`:

| Output | Description |
|---|---|
| `Deployment ID [LLM_APP_NAME]` | ID of the deployed LLM |
| `Deployment Console [LLM_APP_NAME]` | URL to the Deployment Console page |
| `RAG Playground URL [LLM_APP_NAME]` | URL to the Playground comparison chat |

## LLM from a Registered Model

Use this option when you have an existing registered model (not yet deployed) that you want to deploy with an LLM Blueprint. This is the path for NVIDIA NIM models: pick a model from the NVIDIA gallery, specify the registered model ID, and this option deploys it, creates an LLM Blueprint around it, then connects it to the application.

This option creates two deployments:

1. **Proxy deployment** -- deploys the registered model so it can be validated.
2. **Blueprint deployment** -- creates an LLM Blueprint from the validated model, builds a new custom model from that blueprint, and deploys it with full monitoring.

### Resources created

| Resource | Type | Description |
|---|---|---|
| LLM Playground | `datarobot.Playground` | Playground linked to the use case |
| Proxy Deployment | `datarobot.Deployment` | Initial deployment of the registered model for validation |
| LLM Validation | `datarobot.CustomModelLlmValidation` | Validates the deployed model can serve as an LLM |
| LLM Blueprint | `datarobot.LlmBlueprint` | Blueprint created from the validated custom model LLM |
| LLM Custom Model | `datarobot.CustomModel` | Text generation custom model sourced from the blueprint |
| Prediction Environment | `datarobot.PredictionEnvironment` | Serverless prediction environment (or existing if `DATAROBOT_DEFAULT_PREDICTION_ENVIRONMENT` is set) |
| Registered Model | `datarobot.RegisteredModel` | New registered model from the blueprint custom model |
| LLM Deployment | `datarobot.Deployment` | Final deployment with monitoring and data collection enabled |

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TEXTGEN_REGISTERED_MODEL_ID` | Yes | -- | ID of the registered model |
| `<LLM>_DEFAULT_MODEL` | No | `datarobot/datarobot-deployed-llm` | Model label. The blueprint deployment routes by its ID, so this is inert; set it to the real model name for reasoning-parameter matching. |
| `DATAROBOT_TIMEOUT_MINUTES` | No | `30` | Timeout in minutes for DataRobot operations. Increase for models that require GPU allocations |

### Stack outputs

Surfaced by `task infra:info` or `pulumi stack output`:

| Output | Description |
|---|---|
| `Deployment ID [LLM_APP_NAME]` | ID of the governed blueprint deployment the app calls (matches `<LLM>_DEPLOYMENT_ID`) |

## Switching between options

### During project setup

Run `dr start` or `dr dotenv setup` and select the desired option from the interactive prompt.

### Using the symlink

```sh
ln -sf ../configurations/<llm_app_name>/CONFIG_FILE infra/infra/<llm_app_name>.py
```

Available configuration files:

| File | Option |
|---|---|
| `gateway_direct.py` | LLM Gateway |
| `deployed_llm.py` | DataRobot Deployed LLM |
| `nim_deployed_llm.py` | DataRobot NIM Deployed LLM |
| `blueprint_with_external_llm.py` | External LLM |
| `blueprint_with_llm_gateway.py` | LLM Blueprint with LLM Gateway |
| `registered_model.py` | LLM from a Registered Model |

### Using the environment variable

Set `INFRA_ENABLE_LLM` in your `.env` file. Choose from the available options in the `infra/configurations/<llm_app_name>` folder.

#### LLM Gateway (default)

```sh
INFRA_ENABLE_LLM=gateway_direct.py
```

#### DataRobot Deployed LLM

```sh
LLM_DEPLOYMENT_ID=<your_deployment_id>
INFRA_ENABLE_LLM=deployed_llm.py
```

When you select DataRobot Deployed LLM during `dr start` (or `dr dotenv setup`), the template sets `USE_DATAROBOT_LLM_GATEWAY=0` automatically so the agent calls your deployment directly instead of routing through the LLM Gateway. You do not need to set `USE_DATAROBOT_LLM_GATEWAY` manually for this option.

#### DataRobot NIM Deployed LLM

```sh
NIM_DEPLOYMENT_ID=<your_nim_deployment_id>
LLM_DEFAULT_MODEL=<your_nim_model_id>
INFRA_ENABLE_LLM=nim_deployed_llm.py
```

#### External LLM (Azure OpenAI example)

```sh
INFRA_ENABLE_LLM=blueprint_with_external_llm.py
LLM_DEFAULT_MODEL="azure/gpt-5-mini-2025-08-07"
OPENAI_API_VERSION='2024-08-01-preview'
OPENAI_API_BASE='https://<your_custom_endpoint>.openai.azure.com'
OPENAI_API_DEPLOYMENT_ID='<your deployment_id>'
OPENAI_API_KEY='<your_api_key>'
```

#### LLM Blueprint with LLM Gateway

```sh
INFRA_ENABLE_LLM=blueprint_with_llm_gateway.py
```

#### LLM from a Registered Model

```sh
TEXTGEN_REGISTERED_MODEL_ID=<your_registered_model_id>
INFRA_ENABLE_LLM=registered_model.py
```

### Editing the configuration directly

In addition to the `.env` file changes, you can also edit the respective configuration file to make additional changes, such as the default LLM, temperature, top_p, etc. The feature-flag, credential, and LLM-verification helpers each configuration module imports come from the [`datarobot-pulumi-utils`](https://github.com/datarobot-oss/datarobot-pulumi-utils) package rather than a generated module in your project.

## Common configuration

All options that deploy resources share these behaviors:

- **Prediction environment** -- if `DATAROBOT_DEFAULT_PREDICTION_ENVIRONMENT` is set, the component uses that existing environment; otherwise, it creates a new serverless environment.
- **Scaling** -- deployments default to `min_computes=0` and `max_computes=2`.
- **Data collection** -- blueprint, gateway, and registered model deployments enable prediction data collection.
- **Association IDs** -- deployments use `association_id` for tracking predictions.

### Required feature flags

All options require these DataRobot feature flags to be enabled:

- `ENABLE_MLOPS`
- `ENABLE_CUSTOM_INFERENCE_MODEL`
- `ENABLE_PUBLIC_NETWORK_ACCESS_FOR_ALL_CUSTOM_MODELS`
- `ENABLE_MLOPS_TEXT_GENERATION_TARGET_TYPE`

LLM Gateway additionally requires:

- `ENABLE_MLOPS_RESOURCE_REQUEST_BUNDLES`

## Variable naming

In the tables above, `<LLM>` is a placeholder for your LLM app name in uppercase (e.g. if your app name is `llm`, variables are prefixed with `LLM_`). This is set by the `llm_app_name` template variable during project setup.

`USE_DATAROBOT_LLM_GATEWAY` tells downstream consumers (e.g. the agent) whether to route LLM calls through the DataRobot LLM Gateway. datarobot-genai defaults it to `True` (gateway) when unset, so every option now sets it explicitly: `1` for the gateway-based options (LLM Gateway, LLM Blueprint with LLM Gateway) and `0` for the deployment-based options (DataRobot Deployed LLM, DataRobot NIM Deployed LLM, External LLM, LLM from a Registered Model). Setting `0` is what makes those options route to their own deployment instead of the gateway.

## Runtime configuration and datarobot-genai

After `pulumi up`, uppercase stack exports (for example `<LLM>_DEPLOYMENT_ID`, `USE_DATAROBOT_LLM_GATEWAY`, `<LLM>_DEFAULT_MODEL`) are written for runtime use. In an App Framework recipe project, these values are typically present in the project `.env` after deploy.

[`datarobot-genai`](https://github.com/datarobot-oss/datarobot-genai) `get_llm()` reads this environment to route calls to the LLM Gateway, a DataRobot deployment, NIM, or an external provider. Ensure the exports for your chosen option are set before starting application or agent code.

The `af-component-llm` component repository runs maintainer end-to-end smoke tests against live endpoints after each configuration deploys. See that repository's README for local and CI setup.

## Further reading

- [Playground overview](https://docs.datarobot.com/en/docs/agentic-ai/playground-tools/playground-overview.html) -- what a Playground is and how LLM blueprints fit in.
- [Build LLM blueprints](https://docs.datarobot.com/en/docs/agentic-ai/playground-tools/build-llm-blueprints.html) -- LLM blueprint settings (base LLM, prompting, vector database).
- [Deploy an LLM](https://docs.datarobot.com/en/docs/agentic-ai/playground-tools/deploy-llm.html) -- deploying an LLM blueprint for production use.
- [LLM gateway model configuration](https://docs.datarobot.com/en/docs/reference/gen-ai-ref/llm-gateway-config.html) -- admin guide to provisioning provider credentials for the LLM gateway.
- [LiteLLM providers](https://docs.litellm.ai/docs/providers) -- reference for the model-string prefixes used by the `verify_llm` check.
