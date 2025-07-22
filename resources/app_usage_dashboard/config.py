import json
import subprocess

from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources import parse_env_vars


class PulumiSettingsSource(EnvSettingsSource):
    """Pulumi stack outputs as a pydantic settings source."""

    _PULUMI_OUTPUTS = None

    def __init__(self, *args, **kwargs):
        self.read_pulumi_outputs()
        super().__init__(*args, **kwargs)

    def read_pulumi_outputs(self):
        try:
            raw_outputs = json.loads(
                subprocess.check_output(
                    ["pulumi", "stack", "output", "-j"], text=True
                ).strip()
            )
            self._PULUMI_OUTPUTS = {
                k: v if isinstance(v, str) else json.dumps(v)
                for k, v in raw_outputs.items()
            }
        except Exception:
            self._PULUMI_OUTPUTS = {}

    def _load_env_vars(self):
        return parse_env_vars(
            self._PULUMI_OUTPUTS,
            self.case_sensitive,
            self.env_ignore_empty,
            self.env_parse_none_str,
        )


class DynamicSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            PulumiSettingsSource(settings_cls),
            env_settings,
        )
