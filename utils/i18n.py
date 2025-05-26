# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gettext as gettext_module
import os
from enum import Enum
from gettext import GNUTranslations, NullTranslations
from typing import Union

from babel.messages import mofile, pofile
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class LanguageCode(str, Enum):
    EN = "en_US"
    JA = "ja_JP"
    ALL = {EN, JA}


# Locale for prompts and app front end
APP_LOCALE: LanguageCode = LanguageCode.JA

app_locale_env_name: str = "APP_LOCALE"


def compile_mo_from_po(locale_folder_path: str) -> None:
    """
    Compile a .po file to a .mo file.
    :param locale_folder_path: Path to the parent locale folder.
    """

    mo_file_path = os.path.join(locale_folder_path, "base.mo")
    po_file_path = os.path.join(locale_folder_path, "base.po")

    if not os.path.exists(po_file_path):
        raise ValueError(f"Invalid locale file: {po_file_path}")

    with open(po_file_path, "r", encoding="utf-8") as po_file:
        catalog = pofile.read_po(po_file)
    with open(mo_file_path, "wb") as mo_file:
        mofile.write_mo(mo_file, catalog)


class LocaleSettings(BaseSettings):
    """Establish locale settings based upon env"""

    app_locale: str = Field(
        validation_alias=AliasChoices(
            "MLOPS_RUNTIME_PARAM_" + app_locale_env_name,
            "MAIN_" + app_locale_env_name,
        ),
        default=APP_LOCALE,
    )

    def setup_locale(self) -> None:
        """Validate Locale code and assets"""
        application_locale = self.app_locale
        if application_locale not in LanguageCode.ALL:
            raise ValueError(f"Invalid locale: {application_locale}")
        if application_locale != LanguageCode.EN:
            locale_folder_path = os.path.join(
                self.get_locale_dir(), self.app_locale, "LC_MESSAGES"
            )
            if not os.path.exists(locale_folder_path):
                raise ValueError(f"Invalid locale path: {locale_folder_path}")
            compile_mo_from_po(locale_folder_path)

    def get_locale_dir(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(base_dir, "locale"))


def get_translation_ctx() -> Union[NullTranslations, GNUTranslations]:
    """Return a Translations instance based on the locale set in the environment"""
    if LocaleSettings().app_locale == LanguageCode.EN:
        return gettext_module.NullTranslations()
    else:
        return gettext_module.translation(
            "base",
            localedir=LocaleSettings().get_locale_dir(),
            languages=[LocaleSettings().app_locale],
            fallback=True,
        )


def gettext_noop(message: str) -> str:
    """
    no-op passthrough for deferred translations.
    """
    return message


def gettext(message: str) -> str:
    """
    Look up the context and message id in the catalog and return the corresponding message string,
    as a Unicode string.
    """
    return get_translation_ctx().gettext(message)