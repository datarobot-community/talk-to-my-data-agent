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

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

DOWNLOADS_FOLDER = Path(__file__).resolve().parent.absolute() / "downloads"


def find_element(
    browser: webdriver.Chrome, by: str, value: str, timeout: int = 10
) -> WebElement:
    """
    Find an element with a given locator.

    :param browser: WebDriver instance
    :param by: Locator type (e.g., By.ID, By.XPATH)
    :param value: Locator value
    :param timeout: Maximum time to wait for the element
    :return: WebElement if found, else raises TimeoutException
    """
    return WebDriverWait(browser, timeout).until(
        EC.presence_of_element_located((by, value))
    )


def click_element(
    browser: webdriver.Chrome, by: str, value: str, timeout: int = 10
) -> None:
    """
    Click an element with a given locator.

    :param browser: WebDriver instance
    :param by: Locator type (e.g., By.ID, By.XPATH)
    :param value: Locator value
    :param timeout: Maximum time to wait for the element
    """
    wait_for_element_to_be_clickable(browser, by, value)
    element = find_element(browser, by, value, timeout)
    element.click()


def enter_text(
    browser: webdriver.Chrome, by: str, value: str, text: str, timeout: int = 10
) -> None:
    """
    Enter text into an input field with a given locator.

    :param browser: WebDriver instance
    :param by: Locator type (e.g., By.ID, By.XPATH)
    :param value: Locator value
    :param text: Text to enter
    :param timeout: Maximum time to wait for the element
    """
    element = find_element(browser, by, value, timeout)
    element.clear()
    element.send_keys(text)


def wait_for_element_to_be_clickable(
    browser: webdriver.Chrome, by: str, value: str, timeout: int = 10
) -> WebElement:
    """
    Wait for an element to be clickable.

    :param browser: WebDriver instance
    :param by: Locator type (e.g., By.ID, By.XPATH)
    :param value: Locator value
    :param timeout: Maximum time to wait for the element
    :return: WebElement if clickable, else raises TimeoutException
    """
    return WebDriverWait(browser, timeout).until(
        EC.element_to_be_clickable((by, value))
    )


def get_element_text(
    browser: webdriver.Chrome, by: str, value: str, timeout: int = 10
) -> str:
    """
    Get the text of an element with a given locator.

    :param browser: WebDriver instance
    :param by: Locator type (e.g., By.ID, By.XPATH)
    :param value: Locator value
    :param timeout: Maximum time to wait for the element
    :return: Text of the WebElement
    """
    element = find_element(browser, by, value, timeout)
    return element.text


def wait_for_element_to_be_visible(
    browser: webdriver.Chrome, by: str, value: str, timeout: int = 10
) -> WebElement | None:
    """
    Wait for an element to be visible.

    :param browser: WebDriver instance
    :param by: Locator type (e.g., By.ID, By.XPATH)
    :param value: Locator value
    :param timeout: Maximum time to wait for the element
    :return: WebElement if visible, else raises TimeoutException
    """
    return WebDriverWait(browser, timeout).until(
        EC.visibility_of_element_located((by, value))
    )


def str_to_bool(s: str) -> bool:
    if s.lower() in ["true", "yes", "1"]:
        return True
    return False


def download_file(browser, url, timeout=60) -> Path:
    file_name = url.split("/")[-1]
    downloaded_file_path = DOWNLOADS_FOLDER / file_name
    if not downloaded_file_path.is_file():
        browser.get(url)
        wait_for_download(downloaded_file_path, timeout)

    return downloaded_file_path


def wait_for_download(file_path, timeout=60) -> bool:
    start_time = time.time()
    while not file_path.is_file():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Download timeout: {file_path}")
        time.sleep(1)
    return True
