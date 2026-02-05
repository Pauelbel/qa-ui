import allure
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

class BaseElement:

    def __init__(self, locator, name, timeout=20):
        self.locator = locator
        self.name = name
        self.timeout = timeout * 1000  # Секунды

    def _wait_for_visible(self, timeout=None):
        wait_timeout = timeout if timeout is not None else self.timeout
        try:
            self.locator.wait_for(state="visible", timeout=wait_timeout)
        except PlaywrightTimeoutError:
            allure.attach(body=f"Элемент → '{self.name}' не появился за → {wait_timeout} мс",
                          name="Ошибка ожидания",
                          attachment_type=allure.attachment_type.TEXT)
            raise AssertionError(f"Элемент → '{self.name}' не появился за → {wait_timeout} мс")

    def click(self, timeout=None):
        with allure.step(f'Клик → "{self.name}"'):
            self._wait_for_visible(timeout)
            try:
                self.locator.click(timeout=timeout if timeout is not None else self.timeout)
            except Exception as e:
                allure.attach(str(e), name="Ошибка клика", attachment_type=allure.attachment_type.TEXT)
                raise

    def fill(self, text, timeout=None):
        with allure.step(f'Ввод текста → {self.name}'):
            self._wait_for_visible(timeout)
            try:
                self.locator.fill(text, timeout=timeout if timeout is not None else self.timeout)
            except Exception as e:
                allure.attach(str(e), name="Ошибка ввода", attachment_type=allure.attachment_type.TEXT)
                raise

    def get_text(self, timeout=None):
        with allure.step(f'Получение текста из → {self.name}'):
            self._wait_for_visible(timeout)
            try:
                return self.locator.inner_text(timeout=timeout if timeout is not None else self.timeout)
            except Exception as e:
                allure.attach(str(e), name="Ошибка получения текста", attachment_type=allure.attachment_type.TEXT)
                raise

    def is_visible(self, timeout=None):
        with allure.step(f'Проверка видимости элемента → {self.name}'):
            self._wait_for_visible(timeout)
            try:
                return self.locator.is_visible(timeout=timeout if timeout is not None else self.timeout)

            except Exception as e:
                allure.attach(str(e), name="Ошибка проверки видимости элемента", attachment_type=allure.attachment_type.TEXT)
                raise


    def press_key(self, key, timeout=None):
        """Нажать указанную клавишу (например, 'Enter', 'Escape', 'ArrowDown' и т.д.)"""
        with allure.step(f'Нажатие клавиши → {key} на элементе → "{self.name}"'):
            self._wait_for_visible(timeout)
            try:
                self.locator.press(key, timeout=timeout if timeout is not None else self.timeout)
            except Exception as e:
                allure.attach(str(e), name=f"Ошибка нажатия клавиши {key}", 
                            attachment_type=allure.attachment_type.TEXT)
                raise