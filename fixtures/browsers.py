from playwright.sync_api import Page, Playwright
import pytest

# ================= Фикстуры для каждого браузера =================
@pytest.fixture()
def chromium_page(playwright: Playwright):
    """Фикстура для запуска Chromium браузера"""
    browser = playwright.chromium.launch(
        headless=False,
        #args=['--window-position=-1919,-457']
        )
    
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        ignore_https_errors=True
    )
    page = context.new_page()

    yield page

    context.close()
    browser.close()


# ================= Основная фикстура page =================

@pytest.fixture()
def page(chromium_page) -> Page:
    """
    Основная фикстура page для тестов.
    По умолчанию использует Chromium, но можно легко переключиться на другой браузер.
    """
    return chromium_page

