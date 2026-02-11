import pytest
import allure
from pages.login_page import LoginPage
from src.screenshot_testing.screen import ScreenshotComparator, ScreenshotManager


@allure.title('Скриншотный тест')
def test_visual(page):
    login_page = LoginPage(page)
    login_page.open()
    
    ScreenshotComparator("visual.png").save_screenshot(page).compare()
