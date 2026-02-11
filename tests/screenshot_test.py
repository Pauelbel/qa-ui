import pytest
import allure
from pages.login_page import LoginPage
from src.screenshot_testing.screen import ScreenshotComparator, ScreenshotManager


@allure.title('Тест визуального регрессионного анализа')
def test_visual(page):
    """
    Пример теста визуального регрессионного анализа
    """
    login_page = LoginPage(page)
    login_page.open()
    
    # Используем ScreenshotManager для управления файлами
    screenshot_manager = ScreenshotManager()
    test_name = "test_visual"
    base_img = screenshot_manager.get_path('base', f'{test_name}.png')
    new_img = screenshot_manager.get_path('new', f'{test_name}.png')
    result_img = screenshot_manager.get_path('result', f'{test_name}.png')
    
    # Делаем текущий скриншот
    screenshot_manager.capture_page_screenshot(page, new_img)
    
    # Выполняем визуальное тестирование
    comparator = ScreenshotComparator()
    comparator.perform_visual_regression_test(
        baseline_path=base_img,
        current_path=new_img,
        output_path=result_img
    )