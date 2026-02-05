import pytest
import allure
from pages.login_page import LoginPage


@allure.title('Авторизация пользователя на сайте')
def test_login(page):
    page = LoginPage(page)
    page.open().login()