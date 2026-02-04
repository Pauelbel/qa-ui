import pytest
import allure
from pages.login_page import LoginPage


@allure.title('Авторизация пользователя на сайте')
@pytest.mark.skip_auth
def test_login(page):
    page = LoginPage(page)
    page.open().login()