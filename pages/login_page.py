import os
import time
import allure
from src.base_page import BasePage


class LoginPage(BasePage):
    
    URL = os.environ.get("URL")
    
    LOC = {
        "username": "//input[@id='user-name']",
        "password": "//input[@id='password']",
        "submit": "//input[@id='login-button']"
    }

    def __init__(self, page):
        super().__init__(page)

        self.config = {
            "url": os.environ.get("URL"),
            "login": os.environ.get("LOGIN"),
            "password": os.environ.get("PASSWORD")
        }


    @allure.step('Переход на страницу авторизации')
    def open(self):
        self.page.goto(self.config["url"])
        return self

    @allure.step('Авторизация пользователя "{username}"')
    def login(self, username=None, password=None):
        username = username or self.config["login"]
        password = password or self.config["password"]

        self.element("Поле ввода 'логин'", self.LOC["username"]).fill(username)
        self.element("Поле ввода 'пароль'", self.LOC["password"]).fill(password)
        self.element("Кнопка 'логин'", self.LOC["submit"]).click()
        return self