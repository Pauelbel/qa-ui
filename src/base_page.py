from src.base_element import BaseElement

class BasePage:
    def __init__(self, page):
        self.page = page

    def element(self, description, selector):
        return BaseElement(self.page.locator(selector), description)
