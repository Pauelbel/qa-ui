import allure, pytest, logging

@pytest.fixture(autouse=True)
def capture_allure_steps():
    """
    Фикстура для перехвата всех вызовов allure.step и вывода их в лог.
    """
    original_allure_step = allure.step

    def custom_allure_step(step_message):
        """
        Обертка над allure.step для записи сообщений в лог.
        """
        logging.info(f"STEP: {step_message}")
        return original_allure_step(step_message)

    allure.step = custom_allure_step
    yield
    allure.step = original_allure_step