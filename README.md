# qa-ui
Автотесты пользовательского интерфейса

**Описание:** Набор UI-автотестов на pytest + Playwright с простыми скриншотными тестами


### **Требования:** 
Python 3.10+, Playwright, OpenCV, остальные зависимости в `requirements.txt`.

### **Установка:**
```bash
pip install -r requirements.txt
playwright install
```

### **Базовая настройка:**
	- Отредактировать переменные переименовать  `.env_example` в `.env` заполнить данные.

### **Как запускать тесты:**
- Запуск всех тестов:
    ```bash
    pytest
    ```
- Запуск одного файла/теста:
	```bash
	pytest tests/screenshot_test.py::test_visual
	```

### **CI/Отчёты:**
	- Allure-результаты сохраняются в `allure-results/` — генерировать отчёты через `allure serve`.

