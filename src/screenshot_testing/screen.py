import cv2
import numpy as np
import os


class ScreenshotComparator:
    # Настройки
    indent = 5              # Отступ вокруг контуров
    rgb = (204, 0, 204)     # Цвет обводки (BGR)
    thickness = 2           # Толщина рамки
    fill_alpha = 0.3        # Прозрачность заливки
    threshold = 0.01        # Порог различий

    def compare_images(self, baseline_path, current_path, threshold=None):
        """
        Сравнивает два изображения и возвращает долю различающихся пикселей и изображение с выделенными различиями
        
        Args:
            baseline_path: путь к опорному изображению
            current_path: путь к текущему изображению
            threshold: порог различий (по умолчанию используется значение по умолчанию)
        
        Returns:
            tuple: (diff_ratio, highlight_image) - доля различий и изображение с выделенными различиями
        """
        if threshold is None:
            threshold = self.threshold
            
        # Загружаем изображения
        base = cv2.imread(baseline_path)
        curr = cv2.imread(current_path)

        if base is None:
            raise FileNotFoundError(f"Исходное изображение не найдено: {baseline_path}")
        if curr is None:
            raise FileNotFoundError(f"Текущее изображение не найдено: {current_path}")
        if base.shape != curr.shape:
            raise ValueError("Размер изображений различается")

        # Абсолютная разница
        diff = cv2.absdiff(base, curr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        # Доля изменённых пикселей
        diff_ratio = np.count_nonzero(mask) / mask.size

        # Копия изображения для выделения различий
        highlight = base.copy()

        # Контуры различий
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = highlight.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(x - self.indent, 0)
            y1 = max(y - self.indent, 0)
            x2 = min(x + w + self.indent, base.shape[1] - 1)
            y2 = min(y + h + self.indent, base.shape[0] - 1)
            
            # Рамка
            cv2.rectangle(highlight, (x1, y1), (x2, y2), self.rgb, self.thickness)
            # Полупрозрачная заливка
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.rgb, -1)

        # Смешиваем overlay с highlight
        cv2.addWeighted(overlay, self.fill_alpha, highlight, 1 - self.fill_alpha, 0, highlight)

        return diff_ratio, highlight

    def perform_visual_regression_test(self, baseline_path, current_path, output_path=None, threshold=None):
        """
        Выполняет визуальный регрессионный тест
        
        Args:
            baseline_path: путь к опорному изображению
            current_path: путь к текущему изображению
            output_path: путь для сохранения результата (необязательно)
            threshold: порог различий (по умолчанию используется значение по умолчанию)
        
        Returns:
            float: процент визуальной разницы
        """
        if threshold is None:
            threshold = self.threshold
            
        diff_ratio, highlight = self.compare_images(baseline_path, current_path, threshold)

        # Сохраняем результат, если указан путь
        if output_path:
            cv2.imwrite(output_path, highlight)
        else:
            # Сохраняем в стандартное место
            output_dir = os.path.dirname(current_path)
            if not output_dir:
                output_dir = "."
            cv2.imwrite(os.path.join(output_dir, "highlight.png"), highlight)

        print(f"Процент визуальной разницы: {diff_ratio*100:.2f}%")
        if diff_ratio >= threshold:
            raise AssertionError(f"Слишком большая визуальная разница: {diff_ratio:.4f}, порог: {threshold:.4f}")

        return diff_ratio

    def compare_images_with_filenames(self, expected_filename, actual_image, diff_filename, threshold=None):
        """
        Сравнивает эталонное изображение с текущим и создает изображение с различиями
        
        Args:
            expected_filename: имя эталонного изображения
            actual_image: текущее изображение (уже загруженное)
            diff_filename: имя файла для сохранения различий
            threshold: порог различий
        """
        if threshold is None:
            threshold = self.threshold
            
        # Используем ScreenshotManager для получения путей
        manager = ScreenshotManager()
        
        # Получаем путь к эталонному изображению
        expected_path = manager.get_path('base', expected_filename)
        
        # Загружаем эталонное изображение
        expected_img = cv2.imread(expected_path)
        
        if expected_img is None:
            raise FileNotFoundError(f"Эталонное изображение не найдено: {expected_path}")
        if actual_image is None:
            raise ValueError("Текущее изображение не предоставлено")
        if expected_img.shape != actual_image.shape:
            raise ValueError("Размер эталонного и текущего изображений различается")
        
        # Абсолютная разница
        diff = cv2.absdiff(expected_img, actual_image)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        # Доля изменённых пикселей
        diff_ratio = np.count_nonzero(mask) / mask.size

        # Копия эталонного изображения для выделения различий
        highlight = expected_img.copy()

        # Контуры различий
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = highlight.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(x - self.indent, 0)
            y1 = max(y - self.indent, 0)
            x2 = min(x + w + self.indent, expected_img.shape[1] - 1)
            y2 = min(y + h + self.indent, expected_img.shape[0] - 1)
            
            # Рамка
            cv2.rectangle(highlight, (x1, y1), (x2, y2), self.rgb, self.thickness)
            # Полупрозрачная заливка
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.rgb, -1)

        # Смешиваем overlay с highlight
        cv2.addWeighted(overlay, self.fill_alpha, highlight, 1 - self.fill_alpha, 0, highlight)
        
        # Получаем путь для сохранения различий
        diff_path = manager.get_path('result', diff_filename)
        cv2.imwrite(diff_path, highlight)

        print(f"Процент визуальной разницы: {diff_ratio*100:.2f}%")
        if diff_ratio >= threshold:
            raise AssertionError(f"Слишком большая визуальная разница: {diff_ratio:.4f}, порог: {threshold:.4f}")

    # --- Fluent convenience API -------------------------------------------------
    def __init__(self, base_filename: str | None = None, manager: 'ScreenshotManager' | None = None):
        """
        Инициализация для упрощённого (цепочного) использования:
        Example: ScreenshotComparator("base.png").save_screenshot(page).compare()
        Можно не передавать `page` в `save_screenshot()` — тогда метод проверит,
        что файл `new` уже существует.
        """
        # Отложенный импорт типа для аннотаций совместимости в рантайме
        self.manager = manager or ScreenshotManager()
        self.base_filename = base_filename
        if base_filename:
            self.base_path = self.manager.get_path('base', base_filename)
            self.new_path = self.manager.get_path('new', base_filename)
            self.result_path = self.manager.get_path('result', base_filename)
        else:
            self.base_path = None
            self.new_path = None
            self.result_path = None

    def save_screenshot(self, page=None, filename: str | None = None):
        """
        Сохраняет (или проверяет наличие) текущего скриншота в `new`.

        Args:
            page: опционально — объект Playwright `page`; если передан — делается снимок.
            filename: опционально — имя файла (если нужно переопределить).

        Возвращает `self` для цепочки вызовов.
        """
        if filename:
            self.new_path = self.manager.get_path('new', filename)
            self.result_path = self.manager.get_path('result', filename)
            if not self.base_filename:
                self.base_filename = filename
                self.base_path = self.manager.get_path('base', filename)

        if page is not None:
            # Сделать скриншот через менеджер
            self.manager.capture_page_screenshot(page, self.new_path)
        else:
            # Проверяем, что файл уже есть
            if not self.new_path or not os.path.exists(self.new_path):
                raise FileNotFoundError(f"Текущий скриншот не найден: {self.new_path}")

        return self

    def compare(self, threshold=None):
        """
        Выполняет сравнение между `base` и `new` (и сохраняет результат в `result`).
        Возвращает значение diff_ratio (float).
        """
        if not self.base_path or not self.new_path:
            raise ValueError("Не указаны пути для сравнения (base/new)")
        # Используем существующий метод для проверки и сохранения
        return self.perform_visual_regression_test(
            baseline_path=self.base_path,
            current_path=self.new_path,
            output_path=self.result_path,
            threshold=threshold,
        )


class ScreenshotManager:
    """
    Класс для управления файлами и директориями скриншотов
    """
    def __init__(self, base_dir="tests/screenshots_testing"):
        """
        Инициализация менеджера скриншотов
        
        Args:
            base_dir: базовая директория для хранения скриншотов
        """
        self.base_dir = base_dir
        self.base_img_dir = os.path.join(base_dir, "base_img")
        self.new_img_dir = os.path.join(base_dir, "new_img")
        self.result_img_dir = os.path.join(base_dir, "result_img")
        
        # Создаем все необходимые директории
        for directory in [self.base_img_dir, self.new_img_dir, self.result_img_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_path(self, img_type, filename):
        """
        Получает путь к изображению
        
        Args:
            img_type: тип изображения ('base', 'new', 'result')
            filename: имя файла
            
        Returns:
            str: полный путь к изображению
        """
        if img_type == 'base':
            return os.path.join(self.base_img_dir, filename)
        elif img_type == 'new':
            return os.path.join(self.new_img_dir, filename)
        elif img_type == 'result':
            return os.path.join(self.result_img_dir, filename)
        else:
            raise ValueError(f"Неизвестный тип изображения: {img_type}")

    def capture_page_screenshot(self, page, filepath):
        """
        Делает скриншот страницы с использованием playwright
        
        Args:
            page: объект страницы playwright
            filepath: путь для сохранения скриншота
        """
        page.screenshot(path=filepath)