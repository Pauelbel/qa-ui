import os
import pytest

@pytest.fixture(scope="session", autouse=True)
def load_env_variables():
    """Загружает переменные окружения из .credentials"""
    try:
        with open('.credentials') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ[key.strip()] = val.strip()
    except FileNotFoundError:
        pass