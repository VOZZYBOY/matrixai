import subprocess
import time
import threading

# Переменная для хранения текущего токена
IAM_TOKEN = None

def update_iam_token():
    """Создает новый IAM токен и сохраняет его в глобальную переменную."""
    global IAM_TOKEN
    try:
        # Выполнение команды `yc iam create-token`
        result = subprocess.run(["yc", "iam", "create-token"], capture_output=True, text=True, check=True)
        IAM_TOKEN = result.stdout.strip()
        print("IAM токен успешно обновлен.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при обновлении IAM токена: {e.stderr}")
        IAM_TOKEN = None

def get_iam_token():
    """Возвращает текущий токен. Если токена нет, обновляет его."""
    global IAM_TOKEN
    if not IAM_TOKEN:
        update_iam_token()
    return IAM_TOKEN

def start_token_updater():
    """Запускает процесс обновления IAM токена каждые 12 часов."""
    def updater():
        while True:
            update_iam_token()
            time.sleep(12 * 60 * 60)  # Обновление токена каждые 12 часов

    thread = threading.Thread(target=updater, daemon=True)
    thread.start()
