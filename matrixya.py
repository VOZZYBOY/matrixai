import subprocess
import time
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import requests
from yandex_chain import YandexLLM, YandexEmbeddings, YandexGPTModel
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

# --- Глобальные переменные ---
IAM_TOKEN = None
ACCESS_TOKEN = None

# --- Константы ---
LOGIN_URL = "https://dev.back.matrixcrm.ru/api/v1/Auth/login"
API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"

LOGIN_PAYLOAD = {
    "Email": "xzolenr6@gmail.com",
    "Password": "Ericman2004",
    "DeviceId": "1234",
    "TenantId": "1234",
}

FOLDER_ID = "b1gb9k14k5ui80g91tnp"
YANDEX_SLEEP_INTERVAL = 0.1
YANDEX_MODEL = YandexGPTModel.ProRC


# --- Функции для работы с токенами ---
def update_iam_token():
    global IAM_TOKEN
    try:
        logger.info("Обновление IAM токена...")
        result = subprocess.run(["yc", "iam", "create-token"], capture_output=True, text=True, check=True)
        IAM_TOKEN = result.stdout.strip()
        logger.info("IAM токен успешно обновлен.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при обновлении IAM токена: {e.stderr}")
        IAM_TOKEN = None


def start_token_updater():
    def updater():
        while True:
            update_iam_token()
            time.sleep(12 * 60 * 60)

    thread = threading.Thread(target=updater, daemon=True)
    thread.start()
    logger.info("Фоновый процесс для обновления IAM токена запущен.")


def get_access_token():
    global ACCESS_TOKEN
    try:
        logger.info("Получение ACCESS токена...")
        response = requests.post(LOGIN_URL, data=LOGIN_PAYLOAD)
        if response.status_code == 200:
            ACCESS_TOKEN = response.json().get("data", {}).get("lclToken")
            logger.info("ACCESS токен успешно получен.")
        else:
            raise ValueError("Ошибка получения ACCESS токена.")
    except Exception as e:
        logger.error(f"Ошибка при получении ACCESS токена: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при получении ACCESS токена.")


def get_context_from_api():
    global ACCESS_TOKEN
    try:
        if not ACCESS_TOKEN:
            get_access_token()

        logger.info("Запрос данных из API...")
        headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        response = requests.get(API_URL, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Ошибка при получении данных API.")

        data = response.json()
        items = data.get("data", {}).get("items", [])

        if not isinstance(items, list):
            raise ValueError("Некорректный формат данных API: 'items' не является списком.")

        # Форматирование данных
        formatted_services = []
        for item in items:
            service_name = item.get("serviceName", "Неизвестная услуга")
            description = item.get("serviceDescription", "Нет описания")
            category = item.get("categoryName", "Нет категории")
            branch = item.get("filialName", "Не указан филиал")
            specialist = item.get("employeeFullName", "Не указан специалист")
            price = item.get("price", "Цена не указана")

            formatted_services.append(
                f"Название: {service_name}, Описание: {description}, Категория: {category}, "
                f"Филиал: {branch}, Специалист: {specialist}, Цена: {price} руб."
            )
        return formatted_services
    except Exception as e:
        logger.error(f"Ошибка при запросе API: {e}")
        return ["Ошибка при подключении к API."]


# --- Основная логика обработки запроса ---
def process_query(query: str):
    logger.info("Начало обработки запроса...")
    update_iam_token()

    embeddings = YandexEmbeddings(folder_id=FOLDER_ID, iam_token=IAM_TOKEN, sleep_interval=YANDEX_SLEEP_INTERVAL)
    llm = YandexLLM(folder_id=FOLDER_ID, iam_token=IAM_TOKEN, model=YANDEX_MODEL)

    services = get_context_from_api()
    if not services:
        raise ValueError("Не удалось получить данные из API.")

    # Векторное хранилище
    vectorstore = InMemoryVectorStore.from_texts(services, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Промпт
    template = """
    Тебя зовут Эрика, ты консультант клиники Эрикмед.
    Вот список доступных услуг клиники:
    {context}

    Вопрос: {question}
    Пожалуйста, предоставь развернутый ответ на основе вышеуказанного списка услуг,
    включая информацию о специалистах и филиалах, если они указаны.
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    logger.info(f"Ответ от Yandex GPT: {response}")
    return response


# --- FastAPI сервер ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestData(BaseModel):
    query: str


@app.post("/process")
async def process_api(request: RequestData):
    logger.info(f"Получен запрос: '{request.query}'")
    try:
        response = process_query(request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Запуск сервера ---
if __name__ == "__main__":
    logger.info("Запуск сервера FastAPI...")
    start_token_updater()
    get_access_token()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
