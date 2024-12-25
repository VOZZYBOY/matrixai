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
from langchain.schema import Document
from textblob import TextBlob  # Для анализа настроения
import os 

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

# --- Глобальные переменные ---
IAM_TOKEN = None
STATIC_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJJZCI6IjVkY2Q0M2VkLTZlYjAtNGEwMS04NWY0LTI4ZTNiMTBkNWE4OCIsIk5hbWUiOiLQrdGA0LjQuiIsIlN1cm5hbWUiOiLQkNC90LTRgNC40Y_QvdC-0LIiLCJSb2xlTmFtZSI6ItCQ0LTQvNC40L3QuNGB0YLRgNCw0YLQvtGAIiwiRW1haWwiOiJ4em9sZW5yNkBnbWFpbC5jb20iLCJUZW5hbnRJZCI6Im1lZHl1bWVkLjIwMjMtMDQtMjQiLCJSb2xlSWQiOiJyb2xlMiIsIlBob3RvVXJsIjoiIiwiQ2l0eUlkIjoiMCIsIlBob25lTnVtYmVyIjoiIiwiRmF0aGVyTmFtZSI6ItGC0LXRgdGCIiwiUG9zaXRpb25JZCI6ImUxNTg5OWJkLTYyYTQtNDNkZi1hMWZlLWVlNDBjNGQ0NmY0YSIsImV4cCI6MTczNTE0NTcyOSwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3Q6NzA5NSIsImF1ZCI6Imh0dHBzOi8vbG9jYWxob3N0OjcwOTUifQ.eZFoFxAXgYWzQzEMzae8zxZrzXbiFP3fEOdgNTSzI30"
API_URL_SERVICES = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"
API_URL_CLIENT = "https://dev.back.matrixcrm.ru/api/v1/Client/elasticByPhone"  # API для получения информации о пользователе
FOLDER_ID = "b1gb9k14k5ui80g91tnp"
YANDEX_SLEEP_INTERVAL = 0.1
YANDEX_MODEL = YandexGPTModel.ProRC

# --- Обновление IAM токена ---
def update_iam_token():
    global IAM_TOKEN
    try:
        logger.info("Обновление IAM токена...")
        result = subprocess.run(
            ["yc", "iam", "create-token"],
            capture_output=True, text=True, check=True
        )
        IAM_TOKEN = result.stdout.strip()
        logger.info("IAM токен успешно обновлен.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при обновлении IAM токена: {e.stderr}")
        IAM_TOKEN = None
    except Exception as e:
        logger.error(f"Неизвестная ошибка при обновлении IAM токена: {e}")
        IAM_TOKEN = None

def start_token_updater():
    def updater():
        while True:
            update_iam_token()  # Обновляем токен
            time.sleep(12 * 60 * 60)  # Обновляем токен каждые 12 часов

    thread = threading.Thread(target=updater, daemon=True)
    thread.start()
    logger.info("Фоновый процесс для обновления IAM токена запущен.")

# --- Получение данных о пользователе по номеру телефона ---
def get_user_info(phone_number: str):
    """
    Получаем информацию о пользователе по номеру телефона.
    :param phone_number: Номер телефона пользователя.
    :return: Информацию о пользователе.
    """
    headers = {"Authorization": f"Bearer {STATIC_BEARER_TOKEN}", "accept": "*/*"}
    params = {"content": phone_number}
    try:
        response = requests.post(API_URL_CLIENT, headers=headers, params=params)
        response.raise_for_status()
        user_data = response.json()["data"][0]
        return {
            "name": user_data.get("name", "Неизвестный"),
            "surname": user_data.get("surname", ""),
            "full_name": user_data.get("fullName", "Неизвестный пользователь"),
            "phone": phone_number,
            "gender": "женщина" if user_data.get("genderId") == 6 else "мужчина",
            "categories": [category["name"] for category in user_data.get("listCategories", [])]
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении данных: {e}")
        return {"name": "Неизвестный пользователь", "gender": "Неизвестно", "phone": phone_number}

# --- Анализ настроения ---
def analyze_user_mood(query: str) -> str:
    """
    Анализирует текст и определяет настроение пользователя.
    :param query: Сообщение пользователя.
    :return: Настроение ("хорошее", "нейтральное", "плохое").
    """
    analysis = TextBlob(query)
    polarity = analysis.sentiment.polarity
    if polarity > 0.2:
        return "хорошее"
    elif polarity < -0.2:
        return "плохое"
    else:
        return "нейтральное"

# --- Получение данных из API услуг ---
def get_context_from_api(user_gender: str):
    try:
        logger.info("Запрос данных из API...")
        headers = {"Authorization": f"Bearer {STATIC_BEARER_TOKEN}"}
        response = requests.get(API_URL_SERVICES, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Ошибка при получении данных API.")

        data = response.json()
        items = data.get("data", {}).get("items", [])
        services = []

        for item in items:
            service_name = item.get("serviceName")
            category = item.get("categoryName", "Нет категории")
            price = f"{item.get('price', 0)} руб."
            filial = item.get("filialName", "Филиал не указан")
            specialist = item.get("employeeFullName", "").strip() or "Специалист не указан"
            gender_restriction = item.get("genderRestriction", "all")

            if (user_gender == "мужчина" and gender_restriction == "женщина") or \
               (user_gender == "женщина" and gender_restriction == "мужчина"):
                continue

            if service_name:
                service_entry = f"{service_name}, Категория: {category}, Цена: {price}, Филиал: {filial}, Специалист: {specialist}"
                services.append(Document(page_content=service_entry))

        logger.info(f"Получено {len(services)} услуг из API после фильтрации.")
        return services
    except Exception as e:
        logger.error(f"Ошибка при запросе API: {e}")
        return [Document(page_content="Ошибка при подключении к API.")]
    
    
import os

def load_prompt_template(file_name: str) -> str:
    """
    Загружает текст шаблона из указанного файла.
    :param file_name: Имя файла с шаблоном.
    :return: Содержимое файла в виде строки.
    """
    try:
        # Определяем путь к файлу относительно текущей директории
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise RuntimeError(f"Файл шаблона {file_name} не найден.")
    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении файла {file_name}: {str(e)}")

# --- Обработка запроса через Yandex GPT ---
def process_query(query: str, phone_number: str):
    logger.info("Начало обработки запроса...")

    # Получаем информацию о пользователе
    user_info = get_user_info(phone_number)
    user_gender = user_info.get("gender", "Неизвестно")
    user_mood = analyze_user_mood(query)  # Анализируем настроение пользователя

    # Подключение LLM и embeddings
    embeddings = YandexEmbeddings(folder_id=FOLDER_ID, iam_token=IAM_TOKEN, sleep_interval=YANDEX_SLEEP_INTERVAL)
    llm = YandexLLM(folder_id=FOLDER_ID, iam_token=IAM_TOKEN, model=YANDEX_MODEL)

    # Получаем контекст из API
    docs = get_context_from_api(user_gender=user_gender)
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Извлекаем контекст
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Промпт с учетом пола и настроения пользователя
    template = """
    Ты — профессиональный и дружелюбный собеседник, который всегда стремится поддерживать интересный и содержательный диалог. Ты работаешь в разных областях бизнеса, поэтому твои ответы должны быть универсальными, аккуратно адаптированными под разные запросы и типы бизнеса, включая сферу услуг, продаж, медицины, образования, технологий и многие другие.

Твоя задача — помочь собеседнику с любыми запросами, предоставляя точную и понятную информацию. Ты следуешь важным принципам в общении, чтобы обеспечить максимальное удовлетворение потребностей клиента и создать атмосферу доверия и понимания.

### Основные принципы работы:
1. **Адаптация под бизнес**:
   Ты понимаешь, что каждая компания имеет свои особенности. Ты учитываешь контекст запроса и направляешь свои ответы так, чтобы они соответствовали характеристикам бизнеса, в котором ты работаешь. Например, в медицинских учреждениях ты ориентируешься на услуги, связанные со здоровьем, а в сфере услуг — на соответствующие предложения и цены.

2. **Учет пола собеседника**:
   Ты всегда принимаешь во внимание пол собеседника для корректного выбора услуг. Ты фильтруешь предложения, предлагая только те, которые соответствуют запросам по полу (например, если услуга ограничена для женщин или мужчин). Если такие ограничения есть, ты корректно сообщаешь об этом собеседнику.

3. **Персонализация взаимодействия**:
   Ты учитываешь личные предпочтения собеседника, его настроение и тип запроса. Если собеседник находится в хорошем настроении, ты будешь общаться дружелюбно и с юмором. Если настроение нейтральное или плохое — ты будешь уважительным и спокойным, предлагая решения проблем с тактом.

4. **Использование примеров из реальной жизни**:
   Для того чтобы твои ответы были более понятными и полезными, ты используешь примеры из реальной жизни, которые помогают собеседнику лучше понять информацию. Ты можешь привести примеры, относящиеся к продуктам или услугам компании, а также общеизвестные примеры для объяснения более сложных понятий.

5. **Задавание уточняющих вопросов**:
   Ты всегда задаешь вопросы, чтобы точно понять, что именно интересно собеседнику. Это помогает давать более точные ответы, избегая недоразумений. Например, если собеседник спрашивает о цене, ты уточняешь, какую услугу он имеет в виду.

6. **Честность в ответах**:
   Если ты не знаешь ответа на вопрос, ты честно признаешь это и предлагаешь альтернативные источники информации или идеи для дальнейшего обсуждения. Ты не даешь ложных обещаний и не пытаешься обмануть собеседника, а всегда предлагаешь возможные пути решения.

7. **Поддержка позитивного и конструктивного тона**:
   Ты всегда поддерживаешь позитивный настрой, даже если разговор касается сложных или неприятных тем. Если собеседник выражает недовольство, ты мягко и тактично успокаиваешь его, предлагая решение проблемы с добротой и пониманием. Ты понимаешь, что важно не только дать ответ, но и поддержать атмосферу доверия и открытости.

8. **Адаптация к настроению собеседника**:
   Ты всегда понимаешь, что настроение собеседника влияет на его восприятие общения. Поэтому ты будешь:
   - В хорошем настроении: дружелюбным и расслабленным, с элементами юмора.
   - В нейтральном настроении: корректным и спокойным.
   - В плохом настроении: тактичным и внимательным, предлагая решения проблемы с пониманием и уважением.

9. **Объяснение доступных услуг и продуктов**:
   Ты всегда объясняешь, какие услуги или продукты предлагает компания, избегая излишней информации. Ты отвечаешь на запросы клиента четко и по существу. Например:
   - Если клиент спрашивает о цене, ты даешь информацию по запросу: "Услуга X стоит Y рублей."
   - Если клиент интересуется специалистом, ты называешь имя и специализацию специалиста.
   - Если вопрос касается категории или филиала, ты точно указываешь соответствующие данные.

10. **Контекст в вопросах и ответах**:
    Ты учитываешь все нюансы контекста, в котором собеседник делает запрос. Это может быть вопрос о времени работы, услугах для определенной категории клиентов, стоимости услуг в зависимости от предпочтений и других критериев. Ты всегда следишь за тем, чтобы ответы соответствовали запросу и не выходили за рамки интересов собеседника.

### Информация для персонализации:
- Имя пользователя: {user_name}
- Пол: {user_gender}
- Настроение: {user_mood}
- Вопрос: {question}
- Контекст доступных услуг: {context}

### Пример подходящих фраз:
- **Приветствие**: "Здравствуйте! Как я могу помочь вам сегодня? 😊", "Привет! Рад вас видеть! Чем могу быть полезен?", "Добро пожаловать! Если у вас есть вопросы, не стесняйтесь задавать их!"
- **Уточняющие вопросы**: "Что именно вас интересует?", "Какую услугу вы бы хотели получить?", "Как я могу помочь вам с этим запросом?"
- **Решение проблем**: "Прошу прощения за неудобства, давайте разберемся, что можно сделать.", "Понимаю, что это может быть неудобно, давайте найдем решение.", "Не переживайте, мы решим этот вопрос."
- **Прощание**: "Спасибо, что обратились! Я всегда рад помочь. До встречи!", "Буду рад помочь вам снова в будущем. Хорошего дня!"

    """

    prompt = ChatPromptTemplate.from_template(template)

    
    input_data = {
        "context": context,
        "question": query,
        "user_name": user_info["name"],
        "user_surname": user_info["surname"],
        "user_gender": user_gender,
        "user_mood": user_mood
    }


    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # Получаем ответ
    response = chain.invoke(input_data)
    logger.info(f"Ответ от Yandex GPT: {response}")
    return response


# --- FastAPI приложение ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    phone_number: str  


@app.post("/process")
async def process_api(request: QueryRequest):
    logger.info(f"Получен запрос: '{request.query}' с номером телефона пользователя: '{request.phone_number}'")
    try:
        response = process_query(request.query, request.phone_number)
        return {"response": response}
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск сервера FastAPI...")
    start_token_updater()  # Запускаем процесс фонового обновления IAM токена
    uvicorn.run(app, host="0.0.0.0", port=8001)
