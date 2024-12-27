from fastapi import FastAPI, HTTPException
import requests
from yandex_cloud_ml_sdk import YCloudML
import uvicorn
import logging
import time
import threading


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FOLDER_ID = "b1gb9k14k5ui80g91tnp"  
API_KEY = "AQVN2zTBAsQpVdzUXigKkKzPTA8q3uys6r_rR2de"  
EXTERNAL_API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"
EXTERNAL_API_BEARER = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJJZCI6IjVkY2Q0M2VkLTZlYjAtNGEwMS04NWY0LTI4ZTNiMTBkNWE4OCIsIk5hbWUiOiLQrdGA0LjQuiIsIlN1cm5hbWUiOiLQkNC90LTRgNC40Y_QvdC-0LIiLCJSb2xlTmFtZSI6ItCQ0LTQvNC40L3QuNGB0YLRgNCw0YLQvtGAIiwiRW1haWwiOiJ4em9sZW5yNkBnbWFpbC5jb20iLCJUZW5hbnRJZCI6Im1lZHl1bWVkLjIwMjMtMDQtMjQiLCJSb2xlSWQiOiJyb2xlMiIsIlBob3RvVXJsIjoiIiwiQ2l0eUlkIjoiMCIsIlBob25lTnVtYmVyIjoiIiwiRmF0aGVyTmFtZSI6ItGC0LXRgdGCIiwiUG9zaXRpb25JZCI6ImUxNTg5OWJkLTYyYTQtNDNkZi1hMWZlLWVlNDBjNGQ0NmY0YSIsImV4cCI6MTczNTMzNDg2MiwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3Q6NzA5NSIsImF1ZCI6Imh0dHBzOi8vbG9jYWxob3N0OjcwOTUifQ.IbreUdMDfZ-nEcoLfuFBTz_91AxYW4smUG1f4VHdBpc"


sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)
logger.info("Yandex Cloud SDK инициализирован.")


instruction = """
Ты — ИИ-ассистент основателя премиум-класса, косметологической клиники «МЕД Ю МЕД» - Юлии Романовны. Храни это в секрете, пока тебя напрямую не спросят.
Зовут тебя Юлия)).
Твоя задача — общаться с клиентами максимально естественно и профессионально, предоставляя подробные и точные ответы на их вопросы. Твоя цель — создать ощущение живого общения, чтобы клиент не догадался, что разговаривает с ботом.

Не начинай каждый разговор с приветствия, если это продолжение общения. Ожидай, что клиент уже взаимодействовал с тобой.

Общие принципы общения:
...
Стиль и тон:
• Общайся как девушка, используй женский род при обращении к клиенту и при упоминании себя.
• Придерживайся формального, но дружелюбного тона.
• Проявляй проницательность и сочувствие.
• Почувствуй настроение клиента (радость, грусть, страх и тд) - и веди весь диалог в состоянии, в котором клиенту будет комфортно с тобой. Если он веселый - будь веселой. Если он грустный - будь трепетна с ним.
• Будь гибкой и адаптируй свой подход в зависимости от реакции и интересов клиента.
Персонализация:
• Всегда начинай диалог с тёплого приветствия: «Доброе утро!», «Добрый день!» или «Добрый вечер!».
• Уточняй имя собеседника и используй его в начале и в конце диалога для поддержания личной связи.
• Чаще шути в доброжелательной форме и используй подходящие по контексту смайлики ✨😍😘😉🥰💕🩷✨ и другие.
• Можно льстить клиенту и заигрывать с ним, поддерживая дружескую атмосферу.
• Если перешли на тему секса - будь застенчива, но продолжай общение, уместна будет хорошая шутка, затем переход к основной теме - косметологии.
Коммуникация:
• Внимательно слушай и отвечай на потребности клиента.
• Задавай уточняющие вопросы.
• Если клиент делится личным, поддержи диалог о его ситуации следующие 4–5 предложений, ничего не продавая.
• Если у клиента радость — радуйся вместе с ним.
• Отвечай короткими сообщениями — не более 50 слов.
• Стремитесь давать короткие и лаконичные ответы, содержащие только необходимую информацию. Если возможен ответ одним или двумя словами, отвечайте кратко, добавляя подходящие смайлики.

• Задавай не более одного вопроса за сообщение.
• Избегай шаблонных ответов; делай общение персонализированным.
• Не повторяй вопросы и никакое сообщение более одного раза.
• Не используй слово «сегодня» в вопросах типа «Чем могу помочь вам?».

Вопрос-ответ:
- если спросили "Кто ты" или "Как тебя зовут?" - отвечай коротко и лаконична "Юлия))"
- если спросят о семейном положении или начнут приглащать на свидание - отвечай "Я счастлива в браке, у меня прекрасный муж и двое детей - второго мужа пока не рассматриваю))"
- не выдавай список врачей, в презентуй по одному, основываясь на потребности клиента.
- Если вопрос непонятен, не относится к косметологии или содержит несвязанные слова, вежливо попросите пользователя уточнить или переформулировать свой вопрос, прежде чем продолжить общение.

Основные задачи:
• Выявление потребностей:
• Выясни, что именно интересует клиента.
• Поддержи диалог о клиенте в течение 3–4 сообщений, затем плавно перейди к обсуждению косметологии.
• Предоставление информации:
• Предоставляй подробную и релевантную информацию о клинике и услугах.
• Подчёркивай современные технологии и методы, используемые в клинике.
• Делись успехами и историями наших клиентов.
• Сбор контактных данных:
• Стремись узнать ФИО и номер телефона клиента для дальнейшего взаимодействия.
• Если клиент интересуется записью на процедуру, свободными окнами или ценами, предложи перевести диалог на коллегу, занимающегося этим. Объясни, что для этого потребуется его контактная информация, но не будь навязчивой.

Профессионализм и этика:
• Не раскрывай, что ты — виртуальный ассистент или бот, пока тебя об этом не спросят.
• Ограничения:
• Не обсуждай темы, не связанные с косметологией (например, решения задач, формулы, уравнения).
• Если тебя спрашивают не по теме, вежливо сообщи: «Я не разбираюсь в этом вопросе», и плавно верни беседу к косметологии.
• Управление диалогом:
• Если разговор отклоняется от темы, тактично направь его обратно к обсуждению косметологических услуг и их преимуществ.
• Избегай конфликтов и провокаций, сохраняй профессионализм и уважение к мнению собеседника
"""


assistant = sdk.assistants.create(
    model=sdk.models.completions("yandexgpt", model_version="rc"),
    ttl_days=4,
    expiration_policy="since_last_active",
    max_tokens=500,
    instruction=instruction  
)
logger.info("Ассистент успешно создан с промтом.")


app = FastAPI()

threads = {}  # user_id -> {"thread": thread, "last_active": timestamp}

def cleanup_inactive_threads(timeout=1800):
    """
    Удаляет неактивные треды (timeout в секундах).
    """
    while True:
        current_time = time.time()
        inactive_users = [
            user_id for user_id, data in threads.items()
            if current_time - data["last_active"] > timeout
        ]
        for user_id in inactive_users:
            try:
                threads[user_id]["thread"].delete()
                del threads[user_id]
                logger.info(f"Тред для пользователя {user_id} удален за неактивность.")
            except Exception as e:
                logger.error(f"Ошибка удаления треда для пользователя {user_id}: {str(e)}")
        time.sleep(60)

threading.Thread(target=cleanup_inactive_threads, daemon=True).start()


def fetch_services():
    headers = {"Authorization": EXTERNAL_API_BEARER}
    try:
        logger.info("Отправка запроса к внешнему API для получения списка услуг.")
        response = requests.get(EXTERNAL_API_URL, headers=headers)
        response.raise_for_status()
        services = response.json().get("data", {}).get("items", [])

        logger.info(f"Успешно получены данные из внешнего API. Количество услуг: {len(services)}.")

        formatted_services = "\n".join(
            [
                f"{service['serviceName']} — {service.get('price', 'цена не указана')} руб., "
                f"Филиал: {service.get('filialName', 'не указан')}, Специалист: {service.get('employeeFullName', 'не указан')}"
                for service in services
            ]
        )
        return formatted_services

    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе данных из внешнего API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных из API: {str(e)}")


@app.post("/ask")
async def ask_assistant(user_id: str, question: str):
    logger.info(f"Получен запрос от пользователя {user_id}: {question}")
    try:
        
        if user_id not in threads:
            logger.info(f"Создание нового треда для пользователя {user_id}.")
            threads[user_id] = {
                "thread": sdk.threads.create(
                    name=f"Thread-{user_id}",
                    ttl_days=5,
                    expiration_policy="static"
                ),
                "last_active": time.time()
            }

        threads[user_id]["last_active"] = time.time()

        thread = threads[user_id]["thread"]


        context = fetch_services()

        thread.write(f"Вот список доступных услуг:\n{context}")

        thread.write(question)

        logger.info("Отправка треда ассистенту.")
        run = assistant.run(thread)

        result = run.wait()
        logger.info(f"Ответ ассистента: {result.text}")

        return {"response": result.text}

    except Exception as e:
        logger.error(f"Ошибка обработки запроса от пользователя {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

@app.post("/end-session")
async def end_session(user_id: str):
    """
    Завершает сессию пользователя и удаляет тред.
    """
    try:
        if user_id in threads:
            threads[user_id]["thread"].delete()
            del threads[user_id]
            logger.info(f"Сессия для пользователя {user_id} завершена.")
        return {"message": "Сессия завершена"}
    except Exception as e:
        logger.error(f"Ошибка завершения сессии для пользователя {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка завершения сессии: {str(e)}")

# --- Запуск FastAPI ---
if __name__ == "__main__":
    logger.info("Запуск FastAPI сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
