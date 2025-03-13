import aiohttp
import asyncio
import aiofiles
import json
import uvicorn
import logging
import time
import os
import numpy as np
import re
import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from voicerecognise import recognize_audio_with_sdk
from yandex_cloud_ml_sdk import YCloudML
from typing import Dict, List, Optional
import faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_DIR = "base"
EMBEDDINGS_DIR = "embeddings_data"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"
YANDEX_FOLDER_ID = "b1gb9k14k5ui80g91tnp"  # Замените на ID вашего каталога Yandex Cloud
YANDEX_API_KEY = "AQVN1bqbPjz_tqxihcUv8sZSBJwXtCC0fBuOuqip" # Замените на ваш API-ключ Yandex Cloud
AI_ASSISTANT_ID = "your_assistant_id_from_yc"  # <---- Вставьте сюда ID своего ассистента из Yandex Cl

if not YANDEX_FOLDER_ID or not YANDEX_API_KEY:
    logger.error("Переменные окружения YANDEX_FOLDER_ID или YANDEX_API_KEY не заданы.")
    raise ValueError("Необходимо задать переменные окружения YANDEX_FOLDER_ID и YANDEX_API_KEY.")

logger.info("Загрузка моделей...")
search_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
logger.info("Модели успешно загружены.")

thread_ids: Dict[str, str] = {}  # Временное хранение thread_id в памяти (для примера!)
sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)
assistant = None  # Глобальная переменная для ассистента

app = FastAPI()

async def setup_assistant():
    """Функция для настройки ассистента, вызывается один раз при первом запросе."""
    global assistant
    if assistant is None:
        logger.info("Настройка ассистента...")
        model = sdk.models.completions("yandexgpt", model_version="rc")
        assistant = await asyncio.get_event_loop().run_in_executor(None, lambda: sdk.assistants.create(
            model, ttl_days=100, expiration_policy="static", max_tokens=1000  # Установлены ttl_days=100 и max_tokens=1000
        ))
        logger.info("Ассистент настроен.")

async def get_or_create_thread_id(user_id: str) -> str:
    """
    Возвращает ID существующего треда для user_id, или создает новый и возвращает его ID.
    В production-приложении thread_ids нужно хранить в базе данных.
    """
    if user_id in thread_ids:
        return thread_ids[user_id]
    else:
        logger.info(f"Создание нового треда для user_id={user_id}")
        try:
            loop = asyncio.get_event_loop()
            # Используем параметры ttl_days, expiration_policy и name при создании треда
            thread = await loop.run_in_executor(
                None,
                lambda: sdk.threads.create(
                    name=f"UserThread-{user_id}",  # Имя треда для идентификации пользователя
                    ttl_days=100,  # Установлен ttl_days=100 дней
                    expiration_policy="static"
                )
            )
            thread_id = thread.id
            thread_ids[user_id] = thread_id
            logger.info(f"Создан тред ID={thread_id} для user_id={user_id}, всего тредов в памяти: {len(thread_ids)}")
            return thread_id
        except Exception as e:
            logger.error(f"Ошибка при создании треда для user_id={user_id}: {e}")
            raise HTTPException(status_code=500, detail="Ошибка создания треда.")


def get_tenant_path(tenant_id: str) -> Path:
    """Создает папку для конкретного тенанта"""
    tenant_path = Path(EMBEDDINGS_DIR) / tenant_id
    tenant_path.mkdir(parents=True, exist_ok=True)
    return tenant_path


def normalize_text(text: str) -> str:
    """Нормализует текст: удаляет лишние символы, приводит к нижнему регистру."""
    text = text.strip()
    text = re.sub(r"[^\w\s\d\n]", "", text)
    return text.lower()


def tokenize_text(text: str) -> List[str]:
    """Токенизирует текст: разбивает на слова, удаляет стоп-слова."""
    stopwords = {
        "и", "в", "на", "с", "по", "для", "как", "что", "это", "но",
        "а", "или", "у", "о", "же", "за", "к", "из", "от", "так", "то", "все"
    }
    tokens = text.split()
    return [word for word in tokens if word not in stopwords]


def extract_text_fields(record: dict) -> str:
    """Формирует многострочное текстовое представление записи."""
    filial = record.get("filialName", "Филиал не указан")
    category = record.get("categoryName", "Категория не указана")
    service = record.get("serviceName", "Услуга не указана")
    service_desc = record.get("serviceDescription", "Описание услуги не указано")
    price = record.get("price", "Цена не указана")
    specialist = record.get("employeeFullName", "Специалист не указан")
    spec_desc = record.get("employeeDescription", "Описание не указано")
    text = (
        f"Филиал: {filial}\n"
        f"Категория: {category}\n"
        f"Услуга: {service}\n"
        f"Описание услуги: {service_desc}\n"
        f"Цена: {price}\n"
        f"Специалист: {specialist}\n"
        f"Описание специалиста: {spec_desc}"
    )
    return normalize_text(text)


async def load_json_data(tenant_id: str) -> List[dict]:
    """Загружает данные из JSON-файла и преобразует их в список записей."""
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл для tenant_id={tenant_id} не найден.")

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при чтении JSON файла {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла {file_path}: некорректный JSON формат.")
    except Exception as e:
        logger.error(f"Ошибка при открытии или чтении файла {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла {file_path}: {str(e)}.")


    records = []
    branches = data.get("data", {}).get("branches", [])
    for branch in branches:
        filial_name = branch.get("name", "Филиал не указан")
        categories = branch.get("categories", [])
        for category in categories:
            category_name = category.get("name", "Категория не указана")
            services = category.get("services", [])
            for service in services:
                service_name = service.get("name", "Услуга не указана")
                price = service.get("price", "Цена не указана")
                service_description = service.get("description", "")
                employees = service.get("employees", [])
                if employees:
                    for emp in employees:
                        employee_full_name = emp.get("full_name", "Специалист не указан")
                        employee_description = emp.get("description", "Описание не указано")
                        record = {
                            "filialName": filial_name,
                            "categoryName": category_name,
                            "serviceName": service_name,
                            "serviceDescription": service_description,
                            "price": price,
                            "employeeFullName": employee_full_name,
                            "employeeDescription": employee_description
                        }
                        records.append(record)
                else:
                    record = {
                        "filialName": filial_name,
                        "categoryName": category_name,
                        "serviceName": service_name,
                        "serviceDescription": service_description,
                        "price": price,
                        "employeeFullName": "Специалист не указан",
                        "employeeDescription": "Описание не указано"
                    }
                    records.append(record)
    return records


async def prepare_data(tenant_id: str):
    """Подготавливает данные для тенанта: загружает JSON, строит эмбеддинги, BM25 и FAISS-индекс."""
    tenant_path = get_tenant_path(tenant_id)
    data_file = tenant_path / "data.json"
    embeddings_file = tenant_path / "embeddings.npy"
    bm25_file = tenant_path / "bm25.pkl"
    faiss_index_file = tenant_path / "faiss_index.index"

    force_rebuild = False # For debugging and testing, set to True to force rebuild
    if force_rebuild or not all([f.exists() for f in [data_file, embeddings_file, bm25_file, faiss_index_file]]):
        logger.info(f"Необходима перестройка данных для tenant_id={tenant_id} или форсированная перестройка.")
    elif all([f.exists() for f in [data_file, embeddings_file, bm25_file, faiss_index_file]]):
        file_age = time.time() - os.path.getmtime(data_file)
        if file_age < 2_592_000 and not force_rebuild:
            logger.info(f"Данные для tenant_id={tenant_id} актуальны (менее 30 дней). Загрузка из кэша.")
            try:
                async with aiofiles.open(data_file, "r", encoding="utf-8") as f:
                    data = json.loads(await f.read())
                embeddings = np.load(embeddings_file)
                with open(bm25_file, "rb") as f:
                    bm25 = pickle.load(f)
                index = faiss.read_index(str(faiss_index_file))
                return data, embeddings, bm25, index
            except Exception as e:
                logger.error(f"Ошибка при загрузке кэшированных данных для tenant_id={tenant_id}: {e}. Перестраиваем данные.")
                force_rebuild = True # Fallback to rebuild if loading from cache fails


    records = await load_json_data(tenant_id)
    documents = [extract_text_fields(record) for record in records]

    loop = asyncio.get_event_loop()
    embeddings, bm25 = await asyncio.gather(
        loop.run_in_executor(None, lambda: search_model.encode(documents, convert_to_tensor=True).cpu().numpy()),
        loop.run_in_executor(None, lambda: BM25Okapi([tokenize_text(doc) for doc in documents]))
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_index_file))

    try:
        async with aiofiles.open(data_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps({
                "records": records,
                "raw_texts": documents,
                "timestamp": time.time()
            }, ensure_ascii=False, indent=4))
    except Exception as e:
        logger.error(f"Ошибка при сохранении data.json: {e}")

    try:
        np.save(embeddings_file, embeddings)
    except Exception as e:
        logger.error(f"Ошибка при сохранении embeddings.npy: {e}")
    try:
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)
    except Exception as e:
        logger.error(f"Ошибка при сохранении bm25.pkl: {e}")

    return {"records": records, "raw_texts": documents}, embeddings, bm25, index


async def update_json_file(mydtoken: str, tenant_id: str):
    """
    Обновляет данные, запрашивая страницы последовательно, но ограничиваясь первыми 500 страницами.
    """
    tenant_path = get_tenant_path(tenant_id)
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")

    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age < 2_592_000: # 30 days
            logger.info(f"Файл {file_path} актуален (менее 30 дней), пропускаем обновление.")
            return

    logger.info(f"Файл {file_path} устарел или отсутствует, начинаем обновление.")
    for f in tenant_path.glob("*"):
        try:
            os.remove(f)
        except Exception as e:
            logger.error(f"Ошибка удаления файла {f}: {e}")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {mydtoken}"}
            params = {"tenantId": tenant_id, "page": 1}
            all_data = []
            max_pages = 500
            while True:
                if params["page"] > max_pages:
                    logger.info(f"Достигнут лимит {max_pages} страниц ({max_pages}), завершаем загрузку.")
                    break
                try:
                    async with session.get(API_URL, headers=headers, params=params, timeout=30) as response: # Added timeout
                        response.raise_for_status()
                        data = await response.json()
                        branches = data.get("data", {}).get("branches", [])
                        if not branches:
                            logger.info(f"Страница {params['page']} пустая, завершаем загрузку.")
                            break
                        all_data.extend(branches)
                        logger.info(f"Получено {len(branches)} записей со страницы {params['page']}.")
                        params["page"] += 1
                except aiohttp.ClientError as e:
                    logger.error(f"Ошибка HTTP запроса на странице {params['page']}: {e}")
                    break # Stop on HTTP error to prevent infinite loop
                except asyncio.TimeoutError:
                    logger.error(f"Превышено время ожидания ответа от API на странице {params['page']}.")
                    break # Stop on timeout

            logger.info(f"Общее число полученных филиалов: {len(all_data)}")
            async with aiofiles.open(file_path, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps(
                    {"code": data.get("code", 200), "data": {"branches": all_data}},
                    ensure_ascii=False,
                    indent=4
                ))
            logger.info(f"Файл {file_path} успешно обновлен.")

    except Exception as e:
        logger.error(f"Ошибка при обновлении файла: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления данных.")


async def rerank_with_cross_encoder(query: str, candidates: List[int], raw_texts: List[str]) -> List[int]:
    """Реранкинг топ-10 кандидатов с использованием кросс-энкодера"""
    cross_inp = [(query, raw_texts[idx]) for idx in candidates]
    loop = asyncio.get_event_loop()
    cross_scores = await loop.run_in_executor(None, lambda: cross_encoder.predict(cross_inp))
    sorted_indices = np.argsort(cross_scores)[::-1].tolist()
    return [candidates[i] for i in sorted_indices]


free_times_function = {
    "name": "getFreeTimesOfEmployeeByChoosenServices",
    "description": "Получить свободное время сотрудника по выбранным услугам",
    "parameters": {
        "type": "object",
        "properties": {
            "employeeId": {"type": "string", "description": "ID сотрудника"},
            "serviceId": {"type": "array", "items": {"type": "string"}, "description": "Список ID услуг"},
            "dateTime": {"type": "string", "description": "Дата в формате YYYY-MM-DD"},
            "tenantId": {"type": "string", "description": "ID тенанта"},
            "filialId": {"type": "string", "description": "ID филиала"},
            "langId": {"type": "string", "description": "Язык (например, ru)"}
        },
        "required": ["employeeId", "serviceId", "dateTime", "tenantId", "filialId", "langId"]
    }
}


async def generate_yandexgpt_response(context: str, question: str, user_id: str) -> str:
    """
    Генерирует ответ с использованием YandexGPT, используя треды API для управления контекстом.
    """
    if assistant is None: # Ensure assistant is setup, call setup_assistant here if not called elsewhere on startup
        await setup_assistant()

    thread_id = await get_or_create_thread_id(user_id) # Получаем или создаем thread_id
    model_uri = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-32k/rc"

    try:
        loop = asyncio.get_event_loop()
        # Записываем вопрос пользователя в тред
        await loop.run_in_executor(None, lambda: sdk.threads.write(thread_id, question))

        result = await loop.run_in_executor(
            None,
            lambda: sdk.threads.completions(thread_id, model_uri) # Запускаем completions ДЛЯ ТРЕДА
                .configure(temperature=0.55, max_tokens=1000) # max_tokens=1000 установлен здесь
                .run([]) # Сообщения передаются через thread.write, здесь пустой список
        )
        if result and result.alternatives:
            response_text = result.alternatives[0].text.strip()
            logger.info(f"Ответ GPT для user_id={user_id}, thread_id={thread_id}: {response_text}")
            return response_text
        return "Извините, мне не удалось сгенерировать ответ."
    except Exception as e:
        logger.error(f"Ошибка YandexGPT API для user_id={user_id}, thread_id={thread_id}: {str(e)}")
        return "Извините, произошла ошибка при генерации ответа."


@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: Optional[str] = Form(None),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    file: UploadFile = File(None)
):
    start_time = time.time()
    try:
        await setup_assistant() 

        recognized_text = None
        if file and file.filename:
            temp_path = f"/tmp/{file.filename}"
            try:
                async with aiofiles.open(temp_path, "wb") as temp_file:
                    await temp_file.write(await file.read())
                loop = asyncio.get_event_loop()
                recognized_text = await loop.run_in_executor(None, lambda: recognize_audio_with_sdk(temp_path))
            except Exception as e:
                logger.error(f"Ошибка при обработке аудиофайла: {e}")
                raise HTTPException(status_code=500, detail=f"Ошибка обработки аудиофайла: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            if not recognized_text:
                raise HTTPException(status_code=500, detail="Ошибка распознавания речи из файла.")

        input_text = recognized_text or question
        if not input_text:
            raise HTTPException(status_code=400, detail="Необходимо передать текст или файл.")

        update_start_time = time.time()
        await update_json_file(mydtoken, tenant_id)
        update_end_time = time.time()
        logger.info(f"Время обновления JSON: {update_end_time - update_start_time:.2f} сек.")


        prepare_start_time = time.time()
        data_dict, embeddings, bm25, faiss_index = await prepare_data(tenant_id)
        prepare_end_time = time.time()
        logger.info(f"Время подготовки данных: {prepare_end_time - prepare_start_time:.2f} сек.")


        normalized_question = normalize_text(input_text)
        tokenized_query = tokenize_text(normalized_question)

        bm25_start_time = time.time()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:50].tolist()
        bm25_end_time = time.time()
        logger.info(f"Время BM25 поиска: {bm25_end_time - bm25_start_time:.2f} сек.")

        faiss_start_time = time.time()
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: search_model.encode(normalized_question, convert_to_tensor=True).cpu().numpy()
        )
        D, I = faiss_index.search(query_embedding.reshape(1, -1), 50)
        DISTANCE_THRESHOLD = 10.0
        filtered_faiss = [idx for idx, dist in zip(I[0].tolist(), D[0].tolist()) if dist < DISTANCE_THRESHOLD]
        if not filtered_faiss:
            filtered_faiss = I[0].tolist()
        top_faiss_indices = filtered_faiss
        faiss_end_time = time.time()
        logger.info(f"Время FAISS поиска: {faiss_end_time - faiss_start_time:.2f} сек.")


        combined_indices = list(set(top_bm25_indices + top_faiss_indices))[:50]
        rerank_start_time = time.time()
        top_10_indices = await rerank_with_cross_encoder(
            query=normalized_question,
            candidates=combined_indices[:30],
            raw_texts=data_dict["raw_texts"]
        )
        rerank_end_time = time.time()
        logger.info(f"Время реранкинга: {rerank_end_time - rerank_start_time:.2f} сек.")


        context_start_time = time.time()
        context = "\n\n".join([
    f"**Документ {i+1}:**\n"
    f"* Филиал: {data_dict['records'][idx].get('filialName', 'Не указан')}\n"
    f"* Категория: {data_dict['records'][idx].get('categoryName', 'Не указана')}\n"
    f"* Услуга: {data_dict['records'][idx].get('serviceName', 'Не указана')}\n"
    f"* Цена: {data_dict['records'][idx].get('price', 'Цена не указана')} руб.\n"
    f"* Специалист: {data_dict['records'][idx].get('employeeFullName', 'Не указан')}\n"
    f"* Описание: {data_dict['records'][idx].get('employeeDescription', 'Описание не указано')}"
    for i, idx in enumerate(top_10_indices[:5])
])
        context_end_time = time.time()
        logger.info(f"Время построения контекста: {context_end_time - context_start_time:.2f} сек.")


        gpt_start_time = time.time()
        response_text = await generate_yandexgpt_response(context, input_text, user_id) #  Вызов generate_yandexgpt_response
        gpt_end_time = time.time()
        logger.info(f"Время генерации ответа YandexGPT: {gpt_end_time - gpt_start_time:.2f} сек.")


        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Общее время обработки запроса: {total_time:.2f} сек.")

        return {"response": response_text, "time_taken": total_time}
    except HTTPException as http_exc:
        logger.warning(f"HTTPException: {http_exc.detail}", exc_info=True)
        raise http_exc

if __name__ == "__main__":
    logger.info("Запуск сервера на порту 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
