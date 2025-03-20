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
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from rank_bm25 import BM25Okapi
from voicerecognise import recognize_audio_with_sdk
from typing import Dict, List, Optional, Sequence, Any, Union, Literal
from typing_extensions import Annotated, TypedDict
import faiss
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.tools.giga_tool import giga_tool, FewShotExamples
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta

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
GIGACHAT_API_KEY = "OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OmNlOGUzMjhmLWQ5MDEtNDBjOS04YWJjLWI0Mjc1NTlkMzdjNg==" 

logger.info("Загрузка моделей...")
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru").to(device)
cross_encoder = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").to(device)
logger.info(f"Модели успешно загружены на устройстве: {device}")

# Словари для хранения истории разговоров и состояния записи
conversation_history: Dict[str, Dict] = {}
appointment_states = {}  # Словарь для хранения состояния процесса записи

app = FastAPI()


# === Модели Pydantic для структур данных записи на прием ===
class AppointmentSlot(BaseModel):
    time: str = Field(description="Время слота в формате HH:MM")
    date: str = Field(description="Дата слота в формате YYYY.MM.DD")

class WorkDateItem(BaseModel):
    date: str = Field(description="Дата в формате YYYY.MM.DD")
    timeSlots: List[str] = Field(description="Список доступных временных слотов в формате HH:MM")
    
class DoctorSlotsInfo(BaseModel):
    doctor_id: str = Field(description="ID врача")
    doctor_name: str = Field(description="Имя врача")
    work_dates: List[WorkDateItem] = Field(description="Доступные даты и временные слоты")
    photo_url: Optional[str] = Field(description="URL фото врача", default=None)

class GetSlotsResult(BaseModel):
    status: str = Field(description="Статус операции: success или error")
    doctor_info: Optional[DoctorSlotsInfo] = Field(description="Информация о враче и доступных слотах", default=None)
    message: Optional[str] = Field(description="Сообщение о результате операции", default=None)

class FormatSlotsResult(BaseModel):
    status: str = Field(description="Статус операции: success или error")
    formatted_message: str = Field(description="Отформатированное сообщение со слотами")
    doctor_id: str = Field(description="ID врача")
    doctor_name: str = Field(description="Имя врача")
    dates: List[Dict[str, Any]] = Field(description="Форматированные данные о датах")
    
class TimeSelectionResult(BaseModel):
    status: str = Field(description="Статус операции: success или error")
    message: str = Field(description="Подтверждение выбора даты и времени")
    selected_date: str = Field(description="Выбранная дата")
    selected_time: str = Field(description="Выбранное время")
    doctor_id: str = Field(description="ID врача")
    service_id: str = Field(description="ID услуги")

class AppointmentResult(BaseModel):
    status: str = Field(description="Статус операции: success или error")
    message: str = Field(description="Сообщение о результате операции")
    appointment_id: Optional[str] = Field(description="ID созданной записи", default=None)

class ClientInfoResult(BaseModel):
    status: str = Field(description="Статус операции: success или error")
    name: str = Field(description="Имя клиента")
    phone: str = Field(description="Номер телефона клиента")
    message: Optional[str] = Field(description="Сообщение о результате", default=None)


# Mean Pooling - для корректного усреднения с учетом маски внимания
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Первый элемент содержит все token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings(texts, batch_size=32):
    logger.info(f"Создание эмбеддингов для {len(texts)} текстов")
    start_time = time.time()
    
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(batch_embeddings.cpu().numpy())
        
        if batch_idx % 10 == 0:
            logger.info(f"Обработано {min(end_idx, len(texts))}/{len(texts)} текстов")
    
    result = np.vstack(embeddings)
    logger.info(f"Эмбеддинги созданы за {time.time() - start_time:.2f}с, размерность: {result.shape}")
    
    return result


def get_tenant_path(tenant_id: str) -> Path:
    tenant_path = Path(EMBEDDINGS_DIR) / tenant_id
    tenant_path.mkdir(parents=True, exist_ok=True)
    return tenant_path


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s\d\n.,?!-]", "", text)
    return text.lower()


def tokenize_text(text: str) -> List[str]:
    from nltk.stem.snowball import SnowballStemmer
    
    try:
        stemmer = SnowballStemmer("russian")
    except Exception as e:
        logger.warning(f"Не удалось создать стеммер: {str(e)}. Используем упрощенную токенизацию.")
        stemmer = None

    stopwords = {
        "и", "в", "на", "с", "по", "для", "как", "что", "это", "но",
        "а", "или", "у", "о", "же", "за", "к", "из", "от", "так", "то", "все"
    }
    
    tokens = text.split()
    if stemmer:
        return [stemmer.stem(word) for word in tokens if word not in stopwords]
    else:
        return [word for word in tokens if word not in stopwords]


def extract_text_fields(record: dict) -> str:
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


async def convert_base_json_to_data_json(tenant_id: str) -> bool:
    try:
        base_file = os.path.join(BASE_DIR, f"{tenant_id}.json")
        if not os.path.exists(base_file):
            logger.error(f"Базовый файл {base_file} не найден")
            return False
            
        tenant_path = get_tenant_path(tenant_id)
        data_file = tenant_path / "data.json"
        
        logger.info(f"Конвертация JSON для тенанта {tenant_id}")
        
        async with aiofiles.open(base_file, "r", encoding="utf-8") as f:
            content = await f.read()
            base_data = json.loads(content)
        
        records = []
        raw_texts = []
        branches = base_data.get("data", {}).get("branches", [])
        
        for branch in branches:
            filial_id = branch.get("id", "")
            filial_name = branch.get("name", "")
            
            categories = branch.get("categories", [])
            
            for category in categories:
                category_id = category.get("id", "")
                category_name = category.get("name", "")
                
                services = category.get("services", [])
                
                for service in services:
                    service_id = service.get("id", "")
                    service_name = service.get("name", "")
                    service_description = service.get("description", "")
                    price = service.get("price", 0)
                    duration = service.get("duration", 0)
                    
                    employees = service.get("employees", [])
                    if employees:
                        for employee in employees:
                            record = {
                                "filialId": filial_id,
                                "filialName": filial_name,
                                "categoryId": category_id,
                                "categoryName": category_name,
                                "serviceId": service_id,
                                "serviceName": service_name,
                                "serviceDescription": service_description,
                                "price": price,
                                "duration": duration,
                                "employeeId": employee.get("id", ""),
                                "employeeFullName": employee.get("full_name", ""),
                                "employeeDescription": employee.get("description", ""),
                                "employeeExperience": employee.get("experience", ""),
                                "employeeTechnologies": employee.get("technologies", [])
                            }
                            records.append(record)
                            raw_texts.append(extract_text_fields(record))
                    else:
                        record = {
                            "filialId": filial_id,
                            "filialName": filial_name,
                            "categoryId": category_id,
                            "categoryName": category_name,
                            "serviceId": service_id,
                            "serviceName": service_name,
                            "serviceDescription": service_description,
                            "price": price,
                            "duration": duration,
                            "employeeId": "",
                            "employeeFullName": "",
                            "employeeDescription": "",
                            "employeeExperience": "",
                            "employeeTechnologies": []
                        }
                        records.append(record)
                        raw_texts.append(extract_text_fields(record))
        
        logger.info(f"Обработано {len(records)} записей")
        
        async with aiofiles.open(data_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps({
                "records": records,
                "raw_texts": raw_texts,
                "timestamp": time.time()
            }, ensure_ascii=False, indent=4))
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при конвертации JSON: {str(e)}")
        return False


async def load_json_data(tenant_id: str) -> List[dict]:
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        logger.error(f"Файл {file_path} не найден")
        raise HTTPException(status_code=404, detail=f"Файл для tenant_id={tenant_id} не найден.")
    
    logger.info(f"Загрузка данных из {file_path}")
    
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)
    
    records = []
    branches = data.get("data", {}).get("branches", [])
    
    for branch in branches:
        filial_id = branch.get("id", "")
        filial_name = branch.get("name", "Филиал не указан")
        categories = branch.get("categories", [])
        for category in categories:
            category_id = category.get("id", "")
            category_name = category.get("name", "Категория не указана")
            services = category.get("services", [])
            for service in services:
                service_id = service.get("id", "")
                service_name = service.get("name", "Услуга не указана")
                price = service.get("price", "Цена не указана")
                service_description = service.get("description", "")
                employees = service.get("employees", [])
                if employees:
                    for emp in employees:
                        employee_id = emp.get("id", "")
                        employee_full_name = emp.get("full_name", "Специалист не указан")
                        employee_description = emp.get("description", "Описание не указано")
                        record = {
                            "filialId": filial_id,
                            "filialName": filial_name,
                            "categoryId": category_id,
                            "categoryName": category_name,
                            "serviceId": service_id,
                            "serviceName": service_name,
                            "serviceDescription": service_description,
                            "price": price,
                            "employeeId": employee_id,
                            "employeeFullName": employee_full_name,
                            "employeeDescription": employee_description
                        }
                        records.append(record)
                else:
                    record = {
                        "filialId": filial_id,
                        "filialName": filial_name,
                        "categoryId": category_id,
                        "categoryName": category_name,
                        "serviceId": service_id,
                        "serviceName": service_name,
                        "serviceDescription": service_description,
                        "price": price,
                        "employeeId": "",
                        "employeeFullName": "Специалист не указан",
                        "employeeDescription": "Описание не указано"
                    }
                    records.append(record)
    
    logger.info(f"Загружено {len(records)} записей")
    return records


async def prepare_data(tenant_id: str):
    tenant_path = get_tenant_path(tenant_id)
    data_file = tenant_path / "data.json"
    embeddings_file = tenant_path / "embeddings.npy"
    bm25_file = tenant_path / "bm25.pkl"
    faiss_index_file = tenant_path / "faiss_index.index"
    
    # Проверяем наличие и актуальность файлов
    if all([f.exists() for f in [data_file, embeddings_file, bm25_file, faiss_index_file]]):
        file_age = time.time() - os.path.getmtime(data_file)
        if file_age < 2_592_000:  # 30 дней
            logger.info(f"Используем кэшированные данные для тенанта {tenant_id}")
            
            async with aiofiles.open(data_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            
            embeddings = np.load(embeddings_file)
            
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)
            
            index = faiss.read_index(str(faiss_index_file))
            
            return data, embeddings, bm25, index

    # Если файлы не существуют или устарели, создаем заново
    logger.info(f"Подготовка новых данных для тенанта {tenant_id}")
    
    # Загружаем и обрабатываем данные
    records = await load_json_data(tenant_id)
    documents = [extract_text_fields(record) for record in records]
    logger.info(f"Подготовлено {len(documents)} документов для индексации")

    # Параллельно создаем эмбеддинги и BM25 индекс
    logger.info("Создание эмбеддингов и BM25 индекса...")
    
    loop = asyncio.get_event_loop()
    embeddings, bm25 = await asyncio.gather(
        loop.run_in_executor(None, lambda: get_embeddings(documents, batch_size=32)),
        loop.run_in_executor(None, lambda: BM25Okapi([tokenize_text(doc) for doc in documents]))
    )

    # Создаем FAISS индекс
    logger.info("Создание FAISS индекса...")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_index_file))

    # Сохраняем все данные
    logger.info("Сохранение данных в файлы...")
    
    async with aiofiles.open(data_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps({
            "records": records,
            "raw_texts": documents,
            "timestamp": time.time()
        }, ensure_ascii=False, indent=4))

    np.save(embeddings_file, embeddings)
    
    with open(bm25_file, "wb") as f:
        pickle.dump(bm25, f)
    
    logger.info("Подготовка данных завершена")

    return {"records": records, "raw_texts": documents}, embeddings, bm25, index


async def update_json_file(mydtoken: str, tenant_id: str):
    tenant_path = get_tenant_path(tenant_id)
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    
    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age < 2_592_000:  # 30 дней
            logger.info(f"Файл {file_path} актуален, пропускаем обновление.")
            return
    
    logger.info(f"Обновление данных для тенанта {tenant_id}")
    
    # Удаляем старые файлы
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
            
            logger.info(f"Загрузка данных из API")
            
            while True:
                if params["page"] > max_pages:
                    logger.info(f"Достигнут лимит {max_pages} страниц")
                    break
                
                async with session.get(API_URL, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    branches = data.get("data", {}).get("branches", [])
                    if not branches:
                        logger.info(f"Страница {params['page']} пустая")
                        break
                    
                    all_data.extend(branches)
                    logger.info(f"Получено {len(branches)} записей с страницы {params['page']}")
                    params["page"] += 1

            logger.info(f"Общее число полученных филиалов: {len(all_data)}")
            
            # Сохраняем данные в JSON
            async with aiofiles.open(file_path, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps(
                    {"code": data.get("code", 200), "data": {"branches": all_data}},
                    ensure_ascii=False,
                    indent=4
                ))
            
    except Exception as e:
        logger.error(f"Ошибка при обновлении файла: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления данных.")


async def rerank_with_cross_encoder(query: str, candidates: List[int], raw_texts: List[str]) -> List[int]:
    logger.info(f"Перераниркование {len(candidates)} кандидатов")
    
    # Создаем функцию для кросс-энкодер ранжирования
    def compute_cross_scores(query, texts):
        # Используем токенизатор и модель кросс-энкодера
        pairs = [(query, text) for text in texts]
        encoded_pairs = tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = cross_encoder(**encoded_pairs)
            # Используем logits из правильной модели
            logits = outputs.logits
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
        
        return scores
    
    # Подготавливаем данные для ранжирования
    texts_to_rank = [raw_texts[idx] for idx in candidates]
    
    # Выполняем ранжирование
    loop = asyncio.get_event_loop()
    cross_scores = await loop.run_in_executor(None, lambda: compute_cross_scores(query, texts_to_rank))
    
    # Сортируем результаты по убыванию релевантности
    sorted_indices = np.argsort(cross_scores)[::-1].tolist()
    result = [candidates[i] for i in sorted_indices]
    
    return result


# Определение типа состояния для чат-бота
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str


# === Функции-инструменты (tools) для записи на прием ===
@giga_tool(
    few_shot_examples=[
        {
            "request": "Меня зовут Иван Петров, телефон +79001234567",
            "params": {"name": "Иван Петров", "phone": "+79001234567"}
        },
        {
            "request": "Запиши меня, я Светлана, 89991112233",
            "params": {"name": "Светлана", "phone": "89991112233"}
        }
    ]
)
async def get_client_info(
    name: str = Field(description="Полное имя клиента"),
    phone: str = Field(description="Номер телефона клиента в формате +7XXXXXXXXXX или 8XXXXXXXXXX")
) -> ClientInfoResult:
    """
    Получает и сохраняет информацию о клиенте для записи на прием.
    Эту функцию нужно вызвать ТОЛЬКО ПОСЛЕ того, как пользователь сам предоставил своё имя и номер телефона.
    НЕ используйте тестовые данные из примеров ("Иван Петров" и "Светлана") - эти имена приведены только как образец.
    """
    # Проверка на тестовые данные из примеров
    test_names = ["Иван Петров", "Светлана"]
    test_phones = ["+79001234567", "89991112233"]
    
    if name in test_names and phone in test_phones:
        return ClientInfoResult(
            status="error",
            name="",
            phone="",
            message="Не используйте тестовые данные. Запросите имя и телефон у клиента."
        )
    
    logger.info(f"Получена информация о клиенте: {name}, {phone}")
    
    # Нормализация номера телефона
    normalized_phone = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    # Сохраняем в состоянии для данного пользователя
    user_id = None
    for uid, state in appointment_states.items():
        if state.get("current") == True:
            user_id = uid
            break
    
    if user_id:
        appointment_states[user_id]["client_name"] = name
        appointment_states[user_id]["client_phone"] = normalized_phone
        appointment_states[user_id]["stage"] = "collecting_info"
    
    return ClientInfoResult(
        status="success",
        name=name,
        phone=normalized_phone,
        message="Информация о клиенте сохранена"
    )

@giga_tool(
    few_shot_examples=[
        {
            "request": "Хочу к Марьям Есиеваа на косметологию",
            "params": {
                "employee_id": "ea541f08-5666-4156-835f-61654184146f", 
                "service_id": "s123", 
                "filial_id": "f1"
            }
        }
    ]
)
async def get_available_slots(
    employee_id: str = Field(description="ID специалиста"),
    service_id: str = Field(description="ID услуги"),
    date: Optional[str] = Field(description="Дата в формате YYYY-MM-DD", default=None),
    filial_id: str = Field(description="ID филиала")
) -> GetSlotsResult:
    """
    Получение доступных временных слотов для записи на прием к указанному специалисту.
    Возвращает информацию о враче и все доступные даты и временные слоты.
    """
    logger.info(f"Запрос доступных слотов: спец={employee_id}, услуга={service_id}, филиал={filial_id}")
    
    try:
        # Здесь будет реальный вызов API getFreeTimesOfEmployeeByChosenServices
        api_url = "https://dev.back.matrixcrm.ru/api/v1/AI/getFreeTimesOfEmployeeByChosenServices"
        tenant_id = "tenant123"  # Используйте tenant_id из контекста вызова или дополнительного параметра
        
        payload = {
            "employeeId": employee_id,
            "serviceId": [service_id],
            "dateTime": date or "",
            "tenantId": tenant_id,
            "filialId": filial_id,
            "langId": "ru"
        }
        
        # Заглушка для тестирования - реальный вызов API будет добавлен позже
        # Тестовые данные
        test_data = {
            "id": "ea541f08-5666-4156-835f-61654184146f",
            "name": "Марьям Есиеваа Хасановна",
            "photoUrl": "https://cdn.matrixcrm.ru/medyumed.2023-04-24/63d121ca-0290-4815-859a-d43bd1a51add.png",
            "workDates": [
                {
                    "date": "2025.3.19",
                    "timeSlots": ["10:00", "10:15", "10:30", "11:00", "11:15", "12:00"]
                },
                {
                    "date": "2025.3.20",
                    "timeSlots": ["10:00", "10:15", "10:30", "14:00", "14:15", "15:00"]
                }
            ]
        }
        
        # Сохраняем информацию о враче в состоянии
        user_id = None
        for uid, state in appointment_states.items():
            if state.get("current") == True:
                user_id = uid
                break
        
        if user_id:
            appointment_states[user_id]["doctor_id"] = test_data.get("id", "")
            appointment_states[user_id]["doctor_name"] = test_data.get("name", "")
            appointment_states[user_id]["available_slots"] = test_data.get("workDates", [])
            appointment_states[user_id]["service_id"] = service_id
            appointment_states[user_id]["filial_id"] = filial_id
            appointment_states[user_id]["stage"] = "selecting_time"
        
        # Создаем правильно типизированные объекты для work_dates
        work_dates = [
            WorkDateItem(date=date_item["date"], timeSlots=date_item["timeSlots"])
            for date_item in test_data.get("workDates", [])
        ]
        
        return GetSlotsResult(
            status="success",
            doctor_info=DoctorSlotsInfo(
                doctor_id=test_data.get("id", ""),
                doctor_name=test_data.get("name", ""),
                work_dates=work_dates,
                photo_url=test_data.get("photoUrl", "")
            )
        )
                    
    except Exception as e:
        logger.error(f"Ошибка при получении слотов: {str(e)}")
        return GetSlotsResult(
            status="error",
            message=f"Ошибка при получении слотов: {str(e)}"
        )

@giga_tool(
    few_shot_examples=[
        {
            "request": "Покажи доступное время",
            "params": {
                "doctor_id": "ea541f08-5666-4156-835f-61654184146f",
                "doctor_name": "Марьям Есиеваа Хасановна"
            }
        }
    ]
)
async def format_available_slots(
    doctor_id: str = Field(description="ID врача"),
    doctor_name: str = Field(description="Имя врача")
) -> FormatSlotsResult:
    """
    Форматирует доступные слоты для удобного отображения пользователю.
    Эта функция вызывается после получения информации о слотах, чтобы пользователь мог выбрать подходящее время.
    """
    # Получаем данные из состояния вместо передачи как параметр
    user_id = None
    for uid, state in appointment_states.items():
        if state.get("current") == True:
            user_id = uid
            break
    
    if not user_id or "available_slots" not in appointment_states[user_id]:
        return FormatSlotsResult(
            status="error",
            formatted_message="Информация о доступном времени не найдена",
            doctor_id=doctor_id,
            doctor_name=doctor_name,
            dates=[]
        )
        
    work_dates = appointment_states[user_id]["available_slots"]
    
    # Преобразование дат формата YYYY.M.D в более читаемый формат
    days_of_week = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    
    formatted_dates = []
    
    for work_date in work_dates:
        date_str = work_date.get("date", "")
        time_slots = work_date.get("timeSlots", [])
        
        # Форматирование даты
        try:
            year, month, day = map(int, date_str.split('.'))
            date_obj = datetime(year, int(month), int(day))
            day_name = days_of_week[date_obj.weekday()]
            month_name = months[int(month) - 1]
            formatted_date = f"{day} {month_name} ({day_name})"
            
            # Группировка временных слотов по часам для более компактного отображения
            time_groups = {}
            for time in time_slots:
                hour = time.split(':')[0]
                if hour not in time_groups:
                    time_groups[hour] = []
                time_groups[hour].append(time)
            
            time_display = []
            for hour, times in sorted(time_groups.items()):
                time_display.append(f"{hour}:00-{hour}:59: {len(times)} слотов")
            
            formatted_dates.append({
                "date": date_str,
                "display_date": formatted_date,
                "available_times": time_display,
                "all_slots": time_slots
            })
        except Exception as e:
            logger.error(f"Ошибка форматирования даты {date_str}: {str(e)}")
            # Если с форматированием проблемы, используем оригинальные данные
            formatted_dates.append({
                "date": date_str,
                "display_date": date_str,
                "available_times": [],
                "all_slots": time_slots
            })
    
    formatted_message = f"Врач: {doctor_name}\n\nДоступное время для записи:\n\n"
    
    for date_info in formatted_dates[:7]:  # Ограничим 7 днями для удобства
        formatted_message += f"📅 {date_info['display_date']}:\n"
        for time_info in date_info['available_times'][:5]:  # Показываем первые 5 групп времени
            formatted_message += f"   ⌚ {time_info}\n"
        formatted_message += "\n"
    
    formatted_message += "\nПожалуйста, выберите дату и время для записи. Например: \"Хочу записаться на 20 марта в 14:30\""
    
    return FormatSlotsResult(
        status="success",
        formatted_message=formatted_message,
        doctor_id=doctor_id,
        doctor_name=doctor_name,
        dates=formatted_dates
    )

@giga_tool(
    few_shot_examples=[
        {
            "request": "Хочу записаться на 20 марта в 14:30",
            "params": {
                "date": "2025.3.20", 
                "time": "14:30", 
                "doctor_id": "ea541f08-5666-4156-835f-61654184146f", 
                "service_id": "s123"
            }
        }
    ]
)
async def select_appointment_time(
    date: str = Field(description="Выбранная дата в формате YYYY.MM.DD или в формате текстового описания"),
    time: str = Field(description="Выбранное время в формате HH:MM"),
    doctor_id: str = Field(description="ID врача"),
    service_id: str = Field(description="ID услуги")
) -> TimeSelectionResult:
    """
    Проверяет доступность выбранного пользователем времени и даты.
    Эта функция вызывается после того, как пользователь выбрал подходящую дату и время из предложенных вариантов.
    """
    # Сохраняем выбранное время в состоянии
    user_id = None
    for uid, state in appointment_states.items():
        if state.get("current") == True:
            user_id = uid
            break
    
    if user_id:
        appointment_states[user_id]["selected_date"] = date
        appointment_states[user_id]["selected_time"] = time
        appointment_states[user_id]["stage"] = "confirmation"
    
    return TimeSelectionResult(
        status="success",
        message=f"Вы выбрали запись на {date} в {time}",
        selected_date=date,
        selected_time=time,
        doctor_id=doctor_id,
        service_id=service_id
    )

@giga_tool(
    few_shot_examples=[
        {
            "request": "Подтверждаю запись",
            "params": {
                "client_name": "Иван Петров",
                "client_phone": "+79001234567",
                "service_id": "s123",
                "employee_id": "ea541f08-5666-4156-835f-61654184146f",
                "filial_id": "f1",
                "date": "2025.3.20",
                "time": "14:30"
            }
        }
    ]
)
async def create_appointment(
    client_name: str = Field(description="Имя клиента"),
    client_phone: str = Field(description="Телефон клиента"),
    service_id: str = Field(description="ID услуги"),
    employee_id: str = Field(description="ID специалиста"),
    filial_id: str = Field(description="ID филиала"),
    date: str = Field(description="Дата приема в формате YYYY.MM.DD"),
    time: str = Field(description="Время начала приема в формате HH:MM")
) -> AppointmentResult:
    """
    Создает запись клиента на прием к специалисту.
    Используйте эту функцию после подтверждения пользователем всех деталей записи.
    """
    logger.info(f"Создание записи: {client_name} к {employee_id} на {date} {time}")
    
    try:
        # Вычисляем время окончания (+30 минут к времени начала)
        start_hour, start_minute = map(int, time.split(':'))
        end_minute = start_minute + 30
        end_hour = start_hour
        
        if end_minute >= 60:
            end_minute -= 60
            end_hour += 1
            
        end_time = f"{end_hour:02d}:{end_minute:02d}"
        
        # Подготовка даты в нужном формате для API
        date_parts = date.split('.')
        formatted_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}" if len(date_parts) == 3 else date
        
        # Заглушка для тестирования - в реальности здесь будет вызов API addRecord
        appointment_id = str(uuid.uuid4())
        
        # Очищаем состояние записи после успешного создания
        user_id = None
        for uid, state in appointment_states.items():
            if state.get("current") == True:
                user_id = uid
                break
        
        if user_id:
            appointment_states[user_id] = {
                "stage": "completed",
                "appointment_id": appointment_id,
                "client_name": client_name,
                "client_phone": client_phone,
                "doctor_id": employee_id,
                "doctor_name": appointment_states[user_id].get("doctor_name", ""),
                "selected_date": date,
                "selected_time": time
            }
        
        return AppointmentResult(
            status="success",
            message="Запись успешно создана!",
            appointment_id=appointment_id
        )
    except Exception as e:
        logger.error(f"Ошибка при создании записи: {str(e)}")
        return AppointmentResult(
            status="error",
            message=f"Ошибка при создании записи: {str(e)}. Пожалуйста, попробуйте позже или свяжитесь с клиникой по телефону."
        )


# Инструменты для LangGraph
@tool
def get_context_info(context: str) -> str:
    """
    Получает информацию об услугах, специалистах и ценах из контекста.
    
    Args:
        context: Контекст из базы данных услуг
        
    Returns:
        Информация об услугах, специалистах и ценах
    """
    return context


# Инициализация чат-бота с LangGraph
def init_chat_agent(context: str = ""):
    try:
        logger.info("Инициализация чат-агента")
        
        giga = GigaChat(
            credentials=GIGACHAT_API_KEY,
            scope="GIGACHAT_API_PERS",
            model="GigaChat",
            verify_ssl_certs=False
        )
        
        # Создаем шаблон промпта с системным сообщением
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """# 🔹 Системный промпт для модели 🔹

Ты – ассистент клиники MED YU MED по имени Аида. Твоя задача – помогать пользователям находить информацию о специалистах, услугах, филиалах и ценах. Ты работаешь как RAG-модель (Retrieval-Augmented Generation), что означает, что:

1. Весь контекст уже загружен – в нём есть подробные данные о специалистах, услугах, ценах и филиалах. Эта информация всегда доступна для поиска в контексте.

2. Тебе не нужно выдумывать информацию – если данных нет в контексте, прямо сообщай об этом. 

3. Если пользователь уточняет запрос, ты обязана искать ответ именно в загруженном контексте, а не только в истории диалога. 

## 📌 1. Анализ запроса

- Прочитай контекст и пойми, к чему относится запрос: услуги, цены, специалисты, филиалы и т. д.
- Если запрос неясен – задай уточняющий вопрос вместо того, чтобы догадываться.

## 🔍 2. Подбор услуг

Если пользователь спрашивает про услуги, ты обязана:
- Найти все подходящие услуги.
- Указать цену (например, "от 12000 рублей 💸").
- Назвать филиал, где доступна услуга (Москва – Ходынка, Москва – Сити, Дубай).
- Перечислить всех специалистов, которые выполняют эту услугу (без слов "и другие", только полные списки!).
- Объяснить пользу услуги в 1-2 предложениях (например: "Эта процедура поможет убрать морщины и сделать кожу более упругой ✨").

В клинике три филиала: Москва (Ходынка, Сити), Дубай (Bluewaters).
Информация о контексте: {context}
"""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Функция для ограничения истории сообщений
        trimmer = trim_messages(
            max_tokens=500,  # Максимальное количество токенов
            strategy="last",   # Стратегия - оставляем последние сообщения
            token_counter=giga, # Счетчик токенов от модели
            include_system=True,  # Включаем системное сообщение
            allow_partial=False
        )
        
        # Создаем workflow для чат-бота
        workflow = StateGraph(state_schema=ChatState)
        
        # Определяем функцию для вызова модели
        async def call_model(state: ChatState):
            try:
                # Сначала обрезаем историю, чтобы не превысить лимиты контекста
                trimmed_messages = trimmer.invoke(state["messages"])
                
                # Передаем сообщения в шаблон промпта
                chain_input = {
                    "messages": trimmed_messages,
                    "context": state["context"]
                }
                
                prompt_output = await prompt.ainvoke(chain_input)
                response = await giga.ainvoke(prompt_output)
                
                return {"messages": [response]}
            except Exception as e:
                logger.error(f"Ошибка при вызове модели: {str(e)}")
                return {"messages": [AIMessage(content="Извините, произошла ошибка при обработке вашего запроса. Попробуйте задать вопрос по-другому.")]}
        
        # Добавляем вершину в граф
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        
        # Компилируем граф с сохранением состояния
        memory = MemorySaver()
        chat_app = workflow.compile(checkpointer=memory)
        
        return chat_app
    
    except Exception as e:
        logger.error(f"Ошибка инициализации чат-агента: {str(e)}")
        return None


# Запасной метод для генерации ответа через прямой вызов GigaChat
async def generate_gigachat_response(context: str, history: List[dict], question: str) -> str:
    try:
        logger.info("Запасной вызов GigaChat")
        
        giga = GigaChat(
            credentials=GIGACHAT_API_KEY,
            scope="GIGACHAT_API_PERS",
            model="GigaChat-Pro",
            verify_ssl_certs=False
        )
        
        # Формирование сообщений для чата
        messages = [
            SystemMessage(content="Ты – ассистент клиники MED YU MED по имени Аида.")
        ]
        
        # Добавляем контекст найденных данных в первое сообщение системы
        messages.append(SystemMessage(content=f"Вот список услуг:\n{context}\n"))
        
        # Добавляем историю диалога
        for entry in history[-10:]:
            messages.append(HumanMessage(content=entry['user_query']))
            messages.append(AIMessage(content=entry['assistant_response']))
        
        # Добавляем текущий вопрос
        messages.append(HumanMessage(content=question))
        
        trimmer = trim_messages(
            max_tokens=1000,
            strategy="last",
            token_counter=giga,
            include_system=True
        )
        
        trimmed_messages = trimmer.invoke(messages)
        
        # Отправляем запрос в GigaChat
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: giga.invoke(trimmed_messages)
        )
        
        # Возвращаем результат
        return response.content
        
    except Exception as e:
        logger.error(f"Ошибка GigaChat API: {str(e)}")
        return "Извините, произошла ошибка при генерации ответа. Пожалуйста, попробуйте позже."


# === Основная функция обработки запроса на запись ===
async def process_appointment_with_functions(user_id: str, input_text: str, tenant_id: str):
    """Обработка запроса на запись с использованием функций GigaChat"""
    
    # Инициализация GigaChat
    giga = GigaChat(
        credentials=GIGACHAT_API_KEY,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        verify_ssl_certs=False
    )
    
    # Инициализация истории диалога
    if user_id not in conversation_history:
        conversation_history[user_id] = {"history": [], "last_active": time.time(), "greeted": False}
    
    # Инициализация или обновление состояния записи
    if user_id not in appointment_states:
        appointment_states[user_id] = {
            "stage": "initial",
            "client_name": None,
            "client_phone": None,
            "service_id": None,
            "service_name": None,
            "doctor_id": None,
            "doctor_name": None,
            "filial_id": None,
            "selected_date": None,
            "selected_time": None,
            "available_slots": None,
            "current": True  # Флаг текущей сессии для доступа из функций
        }
    else:
        # Обновляем флаг текущей сессии
        for uid in appointment_states:
            appointment_states[uid]["current"] = (uid == user_id)
    
    # Список функций для GigaChat
    functions = [
        get_client_info, 
        get_available_slots, 
        format_available_slots,
        select_appointment_time,
        create_appointment
    ]
    
    # Привязка функций к GigaChat
    giga_with_functions = giga.bind_functions(functions)
    
    # Создание агента с памятью
    memory = MemorySaver()
    thread_id = f"appointment_{user_id}"
    
    # Настраиваем системное сообщение с учетом текущего этапа записи
    state = appointment_states[user_id]
    stage = state.get("stage", "initial")
    
    # Адаптируем системное сообщение в зависимости от этапа
    system_prompts = {
        "initial": """
        Ты – ассистент клиники MED YU MED по имени Аида. Помоги клиенту записаться на прием.
        
        ВАЖНО! Если клиент просто пишет "запиши меня" или подобное, обязательно сначала спроси его имя и телефон
        для контакта. НЕ ИСПОЛЬЗУЙ тестовые данные из примеров. Используй функцию get_client_info ТОЛЬКО ПОСЛЕ того,
        как пользователь сам предоставил свои данные.
        
        Пример правильного диалога:
        Клиент: "Запиши меня"
        Ты: "Конечно! Чтобы записать вас на прием, мне нужны ваши контактные данные. Подскажите, пожалуйста, ваше имя и номер телефона."
        
        Сначала получи имя и телефон клиента, затем выясни, к какому специалисту и на какую услугу 
        он хочет записаться, а также в какой филиал. Используй функцию get_client_info для сохранения 
        данных клиента, а затем get_available_slots для получения доступных слотов.
        """,
        "collecting_info": """
        Ты – ассистент клиники MED YU MED по имени Аида. Продолжаем запись на прием.
        
        Уже известно:
        - Имя: {client_name}
        - Телефон: {client_phone}
        
        Теперь выясни, к какому специалисту и на какую услугу клиент хочет записаться,
        а также в какой филиал. Используй функцию get_available_slots для получения доступных слотов.
        """,
        "selecting_time": """
        Ты – ассистент клиники MED YU MED по имени Аида. Необходимо выбрать время приема.
        
        Клиент: {client_name}
        Телефон: {client_phone}
        Врач: {doctor_name}
        
        Используй функцию format_available_slots для отображения доступных слотов,
        а затем select_appointment_time для сохранения выбора пользователя.
        
        Не передавай слоты в функцию format_available_slots, она получит их из состояния.
        """,
        "confirmation": """
        Ты – ассистент клиники MED YU MED по имени Аида. Все данные для записи собраны.
        
        Сводка:
        - Клиент: {client_name}
        - Телефон: {client_phone}
        - Врач: {doctor_name}
        - Дата: {selected_date}
        - Время: {selected_time}
        
        Подтверди детали записи с клиентом и используй функцию create_appointment 
        для создания записи после подтверждения.
        """,
        "completed": """
        Ты – ассистент клиники MED YU MED по имени Аида. Запись успешно создана!
        
        Запись подтверждена:
        - Клиент: {client_name}
        - Телефон: {client_phone}
        - Врач: {doctor_name}
        - Дата: {selected_date}
        - Время: {selected_time}
        
        Номер записи: {appointment_id}
        
        Поблагодари клиента за запись и сообщи, что ему нужно будет прийти на 10-15 минут раньше для оформления.
        """
    }
    
    # Форматируем системное сообщение с учетом данных пользователя
    system_message = system_prompts.get(stage, system_prompts["initial"]).format(
        client_name=state.get("client_name", "не указано"),
        client_phone=state.get("client_phone", "не указано"),
        doctor_name=state.get("doctor_name", "не указано"),
        selected_date=state.get("selected_date", "не указано"),
        selected_time=state.get("selected_time", "не указано"),
        appointment_id=state.get("appointment_id", "не указано")
    )
    
    try:
        # Создаем системное сообщение
        system_prompt = SystemMessage(content=system_message)
        
        # Создаем агента для обработки запроса
        agent_executor = create_react_agent(
            giga_with_functions, 
            functions, 
            checkpointer=memory
        )
        
        # Конфигурация для сохранения контекста
        config = {"configurable": {"thread_id": thread_id}}
        
        # Получаем историю сообщений
        history_messages = []
        for entry in conversation_history[user_id]["history"][-5:]:
            history_messages.append(HumanMessage(content=entry["user_query"]))
            history_messages.append(AIMessage(content=entry["assistant_response"]))
        
        # Формируем сообщения для запроса
        messages = [system_prompt] + history_messages + [HumanMessage(content=input_text)]
        
        # Вызываем агента
        response = await agent_executor.ainvoke(
            {"messages": messages}, 
            config=config
        )
        
        # Получаем ответ
        response_text = response["messages"][-1].content if "messages" in response and response["messages"] else "Извините, произошла ошибка при обработке запроса."
        
        # Сохраняем в историю диалога
        conversation_history[user_id]["history"].append({
            "user_query": input_text,
            "assistant_response": response_text
        })
        
        return response_text
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса на запись: {str(e)}")
        return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}. Пожалуйста, попробуйте еще раз или позвоните в клинику."


@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: Optional[str] = Form(None),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        logger.info(f"Обработка запроса от пользователя {user_id}")
        
        # Очистка неактивных пользователей
        current_time = time.time()
        expired_users = [uid for uid, data in conversation_history.items() if current_time - data["last_active"] > 22296]
        for uid in expired_users:
            del conversation_history[uid]
            if uid in appointment_states:
                del appointment_states[uid]

        # Обработка голосового файла
        recognized_text = None
        if file and file.filename:
            logger.info(f"Получен аудиофайл: {file.filename}")
            temp_path = f"/tmp/{file.filename}"
            try:
                async with aiofiles.open(temp_path, "wb") as temp_file:
                    await temp_file.write(await file.read())
                
                loop = asyncio.get_event_loop()
                recognized_text = await loop.run_in_executor(None, lambda: recognize_audio_with_sdk(temp_path))
                
                logger.info(f"Распознано: '{recognized_text}'")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if not recognized_text:
                logger.error("Не удалось распознать речь в файле")
                raise HTTPException(status_code=500, detail="Ошибка распознавания речи из файла.")

        input_text = recognized_text or question
        if not input_text:
            logger.error("Запрос не содержит ни текста, ни распознанного файла")
            raise HTTPException(status_code=400, detail="Необходимо передать текст или файл.")

        logger.info(f"Входной запрос: '{input_text}'")

        # ИСПРАВЛЕНО: Более точная проверка намерения записи
        # Более точные ключевые слова для определения намерения записи
        strong_appointment_keywords = ["запись", "записаться", "записать", "запиши", "запишите"]
        
        # Дополнительные ключевые слова, которые в сочетании указывают на запись
        context_appointment_keywords = ["прием", "консультация", "врач", "доктор", "специалист", "талон", "время"]
        
        # Более строгая проверка на намерение записи
        has_strong_keyword = any(keyword in input_text.lower() for keyword in strong_appointment_keywords)
        context_keyword_count = sum(1 for keyword in context_appointment_keywords if keyword in input_text.lower())
        
        # Запрос на запись, если есть сильное ключевое слово или много контекстных ключевых слов
        is_appointment_intent = has_strong_keyword or context_keyword_count >= 2
        is_in_appointment_process = user_id in appointment_states and appointment_states[user_id].get("stage") != "completed"
        
        if is_appointment_intent or is_in_appointment_process:
            # Используем механизм функций для записи на прием
            logger.info(f"Обработка запроса на запись: {input_text}")
            response_text = await process_appointment_with_functions(user_id, input_text, tenant_id)
            return {"response": response_text}

        # Если запрос не связан с записью, используем обычную обработку RAG
        # Сначала обновляем данные если необходимо
        force_update = False
        if force_update or not (get_tenant_path(tenant_id) / "data.json").exists():
            logger.info(f"Требуется обновление данных для тенанта {tenant_id}")
            await update_json_file(mydtoken, tenant_id)
        
        # Подготовка данных
        data_dict, embeddings, bm25, faiss_index = await prepare_data(tenant_id)
        
        # Нормализация и токенизация запроса
        normalized_question = normalize_text(input_text)
        tokenized_query = tokenize_text(normalized_question)
        
        # BM25 поиск
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:50].tolist()
        
        # Получение эмбеддинга запроса и поиск по FAISS
        async def encode_query():
            encoded_input = tokenizer([normalized_question], padding=True, truncation=True, 
                                     max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
            query_embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
            return query_embedding
        
        query_embedding = await encode_query()
        
        # Поиск по FAISS
        D, I = faiss_index.search(query_embedding, 50)
        
        # Адаптивный порог для релевантности
        mean_distance = np.mean(D[0])
        DISTANCE_THRESHOLD = mean_distance * 1.5
        
        filtered_faiss = [idx for idx, dist in zip(I[0].tolist(), D[0].tolist()) if dist < DISTANCE_THRESHOLD]
        if not filtered_faiss:
            filtered_faiss = I[0].tolist()[:50]
        
        # Комбинирование и взвешивание результатов
        norm_bm25 = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
        norm_faiss = 1 - (D[0] / np.max(D[0])) if np.max(D[0]) > 0 else D[0]
        
        combined_scores = {}
        for idx, score in zip(top_bm25_indices, norm_bm25[top_bm25_indices]):
            combined_scores[idx] = score * 0.4
        
        for idx, score in zip(I[0], norm_faiss):
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.6
        
        # Получаем итоговый список индексов для ранжирования
        combined_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:50]
        logger.info(f"Найдено {len(combined_indices)} релевантных документов")
        
        # Перераниркование кросс-энкодером для финального улучшения релевантности
        top_10_indices = await rerank_with_cross_encoder(
            query=normalized_question,
            candidates=combined_indices[:30],
            raw_texts=data_dict["raw_texts"]
        )
        
        # Формирование контекста для ответа
        context_docs = []
        
        for i, idx in enumerate(top_10_indices[:10]):  # Увеличено с 5 до 10
            doc = (
                f"**Документ {i+1}:**\n" 
                f"* Филиал: {data_dict['records'][idx].get('filialName', 'Не указан')}\n"
                f"* Категория: {data_dict['records'][idx].get('categoryName', 'Не указана')}\n"
                f"* Услуга: {data_dict['records'][idx].get('serviceName', 'Не указана')}\n"
                f"* Цена: {data_dict['records'][idx].get('price', 'Цена не указана')} руб.\n"
                f"* Специалист: {data_dict['records'][idx].get('employeeFullName', 'Не указан')}\n"
                f"* Описание: {data_dict['records'][idx].get('employeeDescription', 'Описание не указано')}"
            )
            context_docs.append(doc)
        
        context = "\n\n".join(context_docs)
        
        # Инициализация истории диалога для пользователя
        if user_id not in conversation_history:
            conversation_history[user_id] = {"history": [], "last_active": time.time(), "greeted": False}

        conversation_history[user_id]["last_active"] = time.time()
        
        # Формирование ответа
        chat_app = init_chat_agent()
        
        if chat_app:
            try:
                thread_id = f"user_{user_id}"
                config = {"configurable": {"thread_id": thread_id}}
                
                # Формируем историю сообщений
                input_messages = []
                for entry in conversation_history[user_id]["history"][-5:]:
                    input_messages.append(HumanMessage(content=entry["user_query"]))
                    input_messages.append(AIMessage(content=entry["assistant_response"]))
                
                # Добавляем текущий запрос
                input_messages.append(HumanMessage(content=input_text))
                
                # Вызываем LangGraph
                response = await chat_app.ainvoke(
                    {"messages": input_messages, "context": context},
                    config=config
                )
                
                # Извлекаем ответ
                if response and "messages" in response and response["messages"]:
                    response_text = response["messages"][-1].content
                else:
                    logger.warning("Не удалось получить ответ через LangGraph, используем запасной метод")
                    response_text = await generate_gigachat_response(context, conversation_history[user_id]["history"], input_text)

            except Exception as e:
                logger.error(f"Ошибка при использовании чат-бота: {str(e)}")
                response_text = await generate_gigachat_response(context, conversation_history[user_id]["history"], input_text)
        else:
            logger.warning("Не удалось инициализировать чат-бота, используем запасной метод")
            response_text = await generate_gigachat_response(context, conversation_history[user_id]["history"], input_text)
        
        # Сохраняем историю диалога
        conversation_history[user_id]["history"].append({
            "user_query": input_text,
            "assistant_response": response_text,
            "search_results": [data_dict['records'][idx] for idx in top_10_indices]
        })
        
        return {"response": response_text}
        
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")


if __name__ == "__main__":
    logger.info("Запуск сервера на порту 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
