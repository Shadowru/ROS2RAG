import os
import requests
from celery_app import app
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import create_retrieval_chain 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Глобальные переменные для кэширования модели в памяти воркера
rag_chain = None

def init_rag():
    """Инициализация RAG (загрузка эмбеддингов и базы)"""
    global rag_chain
    print("Worker: Загрузка моделей...")
    
    DB_PATH = "/app/chroma_db_local" # Путь внутри Docker
    
    # 1. Эмбеддинги (CPU)
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. База
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embedding_function
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 3. LLM (Ollama)
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("MODEL_NAME", "llama3"),
        temperature=0
    )
    
    # 4. Промпт
    system_prompt = (
        "Ты эксперт по ROS 2. Отвечай на русском языке. "
        "Используй контекст ниже. Если не знаешь, скажи 'Не знаю'.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("Worker: RAG готов к работе.")

@app.task(name="process_ros2_query", bind=True)
def process_ros2_query(self, chat_id, user_query):
    """Задача Celery"""
    global rag_chain
    if rag_chain is None:
        init_rag()
        
    try:
        # Генерация ответа
        response = rag_chain.invoke({"input": user_query})
        answer = response["answer"]
        
        # Формирование источников
        sources = set([doc.metadata.get('source', 'unknown').split('/')[-1] for doc in response['context']])
        source_text = "\n\n📚 Источники:\n" + "\n".join([f"- {s}" for s in sources])
        
        final_text = answer + source_text
        
    except Exception as e:
        final_text = f"⚠️ Ошибка при генерации: {str(e)}"

    # Отправка ответа обратно в Telegram (напрямую через API, минуя бота-приемщика)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": final_text,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=data)
    
    return "OK"