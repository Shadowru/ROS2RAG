import os
import requests
import re
import html
from celery_app import app
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import create_retrieval_chain 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

rag_chain = None

def init_rag():
    global rag_chain
    print("Worker: Загрузка моделей...")
    
    DB_PATH = "/app/chroma_db"
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embedding_function
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("MODEL_NAME", "llama3"),
        temperature=0
    )
    
    # ИЗМЕНЕНИЕ 1: Просим Markdown, а не HTML. Это намного надежнее.
    system_prompt = (
    "Ты эксперт по ROS 2. Отвечай на русском языке.\n"
    "Используй Markdown для форматирования ответа.\n"
    "Код ОБЯЗАТЕЛЬНО оборачивай в тройные обратными кавычки с указанием языка, например:\n"
    "```cpp\n"
    "rclcpp::init(argc, argv);\n"
    "```\n"
    "Жирный текст выделяй двойными звездочками (**text**).\n"
    "Для списков используй дефисы (-).\n"
    "\nКонтекст:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("Worker: RAG готов к работе.")

def text_to_telegram_html(text):
    """
    Преобразует Markdown-подобный текст от LLM в валидный Telegram HTML.
    Алгоритм:
    1. Экранируем весь текст (защита от инъекций и < > в коде).
    2. Разбиваем по блокам кода (```).
    1. В текстовых частях обрабатываем жирный, курсив, код, ссылки.
    2. В блоках кода оборачиваем в <pre>.
    """
    # 1. Сначала экранируем всё, чтобы <ros2/rclcpp.h> не сломал HTML
    text = html.escape(text, quote=False)

    # 2. Разбиваем текст на части: код и не код
    # Паттерн ищет ```язык ... ```
    # Используем split, чтобы сохранить порядок
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    
    final_parts = []
    
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            # --- ЭТО БЛОК КОДА ---
            # Убираем кавычки
            content = part[3:-3].strip()
            # Если первая строка это язык (например cpp), убираем её
            first_line_end = content.find('\n')
            if first_line_end != -1 and first_line_end < 20: # защита от длинных строк
                # Проверяем, похоже ли это на название языка (буквы, цифры, +)
                lang_candidate = content[:first_line_end].strip()
                if re.match(r'^[a-zA-Z0-9+]+$', lang_candidate):
                    content = content[first_line_end+1:]
            
            # Оборачиваем в pre (code внутри pre не обязателен для Telegram, но pre обязателен для блока)
            final_parts.append(f"<pre>{content}</pre>")
        else:
            # --- ЭТО ОБЫЧНЫЙ ТЕКСТ ---
            # Обрабатываем Markdown элементы
            
            # Жирный: **text** -> <b>text</b>
            part = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', part)
            
            # Курсив: *text* -> <i>text</i> (аккуратно, чтобы не задеть списки)
            # part = re.sub(r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)', r'<i>\1</i>', part)
            
            # Инлайн код: `text` -> <code>text</code>
            part = re.sub(r'`([^`]+)`', r'<code>\1</code>', part)
            
            # Ссылки: [text](url) -> <a href="url">text</a>
            part = re.sub(r'$$(.*?)$$$(.*?)$', r'<a href="\2">\1</a>', part)
            
            final_parts.append(part)
            
    return "".join(final_parts)

def split_html_safe(text, limit=3000):
    """
    Простое и безопасное разбиение.
    Разбиваем по параграфам (\n\n), чтобы минимизировать шанс разрезать тег.
    Если параграф гигантский (больше лимита) — режем жестко, но это редкость.
    """
    chunks = []
    current_chunk = ""
    
    # Разбиваем по двойным переносам (абзацам)
    paragraphs = text.split('\n\n')
    
    for p in paragraphs:
        # Если добавление параграфа превысит лимит
        if len(current_chunk) + len(p) + 2 > limit:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Если сам параграф больше лимита (например, огромный код)
            if len(p) > limit:
                # Тут ничего не поделаешь, режем кусками
                for i in range(0, len(p), limit):
                    chunks.append(p[i:i+limit])
            else:
                current_chunk = p
        else:
            if current_chunk:
                current_chunk += "\n\n" + p
            else:
                current_chunk = p
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def send_chunk(chat_id, text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True}
    }
    
    response = requests.post(url, json=data)
    
    if not response.ok:
        print(f"HTML failed: {response.text}. Trying plain text.")
        # Если HTML не прошел (например, незакрытый тег), шлем как есть без форматирования
        # Предварительно убрав теги, чтобы не выглядело мусором
        clean_text = re.sub(r'<[^>]+>', '', text) 
        data["text"] = clean_text
        del data["parse_mode"]
        requests.post(url, json=data)

@app.task(name="process_ros2_query", bind=True)
def process_ros2_query(self, chat_id, user_query):
    global rag_chain
    if rag_chain is None:
        init_rag()
        
    try:
        response = rag_chain.invoke({"input": user_query})
        answer = response["answer"]
        
        # Формируем источники
        sources = set([doc.metadata.get('source', 'unknown').split('/')[-1] for doc in response['context']])
        if sources:
            answer += "\n\n**📚 Источники:**\n" + "\n".join([f"- {s}" for s in sources])
        
        # --- КОНВЕРТАЦИЯ ---
        # 1. LLM дала Markdown -> превращаем в безопасный HTML
        html_text = text_to_telegram_html(answer)
        
        # 2. Разбиваем на части
        chunks = split_html_safe(html_text, limit=3500)
        
        for chunk in chunks:
            send_chunk(chat_id, chunk)
            
    except Exception as e:
        error_msg = f"⚠️ Ошибка при генерации: {str(e)}"
        send_chunk(chat_id, error_msg)
    
    return "OK"