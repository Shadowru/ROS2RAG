import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredRSTLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Настройки
DOCS_PATH = "./ros2_documentation" # Путь к склонированному репо
DB_PATH = "./chroma_db"
OPENAI_API_KEY = "sk-..." # Ваш ключ

# 1. Загрузка документов (RST и MD)
print("Загрузка документов...")
# Для Markdown
md_loader = DirectoryLoader(DOCS_PATH, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
# Для reStructuredText (основной формат ROS)
rst_loader = DirectoryLoader(DOCS_PATH, glob="**/*.rst", loader_cls=UnstructuredRSTLoader)

docs = []
docs.extend(md_loader.load())
docs.extend(rst_loader.load())

print(f"Загружено {len(docs)} документов.")

# 2. Чанкинг (разбиение на куски)
# Важно: для кода и конфигов ROS лучше делать чанки побольше с перекрытием
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_documents(docs)

# 3. Создание эмбеддингов и сохранение в ChromaDB
embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

print("Создание векторной базы...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_function,
    persist_directory=DB_PATH
)

print("Готово! База создана.")