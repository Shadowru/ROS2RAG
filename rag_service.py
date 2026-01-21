from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.chains import create_retrieval_chain
from langchain_classic.chains import create_retrieval_chain 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


DB_PATH = "./chroma_db"
OPENAI_API_KEY = "sk-..."

# 1. Подключение к существующей базе
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Берем 5 самых релевантных кусков

# 2. Настройка LLM
llm = ChatOllama(model="gpt-oss:120b", temperature=0)

# 3. Промпт (инструкция боту)
system_prompt = (
    "Ты эксперт по ROS 2 (Robot Operating System). "
    "Используй предоставленный контекст, чтобы ответить на вопрос. "
    "Если в контексте нет ответа, скажи, что не знаешь, не выдумывай. "
    "Приводи примеры кода, если это уместно. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Сборка цепи
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Тест
response = rag_chain.invoke({"input": "Что делает geometry в urdf (ROS 2 Humble)?"})
print(response["answer"])