from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DB_PATH = "./chroma_db"
OPENAI_API_KEY = "sk-..."

# 1. Подключение к существующей базе
embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Берем 5 самых релевантных кусков

# 2. Настройка LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

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
response = rag_chain.invoke({"input": "Как создать ноду на Python в ROS 2 Humble?"})
print(response["answer"])