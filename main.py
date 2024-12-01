import re        # для работы с регулярными выражениями
import codecs
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
import openai
from litellm import completion
import configparser
import os

config = configparser.ConfigParser()
config.read("config.ini")


# Функция создания индексной базы знаний
def create_index_db(database):
    source_chunks = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=4096, chunk_overlap=1)

    for chunk in splitter.split_text(database):
      source_chunks.append(Document(page_content=chunk, metadata={}))

    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    # model_kwargs = {'device': 'cpu'}
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
      model_name=model_id,
      model_kwargs=model_kwargs
    )

    db = FAISS.from_documents(source_chunks, embeddings)
    return db

# Функция загруки содержимого текстового файла
def load_text(file_path):
    # Открытие файла для чтения
    with codecs.open(file_path, "r", encoding="utf-8", errors="ignore") as input_file:
        # Чтение содержимого файла
        content = input_file.read()
    return content

# Функция получения релевантные чанков из индексной базы знаний на основе заданной темы
def get_message_content(topic, index_db, k_num):
    # Поиск релевантных отрезков из базы знаний
    docs = index_db.similarity_search(topic, k = k_num)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#### Document excerpt №{i+1}####\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    print(f"message_content={message_content}")
    return message_content

# Функция отправки запроса в модель и получения ответа от модели
def answer_index(system, topic, message_content, temp):
    openai.api_type = "open_ai"
    openai.base_url = "http://127.0.0.1:1234"
    openai.api_key = "no need anymore"
    openai.api_version = "v1"


    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Here is the document with information to respond to the client: {message_content}\n\n Here is the client's question: \n{topic}"}
    ]

    response = completion(
        model="ollama/llama3.2:1b",
        messages=messages,
        api_base="http://localhost:11434"
    )
    # completion = openai.chat.completions.create(
    #     model='llama-3.2-3b-instruct',
    #     messages=messages,
    #     temperature=temp
    # )
    # cohere_key = config.get("Credentials", "COHARE_TRIAL_KEY")
    # print(cohere_key)
    # os.environ["COHERE_API_KEY"] = cohere_key
    # response = completion(
    #     model="command-r",
    #     messages= messages
    # )
    # deepseek_key = config.get("Credentials", "DEEPSEEK_API_KEY")
    # print(deepseek_key)
    # os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    # response = completion(
    #     model="deepseek/deepseek-chat",
    #     messages=messages
    # )
    answer = response.choices[0].message.content

    return answer  # возвращает ответ

# Загружаем текст Базы Знаний из файла
database = load_text('G:\PyPG\Big_LLM_Project\Data\OrderDeliciousBot_KnowledgeBase_01.txt')
# Создаем индексную Базу Знаний
index_db = create_index_db(database)
# Загружаем промпт для модели, который будет подаваться в system
system = load_text('G:\PyPG\Big_LLM_Project\Data\OrderDeliciousBot_Prompt_01.txt')



def answer_user_question(topic):
    # Ищем реливантные вопросу чанки и формируем контент для модели, который будет подаваться в user
    message_content = get_message_content(topic, index_db, k_num=2)
    # Делаем запрос в модель и получаем ответ модели
    ans = answer_index(system, topic, message_content, temp=0.2)
    return ans

if __name__ == '__main__':
    topic ="Whats is the restouran name , wherer i can eat Labneh?  Describe Labhen"
    ans = answer_user_question(topic)
    print(ans)
