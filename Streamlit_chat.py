# Install the necessary libraries with the following command before running the script:
# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_vectorstore_from_url(url):
    """
    Create a vector store by loading text from a given URL and splitting it into chunks.
    """
    logging.info(f"Loading website content from URL: {url}")
    loader = WebBaseLoader(url)
    document = loader.load()
    logging.info("Website content loaded successfully.")

    # Split the document into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    logging.info(f"Document split into {len(document_chunks)} chunks.")

    # Create a vector store from the document chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    logging.info("Vector store created successfully.")

    return vector_store

def get_context_retriever_chain(vector_store):
    """
    Create a retriever chain that uses a language model to refine search queries based on chat history.
    """
    logging.info("Initializing context-aware retriever chain.")
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    # Define the prompt for generating search queries
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query to get information relevant to the conversation above.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    logging.info("Context-aware retriever chain created successfully.")
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    """
    Create a retrieval-augmented generation (RAG) chain for answering user questions based on retrieved context.
    """
    logging.info("Initializing conversational RAG chain.")
    llm = ChatOpenAI()

    # Define the prompt for answering user questions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # Combine the retriever chain with the document processing chain
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    logging.info("Conversational RAG chain created successfully.")
    return rag_chain

def get_response(user_input):
    """
    Generate a response to the user's input using the RAG chain.
    """
    logging.info("Generating response for user input.")
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    logging.info("Response generated successfully.")
    return response['answer']

# Streamlit app configuration
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Main application logic
if website_url is None or website_url == "":
    st.info("Please enter a website URL to start.")
else:
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]
    if "vector_store" not in st.session_state:
        logging.info("Creating vector store for the first time.")
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # Capture user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        # Generate and display response
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

logging.info("Application executed successfully.")
