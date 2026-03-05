import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def build_index(raw_dir="data/raw_docs", db_dir="data/chroma_db"):
    # Busca chave de forma híbrida (Secrets ou Env)
    api_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    # Inicializa Embeddings com o modelo funcional confirmado
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="retrieval_document"
    )

    # Carregamento e Split (Lógica simplificada para brevidade)
    # ... (mantenha sua lógica de load_all_docs aqui) ...

    # Criação do banco
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name="feridas_cronicas"
    )
    return len(chunks)

if __name__ == "__main__":
    build_index()