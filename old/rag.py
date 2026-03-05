import os
import streamlit as st
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma


DB_DIR = "data/chroma_db"
COLLECTION = "feridas_cronicas"


def get_api_key():
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except:
        pass
    return os.getenv("GOOGLE_API_KEY")


# ---- CACHE ----

@st.cache_resource(show_spinner=False)
def get_embeddings():
    api_key = get_api_key()
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="retrieval_query",
    )


@st.cache_resource(show_spinner=False)
def get_db():
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION,
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = get_api_key()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )


# ---- RAG ----

def answer(question: str, patient_summary: str = "", k: int = 4):

    db = get_db()

    if db._collection.count() == 0:
        raise RuntimeError(
            "Banco vetorial vazio. Rode ingest.py ou 'Recriar índice' no admin."
        )

    docs = db.similarity_search(question, k=k)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Você é um assistente educacional sobre feridas crônicas.

Contexto das aulas e materiais:

{context}

Resumo do paciente:
{patient_summary}

Pergunta:
{question}

Responda de forma clara e objetiva.
"""

    llm = get_llm()

    resp = llm.invoke(prompt)

    hits = [
        {
            "source": d.metadata.get("source", ""),
            "snippet": d.page_content[:300],
        }
        for d in docs
    ]

    return resp.content, hits