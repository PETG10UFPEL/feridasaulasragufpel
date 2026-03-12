import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

DB_DIR = "data/chroma_db"
COLLECTION = "feridas_cronicas"

# Modelo multilíngue — mesmo usado no ingest.py (obrigatório ser idêntico)
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Modelo Groq: llama-3.3-70b-versatile tem boa compreensão de português
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_groq_api_key() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def _emb():
    # Embeddings locais — sem API key, roda completamente no servidor
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def _db():
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=_emb(),
        collection_name=COLLECTION,
    )


@st.cache_resource(show_spinner=False)
def _llm():
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "Sem GROQ_API_KEY. Configure em .streamlit/secrets.toml ou variável de ambiente."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.2,
    )


def answer(question: str, patient_summary: str = "", k: int = 4):
    db = _db()

    # Verifica se o banco está vazio e retorna mensagem amigável
    try:
        count = db._collection.count()  # type: ignore[attr-defined]
        if count == 0:
            return (
                "⚠️ Banco vetorial vazio. Use o menu Admin → 'Recriar índice' para indexar os documentos.",
                [],
            )
    except Exception:
        pass

    docs = db.similarity_search(question, k=k)

    context = "\n\n---\n\n".join((d.page_content or "")[:800] for d in docs)

    prompt = f"""Você é um assistente educacional para estudantes de graduação em Enfermagem sobre feridas crônicas.

Regras:
- Responda em português do Brasil.
- Use somente o material do CONTEXTO (trechos recuperados). Não invente protocolos, doses, números ou condutas.
- Se o CONTEXTO não trouxer base suficiente, diga explicitamente: "não encontrei no material indexado" e sugira o que consultar.
- Estruture a resposta em: (1) Resumo em 3 linhas, (2) Passos práticos, (3) Alertas/limites.
- Quando possível, mencione a origem como: (Fonte: <arquivo> | <página/trecho>).

CONTEXTO CLÍNICO (se houver):
{patient_summary}

CONTEXTO (trechos recuperados, recortados):
{context}

PERGUNTA:
{question}

RESPOSTA:
"""

    resp = _llm().invoke(prompt)
    text = getattr(resp, "content", resp)

    hits = []
    for d in docs:
        meta = d.metadata or {}
        hits.append({
            "metadata": meta,
            "page_content": d.page_content or "",
            "snippet": (d.page_content or "")[:300],
        })

    return text, hits
