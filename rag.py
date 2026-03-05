import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

DB_DIR = "data/chroma_db"
COLLECTION = "feridas_cronicas"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_api_key() -> str | None:
    # Prefer secrets (Streamlit Cloud), fallback to env/.env
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


@st.cache_resource(show_spinner=False)
def _emb(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="retrieval_query",
    )


@st.cache_resource(show_spinner=False)
def _db(api_key: str):
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=_emb(api_key),
        collection_name=COLLECTION,
    )


@st.cache_resource(show_spinner=False)
def _llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )


def answer(question: str, patient_summary: str = "", k: int = 4):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Sem GOOGLE_API_KEY/GEMINI_API_KEY. Configure em .env ou .streamlit/secrets.toml.")

    db = _db(api_key)

    # Falha explícita se o banco estiver vazio
    try:
        if db._collection.count() == 0:  # type: ignore[attr-defined]
            raise RuntimeError("Banco vetorial vazio. Rode ingest.py ou 'Recriar índice' no Admin.")
    except Exception:
        pass

    docs = db.similarity_search(question, k=k)

    # Contexto enxuto (evita mandar texto demais pro LLM)
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

    resp = _llm(api_key).invoke(prompt)
    text = getattr(resp, "content", resp)

    hits = []
    for d in docs:
        meta = d.metadata or {}
        hits.append(
            {
                "metadata": meta,
                "page_content": d.page_content or "",
                "snippet": (d.page_content or "")[:300],
            }
        )

    return text, hits
