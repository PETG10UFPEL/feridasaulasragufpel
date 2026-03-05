import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders (tentamos os mais comuns; se não existir, a gente pula com aviso)
from langchain_community.document_loaders import TextLoader

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None  # type: ignore

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
except Exception:
    UnstructuredWordDocumentLoader = None  # type: ignore

# Streamlit é opcional (ingest deve rodar no terminal sem UI)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore


load_dotenv()


def _get_api_key() -> Optional[str]:
    """Pega chave do Streamlit secrets (se disponível) ou do .env / variáveis de ambiente."""
    api_key = None
    if st is not None:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return api_key


def _load_docs_recursive(raw_dir: str) -> List:
    """Carrega documentos recursivamente em raw_dir (subpastas inclusas)."""
    base = Path(raw_dir)
    if not base.exists():
        print(f"[ERRO] Pasta não existe: {base.resolve()}")
        return []

    files = [p for p in base.rglob("*") if p.is_file()]
    if not files:
        print(f"[ERRO] 0 arquivos encontrados em: {base.resolve()} (incluindo subpastas).")
        return []

    docs = []
    for p in files:
        ext = p.suffix.lower()

        try:
            if ext == ".pdf":
                if PyPDFLoader is None:
                    print(f"[AVISO] Pulando PDF (PyPDFLoader indisponível): {p}")
                    continue
                docs.extend(PyPDFLoader(str(p)).load())

            elif ext in (".txt", ".md"):
                # encoding latin-1 às vezes salva a pele em material antigo
                try:
                    docs.extend(TextLoader(str(p), encoding="utf-8").load())
                except UnicodeDecodeError:
                    docs.extend(TextLoader(str(p), encoding="latin-1").load())

            elif ext == ".docx":
                if UnstructuredWordDocumentLoader is None:
                    print(f"[AVISO] Pulando DOCX (loader indisponível): {p}")
                    continue
                docs.extend(UnstructuredWordDocumentLoader(str(p)).load())

            else:
                # ignora o resto (imagens, etc.)
                continue

        except Exception as e:
            print(f"[AVISO] Falha ao carregar {p}: {e}")

    return docs


def build_index(raw_dir: str = "data/raw_docs", db_dir: str = "data/chroma_db") -> int:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Sem GOOGLE_API_KEY/GEMINI_API_KEY. Defina no .env ou variáveis de ambiente.")

    print("Indexando a partir de:", Path(raw_dir).resolve())
    docs = _load_docs_recursive(raw_dir)

    if not docs:
        print("[FIM] Nada para indexar (nenhum documento carregado).")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        print("[FIM] Documentos carregados, mas 0 chunks gerados. (Checa loaders/conteúdo)")
        return 0

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="retrieval_document",
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name="feridas_cronicas",
    )
    # versões novas persistem automaticamente; manter compatibilidade:
    try:
        vectordb.persist()
    except Exception:
        pass

    print(f"[OK] Index criado: {len(chunks)} chunks em {Path(db_dir).resolve()} (coleção: feridas_cronicas)")
    return len(chunks)


if __name__ == "__main__":
    build_index()