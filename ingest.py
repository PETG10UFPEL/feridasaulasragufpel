import os
import ftfy
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None  # type: ignore

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
except Exception:
    UnstructuredWordDocumentLoader = None  # type: ignore

try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore


load_dotenv()

# Modelo multilíngue: suporta PT, EN, ES e +50 línguas — roda local, sem API key
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def _fix_encoding(docs: List) -> List:
    """
    Corrige problemas de encoding em textos extraídos de PDFs e DOCX.
    Usa ftfy para reparar bytes mal interpretados — comum em documentos PT-BR
    com caracteres como ç, ã, é salvos em latin-1 mas lidos como UTF-8.
    """
    for doc in docs:
        try:
            doc.page_content = ftfy.fix_text(doc.page_content)
        except Exception:
            # Fallback: força re-encode latin-1 → utf-8
            try:
                doc.page_content = (
                    doc.page_content.encode("latin-1", errors="replace")
                    .decode("utf-8", errors="replace")
                )
            except Exception:
                pass
    return docs


def _load_docs_recursive(raw_dir: str) -> List:
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
                loaded = PyPDFLoader(str(p)).load()
                docs.extend(_fix_encoding(loaded))

            elif ext in (".txt", ".md"):
                try:
                    loaded = TextLoader(str(p), encoding="utf-8").load()
                except UnicodeDecodeError:
                    loaded = TextLoader(str(p), encoding="latin-1").load()
                docs.extend(_fix_encoding(loaded))

            elif ext == ".docx":
                if UnstructuredWordDocumentLoader is None:
                    print(f"[AVISO] Pulando DOCX (loader indisponível): {p}")
                    continue
                loaded = UnstructuredWordDocumentLoader(str(p)).load()
                docs.extend(_fix_encoding(loaded))

            else:
                continue

        except Exception as e:
            print(f"[AVISO] Falha ao carregar {p}: {e}")

    return docs


def build_index(
    raw_dir: str = "data/raw_docs",
    db_dir: str = "data/chroma_db",
    gdrive_folder_id: str = "",
) -> int:
    """
    Indexa documentos localmente com HuggingFace (sem API paga, sem cota).
    Suporta documentos em português, inglês e outras línguas simultaneamente.
    Corrige automaticamente problemas de encoding em documentos PT-BR.

    gdrive_folder_id — se informado, faz upload do índice ao Drive após criar,
                       permitindo que o Streamlit recupere o índice após dormir.
    """
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
        print("[FIM] Documentos carregados, mas 0 chunks gerados.")
        return 0

    # Embeddings locais — multilíngue, sem API key, sem limite de cota
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[OK] Gerando embeddings para {len(chunks)} chunks (modelo local)...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name="feridas_cronicas",
    )

    try:
        vectordb.persist()
    except Exception:
        pass

    print(f"[OK] Índice criado: {len(chunks)} chunks — {Path(db_dir).resolve()}")

    # Salva índice no Drive para sobreviver ao sleep do Streamlit
    if gdrive_folder_id:
        print("[Drive] Salvando índice no Google Drive...")
        try:
            from drive_sync import upload_index_to_drive
            ok = upload_index_to_drive(db_dir, gdrive_folder_id)
            if ok:
                print("[Drive] Índice salvo com sucesso no Drive.")
            else:
                print("[Drive] Falha ao salvar índice no Drive (verifique logs acima).")
        except Exception as e:
            print(f"[Drive] Erro ao salvar índice: {e}")

    return len(chunks)


if __name__ == "__main__":
    build_index()
