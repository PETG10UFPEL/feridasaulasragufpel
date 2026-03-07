import os
import io
import streamlit as st
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from dotenv import load_dotenv

load_dotenv()

# ← MUDANÇA: adicionado drive.file para poder criar/atualizar arquivos no Drive
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
FOLDER_MIME = "application/vnd.google-apps.folder"

# Nome da pasta onde o índice Chroma será salvo no Drive
INDEX_FOLDER_NAME = "_chroma_index"


def get_drive_service():
    """
    Autentica no Google Drive de forma flexível:
    1. Tenta usar st.secrets (Streamlit Cloud)
    2. Tenta carregar o arquivo JSON local definido em variáveis de ambiente
    3. Tenta procurar o arquivo JSON padrão na pasta raiz
    """
    creds = None

    # 1) Streamlit Secrets (Produção)
    try:
        if "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        pass

    # 2) Arquivo Local (Desenvolvimento)
    if not creds:
        json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "disco-retina-371501-525385725005.json")
        if os.path.exists(json_path):
            creds = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)

    if not creds:
        raise RuntimeError(
            "Credenciais do Google Cloud não encontradas. "
            "Configure 'gcp_service_account' nos Secrets do Streamlit ou tenha o arquivo JSON localmente."
        )

    return build("drive", "v3", credentials=creds)


def _list_children(service, folder_id: str):
    """Lista itens diretamente dentro de folder_id (com paginação)."""
    q = f"'{folder_id}' in parents and trashed = false"
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _download_file(service, file_id: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _sync_folder_recursive(service, folder_id: str, out_root: Path, rel: Path = Path(".")) -> list[Path]:
    """Sincroniza recursivamente: percorre subpastas e baixa arquivos preservando a estrutura."""
    downloaded: list[Path] = []
    items = _list_children(service, folder_id)

    for it in items:
        name = it["name"]
        mime = it.get("mimeType", "")
        it_id = it["id"]

        if mime == FOLDER_MIME:
            downloaded.extend(_sync_folder_recursive(service, it_id, out_root, rel / name))
            continue

        if mime.startswith("application/vnd.google-apps"):
            continue

        dest = out_root / rel / name

        if not dest.exists():
            _download_file(service, it_id, dest)
            downloaded.append(dest)

    return downloaded


def sync_folder(folder_id: str, out_dir: str, recursive: bool = True) -> list[Path]:
    """
    Sincroniza uma pasta do Google Drive com um diretório local.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        service = get_drive_service()

        if recursive:
            downloaded = _sync_folder_recursive(service, folder_id, out, Path("."))
        else:
            downloaded = []
            items = _list_children(service, folder_id)
            for f in items:
                name = f["name"]
                file_id = f["id"]
                mime = f.get("mimeType", "")

                if mime == FOLDER_MIME or mime.startswith("application/vnd.google-apps"):
                    continue

                dest = out / name
                if not dest.exists():
                    _download_file(service, file_id, dest)
                    downloaded.append(dest)

        print(f"Baixados {len(downloaded)} arquivos para {out.resolve()}")
        return downloaded

    except Exception as e:
        print(f"Erro na sincronização: {e}")
        return []


# ==============================
# FUNÇÕES DE PERSISTÊNCIA DO ÍNDICE
# ==============================

def _get_or_create_index_folder(service, parent_folder_id: str) -> str:
    """
    Retorna o ID da pasta '_chroma_index' dentro de parent_folder_id.
    Cria a pasta se não existir.
    """
    q = (
        f"'{parent_folder_id}' in parents "
        f"and name = '{INDEX_FOLDER_NAME}' "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    resp = service.files().list(q=q, fields="files(id, name)").execute()
    files = resp.get("files", [])

    if files:
        return files[0]["id"]

    # Cria a pasta
    folder_meta = {
        "name": INDEX_FOLDER_NAME,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(body=folder_meta, fields="id").execute()
    print(f"[Drive] Pasta '{INDEX_FOLDER_NAME}' criada no Drive.")
    return folder["id"]


def upload_index_to_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    """
    Faz upload de todos os arquivos do diretório db_dir
    para a pasta '_chroma_index' no Google Drive.

    Arquivos existentes são substituídos (update); novos são criados.
    Retorna True em caso de sucesso.
    """
    db_path = Path(db_dir)
    if not db_path.exists():
        print(f"[ERRO] Pasta do índice não existe: {db_path}")
        return False

    files_to_upload = list(db_path.rglob("*"))
    files_to_upload = [f for f in files_to_upload if f.is_file()]

    if not files_to_upload:
        print("[AVISO] Nenhum arquivo encontrado para upload.")
        return False

    try:
        service = get_drive_service()
        index_folder_id = _get_or_create_index_folder(service, gdrive_folder_id)

        # Lista arquivos já existentes na pasta de índice (para update vs create)
        existing = {
            f["name"]: f["id"]
            for f in _list_children(service, index_folder_id)
            if f.get("mimeType") != FOLDER_MIME
        }

        uploaded = 0
        for file_path in files_to_upload:
            name = file_path.name
            media = MediaFileUpload(str(file_path), resumable=False)

            if name in existing:
                # Atualiza arquivo existente
                service.files().update(
                    fileId=existing[name],
                    media_body=media,
                ).execute()
            else:
                # Cria novo arquivo
                file_meta = {"name": name, "parents": [index_folder_id]}
                service.files().create(
                    body=file_meta,
                    media_body=media,
                    fields="id",
                ).execute()

            uploaded += 1

        print(f"[Drive] Índice salvo: {uploaded} arquivo(s) em '{INDEX_FOLDER_NAME}'.")
        return True

    except Exception as e:
        print(f"[ERRO] Falha ao salvar índice no Drive: {e}")
        return False


def download_index_from_drive(db_dir: str, gdrive_folder_id: str) -> bool:
    """
    Baixa todos os arquivos da pasta '_chroma_index' do Drive para db_dir.

    Retorna True se o índice existia e foi baixado com sucesso.
    Retorna False se a pasta de índice não existir no Drive.
    """
    try:
        service = get_drive_service()

        # Verifica se a pasta de índice existe
        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{INDEX_FOLDER_NAME}' "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        resp = service.files().list(q=q, fields="files(id, name)").execute()
        folders = resp.get("files", [])

        if not folders:
            print("[Drive] Pasta de índice '_chroma_index' não encontrada no Drive.")
            return False

        index_folder_id = folders[0]["id"]
        items = _list_children(service, index_folder_id)

        if not items:
            print("[Drive] Pasta '_chroma_index' está vazia no Drive.")
            return False

        db_path = Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)

        for item in items:
            if item.get("mimeType") == FOLDER_MIME:
                continue
            dest = db_path / item["name"]
            _download_file(service, item["id"], dest)

        print(f"[Drive] Índice baixado: {len(items)} arquivo(s) para '{db_path}'.")
        return True

    except Exception as e:
        print(f"[ERRO] Falha ao baixar índice do Drive: {e}")
        return False


def index_exists_on_drive(gdrive_folder_id: str) -> bool:
    """
    Verifica rapidamente se o índice já existe no Drive (sem baixar).
    """
    try:
        service = get_drive_service()
        q = (
            f"'{gdrive_folder_id}' in parents "
            f"and name = '{INDEX_FOLDER_NAME}' "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        resp = service.files().list(q=q, fields="files(id)").execute()
        folders = resp.get("files", [])
        if not folders:
            return False
        # Verifica se tem arquivos dentro
        items = _list_children(service, folders[0]["id"])
        return len(items) > 0
    except Exception:
        return False


if __name__ == "__main__":
    pass
