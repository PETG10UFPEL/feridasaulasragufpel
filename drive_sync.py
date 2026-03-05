import os
import streamlit as st
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

# Carrega variáveis de ambiente para uso local
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"


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
    """
    Sincroniza recursivamente: percorre subpastas e baixa arquivos preservando a estrutura.
    """
    downloaded: list[Path] = []
    items = _list_children(service, folder_id)

    for it in items:
        name = it["name"]
        mime = it.get("mimeType", "")
        it_id = it["id"]

        # Subpasta
        if mime == FOLDER_MIME:
            downloaded.extend(_sync_folder_recursive(service, it_id, out_root, rel / name))
            continue

        # Ignora ficheiros nativos do Google (Docs, Sheets, Slides) que exigem exportação
        if mime.startswith("application/vnd.google-apps"):
            continue

        dest = out_root / rel / name

        # Só descarrega se o ficheiro não existir localmente (otimização)
        if not dest.exists():
            _download_file(service, it_id, dest)
            downloaded.append(dest)

    return downloaded


def sync_folder(folder_id: str, out_dir: str, recursive: bool = True) -> list[Path]:
    """
    Sincroniza uma pasta do Google Drive com um diretório local.

    - Se recursive=True (padrão): baixa também as subpastas e preserva a estrutura.
    - Se recursive=False: baixa apenas o que estiver no "raiz" da pasta.
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


if __name__ == "__main__":
    # Exemplo de uso:
    # ID_PASTA = "seu_id_aqui"
    # sync_folder(ID_PASTA, "data/raw_docs", recursive=True)
    pass
