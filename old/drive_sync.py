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

def get_drive_service():
    """
    Autentica no Google Drive de forma flexível:
    1. Tenta usar st.secrets (Streamlit Cloud)
    2. Tenta carregar o arquivo JSON local definido em variáveis de ambiente
    3. Tenta procurar o arquivo JSON padrão na pasta raiz
    """
    creds = None
    
    # 1. Tenta via Streamlit Secrets (Produção)
    try:
        if "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except:
        pass

    # 2. Tenta via Arquivo Local (Desenvolvimento no Windows)
    if not creds:
        # Busca o nome do arquivo no .env ou usa o nome que você forneceu
        json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "disco-retina-371501-525385725005.json")
        if os.path.exists(json_path):
            creds = service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)
    
    if not creds:
        raise RuntimeError(
            "Credenciais do Google Cloud não encontradas. "
            "Configure 'gcp_service_account' nos Secrets do Streamlit ou tenha o arquivo JSON localmente."
        )

    return build("drive", "v3", credentials=creds)

def sync_folder(folder_id: str, out_dir: str) -> list[Path]:
    """Sincroniza uma pasta do Google Drive com um diretório local."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    try:
        service = get_drive_service()
        q = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=q, fields="files(id, name, mimeType)").execute()
        files = results.get("files", [])

        downloaded = []
        for f in files:
            name = f["name"]
            file_id = f["id"]
            mime = f["mimeType"]

            # Ignora pastas e ficheiros nativos do Google (Docs, Sheets) que exigem exportação
            if mime.startswith("application/vnd.google-apps"):
                continue

            dest = out / name
            
            # Só descarrega se o ficheiro não existir localmente (otimização)
            if not dest.exists():
                request = service.files().get_media(fileId=file_id)
                with open(dest, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                downloaded.append(dest)
        
        return downloaded
    except Exception as e:
        print(f"Erro na sincronização: {e}")
        return []

if __name__ == "__main__":
    # Exemplo de uso local: coloque o ID da sua pasta do Drive aqui
    # ID_PASTA = "seu_id_aqui"
    # sync_folder(ID_PASTA, "data/raw_docs")
    pass