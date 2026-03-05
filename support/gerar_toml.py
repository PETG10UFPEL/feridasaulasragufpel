import json

# Coloque o nome do seu arquivo JSON aqui
with open("gen-lang-client-0563640032-63af8a02ff8c.json", "r") as f:
    creds = json.load(f)

toml = f'''ADMIN_PASSWORD = "sua_senha_aqui"
GOOGLE_API_KEY = "sua_google_api_key_aqui"
GEMINI_MODEL   = "gemini-2.5-flash"
GDRIVE_FOLDER_ID = "seu_folder_id_aqui"

[gcp_service_account]
type                        = {json.dumps(creds["type"])}
project_id                  = {json.dumps(creds["project_id"])}
private_key_id              = {json.dumps(creds["private_key_id"])}
private_key                 = {json.dumps(creds["private_key"])}
client_email                = {json.dumps(creds["client_email"])}
client_id                   = {json.dumps(creds["client_id"])}
auth_uri                    = {json.dumps(creds["auth_uri"])}
token_uri                   = {json.dumps(creds["token_uri"])}
auth_provider_x509_cert_url = {json.dumps(creds["auth_provider_x509_cert_url"])}
client_x509_cert_url        = {json.dumps(creds["client_x509_cert_url"])}
universe_domain             = {json.dumps(creds.get("universe_domain", "googleapis.com"))}
'''

with open("secrets.toml", "w", encoding="utf-8") as f:
    f.write(toml)

print("secrets.toml gerado com sucesso!")