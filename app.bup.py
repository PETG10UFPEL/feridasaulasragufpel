import os
from dotenv import load_dotenv
load_dotenv(override=True)  # DEVE ser antes de qualquer outro import

import base64
import datetime
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud: carrega secrets se disponíveis
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
    if "GDRIVE_FOLDER_ID" in st.secrets:
        os.environ["GDRIVE_FOLDER_ID"] = st.secrets["GDRIVE_FOLDER_ID"]
except Exception:
    pass

from drive_sync import sync_folder
from ingest import build_index
from rag import answer


st.set_page_config(page_title="Feridas Crônicas - PET G10 UFPel", layout="wide")

# ==============================
# CSS global
# ==============================
st.markdown("""
<style>
  header {visibility: hidden;}
  [data-testid="stToolbar"] {visibility: hidden;}
  [data-testid="stHeader"] {display:none;}
  [data-testid="stDecoration"] {display:none;}
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .block-container { padding-top: 0.25rem !important; }

  .field-label {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    margin-top: 0.8rem;
    color: #1a1a2e;
  }

  .info-text {
    font-size: 1.0rem;
    line-height: 1.6;
    color: #222;
    margin: 0.3rem 0;
  }

  .prompt-box {
    background: #f0f4ff;
    border-left: 4px solid #4a6fa5;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-family: monospace;
    font-size: 0.92rem;
    color: #1a1a2e;
    white-space: pre-wrap;
    margin-top: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

# ==============================
# Helpers de imagem
# ==============================
def img_b64(filename: str) -> str:
    p = Path(__file__).parent / "assets" / filename
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""

# ==============================
# Banner — centralizado
# ==============================
banner_b64 = img_b64("banner.png")
if banner_b64:
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:8px;">
      <img src="data:image/png;base64,{banner_b64}"
           style="width:52%; height:auto; border-radius:6px;">
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("🩹 Feridas Crônicas - PET G10 UFPel")

# ==============================
# Links institucionais
# ==============================
insta_b64 = img_b64("instagram.png")
enf_b64   = img_b64("logo.enfermagem.png")

insta_tag = f'<img src="data:image/png;base64,{insta_b64}" width="22" style="vertical-align:middle; margin-right:4px;">' if insta_b64 else "📷"
enf_tag   = f'<img src="data:image/png;base64,{enf_b64}" width="22" style="vertical-align:middle; margin-right:4px;">'   if enf_b64   else "🎓"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;
            margin-top:4px; margin-bottom:10px; font-size:0.95rem;">
  {insta_tag}
  <a href="https://www.instagram.com/amorapele_ufpel/" target="_blank"
     style="text-decoration:none; font-weight:500;">Amor à Pele</a>
  <span style="color:#aaa;">|</span>
  <a href="https://www.instagram.com/g10petsaude/" target="_blank"
     style="text-decoration:none; font-weight:500;">PET G10</a>
  <span style="color:#aaa;">|</span>
  {enf_tag}
  <a href="https://wp.ufpel.edu.br/fen/" target="_blank"
     style="text-decoration:none; font-weight:500;">Faculdade de Enfermagem – UFPel</a>
</div>
""", unsafe_allow_html=True)

# ==============================
# Assinatura
# ==============================
st.markdown("""
<p class="info-text" style="margin-bottom:0.8rem;">
  Sistema de apoio ao ensino &amp; aprendizagem sobre <strong>Feridas Crônicas</strong> — baseado em RAG-AI
  sobre o material das disciplinas oferecidas pelas professoras
  <a href="https://institucional.ufpel.edu.br/servidores/id/176265" target="_blank">Marina Soares Mota</a>
  e
  <a href="https://institucional.ufpel.edu.br/servidores/id/2858" target="_blank">Adrize Rutz Porto</a>,
  da Faculdade de Enfermagem – UFPel.<br>
  Desenvolvimento conjunto em P&amp;D dos Projetos
  <a href="https://www.instagram.com/amorapele_ufpel/" target="_blank">Amor à Pele</a>
  e
  <a href="https://www.instagram.com/g10petsaude/" target="_blank">PET UFPel Saúde Digital</a>.
</p>
""", unsafe_allow_html=True)

# ==============================
# Bloco explicativo RAG + Embeddings (mantido do app original)
# ==============================
def _html_com_imagens_embutidas(html_path: Path) -> str:
    """Lê o HTML e substitui src de imagens locais por base64."""
    import re
    html = html_path.read_text(encoding="utf-8", errors="replace")
    assets_dir = html_path.parent

    def substituir(match):
        src = match.group(1)
        if src.startswith("http"):
            return match.group(0)
        img_path = assets_dir / src
        if img_path.exists():
            mime = "image/png" if src.lower().endswith(".png") else "image/jpeg"
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return f'src="data:{mime};base64,{b64}"'
        return match.group(0)

    html = re.sub(r'src="([^"]+)"', substituir, html)
    return html

st.markdown("""
<p class="info-text" style="margin-top:0.8rem;">
  <strong>Retrieval-Augmented Generation (RAG-AI)</strong> melhora a precisão dos modelos de linguagem (LLMs)
  ao recuperar dados externos — como documentos e bases de dados — para responder perguntas,
  reduzindo as chamadas alucinações.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  <strong>Embeddings</strong> são representações numéricas (vetores) de textos que capturam seu significado
  semântico, permitindo que a IA encontre informações relevantes por similaridade de sentido,
  e não apenas por palavras-chave.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  Quer visualizar a representação gráfica do conhecimento indexado sobre feridas crônicas?
</p>
""", unsafe_allow_html=True)

_emb_path = Path(__file__).parent / "assets" / "embeddings_feridas.html"
if _emb_path.exists():
    with st.expander("📊 Ver Embeddings da Base de Conhecimento", expanded=False):
        _html_content = _html_com_imagens_embutidas(_emb_path)
        components.html(_html_content, height=700, scrolling=True)
else:
    st.caption("_(arquivo embeddings_feridas.html não encontrado em assets/)_")

st.divider()

# ==============================
# Sidebar — administração
# ==============================
with st.sidebar:
    st.header("Base de conhecimento (Google Drive)")

    admin_pass = st.text_input("Senha admin", type="password")
    is_admin = admin_pass == st.secrets.get("ADMIN_PASSWORD", "")

    if is_admin:
        folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
        st.text_input("Folder ID", value=folder_id, key="folder_id")

        if st.button("1) Sincronizar arquivos do Drive"):
            with st.spinner("Baixando..."):
                files = sync_folder(st.session_state.folder_id, "data/raw_docs")
            st.success(f"Baixados {len(files)} arquivos para data/raw_docs")

        if st.button("2) Recriar índice (embeddings)"):
            with st.spinner("Indexando..."):
                n = build_index("data/raw_docs", "data/chroma_db")
            st.success(f"Índice criado com {n} trechos.")
    else:
        if admin_pass:
            st.error("Senha incorreta")

# ==============================
# Componente de voz (reutilizável)
# ==============================
def mic_component(target_label: str, field_index: int):
    """Injeta botão de microfone que preenche o textarea pelo índice."""
    key = f"mic_{field_index}"
    components.html(f"""
    <style>
      .mic-row {{ display:flex; align-items:center; gap:10px; margin-bottom:2px; }}
      .mic-btn {{
        background:#0066cc; color:white; border:none;
        width:36px; height:36px; border-radius:50%;
        font-size:1.1rem; cursor:pointer; flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        transition:background .2s;
      }}
      .mic-btn.listening {{ background:#e53935; animation:pulse 1s infinite; }}
      @keyframes pulse {{
        0%,100%{{box-shadow:0 0 0 0 rgba(229,57,53,.35);}}
        50%{{box-shadow:0 0 0 8px rgba(229,57,53,0);}}
      }}
      #micStatus_{key} {{ font-size:.80rem; color:#555; }}
    </style>
    <div class="mic-row">
      <button class="mic-btn" id="micBtn_{key}" title="Falar">🎤</button>
      <span id="micStatus_{key}">Toque em 🎤 para falar (Chrome/Edge)</span>
    </div>
    <script>
    (function(){{
      const btn    = document.getElementById('micBtn_{key}');
      const status = document.getElementById('micStatus_{key}');
      let listening = false;

      function fillTextarea(text) {{
        try {{
          const doc = window.parent.document;
          const allTa = doc.querySelectorAll('[data-testid="stTextArea"] textarea');
          const ta = allTa[{field_index}] || doc.querySelector('textarea[aria-label="{target_label}"]');
          if (!ta) {{ status.textContent = '⚠️ Campo não encontrado.'; return; }}
          const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
          setter.call(ta, text);
          ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
          ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
          ta.focus();
          status.textContent = '✅ Texto inserido — edite se quiser.';
        }} catch(e) {{
          status.textContent = '⚠️ Erro: ' + e.message;
        }}
      }}

      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {{
        status.textContent = '⚠️ Voz indisponível — use Chrome ou Edge.';
        btn.style.opacity = '.4';
        return;
      }}
      const rec = new SR();
      rec.lang = 'pt-BR';
      rec.continuous = false;
      rec.interimResults = true;

      rec.onstart = () => {{
        listening = true;
        btn.classList.add('listening');
        btn.innerHTML = '🔴';
        status.textContent = '🎙️ Ouvindo…';
      }};
      rec.onresult = (e) => {{
        let interim = '', final = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {{
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) final += t; else interim += t;
        }}
        if (final) fillTextarea(final.trim());
        else status.textContent = '…' + interim;
      }};
      rec.onerror = (e) => {{
        status.textContent = '❌ Erro: ' + e.error;
        btn.classList.remove('listening'); btn.innerHTML = '🎤'; listening = false;
      }};
      rec.onend = () => {{
        listening = false; btn.classList.remove('listening'); btn.innerHTML = '🎤';
      }};
      btn.addEventListener('click', () => {{ if (listening) rec.stop(); else rec.start(); }});
    }})();
    </script>
    """, height=48, scrolling=False)


# ==============================
# Gerador de prompt de imagem
# ==============================
def gerar_prompt_imagem(pergunta: str, resposta: str) -> str:
    """Monta um prompt em inglês para geração de imagem em outra IA."""
    # Extrai palavras-chave da pergunta para contextualizar o prompt
    contexto = pergunta.strip().rstrip("?").strip()
    prompt = (
        f"Medical illustration, educational style, high detail. "
        f"Topic: chronic wound care — {contexto}. "
        f"Show anatomical layers of skin, wound bed, granulation tissue, "
        f"professional clinical setting, clean background, labeled diagram, "
        f"no text overlays, realistic and respectful depiction suitable for nursing students."
    )
    return prompt


# ==============================
# Componente Text-to-Speech (TTS)
# ==============================
def tts_component(texto: str, key: str = "tts_0"):
    """Injeta botão 🔊 Ouvir / ⏹ Parar com controle de velocidade — pt-BR via Web Speech API."""
    texto_js = texto.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(f"""
    <style>
      .tts-wrap  {{ display:flex; flex-wrap:wrap; align-items:center;
                   gap:10px; margin:8px 0 4px 0; }}
      .tts-btn {{
        background:#2e7d32; color:white; border:none;
        padding:6px 18px; border-radius:20px;
        font-size:0.95rem; cursor:pointer;
        display:flex; align-items:center; gap:6px;
        transition:background .2s; white-space:nowrap;
      }}
      .tts-btn.speaking {{ background:#c62828; }}
      .tts-speed-label {{
        font-size:0.82rem; color:#444; white-space:nowrap;
      }}
      .tts-slider {{
        -webkit-appearance:none; appearance:none;
        width:130px; height:6px;
        border-radius:3px; background:#bbb; outline:none; cursor:pointer;
      }}
      .tts-slider::-webkit-slider-thumb {{
        -webkit-appearance:none; width:16px; height:16px;
        border-radius:50%; background:#2e7d32; cursor:pointer;
      }}
      .tts-slider::-moz-range-thumb {{
        width:16px; height:16px;
        border-radius:50%; background:#2e7d32; cursor:pointer; border:none;
      }}
      #ttsStatus_{key} {{ font-size:0.78rem; color:#666; width:100%; }}
    </style>

    <div class="tts-wrap">
      <button class="tts-btn" id="ttsBtn_{key}">🔊 Ouvir resposta</button>
      <span class="tts-speed-label">🐢</span>
      <input  class="tts-slider" type="range" id="ttsRate_{key}"
              min="0.5" max="2.0" step="0.1" value="1.0">
      <span class="tts-speed-label">🐇</span>
      <span   class="tts-speed-label" id="ttsRateVal_{key}">1.0×</span>
    </div>
    <div><span id="ttsStatus_{key}"></span></div>

    <script>
    (function(){{
      const btn      = document.getElementById('ttsBtn_{key}');
      const status   = document.getElementById('ttsStatus_{key}');
      const slider   = document.getElementById('ttsRate_{key}');
      const rateVal  = document.getElementById('ttsRateVal_{key}');
      const texto    = `{texto_js}`;
      let speaking   = false;

      slider.addEventListener('input', () => {{
        rateVal.textContent = parseFloat(slider.value).toFixed(1) + '×';
      }});

      if (!window.speechSynthesis) {{
        status.textContent = '⚠️ TTS não disponível neste navegador.';
        btn.style.opacity = '.4';
        btn.disabled = true;
        return;
      }}

      function iniciar() {{
        window.speechSynthesis.cancel();
        const utter   = new SpeechSynthesisUtterance(texto);
        utter.lang    = 'pt-BR';
        utter.rate    = parseFloat(slider.value);
        utter.pitch   = 1.0;

        const vozes = window.speechSynthesis.getVoices();
        const vozPT = vozes.find(v => v.lang.startsWith('pt'));
        if (vozPT) utter.voice = vozPT;

        utter.onstart = () => {{
          speaking = true;
          btn.classList.add('speaking');
          btn.innerHTML = '⏹ Parar leitura';
          status.textContent = '🔊 Lendo… (velocidade ' + parseFloat(slider.value).toFixed(1) + '×)';
        }};
        utter.onend = utter.onerror = () => {{
          speaking = false;
          btn.classList.remove('speaking');
          btn.innerHTML = '🔊 Ouvir resposta';
          status.textContent = '';
        }};

        window.speechSynthesis.speak(utter);
      }}

      if (window.speechSynthesis.getVoices().length === 0) {{
        window.speechSynthesis.addEventListener('voiceschanged', () => {{}}, {{ once: true }});
      }}

      btn.addEventListener('click', () => {{
        if (speaking) {{ window.speechSynthesis.cancel(); }}
        else          {{ iniciar(); }}
      }});

      window.addEventListener('beforeunload', () => window.speechSynthesis.cancel());
    }})();
    </script>
    """, height=72, scrolling=False)


# ==============================
# Campos de entrada
# ==============================
st.markdown('<p class="field-label">🩹 Contexto clínico / situação (opcional)</p>', unsafe_allow_html=True)
mic_component("Contexto clínico", 0)
patient = st.text_area(
    "Contexto clínico",
    height=120,
    placeholder=(
        "Opcional: descreva o contexto clínico ou o perfil do paciente que motivou sua dúvida.\n"
        "Ex.: paciente diabético, úlcera venosa em membro inferior, 3ª semana de tratamento."
    ),
    label_visibility="collapsed",
)

st.markdown('<p class="field-label">❓ Sua pergunta sobre feridas crônicas</p>', unsafe_allow_html=True)
mic_component("Pergunta sobre feridas crônicas", 1)
q = st.text_area(
    "Pergunta sobre feridas crônicas",
    height=100,
    placeholder=(
        "Ex.: Quais são as fases da cicatrização? "
        "Como classificar uma úlcera por pressão? "
        "Qual o curativo indicado para ferida com exsudato abundante?"
    ),
    label_visibility="collapsed",
)

# ==============================
# Cache
# ==============================
@st.cache_data(show_spinner=False, ttl=3600)
def cached_answer(q, patient, k):
    return answer(q, patient, k=k)

# ==============================
# Botões de ação
# ==============================
col_btn1, col_btn2, col_btn3, _ = st.columns([1.2, 1.2, 1.2, 2])
with col_btn1:
    gerar = st.button("🚀 Gerar resposta", type="primary")
with col_btn2:
    gerar_img = st.button("🎨 Prompt de imagem")
with col_btn3:
    if st.button("🗑️ Limpar cache"):
        st.cache_data.clear()
        st.session_state.pop("ultima_resposta", None)
        st.session_state.pop("ultima_pergunta", None)
        st.session_state.pop("hits", None)
        st.success("Cache limpo!")

# ==============================
# Execução — Resposta RAG
# ==============================
if gerar:
    if not q.strip():
        st.error("Digite uma pergunta sobre feridas crônicas.")
    else:
        with st.spinner("Buscando nos documentos da aula e gerando resposta…"):
            patient_short = patient[:2000] if patient.strip() else "Sem contexto clínico informado."
            resp, hits = cached_answer(q, patient_short, 4)

        # Armazena na sessão para download e geração de prompt
        st.session_state["ultima_resposta"] = resp
        st.session_state["ultima_pergunta"] = q
        st.session_state["hits"] = hits

col1, col2 = st.columns([2, 1])

if "ultima_resposta" in st.session_state:
    resp = st.session_state["ultima_resposta"]
    hits = st.session_state.get("hits", [])

    with col1:
        st.markdown("### 📖 Resposta")
        st.markdown(resp)
        # ── Leitura em voz ──
        tts_component(resp, key="tts_resposta")


        # ── Download da resposta ──
        st.markdown("---")
        agora = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        nome_arquivo = f"feridas_cronicas_{agora}.txt"

        conteudo_txt = (
            f"Sistema RAG — Feridas Crônicas | PET G10 UFPel\n"
            f"{'='*55}\n"
            f"Data/hora: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            f"PERGUNTA:\n{st.session_state.get('ultima_pergunta','')}\n\n"
            f"CONTEXTO CLÍNICO:\n{patient if patient.strip() else '(não informado)'}\n\n"
            f"RESPOSTA:\n{resp}\n\n"
        )
        if hits:
            conteudo_txt += "FONTES RECUPERADAS:\n"
            for d in hits:
                src  = d.metadata.get("source", "arquivo_desconhecido")
                page = d.metadata.get("page", "trecho")
                conteudo_txt += f"  • {src} | {page}\n"

        st.download_button(
            label="💾 Salvar resposta (.txt)",
            data=conteudo_txt.encode("utf-8"),
            file_name=nome_arquivo,
            mime="text/plain",
        )

    with col2:
        st.markdown("### 📚 Fontes recuperadas")
        if hits:
            for d in hits:
                src  = d.metadata.get("source", "arquivo_desconhecido")
                page = d.metadata.get("page", "trecho")
                st.write(f"- {src} | {page}")
        else:
            st.info("Nenhuma fonte indexada utilizada — resposta baseada em conhecimento geral.")

# ==============================
# Gerador de prompt de imagem
# ==============================
if gerar_img:
    if not q.strip():
        st.warning("Digite uma pergunta primeiro para gerar um prompt de imagem relacionado.")
    else:
        prompt_img = gerar_prompt_imagem(
            q,
            st.session_state.get("ultima_resposta", "")
        )
        st.markdown("---")
        st.markdown("### 🎨 Prompt para geração de imagem")
        st.markdown("""
        <p class="info-text">
          Copie o prompt abaixo e cole em uma IA geradora de imagens como
          <a href="https://www.bing.com/images/create" target="_blank">Bing Image Creator</a>,
          <a href="https://ideogram.ai" target="_blank">Ideogram</a> ou
          <a href="https://leonardo.ai" target="_blank">Leonardo.ai</a>
          para criar uma ilustração educacional sobre o tema da sua pergunta.
        </p>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="prompt-box">{prompt_img}</div>', unsafe_allow_html=True)

        # Botão para copiar / baixar o prompt
        st.download_button(
            label="📋 Baixar prompt (.txt)",
            data=prompt_img.encode("utf-8"),
            file_name="prompt_imagem_feridas.txt",
            mime="text/plain",
        )
