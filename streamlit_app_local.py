# ========================
# IMPORTS
# ========================
import streamlit as st
import datetime
import textwrap
import time
import pandas as pd
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from groq import Groq

st.set_page_config(page_title="Olá, sou GeiLine sua Assistente Virtual", page_icon="🏫")

# ========================
# CONFIGURAÇÕES
# ========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

MODEL = "llama-3.3-70b-versatile"

ARQUIVO_DADOS = "dados/SÍNTESE GERAL DA REDE ESTADUAL DE ENSINO.xlsx"

HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=2)


# ========================
# CARREGA O EXCEL
# ========================
@st.cache_data
def carregar_dados():
    """Lê todas as abas do Excel e transforma em texto para contexto."""
    xl = pd.ExcelFile(ARQUIVO_DADOS)
    partes = []

    for aba in xl.sheet_names:
        df = xl.parse(aba)
        df = df.dropna(how="all").fillna("")
        texto = f"### Aba: {aba}\n"
        texto += df.to_string(index=False)
        partes.append(texto)

    return "\n\n".join(partes)


# ========================
# CLIENTE GROQ
# ========================
@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_API_KEY)


# ========================
# HELPERS
# ========================
TaskInfo = namedtuple("TaskInfo", ["name", "function", "args"])
TaskResult = namedtuple("TaskResult", ["name", "result"])

executor = ThreadPoolExecutor(max_workers=5)


def history_to_text(chat_history):
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


def build_prompt(**kwargs):
    prompt = []
    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")
    return "\n".join(prompt)


def generate_chat_summary(messages):
    prompt = build_prompt(
        instructions="Summarize this conversation as concisely as possible.",
        conversation=history_to_text(messages),
    )
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def build_question_prompt(question, dados_contexto):
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]

    recent_history_str = history_to_text(recent_history) if recent_history else None

    task_infos = []
    if SUMMARIZE_OLD_HISTORY and old_history:
        task_infos.append(
            TaskInfo("old_message_summary", generate_chat_summary, (old_history,))
        )

    results = executor.map(
        lambda t: TaskResult(name=t.name, result=t.function(*t.args)),
        task_infos,
    )
    context = {name: result for name, result in results}

    instructions = textwrap.dedent(f"""
        - Você é um assistente especializado nos dados da Rede Estadual de Ensino do Espírito Santo.
        - Responda APENAS com base nos dados fornecidos na tag <dados_planilha>.
        - Se a pergunta não puder ser respondida com os dados disponíveis, diga claramente que a informação não está na planilha.
        - Use linguagem clara e objetiva em português.
        - Quando relevante, cite números e valores exatos presentes nos dados.
        - Use markdown: tabelas, listas e negrito para destacar informações importantes.
        - Não invente dados. Não use conhecimento externo.
                                   - forneca somente a resposta direta à pergunta, sem rodeios ou explicações adicionais, mesmo que a pergunta seja complexa, sem resumoes ou simplificações, sem explicações sobre como chegou à resposta, sem suposições ou inferências que não estejam explicitamente nos dados.
        - responda apenas à pergunta feita, evite informações adicionais que não foram solicitadas.
                                   - Se a pergunta for ambígua, peça esclarecimentos em vez de assumir algo.
                                   - Se a pergunta for sobre tendências ou comparações, baseie-se apenas nos dados atuais, sem especular sobre o futuro ou o passado.
                                    - Se a pergunta envolver cálculos, faça-os apenas com os dados fornecidos, e mostre o passo a passo do cálculo.
                                   - Se a pergunta for sobre uma categoria específica (ex: "número de alunos por município"), responda apenas com essa categoria, sem incluir outras informações que não foram solicitadas.

    """)

    return build_prompt(
        instructions=instructions,
        dados_planilha=dados_contexto,
        **context,
        recent_messages=recent_history_str,
        question=question,
    )


def get_response_stream(prompt):
    client = get_client()
    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def show_feedback_controls(message_index):
    st.write("")
    with st.popover("Como fui?"):
        with st.form(key=f"feedback-{message_index}", border=False):
            st.markdown(":small[Avaliação]")
            st.feedback(options="stars")
            st.text_area("Mais informações (opcional)")
            ""
            if st.form_submit_button("Enviar feedback"):
                st.success("Obrigado pelo feedback!")


# ========================
# UI
# ========================
st.title("🏫 Olá, sou GeiLine sua Assistente Virtual", anchor=False)
st.caption("Respostas baseadas exclusivamente na **Síntese Geral da Rede Estadual de Ensino**.")

# Carrega dados com spinner
with st.spinner("Carregando planilha..."):
    try:
        dados_contexto = carregar_dados()
        st.success(f"Planilha carregada com sucesso!", icon="✅")
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: `{ARQUIVO_DADOS}`")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar planilha: {e}")
        st.stop()

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# Botão reiniciar
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🔄 Reiniciar"):
        st.session_state.messages = []
        st.rerun()

# Exibe histórico
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_feedback_controls(i)

# Input do usuário
user_message = st.chat_input("Faça uma pergunta sobre a Síntese Geral...")

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Aguardando..."):
            now = datetime.datetime.now()
            diff = now - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = now
            if diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(diff.seconds + diff.microseconds * 0.001)

        with st.spinner("Consultando dados..."):
            full_prompt = build_question_prompt(user_message, dados_contexto)

        with st.container():
            response = st.write_stream(get_response_stream(full_prompt))

            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})

            show_feedback_controls(len(st.session_state.messages) - 1)
