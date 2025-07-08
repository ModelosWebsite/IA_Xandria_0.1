import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain e LLM
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit 
from langchain_groq import ChatGroq

# ===========================================
# Carregamento de ambiente e banco de dados
# ===========================================
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "mysql+pymysql://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ===========================================
# Inicialização da API
# ===========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===========================================
# Modelo de entrada da API
# ===========================================
class User(BaseModel):
    prompt: str
    companyid: int

# ===========================================
# Memória de conversa por empresa
# ===========================================
conversation_memory = {}

# ===========================================
# Instância do LLM (Groq + LLaMA)
# ===========================================
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# ===========================================
# Tools Personalizadas
# ===========================================
@tool
def documentation_tool(url: str, question: str) -> str:
    """Consulta uma URL de documentação para responder perguntas técnicas sobre o sistema."""
    return f"[Mock] Resposta da documentação para a pergunta: {question}"

@tool
def black_formatter_tool(path: str) -> str:
    """Formata automaticamente o código Python no caminho fornecido usando o Black."""
    try:
        os.system(f"black {path}")
        return "Formatado com sucesso!"
    except Exception as e:
        return f"Erro ao formatar: {str(e)}"

# ===========================================
# Leitura do Prompt Externo
# ===========================================
promptzero_path = "promptzero.txt"
promptzero_text = ""
if os.path.exists(promptzero_path):
    with open(promptzero_path, "r", encoding="utf-8") as f:
        promptzero_text = f.read()

# ===========================================
# Inicialização da Ferramenta SQL
# ===========================================
db_sync = SQLDatabase(engine)
sql_toolkit = SQLDatabaseToolkit(db=db_sync, llm=llm)
toolkit = sql_toolkit.get_tools() + [documentation_tool, black_formatter_tool]

# ===========================================
# Prompt Base
# ===========================================
base_prompt = f"""
Seu nome é Xándria, tu és um assistente de Inteligência Artificial especializado em:
- Consultas SQL precisas e seguras;
- Cálculos matemáticos e estatísticos corretos;
- Geração de análises claras, profundas, humanas e bem estruturadas em português formal.
- Se é lhe perguntado algo e caso a informação exista no banco de dados, traga as respostas.

=== REGRAS OBRIGATÓRIAS ===
- Todas as consultas ao banco de dados devem obrigatoriamente incluir a cláusula: WHERE companyid = {{companyid}}, de forma segura e correta.
- (regras continuam...)

=== INSTRUÇÕES ADICIONAIS === 
{promptzero_text}
"""

# ===========================================
# Markdown para HTML
# ===========================================
def format_markdown(text: str) -> str:
    # Substitui negrito (**texto**) por <strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Converte quebras de linha
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return text

# ===========================================
# Rota principal de chat
# ===========================================
@app.post("/chat", response_class=HTMLResponse)
def chat(user: User):
    memory_key = f"user_{user.companyid}"

    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    memory = conversation_memory[memory_key]

    # Injeta companyid no prompt
    dynamic_prompt_text = base_prompt.replace("{companyid}", str(user.companyid))

    # Monta prompt com histórico
    prompt_dynamic = ChatPromptTemplate.from_messages([
        ("system", dynamic_prompt_text),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm=llm, tools=toolkit, prompt=prompt_dynamic)
    executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

    # Executa
    response = executor.invoke({
        "input": user.prompt,
        "chat_history": memory.chat_memory.messages,
    })

    memory.save_context({"input": user.prompt}, {"output": response["output"]})

    return format_markdown(response["output"])
