import os
import re
import tempfile
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from PIL import Image, ImageEnhance, UnidentifiedImageError
import fitz  # PyMuPDF
import easyocr
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# ========================
# Inicialização
# ========================
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
)
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xzero.ao", "http://192.168.100.89:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class User(BaseModel):
    prompt: str
    companyid: str

conversation_memory = {}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========================
# OCR Embutido
# ========================
reader = easyocr.Reader(["pt"], gpu=False)

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(2.0)

def ocr_image(img: Image.Image) -> str:
    img = preprocess_image(img)
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return "\n".join(reader.readtext(img_bytes.read(), detail=0)).strip()

def extract_text(file_bytes: bytes, filename: str) -> str:
    try:
        if filename.lower().endswith(".pdf"):
            text = ""
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for num, page in enumerate(doc, 1):
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                page_text = ocr_image(img)
                text += f"\n--- Página {num} ---\n{page_text}"
            return text.strip()
        else:
            img = Image.open(BytesIO(file_bytes))
            return ocr_image(img)
    except UnidentifiedImageError:
        return "Erro: Tipo de imagem não identificado."
    except Exception as e:
        return f"Erro durante OCR: {e}"

# ========================
# Ferramentas personalizadas
# ========================
@tool
def documentation_tool(url: str, question: str) -> str:
    return f"[Mock] Resposta da documentação para a pergunta: {question}"

@tool
def black_formatter_tool(path: str) -> str:
    try:
        os.system(f"black {path}")
        return "Formatado com sucesso!"
    except Exception as e:
        return f"Erro ao formatar: {str(e)}"

@tool
def ocr_invoice(file_path: str) -> str:
    """Extrai o texto da fatura (PDF ou imagem)."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return extract_text(file_bytes, file_path)

# ========================
# Prompt Base
# ========================
promptzero_path = "promptzero.txt"
promptzero_text = ""
if os.path.exists(promptzero_path):
    with open(promptzero_path, encoding="utf-8") as f:
        promptzero_text = f.read()

base_prompt = f"""
Seu nome é Xándria, tu és um assistente especializado em:
- Consultas SQL seguras
- Análise humana e estruturada
=== REGRAS ===
1. Todas as queries devem ter WHERE companyid = {{companyid}}.
2. Para faturas e receitas, use a tabela sales.
3. Para interações, use a tabela interactions.
4. Calcule com precisão, nunca invente.
5. OCR: quando receber fatura (imagem/PDF), extraia nome da empresa emissora, NIF/IVA e data.
6. {promptzero_text}
"""

# ========================
# Config SQL Toolkit
# ========================
db_sync = SQLDatabase(engine)
sql_toolkit = SQLDatabaseToolkit(db=db_sync, llm=llm)
toolkit = sql_toolkit.get_tools() + [documentation_tool, black_formatter_tool, ocr_invoice]

def format_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    return text.replace("\n\n", "<br><br>").replace("\n", "<br>")

# ========================
# Endpoints
# ========================
@app.post("/chat", response_class=HTMLResponse)
def chat(user: User):
    memory_key = f"user_{user.companyid}"
    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    memory = conversation_memory[memory_key]

    dynamic_prompt_text = base_prompt.replace("{companyid}", user.companyid)
    prompt_dynamic = ChatPromptTemplate.from_messages(
        [
            ("system", dynamic_prompt_text),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm=llm, tools=toolkit, prompt=prompt_dynamic)
    executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)
    response = executor.invoke({"input": user.prompt, "chat_history": memory.chat_memory.messages})
    memory.save_context({"input": user.prompt}, {"output": response["output"]})
    return format_markdown(response["output"])

@app.post("/chat-with-file", response_class=HTMLResponse)
def chat_with_file(companyid: str = Form(...), prompt: str = Form(...), file: UploadFile = File(...)):
    file_bytes = file.file.read()
    extracted_text = extract_text(file_bytes, file.filename)
    prompt_final = f"{prompt}\n\n=== CONTEÚDO EXTRAÍDO ===\n{extracted_text}"

    memory_key = f"user_{companyid}"
    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    memory = conversation_memory[memory_key]

    dynamic_prompt_text = base_prompt.replace("{companyid}", companyid)
    prompt_dynamic = ChatPromptTemplate.from_messages(
        [
            ("system", dynamic_prompt_text),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm=llm, tools=toolkit, prompt=prompt_dynamic)
    executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)
    response = executor.invoke({"input": prompt_final, "chat_history": memory.chat_memory.messages})
    memory.save_context({"input": prompt_final}, {"output": response["output"]})
    return format_markdown(response["output"])
