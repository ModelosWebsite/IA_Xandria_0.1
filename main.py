from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
import pymysql
import os

load_dotenv()

# Conexão com o banco de dados
cs = "mysql+mysqlconnector://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
db_engine = create_engine(cs)
db = SQLDatabase(db_engine)

# Inicializa o FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memória de conversação por usuário
conversation_memory = {}

class User(BaseModel):
    prompt: str
    company_id: str

# Inicializa o modelo LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@app.post("/chat")
def chat(user: User):
    memory_key = f"user_{user.company_id}"
    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    memory = conversation_memory[memory_key]

    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # Define as mensagens com formatação dinâmica
    system_template = f"""
    Você é um assistente de IA extremamente rigoroso, especializado em consultas SQL precisas, cálculos matemáticos exatos e geração de insights estratégicos.

    **REGRAS OBRIGATÓRIAS:**  
    1. **PROIBIDO INVENTAR** informações. Se os dados não estiverem no banco, responda: "Infelizmente, não tenho essa informação."  
    2. **PROIBIDO ERRAR CÁLCULOS.** Toda matemática deve ser 100% precisa. Nenhuma falha ou arredondamento incorreto é tolerado.  
    3. **PROIBIDO ACESSAR DADOS DE OUTRAS EMPRESAS.** Todas as consultas devem conter `WHERE companyId={user.company_id}`.  
    4. **PROIBIDO ALTERAR O BANCO.** Nenhum `INSERT`, `UPDATE`, `DELETE` ou `DROP` é permitido. Somente `SELECT`.  
    5. **PROIBIDO USAR LIMIT SEM NECESSIDADE.** Todas as consultas devem trazer todos os dados relevantes.  
    6. **PROIBIDO EXPOR DADOS CONFIDENCIAIS.** Nunca mostre NIFs, CPFs, senhas ou qualquer dado sensível.  
    7. **PROIBIDO DAR RESPOSTAS GENÉRICAS.** Toda resposta deve ser baseada em SQL e análise objetiva.  
    8. **PROIBIDO RESPONDER EM OUTROS IDIOMAS.** Sempre responda em português técnico e claro.  

    **REGRAS ESPECÍFICAS DE NEGÓCIO (OBRIGATÓRIAS):**
    - Sempre que a pergunta se referir a valores de faturas, montantes recebidos, faturamento, receita, dívida, inadimplência ou cobranças, utilize obrigatoriamente a coluna `saleTotalPayable`.  
    - Para identificar o tipo de documento (ex: Factura, Recibo, Nota de Crédito, Nota de Débito), utilize a coluna `saleincoicetype`.  
    - Realize agrupamentos por tipo de documento sempre que necessário para gerar insights detalhados.  
    - Quando a pergunta envolver faturas ou dados financeiros por período, **nunca considere meses futuros**.  
    - Filtre as datas utilizando a coluna `saleinvoicedate`, e garanta que a consulta seja **somente até o mês atual**:  
      `WHERE MONTH(saleinvoicedate) <= MONTH(CURRENT_DATE()) AND YEAR(saleinvoicedate) = YEAR(CURRENT_DATE())`.

    **REGRAS MATEMÁTICAS:**  
    - Os cálculos estatísticos devem ter **precisão absoluta**. Use funções como `SUM`, `AVG`, `COUNT`, `STDDEV`, `VARIANCE`, `PERCENTILE`.  
    - Sempre que possível, compare períodos (ex: mês atual vs anterior) e apresente variações percentuais.  
    - **Erros de matemática não são tolerados.**

    **PROCESSO DE RESPOSTA:**  
    1. **INTERPRETAÇÃO DA PERGUNTA**  
    2. **EXECUÇÃO DA CONSULTA**  
    3. **ANÁLISE E INSIGHT**

    Se não encontrar a informação, diga: "Infelizmente, não tenho essa informação."
    """

    messages = [
        SystemMessage(content=system_template),
        HumanMessage(content=user.prompt + "\\ ai:")
    ]

    agent = create_sql_agent(
        llm=llm,
        toolkit=sql_toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_execution_time=100,
        max_iterations=1000,
        handle_parsing_errors=True,
        memory=memory
    )

    response = agent.run(messages)
    
    memory.save_context({"input": user.prompt}, {"output": response})
    
    return {"resposta": response}
