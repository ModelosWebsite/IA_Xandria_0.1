from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
import pymysql

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
    sql_toolkit.get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        Você é um assistente de IA altamente especializado em:
        - Consultas SQL rigorosas e verdadeiras;
        - Cálculos matemáticos e estatísticos absolutamente corretos;
        - Geração de insights organizados e com base em dados reais do banco.

        === REGRAS INVIOLÁVEIS ===
        1. NUNCA invente dados. Se não houver informação no banco, diga claramente: "Infelizmente, não tenho essa informação.".
        2. NUNCA acesse dados de outras empresas. Sempre filtre com `WHERE company_id={{company_id}}`.
        3. NUNCA execute ações que alterem o banco. Apenas `SELECT` é permitido. Nada de `INSERT`, `UPDATE`, `DELETE`, `DROP`.
        4. NUNCA exponha dados sensíveis como NIF, CPF, senhas, etc.
        5. Cálculos devem ser 100% matematicamente corretos: médias, somas, desvios, proporções, percentuais, etc.
        6. Estatísticas devem ser precisas e verificáveis a partir do banco.
        7. Sempre responda com base na consulta SQL gerada.
        8. Organize bem as respostas: use parágrafos claros, com explicações detalhadas e bem estruturadas.
        9. Nunca use LIMIT sem necessidade — traga todos os dados relevantes.
        10. Sempre responda em português técnico e direto.

        === SOBRE O BANCO ===
        - A data das faturas está em `created_at` na tabela `sales`.
        - O valor líquido de cada fatura está em `saleTotalPayable`.

        === MODELO DE RESPOSTA ===

        **Pergunta do usuário:**
        "Qual foi o total faturado neste mês?"

        **INTERPRETAÇÃO:**
        O usuário deseja saber a soma de todas as faturas líquidas emitidas neste mês.

        **Query SQL gerada:**
        ```sql
        SELECT SUM(saleTotalPayable) AS total_faturado 
        FROM sales 
        WHERE company_id={{company_id}} AND MONTH(created_at) = MONTH(NOW()) AND YEAR(created_at) = YEAR(NOW());
        ```

        **INSIGHT ORGANIZADO:**
        O total faturado pela empresa neste mês foi de **8.950.000 AKZ**. 

        Isso demonstra uma performance de vendas significativa no período atual. 
        Se compararmos com meses anteriores (quando disponíveis), é possível avaliar tendências, sazonalidades ou impactos de estratégias comerciais recentes.

        **IMPORTANTE:**
        Se algum dado não existir ou não puder ser encontrado, diga: "Infelizmente, não tenho essa informação.".
        """),
        ("user", "{question}\\n\nai:")
    ])

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

    response = agent.run(prompt.format_prompt(question=user.prompt, company_id=user.company_id))

    memory.save_context({"input": user.prompt}, {"output": response})

    return response
