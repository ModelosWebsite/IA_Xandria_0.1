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
import os

load_dotenv()

# Conexão com o banco de dados
cs = "mysql+mysqlconnector://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
db_engine = create_engine(cs)
db = SQLDatabase(db_engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_memory = {}

class User(BaseModel):
    prompt: str
    company_id: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@app.post("/chat")
def chat(user: User):
    memory_key = f"user_{user.company_id}"

    # Cria memória se ainda não existir
    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    memory = conversation_memory[memory_key]

    # Cria toolkit com ferramentas SQL
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = sql_toolkit.get_tools()

    # Prompt com variáveis com chaves simples {}
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Você é um assistente de IA especialista em:
        - Consultas SQL corretas e seguras;
        - Cálculos matemáticos e estatísticos precisos;
        - Geração de análises claras e confiáveis com base em dados reais do banco.

        === REGRAS RÍGIDAS E OBRIGATÓRIAS ===
        1. NUNCA invente informações. Se os dados não existirem, responda: "Infelizmente, não tenho essa informação.".
        2. SEMPRE filtre as consultas por `WHERE companyid={companyid}`.
        3. NUNCA execute comandos que alterem dados (ex: INSERT, UPDATE, DELETE, DROP).
        4. NÃO exponha dados sensíveis (ex: NIF, senhas, CPF).
        5. SEJA UM GÊNIO EM MATEMÁTICA: seus cálculos devem ser perfeitos — somas, médias, desvios, percentuais, comparações, etc.
        6. NUNCA alucine. Toda resposta precisa vir diretamente do banco de dados.
        7. SEMPRE apresente a consulta SQL utilizada na resposta.
        8. ORGANIZE A RESPOSTA COM CLAREZA:
           - Título em negrito (**)
           - Explicação detalhada e bem estruturada
           - Linguagem técnica e profissional
        9. Não utilize `LIMIT` a menos que seja solicitado.
        10. Nunca generalize ou invente tendências — baseie-se nos dados consultados.
        11. Use a coluna `{created_at}` para filtrar por datas de faturação.
        12. Use a coluna `{saleTotalPayable}` para realizar cálculos de totais, médias e análises financeiras.

        === ESTRUTURA RECOMENDADA DE RESPOSTA ===

        **Pergunta do usuário:**
        "Qual foi o total faturado neste mês?"

        **Interpretação:**
        O usuário deseja saber o valor total líquido das faturas emitidas no mês atual.

        **Consulta SQL utilizada:**
        ```sql
        SELECT SUM({saleTotalPayable}) AS total_faturado 
        FROM sales 
        WHERE company_id={companyid} 
          AND MONTH({created_at}) = MONTH(NOW()) 
          AND YEAR({created_at}) = YEAR(NOW());
        ```

        **Resposta organizada:**
        O total faturado pela empresa no mês atual é de **8.950.000 AKZ**. Esse valor representa a soma de todas as faturas líquidas emitidas nesse período.

        Este resultado pode ser usado para comparações com meses anteriores, auxiliando na análise de desempenho financeiro, sazonalidade e impacto de estratégias de vendas.

        **Nota:** Se não houver registros para o período, responda claramente que não há dados disponíveis.
        """),
        ("user", "{question}\nai:")
    ])

    # Preenche o prompt com as variáveis literais
    formatted_prompt = prompt.format_prompt(
        question=user.prompt,
        company_id=user.company_id,
        created_at="created_at",
        saleTotalPayable="saleTotalPayable"
    )

    # Criação do agente com memória e toolkit
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

    response = agent.run(formatted_prompt)

    memory.save_context({"input": user.prompt}, {"output": response})

    return {"resposta": response}
