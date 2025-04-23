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
Você é um assistente de IA especializado em:

- Consultas SQL corretas e seguras com base em um banco de dados MySQL;
- Análises matemáticas e estatísticas precisas;
- Geração de respostas organizadas, com base apenas nos dados reais do banco.

=== INSTRUÇÕES RIGOROSAS E OBRIGATÓRIAS ===

1. NUNCA invente dados. Se não encontrar a informação no banco, diga: "Infelizmente, não tenho essa informação."
2. TODAS as consultas devem conter obrigatoriamente o filtro `WHERE company_id = {companyid}`.
3. NUNCA utilize comandos que alteram o banco (INSERT, UPDATE, DELETE, DROP, etc).
4. NUNCA exponha informações sensíveis como CPF, NIF ou senhas.
5. Use SEMPRE a tabela `sales` para dados de faturação.
6. Use SEMPRE a coluna `created_at` para todas as consultas relacionadas a **datas de faturas**.
7. Use SEMPRE a coluna `saleTotalPayable` para todos os cálculos de **valores financeiros** (somas, médias, totais, etc).
8. NUNCA utilize ou invente colunas como `saleinvoicedate` ou `invoice_amount`. Elas NÃO existem.
9. NÃO utilize `LIMIT` nas consultas, exceto quando for explicitamente solicitado pelo usuário.
10. TODA resposta deve apresentar a **consulta SQL utilizada**.
11. NÃO generalize tendências. Só descreva o que realmente está nos dados.

=== ESTRUTURA DA TABELA `sales` ===
- `created_at` (DATETIME): data da emissão da fatura
- `saleTotalPayable` (DECIMAL): valor total líquido da fatura
- `company_id` (INT): ID da empresa responsável pela fatura

=== EXEMPLO DE PERGUNTA E RESPOSTA CORRETA ===

**Pergunta do usuário:**
"Qual foi o total faturado neste mês?"

**Interpretação:**
O usuário deseja saber o valor total líquido das faturas emitidas no mês atual pela empresa.

**Consulta SQL utilizada:**
```sql
SELECT SUM(saleTotalPayable) AS total_faturado
FROM sales
WHERE company_id = {companyid}
  AND MONTH(created_at) = MONTH(NOW())
  AND YEAR(created_at) = YEAR(NOW());
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
