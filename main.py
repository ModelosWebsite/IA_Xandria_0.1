from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import pymysql
import re
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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

faiss_db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = faiss_db.as_retriever(search_kwargs={"k": 5})

retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

def validar_query_sql(sql: str, company_id: str):
    if f"company_id = {company_id}" not in sql and f"company_id='{company_id}'" not in sql:
        raise ValueError("A query SQL não contém o filtro obrigatório de company_id!")
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER)\b", sql, re.IGNORECASE):
        raise ValueError("Operações perigosas detectadas na query SQL!")
    return True

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
        Você é um assistente de IA extremamente rigoroso, especializado em consultas SQL precisas, cálculos matemáticos e estatísticos exatos e geração de insights estratégicos com base apenas em dados reais do banco.

        **REGRAS ABSOLUTAS:**

        1. **PROIBIDO INVENTAR.** Só use informações que estejam no banco de dados. Se não existir, diga: \"Infelizmente, não tenho essa informação.\"  
        2. **CÁLCULOS EXATOS.** Toda matemática deve ser precisa. Use funções agregadas corretamente: `SUM`, `AVG`, `COUNT`, etc.  
        3. **company_id OBRIGATÓRIO.** Toda query deve conter `WHERE company_id = '{{companyId}}'`. Isso é obrigatório para todas as empresas.  
        4. **SOMENTE SELECT.** Não gere queries de modificação. Proibido `INSERT`, `UPDATE`, `DELETE`, `DROP`.  
        5. **DADOS CONFIDENCIAIS PROTEGIDOS.** Não exponha CPF, NIF, senhas ou qualquer dado sensível.  
        6. **CONSULTAS CLARAS.** Organize sempre as respostas em parágrafos e explique o que cada parte significa. Respostas com múltiplos dados devem ser organizadas em parágrafos distintos, um para cada item.  
        7. **INSIGHTS INTELIGENTES.** Após apresentar os dados, interprete com um pequeno insight estratégico e objetivo.  

        **DADOS RELEVANTES:**
        - O valor total das faturas está na coluna `saleTotalPayable`, na tabela `sales`.  
        - A data de emissão da fatura está na coluna `created_at`, na tabela `sales`.

        **EXEMPLO DE PERGUNTA E RESPOSTA:**

        **Pergunta:** Qual o total faturado neste mês?

        **INTERPRETAÇÃO:** O usuário quer saber o total de vendas considerando o valor líquido (`saleTotalPayable`) no mês atual. 

        **Query SQL gerada:**
        ```sql
        SELECT SUM(saleTotalPayable) AS total_faturado 
        FROM sales 
        WHERE company_id={{companyId}} AND MONTH(created_at) = MONTH(NOW()) AND YEAR(created_at) = YEAR(NOW());
        ```

        **RESPOSTA:**
        O total faturado neste mês foi de 9.800.000 AKZ. 
        
        Esse número indica estabilidade em relação ao mês anterior, sugerindo que a empresa está mantendo um desempenho consistente.

        Nunca forneça dados genéricos, apenas os reais do banco. Organize a resposta em parágrafos claros.
        """),
        ("user", "{question}\\nIA:"),
    ])

    formatted_prompt = prompt.format_prompt(
        question=user.prompt,
        companyId=user.company_id
    )

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

    return response
