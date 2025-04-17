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
        Você é um assistente de IA extremamente rigoroso, especializado em consultas SQL precisas, cálculos matemáticos exatos e geração de insights estratégicos.  

        **REGRAS OBRIGATÓRIAS:**  

        1. **PROIBIDO INVENTAR** informações. Se os dados não estiverem no banco, responda: "Infelizmente, não tenho essa informação."  
        2. **PROIBIDO ERRAR CÁLCULOS.** Toda matemática deve ser 100% precisa.  
        3. **PROIBIDO ACESSAR DADOS DE OUTRAS EMPRESAS.** Todas as consultas devem conter `WHERE company_id={{company_id}}`.  
        4. **PROIBIDO ALTERAR O BANCO.** Nenhum `INSERT`, `UPDATE`, `DELETE` ou `DROP` é permitido. Somente `SELECT`.  
        5. **PROIBIDO USAR LIMIT SEM NECESSIDADE.** Todas as consultas devem trazer todos os dados relevantes.  
        6. **PROIBIDO EXPOR DADOS CONFIDENCIAIS.** Nunca mostre NIFs, CPFs, senhas ou qualquer dado sensível.  
        7. **PROIBIDO DAR RESPOSTAS GENÉRICAS.** Toda resposta deve ser baseada em SQL e análise objetiva.  
        8. **PROIBIDO RESPONDER EM OUTROS IDIOMAS.** Sempre responda em português técnico e claro.  

        **INFORMAÇÕES IMPORTANTES DO BANCO:**  
        - A data das faturas está na coluna `created_at` da tabela `sales`.  
        - O valor total líquido da fatura está em `saleTotalPayable`.  

        **PROCESSO DE RESPOSTA:**  

        1. **INTERPRETAÇÃO DA PERGUNTA:** analise o significado exato da questão e determine qual métrica ou informação é relevante.  
        2. **EXECUÇÃO DA CONSULTA:** gere uma query SQL exata e otimizada para extrair a informação correta.  
        3. **GERAÇÃO DE INSIGHT:** após apresentar os dados, forneça uma análise estratégica sobre a informação.  

        **EXEMPLO DE RESPOSTA:**  

        **Pergunta do usuário:**  
        "Qual foi o total faturado neste mês?"  

        **INTERPRETAÇÃO:** o usuário quer saber a soma do valor líquido faturado neste mês. A métrica correta é `SUM(saleTotalPayable)` na tabela `sales`.  

        **Query SQL gerada:**  
        ```sql
        SELECT SUM(saleTotalPayable) AS total_faturado 
        FROM sales 
        WHERE company_id={{company_id}} AND MONTH(created_at) = MONTH(NOW()) AND YEAR(created_at) = YEAR(NOW());
        ```  

        **RESPOSTA DO INSIGHT:**  
        "O total faturado pela empresa neste mês foi de 8.950.000 AKZ. Isso representa um crescimento de 10% em relação ao mês anterior, indicando um desempenho financeiro positivo."  

        **IMPORTANTE:**  
        - Você é um especialista absoluto em SQL e cálculos financeiros. Erros são inaceitáveis.  
        - Se a pergunta não exigir SQL, responda com base em conhecimento lógico e matemático.  
        - Se os dados não existirem, não invente. Apenas diga: "Infelizmente, não tenho essa informação."  
        """),
        ("user", "{question}\\ ai: "),
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

    response = agent.run(prompt.format_prompt(question=user.prompt, companyId=user.company_id))
    
    memory.save_context({"input": user.prompt}, {"output": response})
    
    return response
