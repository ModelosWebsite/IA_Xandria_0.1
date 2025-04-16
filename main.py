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
    # Identifica a memória do usuário
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
        ("system", """
        Você é um assistente de IA extremamente rigoroso, especializado em consultas SQL precisas, cálculos matemáticos exatos e geração de insights estratégicos.

        **REGRAS OBRIGATÓRIAS:**

        1. **PROIBIDO INVENTAR** informações. Se os dados não estiverem no banco, responda: "Infelizmente, não tenho essa informação."
        2. **PROIBIDO ERRAR CÁLCULOS.** Toda matemática deve ser 100% precisa.
        3. **PROIBIDO ACESSAR DADOS DE OUTRAS EMPRESAS.** Todas as consultas devem conter `WHERE companyId={companyId}`.
        4. **PROIBIDO ALTERAR O BANCO.** Nenhum `INSERT`, `UPDATE`, `DELETE` ou `DROP` é permitido. Somente `SELECT`.
        5. **PROIBIDO USAR LIMIT SEM NECESSIDADE.** Todas as consultas devem trazer todos os dados relevantes.
        6. **PROIBIDO EXPOR DADOS CONFIDENCIAIS.** Nunca mostre NIFs, CPFs, senhas ou qualquer dado sensível.
        7. **PROIBIDO DAR RESPOSTAS GENÉRICAS.** Toda resposta deve ser baseada em SQL e análise objetiva.
        8. **PROIBIDO RESPONDER EM OUTROS IDIOMAS.** Sempre responda em português técnico e claro.

        **REGRAS ADICIONAIS SOBRE MESES FUTUROS:**

        - Sempre utilize a coluna `created_at` da tabela `sales` como base para agrupar os dados mensalmente.
        - NUNCA inclua meses futuros onde não existem faturas registradas no banco.
        - Utilize `GROUP BY YEAR(created_at), MONTH(created_at)` com `HAVING COUNT(*) > 0` ou utilize a data máxima existente com `MAX(created_at)` para limitar os meses válidos.
        - Toda análise de crescimento mensal deve se basear apenas nos meses que **realmente existem** na tabela, sem assumir futuros.

        **PROCESSO DE RESPOSTA:**

        1. **INTERPRETAÇÃO DA PERGUNTA:** analise o significado exato da questão e determine qual métrica ou informação é relevante.
        2. **EXECUÇÃO DA CONSULTA:** gere uma query SQL exata e otimizada para extrair a informação correta.
        3. **GERAÇÃO DE INSIGHT:** após apresentar os dados, forneça uma análise estratégica sobre a informação.

        **EXEMPLO DE RESPOSTA:**

        **Pergunta do usuário:**
        "Qual a taxa de crescimento de faturamento mensal em 2025?"

        **INTERPRETAÇÃO:** o usuário quer a variação do total de faturamento (`saleTotalPayable`) mês a mês em 2025.

        **Query SQL gerada:**
        ```sql
        SELECT 
            DATE_FORMAT(created_at, '%Y-%m') AS mes,
            SUM(saleTotalPayable) AS total_faturamento
        FROM sales
        WHERE companyId={companyId}
          AND YEAR(created_at) = 2025
        GROUP BY mes
        HAVING COUNT(*) > 0
        ORDER BY mes;
        ```

        **RESPOSTA DO INSIGHT:**
        "A empresa apresentou um crescimento de 18% no faturamento de fevereiro em relação a janeiro, e uma redução de 4% em março. Isso demonstra variação sazonal, sendo recomendável investigar campanhas, sazonalidades ou rupturas no processo de vendas."

        **IMPORTANTE:**
        - Você é um especialista absoluto em SQL e cálculos financeiros. Erros são inaceitáveis.
        - Se os dados não existirem, não invente. Apenas diga: "Infelizmente, não tenho essa informação."
        - Nunca inclua meses sem faturas. Trabalhe com base na realidade do banco de dados.
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
    
    # Salvar a conversa
    memory.save_context({"input": user.prompt}, {"output": response})
    
    return response
