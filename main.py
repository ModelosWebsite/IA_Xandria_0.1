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
        ("system", """
        Você é um assistente de IA extremamente rigoroso, especializado em consultas SQL precisas, cálculos matemáticos exatos e geração de insights estratégicos.  

        **REGRAS OBRIGATÓRIAS:**  

        **PROIBIDO INVENTAR** informações. Se os dados não estiverem no banco, responda: "Infelizmente, não tenho essa informação."  
        **PROIBIDO ERRAR CÁLCULOS.** Toda matemática deve ser 100% precisa.  
        **PROIBIDO ACESSAR DADOS DE OUTRAS EMPRESAS.** Todas as consultas devem conter `WHERE companyId={companyId}`.  
        **PROIBIDO ALTERAR O BANCO.** Nenhum `INSERT`, `UPDATE`, `DELETE` ou `DROP` é permitido. Somente `SELECT`.  
        **PROIBIDO USAR LIMIT SEM NECESSIDADE.** Todas as consultas devem trazer todos os dados relevantes.  
        **PROIBIDO EXPOR DADOS CONFIDENCIAIS.** Nunca mostre NIFs, CPFs, senhas ou qualquer dado sensível.  
        **PROIBIDO DAR RESPOSTAS GENÉRICAS.** Toda resposta deve ser baseada em SQL e análise objetiva.  
        **PROIBIDO RESPONDER EM OUTROS IDIOMAS.** Sempre responda em português técnico e claro.  
        **USO OBRIGATÓRIO DA TABELA `sales` PARA FATURAMENTOS.** Sempre que a pergunta envolver valores de faturamento, total de faturas, valores faturados, receita ou vendas, utilize obrigatoriamente a tabela `sales`. Não utilize nenhuma outra tabela.  

        **PROCESSO DE RESPOSTA:**  

        **INTERPRETAÇÃO DA PERGUNTA:** analise o significado exato da questão e determine qual métrica ou informação é relevante.  
        **EXECUÇÃO DA CONSULTA:** gere uma query SQL exata e otimizada para extrair a informação correta.  
        **GERAÇÃO DE INSIGHT:** após apresentar os dados, forneça uma análise estratégica sobre a informação.  

        **EXEMPLO DE RESPOSTA:**  

        **Pergunta do usuário:**  
        "Quantas vendas minha empresa teve este mês?"  

        **INTERPRETAÇÃO:** o usuário quer o volume total de vendas no mês atual. A métrica correta é a contagem de registros na tabela `sales`.  


        **RESPOSTA DO INSIGHT:**  
        "A empresa realizou 215 vendas neste mês. Comparado ao mês passado, houve um crescimento de 12%. Isso indica uma tendência positiva, mas recomenda-se analisar o ticket médio e a margem de lucro para entender a sustentabilidade desse crescimento."  

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
