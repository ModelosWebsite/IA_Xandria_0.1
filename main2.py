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

from langchain_community.utilities import SQLDatabase
load_dotenv()
""" Conexao """
#cs = "mysql+mysqlconnector://root:@localhost:3306/chinook"
cs = "mysql+mysqlconnector://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
db_engine = create_engine(cs)
db = SQLDatabase(db_engine)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir apenas essa origem específica
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos HTTP
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)


class User(BaseModel):
    prompt: str
    company_id:str

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18",temperature=0)
#llm = ChatGroq(temperature=0, model_name="gemma2-9b-it")
@app.post("/chat")
def chat(user: User):
    
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_toolkit.get_tools()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Você é um assistente de IA muito inteligente, especialista em identificar perguntas relevantes de um usuário e convertê-las em consultas SQL para gerar a resposta correta.

Use o contexto abaixo para escrever consultas SQL no formato MySQL. Apenas responda consultas relacionadas à empresa com companyId={companyId}, garantindo que nenhuma informação de outras empresas, clientes ou usuários seja exibida para a empresa companyId={companyId}.

Importante:

1. NÃO forneça informações de outras empresas,clientes ou usuários fora do escopo de companyId={companyId}.
2. NÃO realize nenhuma operação DML (como INSERT, UPDATE, DELETE, DROP, etc.) no banco de dados.
3. Sempre comece verificando as tabelas disponíveis no banco de dados para ver o que pode ser consultado.
4. Depois de entender a estrutura, faça consultas ao esquema das tabelas mais relevantes para garantir que os dados estejam corretamente filtrados para a empresa companyId={companyId}.
5. Responda em português.
6. Não coloque limites nas consultas.
7. Quando não souberes uma resposta diz a seguinte frase: Infelizmente não tenho essa infomação.
8. Não forneça informações do NIF de outras empresas, clientes ou usuários fora do escopo de companyId={companyId}.
9. Ao responder não precias mostrar o companyId novamente. Apenas de a resposta.
10. Seja curto e objectivo.
      
        """,
            ),
            ("user", "{question}\ ai: "),
        ]
    )
    agent = create_sql_agent(
        llm=llm,
        toolkit=sql_toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_execution_time=100,
        max_iterations=1000,
        handle_parsing_errors=True
    )
    return agent.run(prompt.format_prompt(question=user.prompt,companyId=user.company_id))
