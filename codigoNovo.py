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

load_dotenv()
cs = "mysql+mysqlconnector://root:@localhost:3306/pb"


# Database connection
#cs = "mysql+mysqlconnector://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
db_engine = create_engine(cs)
db = SQLDatabase(db_engine)

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation memory for each user
conversation_memory = {}

class User(BaseModel):
    prompt: str
    company_id: str

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)

@app.post("/chat")
def chat(user: User):
    # Create or get memory for this user/company
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
Você é um assistente de IA muito inteligente, especialista em identificar perguntas relevantes de um usuário e convertê-las em consultas SQL para gerar a resposta correta. Além disso, podes responder a perguntas de forma livres sem consultar a base de dados.

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
11. Se não souberes a resposta, podes responder de forma livre sem consultar a base de dados.
12. Se te saudarem responda, Ola nao posso ter conversas fora do escopo da base de dados, em que posso ajudar.

        """),
        ("user", "{question}\ ai: "),
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
    
    # Store the conversation
    memory.save_context({"input": user.prompt}, {"output": response})
    
    return response
