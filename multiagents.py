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

def is_sql_related(prompt: str) -> bool:
    # Keywords that might indicate SQL-related queries
    sql_keywords = [
        'select', 'mostrar', 'listar', 'quanto', 'quantos', 'valor', 'dados', 
        'tabela', 'relatório', 'vendas', 'clientes', 'produtos', 'registros',
        'total', 'soma', 'média', 'consultar'
    ]
    return any(keyword.lower() in prompt.lower() for keyword in sql_keywords)

# Initialize two different LLMs
general_llm = ChatOpenAI(model="gpt-4", temperature=0.7)
sql_llm = ChatOpenAI(model="gpt-4", temperature=0)

@app.post("/chat")
def chat(user: User):
    memory_key = f"user_{user.company_id}"
    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    memory = conversation_memory[memory_key]
    
    # Check if the query is SQL-related
    if is_sql_related(user.prompt):
        # Use SQL LLM and toolkit
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=sql_llm)
        sql_toolkit.get_tools()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """... your existing SQL system prompt ..."""),
            ("user", "{question}\ ai: "),
        ])

        agent = create_sql_agent(
            llm=sql_llm,
            toolkit=sql_toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_execution_time=100,
            max_iterations=1000,
            handle_parsing_errors=True,
            memory=memory
        )
        
        response = agent.run(prompt.format_prompt(question=user.prompt, companyId=user.company_id))
    else:
        # Use general LLM for non-SQL queries
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente geral amigável. Responda perguntas gerais de forma clara e concisa.
            Se a pergunta exigir dados específicos do banco de dados, sugira que o usuário reformule a pergunta."""),
            ("user", "{question}")
        ])
        
        chain = prompt | general_llm
        response = chain.invoke({"question": user.prompt})
        response = response.content

    memory.save_context({"input": user.prompt}, {"output": response})
    return response
