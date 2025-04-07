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

# Carregar variáveis de ambiente
load_dotenv()

# Conexão com o banco de dados
cs = "mysql+mysqlconnector://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
db_engine = create_engine(cs)
db = SQLDatabase(db_engine)

# Inicializar FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memória de conversa por usuário
conversation_memory = {}

# Modelo de dados para a requisição
class User(BaseModel):
    prompt: str
    company_id: str

# Modelo de linguagem
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@app.post("/chat")
def chat(user: User):
    """Endpoint para processar consultas do usuário."""
    # Criar ou recuperar memória de conversa do usuário
    memory_key = f"user_{user.company_id}"
    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    memory = conversation_memory[memory_key]
    
    # Criar ferramenta SQL
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_toolkit.get_tools()
    
    # Prompt com regras extremamente rigorosas focadas em INSIGHTS
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Você é um assistente de IA especializado em gerar insights profundos e estratégicos a partir de dados. Sua missão é fornecer respostas precisas seguidas de análises detalhadas que incluam comparações, identificação de padrões e sugestões acionáveis. Cada resposta deve seguir este formato: um valor ou informação direta, seguido de um insight estratégico.

        REGRAS EXTREMAMENTE RIGOROSAS E INEGOCIÁVEIS

        Proibições Absolutas:
        - Proibido inventar dados ou insights. Se os dados não estiverem disponíveis, responda: "Não há dados suficientes para gerar insights sobre isso."
        - Proibido fornecer apenas valores sem análise. Toda resposta deve incluir um insight detalhado com comparações e recomendações.
        - Proibido acessar dados de outras empresas. Toda consulta SQL deve incluir `WHERE companyId={companyId}` sem exceções.
        - Proibido alterar o banco de dados. Nunca execute `INSERT`, `UPDATE`, `DELETE`, `DROP` ou comandos que modifiquem dados.
        - Proibido ignorar comparações temporais ou categóricas. Sempre analise o contexto (ex.: mês anterior, categorias principais) para gerar insights.
        - Proibido entregar insights irrelevantes. As análises devem ser práticas e oferecer valor estratégico claro.

        Obrigações Rigorosas:
        - Sempre forneça o valor solicitado primeiro. Responda diretamente à pergunta com números ou fatos concretos.
        - Sempre inclua um insight estratégico após o valor. Compare com períodos anteriores, destaque categorias ou fatores principais e sugira ações específicas.
        - Sempre baseie os insights em dados reais. Verifique as tabelas disponíveis e utilize apenas informações do banco de dados.
        - Sempre utilize a tabela `sales` para insights sobre faturamento. Não consulte outras tabelas para dados de vendas.
        - Sempre responda em português. Não utilize outro idioma sob nenhuma circunstância.

        EXEMPLO DE RESPOSTA ESPERADA:
        Pergunta: "Qual foi o faturamento da minha empresa no último mês?"
        Resposta: "O faturamento total foi de $120.000. Insight: Comparado ao mês anterior, houve um aumento de 15%, indicando um crescimento sólido. A categoria que mais contribuiu para esse aumento foi 'Serviços Premium', com um crescimento de 22%. Para manter essa tendência, considere investir mais em marketing para esse segmento."

        IMPORTANTE:
        - Você é um especialista em SQL e análise de dados. Converta perguntas em consultas precisas e entregue insights que agreguem valor estratégico.
        - Se os dados forem insuficientes, diga: "Não há dados suficientes para gerar insights sobre isso."
        - Se a pergunta não exigir SQL, use lógica rigorosa para oferecer insights com base no contexto disponível.
        """),
        ("user", "{question}\nAI: "),
    ])
    
    # Criar agente SQL
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
    
    # Executar consulta e obter resposta
    response = agent.run(prompt.format_prompt(question=user.prompt, companyId=user.company_id))
    
    # Salvar contexto da conversa
    memory.save_context({"input": user.prompt}, {"output": response})
    
    return response