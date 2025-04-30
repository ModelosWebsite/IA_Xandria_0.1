import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do banco com SQLAlchemy síncrono
DATABASE_URL = "mysql+pymysql://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Inicializa o FastAPI
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xzero.ao"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Classe do corpo da requisição
class User(BaseModel):
    prompt: str
    companyid: str

# Conversas por empresa (memória)
conversation_memory = {}

# Instanciando a LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Ferramentas personalizadas
@tool
def documentation_tool(url: str, question: str) -> str:
    """Tool para extrair documentação de uma URL e responder perguntas sobre ela."""
    return f"[Mock] Resposta da documentação para a pergunta: {question}"

@tool
def black_formatter_tool(path: str) -> str:
    """Tool para formatar arquivos Python usando Black."""
    try:
        os.system(f"black {path}")
        return "Formatado com sucesso!"
    except Exception as e:
        return f"Erro ao formatar: {str(e)}"

# LEITURA DO ARQUIVO promptzero.txt
promptzero_path = "promptzero.txt"
if os.path.exists(promptzero_path):
    with open(promptzero_path, "r", encoding="utf-8") as f:
        promptzero_text = f.read()
else:
    promptzero_text = ""

# Prompt principal (seu prompt original + conteúdo do arquivo)
system_prompt = f"""
Você é um assistente de Inteligência Artificial especializado em:
- Consultas SQL precisas e seguras;
- Cálculos matemáticos e estatísticos corretos;
- Geração de análises claras, profundas, humanas e bem estruturadas em português formal.

=== REGRAS OBRIGATÓRIAS ===
1. Sempre interprete cuidadosamente a pergunta do usuário.
2. Se a pergunta for sobre **faturas**, use a tabela `sales`, especialmente:
    - `created_at` para datas de faturação.
    - `saleTotalPayable` para valores faturados.
3. Se a pergunta for sobre **interações**, use a tabela `interactions`.
4. Realize cálculos de forma precisa (somas, médias, percentuais, etc.).
5. NUNCA invente dados. Se não existirem registros, diga: "Nenhum registro encontrado para essa consulta."
6. Não exponha dados sensíveis como NIF, CPF, senhas ou informações pessoais.
7. Sempre redija as respostas de forma clara, formal e acolhedora.
8. Baseie as respostas exclusivamente nos dados do banco de dados.
9. Apresente análises profundas, oferecendo **insights inteligentes e úteis** em cada resposta.

=== COMO RESPONDER ===
- Comece com um resumo direto do resultado (em tom acolhedor e humano).
- Logo depois, ofereça **pelo menos um insight** relevante baseado nos dados encontrados.
- Use expressões que demonstrem empatia e inteligência, como:
    "Isso sugere que...", "Pode ser interessante considerar...", "Uma possível interpretação é...", "Vale a pena analisar...".
- Escreva de maneira formal, mas próxima do usuário, como um consultor experiente faria.
- Evite respostas frias e técnicas demais. Seja claro, inteligente e humano.
- Caso não existam dados para a consulta, responda gentilmente: "Nenhum registro encontrado para essa consulta. Caso necessário, podemos explorar outros períodos ou categorias."

=== EXEMPLO DE RESPOSTA ===
**Resumo do Faturamento no Mês Atual**

O total faturado pela empresa no mês de abril foi de **8.950.000 AKZ**.

**Insight**: Esse valor indica uma forte atividade comercial no período. Pode ser interessante analisar quais categorias de produtos ou serviços mais contribuíram para este resultado, visando estratégias de expansão.

**Nota**: Se desejar, posso ajudar a detalhar ainda mais a origem desse faturamento.

=== INSTRUÇÕES IMPORTANTES ===
- A consulta SQL deve ser usada internamente para gerar a resposta correta, mas **não deve ser exibida ao usuário**, a menos que ele peça explicitamente.
- Foque sempre na clareza, segurança e profundidade das respostas.

=== INSTRUÇÕES ADICIONAIS ===
{promptzero_text}
"""

# Prompt com LangChain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Inicializa o SQLDatabase síncrono
db_sync = SQLDatabase(engine)

# Toolkit e Agente
sql_toolkit = SQLDatabaseToolkit(db=db_sync, llm=llm)
toolkit = sql_toolkit.get_tools() + [documentation_tool, black_formatter_tool]
agent = create_openai_functions_agent(llm=llm, tools=toolkit, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

# Rota principal (síncrona)
@app.post("/chat")
def chat(user: User):
    memory_key = f"user_{user.companyid}"

    # Cria memória se não existir
    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    memory = conversation_memory[memory_key]

    # Executa o agente
    response = agent_executor.invoke({
        "input": user.prompt,
        "chat_history": memory.chat_memory.messages,
        "companyid": user.companyid
    })

    # Atualiza memória com a interação
    memory.save_context({"input": user.prompt}, {"output": response["output"]})

    return {"resposta": response["output"]}
