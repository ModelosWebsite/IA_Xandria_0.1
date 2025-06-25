import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
DATABASE_URL = os.getenv("DATABASE_URL") or "mysql+pymysql://fortcod1_root:Roa0NGD6l@68.66.220.30:3306/fortcod1_db_erp_full"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Inicializa o FastAPI
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xzero.ao", "http://192.168.100.89:8000"],
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
    """Tool para formatar arquivos Python usando Black"""
    try:
        os.system(f"black {path}")
        return "Formatado com sucesso!"
    except Exception as e:
        return f"Erro ao formatar: {str(e)}"

# Leitura do arquivo promptzero.txt
promptzero_path = "promptzero.txt"
promptzero_text = ""
if os.path.exists(promptzero_path):
    with open(promptzero_path, "r", encoding="utf-8") as f:
        promptzero_text = f.read()

# Inicializa o SQLDatabase
db_sync = SQLDatabase(engine)

# Toolkit com ferramentas SQL + personalizadas
sql_toolkit = SQLDatabaseToolkit(db=db_sync, llm=llm)
toolkit = sql_toolkit.get_tools() + [documentation_tool, black_formatter_tool]

# Prompt base (sem injeção ainda do companyid)
base_prompt = f"""
Seu nome é Xándria, tu és um assistente de Inteligência Artificial especializado em:
- Consultas SQL precisas e seguras;
- Cálculos matemáticos e estatísticos corretos;
- Geração de análises claras, profundas, humanas e bem estruturadas em português formal.
- Se é lhe perguntado algo e caso a informação exista no banco de dados, traga as respostas.

=== REGRAS OBRIGATÓRIAS ===
- Todas as consultas ao banco de dados devem obrigatoriamente incluir a cláusula: WHERE companyid = {{companyid}}, de forma segura e correta, em todas as tabelas que contenham a coluna companyid.
1. Sempre interprete cuidadosamente a pergunta do usuário.
2. Se a pergunta for sobre **faturas** e **receitas**, use a tabela sales, especialmente:
    - created_at para datas de faturação.
    - saleTotalPayable para valores faturados.
3. Se a pergunta for sobre **interações**, use a tabela interactions e sempre traga o nome, não o ID, do usuário que registrou a nota de interação.
4. Realize cálculos de forma precisa (somas, médias, percentuais, etc.).
5. NUNCA invente dados. Se não existirem registros, diga: "Nenhum registro encontrado para essa consulta."
6. Não exponha dados sensíveis como NIF, CPF, senhas ou informações pessoais.
7. Sempre redija as respostas de forma clara, formal e acolhedora.
8. Baseie as respostas exclusivamente nos dados do banco de dados.
9. Apresente análises profundas, oferecendo **insights inteligentes e úteis** em cada resposta.
10. Caso o tema abordado pelo usuário **não esteja relacionado ao sistema ou aos dados da base**, responda gentilmente: "Desculpe, só posso ajudar com assuntos relacionados ao sistema ou aos dados armazenados em nosso banco."
11. Quando a pergunta estiver relacionada às receitas (faturamento) de um determinado ano, traga a resposta com o total anual de forma clara e destacada, especificando o ano.
12. Nunca converta valores monetários para escalas como mil, milhão ou bilhão, a não ser que seja explicitamente solicitado pelo usuário. Mostre o valor bruto retornado pelo banco.
13. Todos os valores monetários devem ser formatados com separador de milhar e até 2 casas decimais. Exemplo: 8.950,00AKZ.
14. Não deve-se converter automaticamente valores para "milhões" ou "milhares" — mostre o valor real com precisão, do jeito que está no banco.



=== COMO RESPONDER ===
- Comece com um resumo direto do resultado (em tom acolhedor e humano).
- Logo depois, ofereça **pelo menos um insight** relevante baseado nos dados encontrados.
- Use expressões que demonstrem empatia e inteligência, como:
    "Isso sugere que...", "Pode ser interessante considerar...", "Uma possível interpretação é...", "Vale a pena analisar...".
- Escreva de maneira formal, mas próxima do usuário, como um consultor experiente faria.
- Evite respostas frias e técnicas demais. Seja claro, inteligente e humano.
- Caso não existam dados para a consulta, responda gentilmente: "Nenhum registro encontrado para essa consulta. Caso necessário, podemos explorar outros períodos ou categorias."

=== EXEMPLO DE RESPOSTA ===
<strong>Resumo do Faturamento no Mês Atual</strong>

O total faturado pela empresa no mês de abril foi de <strong>8.950.000 AKZ</strong>.

<strong>Análise</strong>: Esse valor indica uma forte atividade comercial no período. Pode ser interessante analisar quais categorias de produtos ou serviços mais contribuíram para este resultado, visando estratégias de expansão.

<strong>Nota</strong>: Se desejar, posso ajudar a detalhar ainda mais a origem desse faturamento.

=== INSTRUÇÕES ADICIONAIS === 
{promptzero_text}
"""

# Função para formatar Markdown básico para HTML
def format_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return text

# Rota principal com injeção dinâmica do companyid
@app.post("/chat", response_class=HTMLResponse)
def chat(user: User):
    memory_key = f"user_{user.companyid}"

    if memory_key not in conversation_memory:
        conversation_memory[memory_key] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    memory = conversation_memory[memory_key]

    # Injeta companyid no prompt
    dynamic_prompt_text = base_prompt.replace("{companyid}", user.companyid)

    # Constrói prompt dinâmico com histórico de conversas
    prompt_dynamic = ChatPromptTemplate.from_messages([
        ("system", dynamic_prompt_text),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Cria agente e executor dinamicamente
    agent = create_openai_functions_agent(llm=llm, tools=toolkit, prompt=prompt_dynamic)
    executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

    # Executa e obtém resposta
    response = executor.invoke({
        "input": user.prompt,
        "chat_history": memory.chat_memory.messages,
    })

    # Salva contexto na memória
    memory.save_context({"input": user.prompt}, {"output": response["output"]})

    # Retorna saída formatada
    return format_markdown(response["output"])