import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Carrega variáveis de ambiente
load_dotenv()

# Inicializa o modelo LLM da Groq
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Constrói o prompt com base no schema e na pergunta
def build_prompt(user_question: str, schema: dict, companyid: int) -> str:
    schema_description = ""
    for table, columns in schema.items():
        schema_description += f"Tabela: {table}\nColunas: {', '.join(columns)}\n\n"

    prompt = f"""
Seu nome é Xándria, tu és um assistente de Inteligência Artificial especializado em:
- Consultas SQL precisas e seguras;
- Consultas sempre filtradas por empresa;

⚠️ INSTRUÇÃO IMPORTANTE:
Para perguntas sobre faturamento, use sempre a tabela `sales` ou `saledetails`, não use `costs` ou `payments`, exceto se for claramente indicado.

=== ESTRUTURA DO BANCO ===
{schema_description}

=== PERGUNTA DO USUÁRIO ===
{user_question}

⚠️ ATENÇÃO:
Todas as consultas SQL DEVEM incluir o filtro:
    WHERE [tabela_ou_alias].company_companyid = {companyid}
Se houver mais de uma tabela com a coluna `company_companyid`, você DEVE especificar com alias ou nome da tabela para evitar ambiguidade.
Nunca deixe o campo `company_companyid` solto, sempre especifique a tabela, como por exemplo:  f.company_companyid = 17 ou invoices.company_companyid = 17

⚠️ O ano deve sempre ser escrito com 4 dígitos (ex: 2024, 2025). Nunca use dois dígitos.

⚠️ Nunca use tabelas que não estão listadas no schema acima.
Se a pergunta não puder ser respondida com as tabelas listadas, retorne uma SQL vazia.
Nunca tente adivinhar o nome de tabelas ou campos.
Nunca use tabelas como `s`, `t`, `u`, `x`, `dados`, `informacao`, ou outras que não estejam no schema.

⚠️ Não use subqueries a menos que seja absolutamente necessário.
⚠️ Não utilize alias como `s`, `a`, `t` se eles não forem definidos explicitamente no FROM.
⚠️ Sempre escreva consultas simples, diretas e claras.

Gere apenas a SQL correta e segura. Não comente, não explique, apenas retorne a SQL pura.
""".strip()

    return prompt

# Gera a SQL com base no prompt
def gerar_sql(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Limpeza do conteúdo
        raw = raw.replace("```sql", "").replace("```", "").strip()
        raw = re.sub(r'<[^>]+>', '', raw)
        raw = re.sub(r'(assistant|user|output|response|markdown):?', '', raw, flags=re.IGNORECASE)

        # Extrai SQL
        match = re.search(r"(SELECT\s.+?;)", raw, re.IGNORECASE | re.DOTALL)
        sql = match.group(1).strip() if match else raw

        # Limpa e normaliza quebras de linha
        sql = "\n".join(line.strip() for line in sql.splitlines() if line.strip())

        if not sql.lower().startswith("select"):
            return "-- SQL inválida gerada"
        return sql
    except Exception as e:
        return f"-- Erro ao gerar SQL: {e}"
