import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def build_prompt(user_question: str, schema: dict, companyid: int) -> str:
    schema_description = ""
    for table, columns in schema.items():
        schema_description += f"Tabela: {table}\nColunas: {', '.join(columns)}\n\n"

    prompt = f"""
Você é Xándria, uma IA que responde com clareza e gera sempre um insight útil com base nos dados reais do sistema.

- Toda SQL deve conter: WHERE company_companyid = {companyid}, se a tabela tiver essa coluna.
- Use sempre filtros de data como MONTH(created_at) e YEAR(created_at) quando a pergunta mencionar período (ex: este mês, últimos meses, ano passado).
- Para receitas: usar tabela `sales`, coluna `saleTotalPayable`.
- Use filtros rigorosos: `saleTotalPayable > 0` e `invoice_status = 'paid'` quando possível.
- Nunca some todos os registros da tabela sem filtrar (isso gera valores falsos).
- Se não houver dados, a SQL pode retornar vazio (não chute valores).

RESTRIÇÕES:
- Não arredonde valores.
- Não use mil/milhão a menos que o usuário peça explicitamente.
- Os valores devem aparecer com separador de milhar e 2 casas decimais (ex: 1.250.000,00 AKZ).
- Nunca responda com valores inventados. Só gere a SQL.

ESQUEMA:
{schema_description}

PERGUNTA:
{user_question}

Sua tarefa é gerar **apenas a SQL** correta e segura, nada mais.
""".strip()

    return prompt


def gerar_sql(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Limpa código desnecessário
        raw = raw.replace("```sql", "").replace("```", "").strip()
        raw = re.sub(r'<[^>]+>', '', raw)  # Remove tags HTML
        raw = re.sub(r'(assistant|user|output|response|markdown):?', '', raw, flags=re.IGNORECASE)

        # Extrai a SQL pura
        match = re.search(r"(SELECT\s.+?;)", raw, re.IGNORECASE | re.DOTALL)
        sql = match.group(1).strip() if match else raw.strip()

        # Limpeza final
        sql = "\n".join(line.strip() for line in sql.splitlines() if line.strip())

        if not sql.lower().startswith("select"):
            return "-- SQL inválida gerada"
        return sql

    except Exception as e:
        return f"-- Erro ao gerar SQL: {e}"
