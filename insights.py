from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# 🔗 Instância única do modelo
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# 🚩 Arquivo de log de insights
INSIGHTS_FILE = Path("insights_log.md")

# --- auxiliares -----------------------------------------------------------


def _to_markdown_table(result: List[Any]) -> str:
    """Converte até 15 linhas do resultado em tabela Markdown minimalista."""
    if not result:
        return "(sem linhas)"

    if isinstance(result[0], (tuple, list)):
        cols = len(result[0])
        headers = [f"col_{i+1}" for i in range(cols)]
        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join(["---"] * cols) + " |"]
        for row in result[:15]:
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        return "\n".join(lines)

    # Resultado simples (ex.: contagem, soma)
    return str(result[0]) if isinstance(result, list) else str(result)


_CURRENCY_TRASH_RE = re.compile(r"\b(?:R\$|USD|AOA|AO\$)\s*", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?)")


def _normalize_currency(text: str) -> str:
    """
    Remove símbolos de moedas indesejados e adiciona ' Kz' apenas a números decimais
    que representem valores monetários, evitando anos e duplicações.
    """
    # Remove símbolos errados como R$, USD, AOA etc.
    text = _CURRENCY_TRASH_RE.sub("", text)

    def _should_add_kz(m: re.Match) -> str:
        num = m.group(1)
        # Evita anos como "2024" (sem vírgula decimal) e já terminados em "Kz"
        if "," not in num:
            return num  # assume ano ou número inteiro
        after = text[m.end(): m.end() + 4].lower()
        if "kz" in after:
            return num  # já possui Kz
        return num + " Kz"

    return _NUMBER_RE.sub(_should_add_kz, text)

# --- geração de insight ---------------------------------------------------


def build_insight(user_prompt: str, sql_result: List[Any]) -> str:
    """Gera texto de insight curto (máx 4–5 linhas) via LLM."""
    
    # 🔒 Proteção contra alucinação com meses simples
    if sql_result and isinstance(sql_result[0], (tuple, list)):
        colunas = len(sql_result[0])
        if colunas == 1:
            valores = [r[0] for r in sql_result]
            if all(isinstance(v, int) and 1 <= v <= 12 for v in valores):
                return "Os dados mostram os meses com faturação, mas sem valores associados para análise aprofundada."

    table_md = _to_markdown_table(sql_result)

    insight_prompt = f"""
Você é um analista financeiro experiente em Angola. Com base na **pergunta** e nos **dados** abaixo,
redija um insight conciso (2‑3 linhas), em português, destacando tendências, picos, quedas ou alertas.
Quando mencionar valores, escreva o número seguido de **“ Kz”** (com espaço antes). Nunca use R$, USD ou outras moedas.
Digite apenas o texto do insight.

**Pergunta:** {user_prompt}

**Dados:**
{table_md}
""".strip()

    response = llm.invoke(insight_prompt)
    raw_insight = response.content.strip()
    return _normalize_currency(raw_insight)

def save_insight(insight: str, separator: bool = True) -> None:
    """Anexa o insight ao arquivo markdown de histórico."""
    with INSIGHTS_FILE.open("a", encoding="utf‑8") as fp:
        if separator:
            fp.write("\n---\n")
        fp.write(insight + "\n")


def generate_and_persist(user_prompt: str, sql_result: List[Any]) -> str:
    """Gera o insight, normaliza, persiste no log e devolve o texto."""
    insight = build_insight(user_prompt, sql_result)
    save_insight(insight)
    return insight


# Teste rápido
if __name__ == "__main__":
    dummy_prompt = "Total faturado em 2024"
    dummy_rows = [(77_574_683.45,)]
    print(generate_and_persist(dummy_prompt, dummy_rows))
