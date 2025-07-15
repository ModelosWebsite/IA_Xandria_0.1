from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from database import fetch_schema_info, execute_sql
from llm_agent import build_prompt, gerar_sql
from typing import List
from insights import generate_and_persist
from utils import (
    formatar_faturas_unicas,
    format_markdown,
    identificar_tabela_do_prompt,
    formatar_faturas,
    formatar_meses,
    resposta_inteligente,
    resposta_numerica_inteligente
)
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    prompt: str
    companyid: int

def dividir_perguntas(prompt: str) -> list[str]:
    partes = re.split(r"(?<=[?.])\s+|\s+e\s+(?=\d{4})", prompt)
    return [p.strip() for p in partes if p.strip()]

def gerar_output_formatado(sub_prompt: str, resultado_bruto):
    if "fatura" in sub_prompt.lower():
        colunas = len(resultado_bruto[0]) if resultado_bruto else 0
        if colunas == 5:
            return formatar_faturas_unicas(resultado_bruto)
        if colunas >= 9:
            return formatar_faturas(resultado_bruto)
    if resultado_bruto and isinstance(resultado_bruto[0][0], int):
        possiveis_meses = [int(r[0]) for r in resultado_bruto]
        if all(1 <= m <= 12 for m in possiveis_meses):
            return format_markdown(formatar_meses(resultado_bruto))
    output = resposta_numerica_inteligente(sub_prompt, resultado_bruto)
    if not output:
        output = resposta_inteligente(sub_prompt, resultado_bruto)
    return format_markdown(output) if isinstance(output, str) else output

def _render_html(output: str, insight: str | None = None) -> str:
    return f"{output}<br><br>{insight}" if insight else output

@app.post("/chat", response_class=HTMLResponse)
def chat(req: ChatRequest):
    schema = fetch_schema_info()
    prompt = build_prompt(req.prompt, schema, req.companyid)

    print("📝 PROMPT GERADO:\n", prompt)
    sql = gerar_sql(prompt)
    print("🧠 SQL GERADA:\n", sql)

    if not sql.lower().startswith("select"):
        texto = f"SQL inválida gerada:<br><br><code>{sql}</code>"
        insight = generate_and_persist(req.prompt, [(texto,)])
        return _render_html(texto, insight)

    resultado_bruto = execute_sql(sql, raw=True)
    resultado_bruto = [tuple(r) for r in resultado_bruto]

    output = gerar_output_formatado(req.prompt, resultado_bruto)

    if not resultado_bruto:
        return _render_html(output)

    insight = generate_and_persist(req.prompt, resultado_bruto)
    return _render_html(output, insight)

@app.post("/chat/multi", response_class=HTMLResponse)
def chat_multi(req: ChatRequest):
    schema = fetch_schema_info()
    sub_perguntas = dividir_perguntas(req.prompt)

    respostas: List[str] = []
    insights: List[str] = []

    for sub in sub_perguntas:
        prompt_llm = build_prompt(sub, schema, req.companyid)
        sql = gerar_sql(prompt_llm)

        if not sql.lower().startswith("select"):
            respostas.append(format_markdown(f"SQL inválida para '{sub}'"))
            insights.append(f"Não foi possível gerar insight para '{sub}'.")
            continue

        resultado = execute_sql(sql, raw=True)
        resultado = [tuple(r) for r in resultado]

        output = gerar_output_formatado(sub, resultado)
        respostas.append(output)

        insight = generate_and_persist(sub, resultado)
        insights.append(insight)

    output_final = "<br><br>".join(respostas)
    insight_final = "<br><br>".join(insights)

    return _render_html(output_final, insight_final)
