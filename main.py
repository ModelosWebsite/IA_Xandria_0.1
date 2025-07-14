from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import fetch_schema_info, execute_sql
from llm_agent import build_prompt, gerar_sql

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Entrada da API
class ChatRequest(BaseModel):
    prompt: str
    companyid: int

class ChatResponse(BaseModel):
    output: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    schema = fetch_schema_info()
    prompt = build_prompt(req.prompt, schema, req.companyid)
    sql = gerar_sql(prompt)

    if not sql.lower().startswith("select"):
        return {"output": f"SQL inválida gerada:\n\n<pre>{sql}</pre>"}

    resultado = execute_sql(sql)
    return {"output": resultado}
