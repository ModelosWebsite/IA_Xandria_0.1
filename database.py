# database.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine: Engine = create_engine(DATABASE_URL)

def fetch_schema_info():
    """
    Busca informações do schema (tabelas e colunas) do banco de dados atual.
    """
    query = """
    SELECT TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'fortcod1_db_erp_full'
    ORDER BY TABLE_NAME, ORDINAL_POSITION;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
    schema = {}
    for table, column in result:
        schema.setdefault(table, []).append(column)
    return schema

def execute_sql(query: str, raw: bool = False):
    """
    Executa a query SQL fornecida e retorna os dados formatados ou crus (raw=True).
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()

            if not rows:
                return "ℹ️ Nenhum dado encontrado." if not raw else []

            if raw:
                return rows

            colunas = result.keys()
            linhas_formatadas = []
            for row in rows:
                linha = " | ".join(str(cel) for cel in row)
                linhas_formatadas.append(f"• {linha}")

            texto_resultado = "\n".join(linhas_formatadas)
            return f"Resultado encontrado:\n\n{texto_resultado}"
    except Exception as e:
        return f"❌ Erro ao executar SQL: {e}"
