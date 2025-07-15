import re

def format_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r"<strong>\1</strong>", text)
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return text

def formatar_faturas_unicas(linhas, mostrar_id=False):
    faturas_processadas = {}
    for linha in linhas:
        try:
            id_fatura, data, referencia, estado, valor = linha
            if id_fatura in faturas_processadas:
                continue
            faturas_processadas[id_fatura] = {
                "data": data.isoformat() if hasattr(data, "isoformat") else str(data),
                "referencia": referencia or None,
                "estado": estado,
                "valor": f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " Kz"
            }
            if mostrar_id:
                faturas_processadas[id_fatura]["id_fatura"] = id_fatura
        except Exception as e:
            faturas_processadas[id_fatura] = {
                "erro": f"Erro ao processar fatura {id_fatura}: {e}"
            }
    return list(faturas_processadas.values())

def identificar_tabela_do_prompt(prompt: str, schema: dict) -> str:
    prompt = prompt.lower()
    for tabela in schema.keys():
        if tabela.lower() in prompt:
            return tabela
    if "empresa" in prompt:
        return "company"
    if "cliente" in prompt:
        return "client"
    if "produto" in prompt:
        return "product"
    if "fatura" in prompt or "venda" in prompt:
        return "sales"
    return ""

meses_pt = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

def formatar_meses(resultado):
    try:
        numeros = [int(row[0]) for row in resultado if str(row[0]).isdigit()]
        nomes = [meses_pt[num] for num in numeros if 1 <= num <= 12]
        return "Meses com faturação:\n\n" + "\n".join(f"- {nome}" for nome in nomes)
    except Exception as e:
        return f"Erro ao formatar meses: {e}"

def formatar_faturas(resultado):
    resposta_formatada = []
    for row in resultado:
        try:
            id_fatura, data, referencia, estado, valor = row[:5]
            valor_formatado = f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " Kz" if valor else "0,00 Kz"
            resposta_formatada.append({
                "id_fatura": id_fatura,
                "data": data.isoformat() if hasattr(data, "isoformat") else str(data),
                "referencia": referencia,
                "estado": estado,
                "valor": valor_formatado
            })
        except Exception as e:
            resposta_formatada.append({"erro": f"Erro ao processar linha: {e}"})
    return resposta_formatada

def resposta_numerica_inteligente(prompt: str, resultado: list):
    if not resultado or not isinstance(resultado[0][0], (int, float)):
        return None
    valor = resultado[0][0]
    prompt = prompt.lower()
    if "fatura" in prompt and "aberto" in prompt:
        return f"Existem <strong>{valor}</strong> faturas em aberto."
    if "fatura" in prompt and "paga" in prompt:
        return f"Existem <strong>{valor}</strong> faturas pagas."
    if "fatura" in prompt:
        return f"Existem <strong>{valor}</strong> faturas no total."
    if "cliente" in prompt:
        return f"Existem <strong>{valor}</strong> clientes registrados."
    if "empresa" in prompt:
        return f"Existem <strong>{valor}</strong> empresas registradas."
    if "produto" in prompt:
        return f"Existem <strong>{valor}</strong> produtos no sistema."
    if "funcionário" in prompt or "colaborador" in prompt:
        return f"Existem <strong>{valor}</strong> funcionários cadastrados."
    return f"<strong>Resultado encontrado:</strong><br><br>{valor}<br>"

def resposta_inteligente(prompt, resultado):
    if not resultado or resultado == "Nenhum dado encontrado.":
        return "Nenhum dado foi encontrado com base na sua pergunta."

    if isinstance(resultado, list) and isinstance(resultado[0], (tuple, list)):
        colunas = len(resultado[0])
        prompt_lc = prompt.lower()

        match = re.search(r"faturado em (\w+)\s+de\s+(\d{4})", prompt_lc)
        if match and colunas == 1:
            mes_nome = match.group(1).capitalize()
            ano = match.group(2)
            try:
                valor = float(resultado[0][0])
                valor_formatado = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                return f"O faturamento em <strong>{mes_nome} de {ano}</strong> foi de <strong>{valor_formatado} Kz</strong>."
            except Exception:
                pass

        if colunas == 2 and "nome" in prompt_lc and "nif" in prompt_lc and "empresa" in prompt_lc:
            nome, nif = resultado[0]
            return f"O nome da sua empresa é <strong>{nome}</strong> e o NIF é <strong>{nif}</strong>."

        simples_map = {
            "nif": "NIF", "número de identificação fiscal": "NIF",
            "telefone": "telefone", "telemóvel": "telefone",
            "email": "e‑mail", "local": "local",
            "endereço": "endereço", "morada": "endereço"
        }
        for chave, rotulo in simples_map.items():
            if chave in prompt_lc and colunas == 1:
                valor = str(resultado[0][0])
                return f"O {rotulo} da sua empresa é <strong>{valor}</strong>."

        if any(kw in prompt_lc for kw in ["nome da empresa", "qual é a minha empresa"]) and colunas == 1:
            nome = str(resultado[0][0])
            return f"Com base nos dados, o nome da sua empresa é <strong>{nome}</strong>."

        # 2️⃣ – Receita / faturamento
        if "faturamento" in prompt_lc or "receita" in prompt_lc:
            try:
                valor = float(resultado[0][0])
                valor_formatado = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                return f"O faturamento total da empresa é de <strong>{valor_formatado} Kz</strong>."
            except Exception:
                pass

        if colunas in [4, 5]:
            linhas = []
            for row in resultado:
                try:
                    if colunas == 5:
                        _, data, referencia, estado, valor = row
                    else:
                        data, referencia, estado, valor = row
                    data_str = data.isoformat() if hasattr(data, "isoformat") else str(data)
                    valor_str = f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " Kz" if valor else "0,00 Kz"
                    linhas.append(f"- {data_str} | {referencia or '—'} | {estado} | {valor_str}")
                except Exception as e:
                    linhas.append(f"- Erro ao processar linha: {e}")
            return "**Faturas encontradas:**\n\n" + "\n".join(linhas)

        texto = "**Resultado encontrado:**\n\n"
        for row in resultado:
            linha = " | ".join(str(cell) if cell is not None else "—" for cell in row)
            texto += f"- {linha}\n"
        return texto

    return str(resultado)