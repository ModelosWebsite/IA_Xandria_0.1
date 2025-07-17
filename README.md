
# Documentação Técnica: Deploy do Projeto FastAPI “Xandria”

## Estrutura do Projeto

O projeto está localizado no diretório:

```bash
/root/IA_Xandria_0.1/
```

Estrutura básica:

```
├── main.py
├── database.py
├── llm_agent.py
├── models.py
├── utils.py
├── requirements.txt
├── promptone.txt
├── promptzero.txt
├── documentario.md
├── sqlAgent.ipynb
└── venv/
```

---

## Rodar servidor manualmente (para testes)

uvicorn main:app --reload --host 0.0.0.0 --port 8001

---

## Criar e configurar o serviço `systemd`

Conteúdo do arquivo de serviço:

```ini
[Unit]
Description=Xandria FastAPI Service
After=network.target

[Service]
User=root
WorkingDirectory=/root/IA_Xandria_0.1
ExecStart=/root/IA_Xandria_0.1/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=5
Environment=PATH=/root/IA_Xandria_0.1/venv/bin

[Install]
WantedBy=multi-user.target


## Comandos para Gerenciamento do Serviço

### Recarregar configurações do systemd

sudo systemctl daemon-reexec
sudo systemctl daemon-reload
```

### Iniciar o serviço

sudo systemctl start xandria.service

### Parar o serviço

sudo systemctl stop xandria.service

### Reiniciar o serviço (após alterações no código)

sudo systemctl restart xandria.service

### Verificar o status do serviço

sudo systemctl status xandria.service


## Acesso Externo

- A aplicação responde na porta: **8001**
- Está publicada por domínio externo apontando para o servidor
- É necessário garantir que a porta 8001 esteja aberta no firewall

---

## Comportamento Esperado

- O serviço sobe automaticamente com o sistema
- Roda com ambiente virtual próprio
- Reinicia sozinho em caso de falha

---
