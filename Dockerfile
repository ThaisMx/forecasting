FROM python:3.9-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar arquivos importantes explicitamente primeiro
COPY credentials.json .

# Configurar variáveis de ambiente a partir dos argumentos do build
ARG TIMEGPT_API_KEY
ARG GOOGLE_SHEET_ID
ENV TIMEGPT_API_KEY=${TIMEGPT_API_KEY}
ENV GOOGLE_SHEET_ID=${GOOGLE_SHEET_ID}

# Depois copiar os arquivos Python
COPY *.py .

# Garantir permissões corretas para o arquivo de credenciais
RUN chmod 644 credentials.json

# Aumentar limite de tempo para inicialização
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Configurar porta
EXPOSE 8501

# Comando para iniciar o Streamlit com timeout maior
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.timeout=60"]