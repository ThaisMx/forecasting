FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expor a porta que o Streamlit usa
EXPOSE 8501

# Comando para iniciar o aplicativo
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"] 