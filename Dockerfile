FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema necessárias para o psycopg2 e pmdarima
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte e os dados
COPY src/ ./src/
COPY dados/ ./dados/

# Define o diretório de trabalho como src/ para os imports relativos funcionarem
WORKDIR /app/src
