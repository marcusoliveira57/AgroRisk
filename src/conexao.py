import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

#Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def obter_conexao():
    #Cria e retorna a engine de conexão com o PostgreSQL.
    usuario = os.getenv("DB_USER")
    senha = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    porta = os.getenv("DB_PORT")
    banco = os.getenv("DB_NAME")
    
    string_conexao = f"postgresql+psycopg2://{usuario}:{senha}@{host}:{porta}/{banco}"
    engine = create_engine(string_conexao)
    
    return engine

if __name__ == "__main__":
    # Teste rápido de conexão
    engine = obter_conexao()
    try:
        with engine.connect() as conn:
            print("✔ Conexão com o banco AgroRisk estabelecida com sucesso!")
    except Exception as e:
        print(f" Erro ao conectar: {e}")