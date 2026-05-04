import os
import pandas as pd
import datetime
from conexao import obter_conexao

MAPA_CULTURAS = {
    'Algodao_MT': 1, 'Arroz_RS': 2, 'Banana_MG': 3,
    'Cafe_MG': 4, 'Cebola_SC': 5, 'Feijao_SP': 6,
    'Laranja_SP': 7, 'Maca_SC': 8, 'Milho_RS': 9,
    'Soja_PR': 10, 'Trigo_PR': 11
}



CAMINHO_DADOS = os.path.join(os.path.dirname(__file__), '..', 'dados')

# Dicionário para traduzir o mês da CONAB para número
MESES_CONAB = {
    'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04',
    'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
    'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
}


def limpar_valor_cepea(valor):
    if pd.isna(valor): 
        return 0.0
    
    if isinstance(valor, str):
        valor = valor.replace('.', '').replace(',', '.').strip()
        
    try: 
        return round(float(valor), 2)
    except: 
        return 0.0

def formatar_data_conab(data_str):
    """Converte 'MAR-2019' para '2019-03-01' (Padrão PostgreSQL)"""
    try:
        mes_sigla, ano = str(data_str).split('-')
        mes_num = MESES_CONAB.get(mes_sigla.upper(), '01')
        return f"{ano}-{mes_num}-01"
    except: return None

def formatar_data_cepea(valor_data):
    """Blindado nível máximo: aceita Timestamp, barras (/) e traços (-)"""
    if pd.isna(valor_data): 
        return None
    
    # 1. Se o Pandas já entendeu que é data (Timestamp nativo)
    if isinstance(valor_data, (pd.Timestamp, datetime.date, datetime.datetime)):
        return valor_data.strftime("%Y-%m-01")
        
    try:
        texto = str(valor_data).strip()
        
        # 2. Se a data veio no padrão de banco/americano (Ex: "2024-09-01" ou "2024-09-01 00:00:00")
        if '-' in texto:
            # Pega só a parte da data e ignora as horas
            parte_data = texto.split(' ')[0] 
            # Verifica se está no formato YYYY-MM-DD
            if len(parte_data.split('-')) == 3:
                ano, mes, dia = parte_data.split('-')
                return f"{ano}-{mes.zfill(2)}-01"
                
        # 3. Se a data veio no padrão brasileiro (Ex: "09/2024" ou "09/09/2024")
        if '/' in texto:
            partes = texto.split('/')
            if len(partes) == 2:
                mes, ano = partes
                return f"{ano.strip()}-{mes.strip().zfill(2)}-01"
            elif len(partes) == 3:
                dia, mes, ano = partes
                return f"{ano.strip()}-{mes.strip().zfill(2)}-01"
                
    except Exception as e: 
        print(f"Erro ao converter data {valor_data}: {e}")
        
    return None

def rodar_pipeline():
    engine = obter_conexao()
    
    for nome_pasta, id_cultura in MAPA_CULTURAS.items():
        caminho_pasta = os.path.join(CAMINHO_DADOS, nome_pasta)
        estado = nome_pasta.split('_')[1]

        # ---------------------------------------------------------
        # 1. PROCESSANDO CUSTOS (CONAB) - CSV
        # ---------------------------------------------------------
        caminho_custo = os.path.join(caminho_pasta, 'custo.csv')
        if os.path.exists(caminho_custo):
            # Lendo o CSV separando por ponto e vírgula
            df_custo = pd.read_csv(caminho_custo, sep=';')
            
            # Como a coluna de valor pode mudar de estado pra estado, pegamos pela posição (índice 1)
            coluna_data = df_custo.columns[0] # "Ano-Mes.Ano-Mes"
            coluna_valor = df_custo.columns[1] # Ex: "PR/Valor"
            
            df_custo_limpo = pd.DataFrame()
            df_custo_limpo['cultura_id'] = [id_cultura] * len(df_custo)
            df_custo_limpo['estado'] = estado
            df_custo_limpo['custo_variavel_unitario'] = df_custo[coluna_valor].astype(float)
            df_custo_limpo['safra'] = df_custo[coluna_data].apply(formatar_data_conab)
            
            # Inserindo no PostgreSQL
            df_custo_limpo.to_sql('custo_producao', engine, if_exists='append', index=False)
            print(f"Custos de {nome_pasta} inseridos: {len(df_custo_limpo)} linhas.")
            
        # ---------------------------------------------------------
        # 2. PROCESSANDO PREÇOS (CEPEA) - EXCEL
        # ---------------------------------------------------------
        # Procure por .xlsx ou .xls
        caminho_preco = os.path.join(caminho_pasta, 'preco.xlsx')
        if not os.path.exists(caminho_preco):
            caminho_preco = os.path.join(caminho_pasta, 'preco.xls')
            
        if os.path.exists(caminho_preco):
            # O pulo do gato: skiprows=3 ignora o cabeçalho sujo da planilha
            df_preco = pd.read_excel(caminho_preco, skiprows=3)

            # Pegando as colunas exatas do CEPEA
            coluna_data = df_preco.columns[0]
            coluna_valor = df_preco.columns[1]
            
            # Remove linhas vazias se houver
            df_preco = df_preco.dropna(subset=[coluna_data, coluna_valor])
            
            df_preco_limpo = pd.DataFrame()
            df_preco_limpo['cultura_id'] = [id_cultura] * len(df_preco)
            df_preco_limpo['data_referencia'] = df_preco[coluna_data].apply(formatar_data_cepea)
            df_preco_limpo['preco_venda'] = df_preco[coluna_valor].apply(limpar_valor_cepea)
            
            # Inserindo no PostgreSQL
            df_preco_limpo.to_sql('historico_preco', engine, if_exists='append', index=False)
            print(f"Preços de {nome_pasta} inseridos: {len(df_preco_limpo)} linhas.")
            
if __name__ == "__main__":
    print("Iniciando injeção de dados no Banco AgroRisk...")
    rodar_pipeline()