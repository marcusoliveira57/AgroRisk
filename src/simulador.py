import pandas as pd
import warnings
import requests
import pmdarima as pm
from conexao import obter_conexao

# Oculta avisos de convergência estatística do terminal
warnings.filterwarnings("ignore") 

def obter_selic_api_bcb():
    """
    Conecta na API pública do Banco Central do Brasil e busca o valor mais 
    recente da Meta Selic anualizada (Série 432).
    """
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json'
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        
        dados = response.json()
        selic_atual = float(dados[0]['valor'])
        data_ref = dados[0]['data']
        
        print(f"✅ Selic atualizada via BCB: {selic_atual}% a.a. (Ref: {data_ref})")
        return selic_atual
        
    except Exception as e:
        print(f"⚠️ Erro ao conectar na API do BCB: {e}")
        print("Usando taxa padrão de contingência (10.5% a.a.)")
        return 10.5

def listar_culturas_disponiveis(engine):
    """
    Consulta o banco de dados e retorna um dicionário com os IDs e Nomes 
    das culturas cadastradas para exibir no menu.
    """
    query = "SELECT id, nome FROM cultura ORDER BY nome ASC;"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return {}
        return dict(zip(df['id'], df['nome']))
    except Exception as e:
        print(f"Erro ao buscar culturas no banco de dados: {e}")
        return {}

def calcular_metricas(id_cultura, engine):
    """
    Motor preditivo com Auto-ARIMA adaptativo e tratamento de buracos temporais.
    """
    query = f"""
        WITH CustoAnual AS (
            SELECT 
                cultura_id, 
                LEFT(safra::TEXT, 4) AS ano_safra, 
                AVG(custo_variavel_unitario) AS custo_base_ano
            FROM custo_producao
            WHERE cultura_id = {id_cultura} AND custo_variavel_unitario > 0
            GROUP BY cultura_id, LEFT(safra::TEXT, 4)
        ),
        PrecoMensal AS (
            SELECT 
                cultura_id,
                DATE_TRUNC('month', data_referencia) AS mes_referencia,
                LEFT(data_referencia::TEXT, 4) AS ano_referencia,
                AVG(preco_venda) AS preco_venda_medio
            FROM historico_preco
            WHERE cultura_id = {id_cultura}
            GROUP BY 
                cultura_id, 
                DATE_TRUNC('month', data_referencia), 
                LEFT(data_referencia::TEXT, 4)
        )
        SELECT 
            dim.nome,
            dim.tempo_safra_meses,
            pm.mes_referencia AS data,
            ((pm.preco_venda_medio - ca.custo_base_ano) / ca.custo_base_ano) AS margem
        FROM PrecoMensal pm
        INNER JOIN CustoAnual ca 
            ON pm.cultura_id = ca.cultura_id 
            AND pm.ano_referencia = ca.ano_safra
        INNER JOIN cultura dim 
            ON pm.cultura_id = dim.id
        ORDER BY pm.mes_referencia ASC;
    """
    
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Erro na extração de dados: {e}")
        return "Erro", 0, 0.0, 0.0
    
    if df.empty:
        return "Sem Dados", 0, 0.0, 0.0
        
    # Salva os metadados antes de manipular o tempo
    nome_cultura = df['nome'].iloc[0]
    meses_safra = df['tempo_safra_meses'].iloc[0]
    
    # 1. Tratamento de Buracos na Linha do Tempo (Resampling)
    df['data'] = pd.to_datetime(df['data'], utc=True) 
    df['data'] = df['data'].dt.tz_localize(None) 
    df.set_index('data', inplace=True)
    
    # Filtra só a margem para o ffill não tentar preencher os textos
    df = df[['margem']].resample('MS').ffill() 
    df.reset_index(inplace=True)

    # 2. Segurança Adaptativa
    total_meses = len(df)
    if total_meses < 6:
        # Fallback de segurança se houver menos de 6 meses no banco
        retorno_esperado = df['margem'].mean() * 100
        risco_volatilidade = df['margem'].std() * 100 if total_meses > 1 else 0.0
        return nome_cultura, meses_safra, retorno_esperado, risco_volatilidade

    usar_sazonalidade = True if total_meses >= 18 else False

    # 3. Modelagem Preditiva Automática
    serie_margem = df['margem'].values
    
    try:
        modelo = pm.auto_arima(
            serie_margem,
            seasonal=usar_sazonalidade,
            m=12 if usar_sazonalidade else 1,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
        
        # O modelo projeta exatamente a quantidade de meses do ciclo da safra
        previsao = modelo.predict(n_periods=meses_safra)
        
        retorno_esperado = previsao.mean() * 100
        risco_volatilidade = df['margem'].std() * 100 
        
    except Exception:
        # Fallback caso a matemática do ARIMA não convirja por extrema anomalia nos dados
        retorno_esperado = df['margem'].mean() * 100
        risco_volatilidade = df['margem'].std() * 100
    
    return nome_cultura, meses_safra, retorno_esperado, risco_volatilidade

def classificar_risco(volatilidade):
    if volatilidade < 10: return "Baixo"
    elif volatilidade < 20: return "Moderado"
    else: return "Alto"

def gerar_matriz_decisao(id_a, id_b, selic_aa, investimento):
    engine = obter_conexao()
    
    nome_a, meses_a, ret_a, risco_a = calcular_metricas(id_a, engine)
    lucro_a = investimento * (ret_a / 100)
    
    nome_b, meses_b, ret_b, risco_b = calcular_metricas(id_b, engine)
    lucro_b = investimento * (ret_b / 100)
    
    selic_periodo_a = selic_aa * (meses_a / 12) if meses_a else 0
    lucro_rf_a = investimento * (selic_periodo_a / 100)
    
    selic_periodo_b = selic_aa * (meses_b / 12) if meses_b else 0
    lucro_rf_b = investimento * (selic_periodo_b / 100)
    
    print("\n=========================================")
    print("🌱 AGRO-RISK TRACKER - PARECER FINAL")
    print("=========================================")
    print(f"Cenário: [1] {nome_a} vs [2] {nome_b}")
    print(f"Custo de Oportunidade (Selic): {selic_aa}% a.a.\n")
    
    print(f">> PROJEÇÃO CULTURA A ({nome_a}):")
    print(f"Tempo de Capital Travado: {meses_a} meses")
    print(f"Retorno Esperado (Safra Futura): {ret_a:.1f}%")
    print(f"Risco Contábil: {classificar_risco(risco_a)} ({risco_a:.1f}%)")
    print(f"Projeção p/ R$ {investimento:,.2f}: Lucro de R$ {lucro_a:.2f}")
    print(f"Cenário Segurança (Selic no mesmo período): R$ {lucro_rf_a:.2f}\n")
    
    print(f">> PROJEÇÃO CULTURA B ({nome_b}):")
    print(f"Tempo de Capital Travado: {meses_b} meses")
    print(f"Retorno Esperado (Safra Futura): {ret_b:.1f}%")
    print(f"Risco Contábil: {classificar_risco(risco_b)} ({risco_b:.1f}%)")
    print(f"Projeção p/ R$ {investimento:,.2f}: Lucro de R$ {lucro_b:.2f}")
    print(f"Cenário Segurança (Selic no mesmo período): R$ {lucro_rf_b:.2f}\n")
    
    print(">> CONCLUSÃO:")
    
    supera_rf_a = lucro_a > lucro_rf_a
    supera_rf_b = lucro_b > lucro_rf_b
    
    if supera_rf_a and supera_rf_b:
        print("Filtro Renda Fixa: Ambas superam a Selic em seus períodos.")
    elif supera_rf_a:
        print(f"Filtro Renda Fixa: Apenas {nome_a} supera a Selic de seu período.")
    elif supera_rf_b:
        print(f"Filtro Renda Fixa: Apenas {nome_b} supera a Selic de seu período.")
    else:
        print("Filtro Renda Fixa: NENHUMA supera a Selic. Operação agrícola não recomendada.")
        
    vencedora = nome_a if lucro_a > lucro_b else nome_b
    maior_lucro = max(lucro_a, lucro_b)
    risco_vencedora = risco_a if lucro_a > lucro_b else risco_b
    
    print(f"\nVencedor por Rentabilidade Absoluta: {vencedora} (R$ {maior_lucro:.2f} de lucro).")
    print(f"Recomendação: Se o produtor suporta risco {classificar_risco(risco_vencedora).upper()}, alocar em {vencedora}.")

def iniciar_sistema():
    engine = obter_conexao()
    culturas_cadastradas = listar_culturas_disponiveis(engine)
    
    if not culturas_cadastradas:
        print("Nenhuma cultura encontrada. Verifique sua conexão e a tabela 'cultura'.")
        return

    print("=========================================")
    print("   BEM-VINDO AO AGRO-RISK TRACKER 1.0    ")
    print("=========================================")
    print("Culturas cadastradas no banco de dados:")
    
    for cid, nome in culturas_cadastradas.items():
        print(f"[{cid}] - {nome}")
    print("-" * 41)

    def solicitar_id(ordem):
        while True:
            try:
                escolha = int(input(f"Digite o ID numérico da {ordem} cultura: "))
                if escolha in culturas_cadastradas:
                    return escolha
                else:
                    print("❌ ID não encontrado. Escolha um número válido da lista.")
            except ValueError:
                print("❌ Entrada inválida. Por favor, digite apenas números.")

    id_a = solicitar_id("PRIMEIRA")
    id_b = solicitar_id("SEGUNDA")

    if id_a == id_b:
        print("\n Aviso: Você selecionou a mesma cultura duas vezes.")

    print("\n" + "-"*41)
    selic_input = input("Taxa Selic [%] (Aperte ENTER para buscar na API do Banco Central ou digite um valor simulado): ")
    
    if selic_input.strip() == "":
        print("⏳ Conectando ao Banco Central...")
        selic_aa = obter_selic_api_bcb()
    else:
        try:
            selic_aa = float(selic_input.replace(',', '.'))
            print(f"🔧 Usando Selic simulada manualmente: {selic_aa}% a.a.")
        except ValueError:
            print("❌ Entrada inválida. Buscando na API como segurança...")
            selic_aa = obter_selic_api_bcb()

    invest_input = input("\nInvestimento Simulado [R$] (Aperte Enter para padrão 1000): ")
    try:
        investimento = float(invest_input.replace(',', '.')) if invest_input.strip() else 1000.0
    except ValueError:
        investimento = 1000.0

    print("\n⏳ Extraindo dados do PostgreSQL e treinando modelos preditivos...")
    gerar_matriz_decisao(id_a, id_b, selic_aa, investimento)

if __name__ == "__main__":
    iniciar_sistema()