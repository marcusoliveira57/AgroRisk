import pandas as pd
import warnings
import pmdarima as pm
from conexao import obter_conexao

# Oculta avisos de convergência estatística
warnings.filterwarnings("ignore") 

def testar_projeção_retorno(id_cultura, meses_para_testar=6):
    """
    Executa o backtesting usando Auto-ARIMA adaptativo para descobrir o melhor modelo
    para a cultura específica, projetar os resultados e calcular o Erro (MAE).
    """
    engine = obter_conexao()
    
    # Query otimizada com agrupamento mensal
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
            pm.mes_referencia AS data,
            ((pm.preco_venda_medio - ca.custo_base_ano) / ca.custo_base_ano) * 100 AS margem_real
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
        print(f"Erro ao conectar ou executar query no banco: {e}")
        return
    
    nome_cultura = df['nome'].iloc[0] if not df.empty else "Desconhecido"
    
    # >>> VALIDAÇÃO DE SEGURANÇA ADAPTATIVA <<<
    total_meses = len(df)
    
    if total_meses < (meses_para_testar + 3):
        # Limite crítico absoluto
        print(f"❌ Erro Crítico: {nome_cultura} tem apenas {total_meses} meses de histórico.")
        print(f"Para testar {meses_para_testar} meses, você precisa de no mínimo {meses_para_testar + 3} registros no PostgreSQL.")
        return
    elif total_meses < (meses_para_testar + 12):
        # Tem dados, mas não o suficiente para ciclo anual (Sazonalidade)
        print(f"⚠️ Aviso: {nome_cultura} tem apenas {total_meses} meses de dados.")
        print("⚙️ Adaptação: Desligando Sazonalidade Anual para forçar a modelagem...")
        usar_sazonalidade = False
    else:
        # Banco de dados rico, roda o modelo completo
        usar_sazonalidade = True

    # 1. DIVISÃO DOS DADOS
    treino = df.iloc[:-meses_para_testar]
    teste = df.iloc[-meses_para_testar:].copy()
    serie_treino = treino['margem_real'].values
    
    print("==================================================")
    print(f"🔬 BACKTESTING DE RETORNO - {nome_cultura.upper()}")
    print(f"Período de Teste: Últimos {meses_para_testar} meses da base")
    print("==================================================")
    print(f"⏳ Procurando o melhor algoritmo para {nome_cultura}...")
    
    # 2. AUTO-ARIMA (Treinamento Automático Inteligente)
    try:
        modelo_auto = pm.auto_arima(
            serie_treino,
            seasonal=usar_sazonalidade,               # Dinâmico
            #m=12 if usar_sazonalidade else 1,         # Dinâmico
            start_p=0, start_q=0,         
            max_p=3, max_q=3,             
            d=None, D=None,               
            trace=False,                  
            stepwise=True,                
            suppress_warnings=True,
            error_action="ignore"
        )
        
        ordem = modelo_auto.order
        ordem_sazonal = modelo_auto.seasonal_order if usar_sazonalidade else "Desativada"
        print(f"✅ Melhor modelo encontrado: SARIMAX {ordem} x Sazonalidade: {ordem_sazonal}\n")

        # 3. PROJEÇÃO
        previsao = modelo_auto.predict(n_periods=meses_para_testar)
        teste['margem_projetada'] = previsao
        
    except Exception as e:
        print(f"❌ Erro crítico na modelagem: {e}")
        return

    # 4. CÁLCULO DE ERRO
    teste['erro_pontual'] = abs(teste['margem_real'] - teste['margem_projetada'])
    
    mae = teste['erro_pontual'].mean()
    retorno_medio_real = teste['margem_real'].mean()
    retorno_medio_projetado = teste['margem_projetada'].mean()
    
    # 5. EXIBIÇÃO
    teste['data'] = pd.to_datetime(teste['data']).dt.strftime('%Y-%m')
    teste['margem_real'] = teste['margem_real'].apply(lambda x: f"{x:.2f}%")
    teste['margem_projetada'] = teste['margem_projetada'].apply(lambda x: f"{x:.2f}%")
    teste['erro_pontual'] = teste['erro_pontual'].apply(lambda x: f"{x:.2f} p.p.")
    
    print(">> Tabela de Comparação (Mês a Mês):")
    print(teste[['data', 'margem_real', 'margem_projetada', 'erro_pontual']].to_string(index=False))
    
    print("\n>> Resumo da Performance do Modelo:")
    print(f"Média do Retorno REAL no período:      {retorno_medio_real:.2f}%")
    print(f"Média do Retorno PROJETADO no período: {retorno_medio_projetado:.2f}%")
    print("-" * 50)
    print(f"Métrica de Erro (MAE): {mae:.2f} pontos percentuais")
    print("==================================================")
    
    margem_aceitavel = abs(retorno_medio_real * 0.25)
    
    if mae <= margem_aceitavel:
        print("✅ Status do Modelo: APROVADO.")
        print("A inteligência artificial encontrou um padrão confiável para projeção.")
    else:
        print("⚠️ Status do Modelo: REQUER AJUSTES OU MAIS DADOS.")
        print("A volatilidade da série é muito alta para o volume de dados disponível.")

if __name__ == "__main__":
    testar_projeção_retorno(id_cultura=5, meses_para_testar=12)