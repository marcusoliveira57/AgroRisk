import pandas as pd
import numpy as np
import math
from conexao import obter_conexao

def auditar_calculo_de_risco(id_cultura):
    """
    Extrai os dados do banco e faz uma auditoria passo a passo do cálculo de 
    volatilidade (Desvio Padrão) para garantir que a métrica de risco é precisa.
    """
    engine = obter_conexao()
    
    # Mesma query otimizada do sistema principal
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
            ((pm.preco_venda_medio - ca.custo_base_ano) / ca.custo_base_ano) * 100 AS margem
        FROM PrecoMensal pm
        INNER JOIN CustoAnual ca 
            ON pm.cultura_id = ca.cultura_id 
            AND pm.ano_referencia = ca.ano_safra
        INNER JOIN cultura dim 
            ON pm.cultura_id = dim.id
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("Erro: Nenhum dado encontrado para esta cultura.")
        return
        
    nome_cultura = df['nome'].iloc[0]
    n_amostras = len(df)
    
    print("==================================================")
    print(f"📊 AUDITORIA DE RISCO (VOLATILIDADE) - {nome_cultura.upper()}")
    print(f"Total de meses analisados (N): {n_amostras}")
    print("==================================================")
    
    # ---------------------------------------------------------
    # MÉTODO 1: O CÁLCULO DIRETO DO PANDAS (O que está no seu sistema)
    # ---------------------------------------------------------
    risco_sistema = df['margem'].std()
    
    # ---------------------------------------------------------
    # MÉTODO 2: A PROVA REAL MATEMÁTICA (Passo a Passo)
    # ---------------------------------------------------------
    # Passo 1: Calcular a Média (Retorno Esperado Histórico)
    media_margem = df['margem'].sum() / n_amostras
    
    # Passo 2: Calcular os desvios quadrados (o quanto cada mês fugiu da média)
    # (Margem do Mês - Média)²
    soma_desvios_quadrados = sum((x - media_margem) ** 2 for x in df['margem'])
    
    # Passo 3: Calcular a Variância Amostral
    # Em finanças, sempre dividimos por (N - 1), chamado de Correção de Bessel.
    # Se dividirmos apenas por N, estaríamos subestimando o risco.
    variancia = soma_desvios_quadrados / (n_amostras - 1)
    
    # Passo 4: O Desvio Padrão (Raiz Quadrada da Variância)
    risco_matematico = math.sqrt(variancia)
    
    # ---------------------------------------------------------
    # EXIBIÇÃO DA AUDITORIA
    # ---------------------------------------------------------
    print("\n>> PASSO A PASSO MATEMÁTICO:")
    print(f"1. Retorno Médio Histórico: {media_margem:.4f}%")
    print(f"2. Soma dos Desvios ao Quadrado: {soma_desvios_quadrados:.4f}")
    print(f"3. Variância Amostral (N-1): {variancia:.4f}")
    print(f"4. Risco Calculado na Mão (√Variância): {risco_matematico:.4f}%")
    
    print("\n>> COMPARAÇÃO FINAL:")
    print(f"Risco gerado pelo Pandas (AgroRisk): {risco_sistema:.4f}%")
    print(f"Risco da Auditoria Manual:           {risco_matematico:.4f}%")
    
    print("-" * 50)
    # Tolerância minúscula para questões de arredondamento de ponto flutuante do processador
    if abs(risco_sistema - risco_matematico) < 0.0001:
        print("✅ STATUS: AUDITORIA PASSOU COM SUCESSO.")
        print("O seu sistema está calculando o risco financeiro perfeitamente, usando o Desvio Padrão Amostral (N-1).")
    else:
        print("⚠️ STATUS: FALHA NA AUDITORIA.")
        print("Há divergência nos cálculos estatísticos.")

if __name__ == "__main__":
    auditar_calculo_de_risco(id_cultura=4)