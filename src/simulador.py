import numpy as np
import pandas as pd
import warnings
import requests
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from conexao import obter_conexao

warnings.filterwarnings("ignore")


def obter_selic_api_bcb():
    """Busca a Meta Selic anualizada atual na API pública do Banco Central (Série 432)."""
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
    """Retorna dicionário {id: nome} das culturas cadastradas."""
    query = "SELECT id, nome FROM cultura ORDER BY nome ASC;"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return {}
        return dict(zip(df['id'], df['nome']))
    except Exception as e:
        print(f"Erro ao buscar culturas no banco de dados: {e}")
        return {}


def _serie_estacionaria(serie):
    """Teste ADF: retorna True se a série é estacionária (p-valor <= 0.05)."""
    try:
        _, p_valor, *_ = adfuller(serie, autolag='AIC')
        return p_valor <= 0.05
    except Exception:
        return True  # assume estacionária em caso de falha


def _prever_arima(serie, meses_safra, usar_sazonalidade):
    """Auto-ARIMA com detecção automática de diferenciação (d=None)."""
    try:
        modelo = pm.auto_arima(
            serie,
            seasonal=usar_sazonalidade,
            m=12 if usar_sazonalidade else 1,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=None,       # auto-detecta integração
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
        return modelo.predict(n_periods=meses_safra)
    except Exception:
        return None


def _prever_ets(serie, meses_safra, usar_sazonalidade):
    """Holt-Winters (ETS) com suavização exponencial tripla."""
    try:
        modelo = ExponentialSmoothing(
            serie,
            trend='add',
            seasonal='add' if usar_sazonalidade else None,
            seasonal_periods=12 if usar_sazonalidade else None,
            initialization_method='estimated'
        ).fit(optimized=True, remove_bias=True)
        return np.array(modelo.forecast(meses_safra))
    except Exception:
        return None


def calcular_metricas(id_cultura, engine):
    """
    Motor preditivo com Ensemble ARIMA + Holt-Winters.
    Retorna: (nome, meses_safra, retorno_esperado%, volatilidade%, var_95%)
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
        return "Erro", 0, 0.0, 0.0, 0.0

    if df.empty:
        return "Sem Dados", 0, 0.0, 0.0, 0.0

    nome_cultura = df['nome'].iloc[0]
    meses_safra = int(df['tempo_safra_meses'].iloc[0])

    # 1. Preenchimento de lacunas com interpolação linear (neutro, sem autocorrelação artificial)
    df['data'] = pd.to_datetime(df['data'], utc=True).dt.tz_localize(None)
    df.set_index('data', inplace=True)
    df = df[['margem']].resample('MS').interpolate(method='linear')
    df.reset_index(inplace=True)

    total_meses = len(df)
    serie = df['margem'].values

    # 2. Fallback para séries muito curtas
    if total_meses < 6:
        retorno = float(np.mean(serie)) * 100
        vol = float(np.std(serie, ddof=1)) * 100 if total_meses > 1 else 0.0
        var95 = float(np.percentile(serie, 5)) * 100
        return nome_cultura, meses_safra, retorno, vol, var95

    # 3. Sazonalidade requer pelo menos 2 ciclos anuais completos
    usar_sazonalidade = total_meses >= 24

    # 4. Teste de estacionariedade ADF (informativo — d=None no ARIMA já cuida disso)
    _serie_estacionaria(serie)

    # 5. Ensemble: combina ARIMA e ETS
    previsoes = []
    prev_arima = _prever_arima(serie, meses_safra, usar_sazonalidade)
    if prev_arima is not None:
        previsoes.append(prev_arima)

    prev_ets = _prever_ets(serie, meses_safra, usar_sazonalidade)
    if prev_ets is not None:
        previsoes.append(prev_ets)

    if previsoes:
        retorno = float(np.mean(previsoes, axis=0).mean()) * 100
    else:
        retorno = float(np.mean(serie)) * 100

    # 6. Métricas de risco
    vol = float(np.std(serie, ddof=1)) * 100       # Volatilidade (desvio padrão amostral)
    var95 = float(np.percentile(serie, 5)) * 100   # VaR 95% (pior 5% dos meses históricos)

    return nome_cultura, meses_safra, retorno, vol, var95


def calcular_sharpe(retorno_pct, selic_aa, meses_safra, volatilidade):
    """
    Índice de Sharpe Agrícola: retorno excedente sobre a Selic (proporcional ao
    período de capital travado) dividido pela volatilidade.
    Permite comparar culturas de ciclos diferentes de forma justa.
    """
    if volatilidade == 0 or meses_safra == 0:
        return 0.0
    selic_periodo = selic_aa * (meses_safra / 12)
    return (retorno_pct - selic_periodo) / volatilidade


def classificar_risco(volatilidade):
    if volatilidade < 10:   return "Baixo"
    elif volatilidade < 20: return "Moderado"
    else:                   return "Alto"


def gerar_matriz_decisao(id_a, id_b, selic_aa, investimento):
    engine = obter_conexao()

    nome_a, meses_a, ret_a, risco_a, var_a = calcular_metricas(id_a, engine)
    nome_b, meses_b, ret_b, risco_b, var_b = calcular_metricas(id_b, engine)

    lucro_a = investimento * (ret_a / 100)
    lucro_b = investimento * (ret_b / 100)

    selic_periodo_a = selic_aa * (meses_a / 12) if meses_a else 0
    selic_periodo_b = selic_aa * (meses_b / 12) if meses_b else 0
    lucro_rf_a = investimento * (selic_periodo_a / 100)
    lucro_rf_b = investimento * (selic_periodo_b / 100)

    sharpe_a = calcular_sharpe(ret_a, selic_aa, meses_a, risco_a)
    sharpe_b = calcular_sharpe(ret_b, selic_aa, meses_b, risco_b)

    print("\n=========================================")
    print("🌱 AGRO-RISK TRACKER - PARECER FINAL")
    print("=========================================")
    print(f"Cenário: [1] {nome_a} vs [2] {nome_b}")
    print(f"Custo de Oportunidade (Selic): {selic_aa}% a.a.\n")

    print(f">> PROJEÇÃO CULTURA A ({nome_a}):")
    print(f"   Tempo de Capital Travado    : {meses_a} meses")
    print(f"   Retorno Esperado (Ensemble) : {ret_a:.1f}%")
    print(f"   Risco (Volatilidade)        : {classificar_risco(risco_a)} ({risco_a:.1f}%)")
    print(f"   VaR 95% (pior mês histórico): {var_a:.1f}%")
    print(f"   Índice de Sharpe Agrícola   : {sharpe_a:.2f}")
    print(f"   Lucro projetado             : R$ {lucro_a:.2f}")
    print(f"   Selic equivalente (período) : R$ {lucro_rf_a:.2f}\n")

    print(f">> PROJEÇÃO CULTURA B ({nome_b}):")
    print(f"   Tempo de Capital Travado    : {meses_b} meses")
    print(f"   Retorno Esperado (Ensemble) : {ret_b:.1f}%")
    print(f"   Risco (Volatilidade)        : {classificar_risco(risco_b)} ({risco_b:.1f}%)")
    print(f"   VaR 95% (pior mês histórico): {var_b:.1f}%")
    print(f"   Índice de Sharpe Agrícola   : {sharpe_b:.2f}")
    print(f"   Lucro projetado             : R$ {lucro_b:.2f}")
    print(f"   Selic equivalente (período) : R$ {lucro_rf_b:.2f}\n")

    print(">> CONCLUSÃO:")

    supera_rf_a = lucro_a > lucro_rf_a
    supera_rf_b = lucro_b > lucro_rf_b

    if supera_rf_a and supera_rf_b:
        print("   Filtro Renda Fixa: Ambas superam a Selic em seus períodos.")
    elif supera_rf_a:
        print(f"   Filtro Renda Fixa: Apenas {nome_a} supera a Selic de seu período.")
    elif supera_rf_b:
        print(f"   Filtro Renda Fixa: Apenas {nome_b} supera a Selic de seu período.")
    else:
        print("   Filtro Renda Fixa: NENHUMA supera a Selic. Operação agrícola não recomendada.")

    vencedora_lucro  = nome_a if lucro_a  > lucro_b  else nome_b
    vencedora_sharpe = nome_a if sharpe_a > sharpe_b else nome_b
    risco_sharpe     = risco_a if sharpe_a > sharpe_b else risco_b

    print(f"\n   Maior Lucro Absoluto        : {vencedora_lucro} (R$ {max(lucro_a, lucro_b):.2f})")
    print(f"   Melhor Risco-Retorno (Sharpe): {vencedora_sharpe} (índice: {max(sharpe_a, sharpe_b):.2f})")
    print(f"\n   Recomendação: Alocar em {vencedora_sharpe} — melhor retorno ajustado pelo risco")
    print(f"   e pelo custo de oportunidade da Selic. (Perfil: {classificar_risco(risco_sharpe).upper()})")


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
        print("\n⚠️ Aviso: Você selecionou a mesma cultura duas vezes.")

    print("\n" + "-"*41)
    selic_input = input("Taxa Selic [%] (ENTER para buscar na API do BCB ou digite um valor): ")

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

    invest_input = input("\nInvestimento Simulado [R$] (ENTER para padrão R$ 1.000): ")
    try:
        investimento = float(invest_input.replace(',', '.')) if invest_input.strip() else 1000.0
    except ValueError:
        investimento = 1000.0

    print("\n⏳ Treinando modelos preditivos (Ensemble ARIMA + Holt-Winters)...")
    gerar_matriz_decisao(id_a, id_b, selic_aa, investimento)


if __name__ == "__main__":
    iniciar_sistema()