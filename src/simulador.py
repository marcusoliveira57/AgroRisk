import numpy as np
import pandas as pd
import warnings
import requests
import pmdarima as pm
import scipy.stats as st
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

    df['data'] = pd.to_datetime(df['data'], utc=True).dt.tz_localize(None)
    df.set_index('data', inplace=True)
    df = df[['margem']].resample('MS').interpolate(method='linear')
    df.reset_index(inplace=True)

    total_meses = len(df)
    serie = df['margem'].values

    if total_meses < 6:
        retorno = float(np.mean(serie)) * 100
        vol = float(np.std(serie, ddof=1)) * 100 if total_meses > 1 else 0.0
        var95 = float(np.percentile(serie, 5)) * 100
        return nome_cultura, meses_safra, retorno, vol, var95

    usar_sazonalidade = total_meses >= 24
    _serie_estacionaria(serie)

    # 5. Ensemble: combina ARIMA e ETS
    previsoes = []
    prev_arima = _prever_arima(serie, meses_safra, usar_sazonalidade)
    if prev_arima is not None:
        # Garante que remove NaNs se o ARIMA cuspir algum por falta de convergência
        prev_arima = prev_arima[~np.isnan(prev_arima)]
        if len(prev_arima) > 0:
            previsoes.append(prev_arima)

    prev_ets = _prever_ets(serie, meses_safra, usar_sazonalidade)
    if prev_ets is not None:
        prev_ets = prev_ets[~np.isnan(prev_ets)]
        if len(prev_ets) > 0:
            previsoes.append(prev_ets)

    # CORREÇÃO DA CONTAMINAÇÃO: Usamos np.nanmean para ignorar qualquer NaN residual
    if previsoes:
        # Achamos a média das projeções futuras
        bloco_previsoes = np.concatenate(previsoes)
        retorno = float(np.nanmean(bloco_previsoes)) * 100
    else:
        retorno = float(np.nanmean(serie)) * 100

    # 6. Métricas de risco usando funções seguras contra NaN (nanmean, nanstd)
    vol = float(np.nanstd(serie, ddof=1)) * 100  # Ignora NaNs no cálculo da volatilidade
    
    # Se a série estiver limpa, calcula o percentil, senão usa um fallback seguro
    serie_limpa = serie[~np.isnan(serie)]
    if len(serie_limpa) > 0:
        var95 = float(np.percentile(serie_limpa, 5)) * 100
    else:
        var95 = 0.0

    return nome_cultura, meses_safra, retorno, vol, var95


# --- NOVAS FUNÇÕES ADICIONADAS ---

def calcular_wacc(selic_aa, perc_financiado, juros_banco_aa):
    """Estrutura de Capital: Calcula o Custo Médio Ponderado de Capital (WACC)."""
    peso_banco = perc_financiado / 100.0
    peso_proprio = 1.0 - peso_banco
    wacc = (peso_proprio * selic_aa) + (peso_banco * juros_banco_aa)
    return wacc


def calcular_hedge_protecao(volatilidade_pct, confianca=0.95):
    """Operações Financeiras: Calcula a amplitude ideal para proteção via Hedge."""
    if volatilidade_pct == 0:
        return 0.0
    z_score = st.norm.ppf(confianca)  # Busca o multiplicador da curva normal (ex: 95% = 1.645)
    amplitude = z_score * volatilidade_pct

    if amplitude > 100.0:
        return 99.9 # Trava no limite teórico
    
    return amplitude


def classificar_risco(volatilidade):
    if volatilidade < 10:   return "Baixo"
    elif volatilidade < 20: return "Moderado"
    else:                   return "Alto"


def gerar_matriz_decisao(id_a, id_b, selic_aa, investimento, perc_financiado, juros_banco):
    engine = obter_conexao()

    # 1. Estrutura de Capital (WACC)
    wacc_aa = calcular_wacc(selic_aa, perc_financiado, juros_banco)
    
    print("\n==================================================")
    print("🏢 ESTRUTURA DE CAPITAL (WACC)")
    print("==================================================")
    print(f"Capital Próprio : {100 - perc_financiado:.1f}% (Custo: Selic {selic_aa}% a.a.)")
    print(f"Capital Banco   : {perc_financiado:.1f}% (Custo: Financiamento {juros_banco}% a.a.)")
    print(f"Custo Médio Ponderado de Capital (WACC): {wacc_aa:.2f}% a.a.")
    print("-> Esta taxa define o retorno mínimo aceitável para mitigar custos de capital.\n")

    nome_a, meses_a, ret_a, risco_a, var_a = calcular_metricas(id_a, engine)
    nome_b, meses_b, ret_b, risco_b, var_b = calcular_metricas(id_b, engine)

   # 2. Gestão de Investimentos (Cálculo do Portfólio 50/50)
    ret_portfolio = (ret_a * 0.5) + (ret_b * 0.5)
    risco_portfolio = ((risco_a**2 * 0.25) + (risco_b**2 * 0.25)) ** 0.5
    meses_portfolio = (meses_a + meses_b) / 2

    lucro_a = investimento * (ret_a / 100)
    lucro_b = investimento * (ret_b / 100)
    lucro_port = investimento * (ret_portfolio / 100)

    # Custos de oportunidade ajustados ao tempo de capital travado pelo WACC
    custo_wacc_a = investimento * ((wacc_aa * (meses_a / 12)) / 100) if meses_a else 0
    custo_wacc_b = investimento * ((wacc_aa * (meses_b / 12)) / 100) if meses_b else 0
    custo_wacc_port = investimento * ((wacc_aa * (meses_portfolio / 12)) / 100) if meses_portfolio else 0

    # --- NOVO: Retorno se investido 100% na Selic (Renda Fixa) ---
    lucro_selic_a = investimento * ((selic_aa * (meses_a / 12)) / 100) if meses_a else 0
    lucro_selic_b = investimento * ((selic_aa * (meses_b / 12)) / 100) if meses_b else 0
    lucro_selic_port = investimento * ((selic_aa * (meses_portfolio / 12)) / 100) if meses_portfolio else 0

    print("==================================================")
    print("🌱 GESTÃO DE INVESTIMENTOS (OPÇÕES DE ALOCAÇÃO)")
    print("==================================================")
    print(f">> ALOCAÇÃO 1: 100% {nome_a}")
    print(f"   Tempo de Capital Travado: {meses_a} meses | Retorno Projetado: {ret_a:.1f}%")
    print(f"   Risco (Volatilidade): {classificar_risco(risco_a)} ({risco_a:.1f}%) | VaR 95%: {var_a:.1f}%")
    print(f"   Lucro da Safra: R$ {lucro_a:,.2f} | Custo do Dinheiro (WACC): R$ {custo_wacc_a:,.2f}")
    print(f"   Comparativo -> Renda Fixa (Selic): R$ {lucro_selic_a:,.2f}\n")

    print(f">> ALOCAÇÃO 2: 100% {nome_b}")
    print(f"   Tempo de Capital Travado: {meses_b} meses | Retorno Projetado: {ret_b:.1f}%")
    print(f"   Risco (Volatilidade): {classificar_risco(risco_b)} ({risco_b:.1f}%) | VaR 95%: {var_b:.1f}%")
    print(f"   Lucro da Safra: R$ {lucro_b:,.2f} | Custo do Dinheiro (WACC): R$ {custo_wacc_b:,.2f}")
    print(f"   Comparativo -> Renda Fixa (Selic): R$ {lucro_selic_b:,.2f}\n")

    print(f">> ALOCAÇÃO 3: FAZENDA DIVERSIFICADA (50% {nome_a} / 50% {nome_b})")
    print(f"   Tempo de Capital Travado: {meses_portfolio:.1f} meses | Retorno Projetado: {ret_portfolio:.1f}%")
    print(f"   Risco Combinado: {classificar_risco(risco_portfolio)} ({risco_portfolio:.1f}%)")
    print(f"   Lucro da Safra: R$ {lucro_port:,.2f} | Custo do Dinheiro (WACC): R$ {custo_wacc_port:,.2f}")
    print(f"   Comparativo -> Renda Fixa (Selic): R$ {lucro_selic_port:,.2f}\n")

    # 3. Operações Financeiras (Hedge)
    print("==================================================")
    print("🛡️ OPERAÇÕES FINANCEIRAS (ESTRATÉGIA DE HEDGE)")
    print("==================================================")
    amp_a = calcular_hedge_protecao(risco_a)
    amp_b = calcular_hedge_protecao(risco_b)

    print("Recomendação para travar o preço mínimo de venda na Bolsa (B3):")
    print(f"- Para {nome_a}: Adquirir Puts (Opções de Venda) com proteção máxima contra quedas além de {amp_a:.1f}%.")
    print(f"- Para {nome_b}: Adquirir Puts (Opções de Venda) com proteção máxima contra quedas além de {amp_b:.1f}%.")
    print("Objetivo: Blindar o caixa do produtor contra oscilações que firam o custo operacional.")
    print("==================================================\n")


def iniciar_sistema():
    engine = obter_conexao()
    culturas_cadastradas = listar_culturas_disponiveis(engine)

    if not culturas_cadastradas:
        print("Nenhuma cultura encontrada. Verifique sua conexão e a tabela 'cultura'.")
        return

    print("=========================================")
    print("   BEM-VINDO AO AGRO-RISK TRACKER  ")
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

    # Coletores para Estrutura de Capital
    print("\n-----------------------------------------")
    print("🏦 PARAMETRIZAÇÃO DE CRÉDITO")
    print("-----------------------------------------")
    financiamento_input = input("Quantos % desse investimento será financiado pelo Banco? (Ex: 70): ")
    try:
        perc_financiado = float(financiamento_input.replace(',', '.')) if financiamento_input.strip() else 0.0
    except ValueError:
        perc_financiado = 0.0

    juros_banco = 0.0
    if perc_financiado > 0:
        juros_input = input("Qual a taxa de juros anual desse financiamento? [% a.a.] (Ex: 8.5): ")
        try:
            juros_banco = float(juros_input.replace(',', '.')) if juros_input.strip() else 8.5
        except ValueError:
            juros_banco = 8.5

    print("\n⏳ Computando dados institucionais e estruturando estratégias...")
    gerar_matriz_decisao(id_a, id_b, selic_aa, investimento, perc_financiado, juros_banco)


if __name__ == "__main__":
    iniciar_sistema()