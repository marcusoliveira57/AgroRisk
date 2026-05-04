import numpy as np
import pandas as pd
import warnings
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from conexao import obter_conexao

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Funções auxiliares de erro
# ─────────────────────────────────────────────

def _mae(real, prev):
    return float(np.mean(np.abs(real - prev)))

def _mape(real, prev):
    """MAPE — ignora valores reais próximos de zero para evitar divisão instável."""
    mascara = np.abs(real) > 0.01
    if not np.any(mascara):
        return float('nan')
    return float(np.mean(np.abs((real[mascara] - prev[mascara]) / real[mascara])) * 100)


# ─────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────

def _prever_arima(serie_treino, n_periodos, usar_sazonalidade):
    try:
        modelo = pm.auto_arima(
            serie_treino,
            seasonal=usar_sazonalidade,
            m=12 if usar_sazonalidade else 1,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=None,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
        return modelo.predict(n_periods=n_periodos), str(modelo.order)
    except Exception:
        return None, "—"

def _prever_ets(serie_treino, n_periodos, usar_sazonalidade):
    """
    Holt-Winters com fallback progressivo:
    1) trend + seasonal  →  2) trend only  →  3) simple (só nível)
    """
    configs = []
    if usar_sazonalidade:
        configs.append(('add', 'add', 12))    # tripla
    configs.append(('add', None, None))        # dupla (tendência)
    configs.append((None, None, None))         # simples (nível)

    for trend, seasonal, sp in configs:
        try:
            modelo = ExponentialSmoothing(
                serie_treino,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=sp,
                initialization_method='estimated'
            ).fit(optimized=True, remove_bias=True)
            return np.array(modelo.forecast(n_periodos))
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────
# Walk-Forward Cross-Validation
# ─────────────────────────────────────────────

def walk_forward_cv(serie, n_folds, horizon, usar_sazonalidade):
    """
    Desliza a janela de treino progressivamente e avalia cada modelo em
    múltiplos folds. Retorna dicionário com listas de resultados por modelo.
    """
    n = len(serie)
    # Tamanho mínimo do primeiro conjunto de treino
    min_treino = n - n_folds * horizon

    resultados = {'arima': [], 'ets': [], 'ensemble': []}

    for fold in range(n_folds):
        fim_treino = min_treino + fold * horizon
        fim_teste  = fim_treino + horizon

        if fim_teste > n:
            break

        treino = serie[:fim_treino]
        teste  = serie[fim_treino:fim_teste]

        prev_arima, ordem_arima = _prever_arima(treino, horizon, usar_sazonalidade)
        prev_ets                = _prever_ets(treino, horizon, usar_sazonalidade)

        # Ensemble: média dos modelos disponíveis
        disponiveis = [p for p in [prev_arima, prev_ets] if p is not None]
        prev_ensemble = np.mean(disponiveis, axis=0) if disponiveis else None

        for chave, prev in [('arima', prev_arima), ('ets', prev_ets), ('ensemble', prev_ensemble)]:
            if prev is not None:
                resultados[chave].append({
                    'fold':        fold + 1,
                    'mae':         _mae(teste, prev),
                    'mape':        _mape(teste, prev),
                    'real_medio':  float(teste.mean() * 100),
                    'prev_medio':  float(prev.mean() * 100),
                    'ordem_arima': ordem_arima if chave == 'arima' else '—',
                    'status':      'OK'
                })
            else:
                resultados[chave].append({
                    'fold':   fold + 1,
                    'status': 'FALHOU',
                    'motivo': f'treino com {len(treino)} obs. insuficientes'
                })

    return resultados


# ─────────────────────────────────────────────
# Função principal
# ─────────────────────────────────────────────

def testar_projeção_retorno(id_cultura, n_folds=3, horizon=6):
    """
    Walk-forward cross-validation comparando Auto-ARIMA, Holt-Winters (ETS) e
    o Ensemble de ambos. Reporta MAE e MAPE por fold e ranking final.
    """
    engine = obter_conexao()

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
        print(f"Erro ao conectar ou executar query: {e}")
        return

    if df.empty:
        print("Nenhum dado encontrado para esta cultura.")
        return

    nome_cultura = df['nome'].iloc[0]
    total_meses  = len(df)

    # ── Auto-adaptação de parâmetros ──
    # Garante pelo menos MIN_TREINO meses para o primeiro fold
    MIN_TREINO = max(8, total_meses // 3)
    max_horizon_possivel = (total_meses - MIN_TREINO) // n_folds

    if max_horizon_possivel < 2:
        # Série muito curta até para 1 fold — reduz para 1 fold
        n_folds = 1
        max_horizon_possivel = (total_meses - MIN_TREINO) // n_folds

    if horizon > max_horizon_possivel:
        horizon_ajustado = max_horizon_possivel
        aviso_ajuste = f"⚙️  Horizonte ajustado de {horizon} → {horizon_ajustado} meses (dados insuficientes para o horizonte solicitado)."
        horizon = horizon_ajustado
    else:
        aviso_ajuste = None

    print("==================================================")
    print(f"🔬 BACKTESTING WALK-FORWARD — {nome_cultura.upper()}")
    print(f"   Total de observações : {total_meses} meses")
    print(f"   Folds                : {n_folds}")
    print(f"   Horizonte por fold   : {horizon} meses")
    if aviso_ajuste:
        print(f"   {aviso_ajuste}")
    print("==================================================")

    usar_sazonalidade = total_meses >= 24
    if not usar_sazonalidade:
        print("⚙️  Sazonalidade desligada (< 24 meses de histórico).\n")

    serie = df['margem_real'].values

    print(f"\n⏳ Executando {n_folds} folds...\n")
    resultados = walk_forward_cv(serie, n_folds, horizon, usar_sazonalidade)

    # ── Detalhe por modelo ──
    modelos = {
        'arima':    'Auto-ARIMA',
        'ets':      'Holt-Winters (ETS)',
        'ensemble': 'Ensemble (média)',
    }

    for chave, nome_modelo in modelos.items():
        folds = resultados[chave]
        if not folds:
            continue

        print(f"  ── {nome_modelo} ──")
        print(f"  {'Fold':<5} {'Real médio':>11} {'Previsto':>11} {'MAE':>9} {'MAPE':>9}")
        print(f"  {'─'*49}")

        folds_ok = []
        for f in folds:
            if f['status'] == 'FALHOU':
                print(f"  {f['fold']:<5} {'—':>11} {'—':>11} {'FALHOU':>9}   ({f['motivo']})")
            else:
                folds_ok.append(f)
                mape_str = f"{f['mape']:.1f}%" if not np.isnan(f['mape']) else "  N/A"
                print(f"  {f['fold']:<5} {f['real_medio']:>9.2f}% {f['prev_medio']:>9.2f}% "
                      f"{f['mae']:>7.2f}pp {mape_str:>9}")

        if folds_ok:
            mae_medio = float(np.mean([f['mae'] for f in folds_ok]))
            mape_vals = [f['mape'] for f in folds_ok if not np.isnan(f['mape'])]
            mape_str  = f"{np.mean(mape_vals):.1f}%" if mape_vals else "N/A"
            print(f"  {'MÉDIA':<5} {'':>11} {'':>11} {mae_medio:>7.2f}pp {mape_str:>9}")
        print()

    # ── Ranking final ──
    print(">> RANKING FINAL (menor MAE médio = melhor)")
    print("─" * 50)

    ranking = []
    for chave, nome_modelo in modelos.items():
        folds_ok = [f for f in resultados[chave] if f.get('status') == 'OK']
        if folds_ok:
            mae_medio = float(np.mean([f['mae'] for f in folds_ok]))
            ranking.append((mae_medio, nome_modelo, chave))

    ranking.sort()
    medalhas = ["🥇", "🥈", "🥉"]
    for i, (mae, nome, _) in enumerate(ranking):
        print(f"  {medalhas[i]} {nome:<25} MAE médio: {mae:.2f} p.p.")

    print("==================================================")
    if ranking:
        print(f"\n✅ Modelo mais preciso para {nome_cultura}: {ranking[0][1]}")
        print("   O Ensemble é geralmente mais estável, mas ARIMA ou ETS pode vencer")
        print("   quando a série tem um padrão dominante claro (tendência ou sazonalidade).")


if __name__ == "__main__":
    testar_projeção_retorno(id_cultura=5, n_folds=3, horizon=12)