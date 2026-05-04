# 🌱 AgroRisk Tracker

> **Sistema de Análise e Simulação de Risco Agrícola**  
> Projeto acadêmico — Administração Financeira · UFMG

AgroRisk é uma ferramenta de suporte à tomada de decisão para produtores rurais e investidores agrícolas. A partir de dados históricos de **preços de mercado (CEPEA)** e **custos de produção (CONAB)**, o sistema calcula a margem de lucratividade de diferentes culturas, projeta retornos futuros com **Ensemble (Auto-ARIMA + Holt-Winters)** e compara o risco de cada investimento em relação à taxa **Selic**.

## 📐 Funcionalidades

| Módulo               | Descrição                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `etl_cepea_conab.py` | Pipeline ETL que lê planilhas CEPEA (`.xlsx`) e arquivos CONAB (`.csv`) e os ingere no PostgreSQL |
| `simulador.py`       | Simulador interativo com Ensemble ARIMA + Holt-Winters, VaR 95% e Índice de Sharpe Agrícola       |
| `validar_risco.py`   | Auditoria matemática passo a passo do cálculo de volatilidade (Desvio Padrão Amostral)            |
| `testar_modelo.py`   | Walk-Forward Cross-Validation comparando ARIMA, ETS e Ensemble (MAE + MAPE)                       |

### Culturas cadastradas

Algodão (MT) · Arroz (RS) · Banana (MG) · Café (MG) · Cebola (SC) · Feijão (SP) · Laranja (SP) · Maçã (SC) · Milho (RS) · Soja (PR) · Trigo (PR)

## 🚀 Setup com Docker (recomendado)

<details>
<summary>Expandir instruções de setup otimizado com Docker</summary>

**Pré-requisitos:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e em execução

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/AgroRisk.git
cd AgroRisk

# 2. Configure o .env com suas credenciais locais
cp .env.example .env # Pré-configurado para Docker, não é necessário alterar nada

# 3. Suba o banco de dados e construa a imagem da aplicação
docker compose up db -d
docker compose build app

# 4. Execute o ETL para popular o banco com dados históricos
docker compose run --rm app python etl_cepea_conab.py
```

## </details>

## 🔧 Setup Manual (tradicional, sem Docker)

<details>
<summary>Expandir instruções de setup sem Docker</summary>

**Pré-requisitos:** Python 3.10+ e PostgreSQL 14+

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/AgroRisk.git
cd AgroRisk

# 2. Ambiente virtual
python -m venv .venv && source .venv/bin/activate

# 3. Dependências
pip install -r requirements.txt

# 4. Configure o .env com suas credenciais locais
cp .env.example .env   # edite DB_HOST=localhost, DB_USER e DB_PASSWORD

# 5. Crie o banco e as tabelas
psql -U seu_usuario -c "CREATE DATABASE agrorisk;"
psql -U seu_usuario -d agrorisk -f sql/01_create_tables.sql
psql -U seu_usuario -d agrorisk -f sql/02_insert_culturas.sql
psql -U seu_usuario -d agrorisk -f sql/03_migration_tempo_safra.sql

# 6. Execute o ETL para popular o banco com dados históricos
cd src
python etl_cepea_conab.py
```

## 📦 Dependências

| Pacote            | Versão | Uso                                                     |
| ----------------- | ------ | ------------------------------------------------------- |
| `pandas`          | 2.2.0  | Manipulação e limpeza de dados                          |
| `SQLAlchemy`      | 2.0.25 | ORM / engine de conexão com PostgreSQL                  |
| `psycopg2-binary` | 2.9.9  | Driver PostgreSQL para Python                           |
| `python-dotenv`   | 1.0.1  | Carregamento de variáveis de ambiente                   |
| `pmdarima`        | 2.0.4  | Auto-ARIMA para modelagem preditiva de séries temporais |
| `requests`        | 2.31.0 | Requisições HTTP para API do Banco Central (Selic)      |
| `openpyxl`        | 3.1.2  | Leitura dos arquivos `.xlsx` do CEPEA                   |

</details>

## ▶️ Como usar

### 🌱 Simulador de Risco (principal)

```bash
docker compose run --rm app python simulador.py
# ou
python simulador.py
```

O sistema exibe as culturas disponíveis, solicita a seleção de duas para comparação, busca a Selic atual via **API do Banco Central** (ou aceita um valor manual) e gera uma **matriz de decisão** com:

- Retorno esperado projetado (Ensemble ARIMA + Holt-Winters)
- Nível de risco (Baixo / Moderado / Alto) + **VaR 95%**
- **Índice de Sharpe Agrícola** (retorno ajustado pelo risco e pela Selic)
- Lucro projetado sobre o investimento simulado

### 🔬 Backtesting do modelo

```bash
docker compose run --rm app python testar_modelo.py
# ou
python testar_modelo.py
```

Walk-Forward Cross-Validation com múltiplos folds comparando ARIMA, Holt-Winters (ETS) e Ensemble. Reporta MAE e MAPE por fold e ranking final dos modelos.

### 📊 Auditoria do cálculo de risco

```bash
docker compose run --rm app python validar_risco.py
# ou
python validar_risco.py
```

Valida matematicamente (passo a passo) o cálculo de volatilidade usando Desvio Padrão Amostral (N-1).

### Encerrar e limpar (Docker)

```bash
docker compose down          # para e remove os containers
docker compose down -v       # também apaga os dados do banco
```

## 🗂️ Estrutura do Projeto

```
AgroRisk/
├── dados/                        # Dados brutos por cultura
│   ├── Algodao_MT/
│   │   ├── custo.csv             # Custos CONAB (separado por ;)
│   │   └── preco.xlsx            # Preços CEPEA (com cabeçalho em skiprows=3)
│   └── ...                       # (demais culturas no mesmo padrão)
│
├── sql/
│   ├── 01_create_tables.sql      # DDL: criação das tabelas do banco
│   └── 02_insert_culturas.sql    # DML: seed das culturas cadastradas
│
├── src/
│   ├── conexao.py                # Fábrica de conexão com PostgreSQL via SQLAlchemy
│   ├── etl_cepea_conab.py        # Pipeline de ingestão de dados
│   ├── simulador.py              # Motor preditivo + interface interativa
│   ├── testar_modelo.py          # Walk-Forward CV (ARIMA vs ETS vs Ensemble)
│   └── validar_risco.py          # Auditoria do cálculo de risco
│
├── .env.example                  # Modelo de variáveis de ambiente
├── .env                          # Variáveis de ambiente (não versionado)
├── docker-compose.yml            # Orquestração Docker (app + banco)
├── Dockerfile                    # Imagem da aplicação Python
├── .gitignore
└── requirements.txt
```

## 🧱 Modelo de Dados

```
cultura          (id, nome, praca_referencia, unidade_conab, unidade_cepea)
    │
    ├──< custo_producao   (cultura_id, estado, custo_variavel_unitario, safra)
    └──< historico_preco  (cultura_id, data_referencia, preco_venda)
```

## 🔬 Metodologia

1. **Margem de lucratividade** mensal = `(preço_venda − custo_variável) / custo_variável`
2. **Risco** = Desvio Padrão Amostral (N-1) + **VaR 95%** (percentil 5% das margens históricas)
3. **Retorno projetado** = média das previsões do **Ensemble (Auto-ARIMA + Holt-Winters/ETS)**
4. **Sharpe Agrícola** = `(retorno - selic_período) / volatilidade` — compara culturas de ciclos diferentes
5. **Benchmark** = rendimento equivalente da **Selic** no mesmo período de capital travado

## 📄 Licença

Projeto acadêmico desenvolvido para a disciplina de Administração da UFMG. Uso educacional.
