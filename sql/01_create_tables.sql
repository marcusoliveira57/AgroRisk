CREATE TABLE cultura (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(50) NOT NULL,
    praca_referencia VARCHAR(100),
    unidade_conab VARCHAR(20),
    unidade_cepea VARCHAR(20)
);

CREATE TABLE custo_producao (
    id SERIAL PRIMARY KEY,
    cultura_id INTEGER REFERENCES cultura(id),
    estado CHAR(2),
    custo_variavel_unitario DECIMAL(10, 2),
    safra VARCHAR(20)
);

CREATE TABLE historico_preco (
    id SERIAL PRIMARY KEY,
    cultura_id INTEGER REFERENCES cultura(id),
    data_referencia DATE,
    preco_venda DECIMAL(10, 2)
);