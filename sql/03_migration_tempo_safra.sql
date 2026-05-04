-- Adiciona a coluna de duração do ciclo de cada cultura (em meses)
ALTER TABLE cultura ADD COLUMN IF NOT EXISTS tempo_safra_meses INTEGER DEFAULT 6;

-- Atualiza cada cultura com seu ciclo produtivo típico
UPDATE cultura SET tempo_safra_meses = 6  WHERE nome = 'Algodão';   -- MT: plantio out-nov, colheita abr-mai
UPDATE cultura SET tempo_safra_meses = 5  WHERE nome = 'Arroz';     -- RS: nov a mar/abr
UPDATE cultura SET tempo_safra_meses = 12 WHERE nome = 'Banana';    -- MG: ciclo anual (perene)
UPDATE cultura SET tempo_safra_meses = 12 WHERE nome = 'Café';      -- MG: ciclo anual (bienal produtivo)
UPDATE cultura SET tempo_safra_meses = 4  WHERE nome = 'Cebola';    -- SC: plantio mar, colheita jun-jul
UPDATE cultura SET tempo_safra_meses = 4  WHERE nome = 'Feijão';    -- SP: ciclo curto ~90-120 dias
UPDATE cultura SET tempo_safra_meses = 12 WHERE nome = 'Laranja';   -- SP: ciclo anual (perene)
UPDATE cultura SET tempo_safra_meses = 6  WHERE nome = 'Maçã';      -- SC: nov a abr/mai
UPDATE cultura SET tempo_safra_meses = 5  WHERE nome = 'Milho';     -- RS: set a jan/fev
UPDATE cultura SET tempo_safra_meses = 5  WHERE nome = 'Soja';      -- PR: out a fev/mar
UPDATE cultura SET tempo_safra_meses = 5  WHERE nome = 'Trigo';     -- PR: abr a ago/set
