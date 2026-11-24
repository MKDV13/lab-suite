# -*- coding: utf-8 -*-
"""
report_extensions/exemplo_extensao_pesquisa.py
Extensao modelo pronta para copiar e adaptar para a sua pesquisa no lab-suite.

Como copiar e usar:
  1. Duplique este arquivo dentro de `report_extensions/` (ex.: `minha_analise.py`).
  2. Ajuste `TITLE`, `DESCRIPTION` e `ORDER`.
  3. Altere apenas o bloco de calculo dentro de `render_extensao(contexto, cfg_plot)`.

Chaves disponiveis em `contexto`:
  - "df_consolidado": pd.DataFrame (1 linha por amostra, colunas "<tecnica> · <metrica>")
  - "df_por_registro": pd.DataFrame (1 linha por par amostra/tecnica)
  - "df_longo": pd.DataFrame (1 linha por metrica)
  - "registros": List[core.resultados.Resultado]
  - "amostras": List[str]
  - "tecnicas": List[str]
"""

import numpy as np
import pandas as pd
import streamlit as st

from core.plotting import botao_download_csv, fmt_num

TITLE = "Analise Derivada da Pesquisa (Modelo)"
DESCRIPTION = "Exemplo pronto de calculo derivado, tratamento seguro de ausencias e exportacao CSV."
ORDER = 10


def _buscar_coluna_metrica(df: pd.DataFrame, padroes: list[str]) -> str | None:
    for padrao in padroes:
        p_low = padrao.lower()
        for col in df.columns:
            if p_low in str(col).lower():
                return col
    return None


def _razao_segura(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerador, errors="coerce")
    den = pd.to_numeric(denominador, errors="coerce")
    valido = num.notna() & den.notna() & (den.abs() > 1e-15)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    out[valido] = num[valido] / den[valido]
    return out


def _formatar_tabela(df: pd.DataFrame, usar_virgula: bool) -> pd.DataFrame:
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_bool_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda v: "Sim" if v else "Nao")
        elif pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(
                lambda v: "—" if pd.isna(v) else fmt_num(float(v), 3, usar_virgula)
            )
    return df_fmt


def render_extensao(contexto: dict, cfg_plot: dict) -> None:
    df_cons: pd.DataFrame = contexto["df_consolidado"]
    if df_cons.empty:
        st.info("Sem amostras registradas para calcular métricas derivadas.")
        return

    usar_virgula = bool(cfg_plot.get("usar_virgula", True))
    fator_foco = float(st.number_input(
        "Peso de sensibilidade no indice composto (modelo)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="ext_modelo_peso_sens",
    ))

    # Localize colunas sem quebrar quando uma amostra nao tiver todas as tecnicas
    col_sens = _buscar_coluna_metrica(df_cons, ["sensibilidade", "sensib"])
    col_t90 = _buscar_coluna_metrica(df_cons, ["t90"])
    col_trec = _buscar_coluna_metrica(df_cons, ["t_rec90", "trec90"])

    df_resumo = pd.DataFrame({"Amostra": df_cons["Amostra"]})

    if col_sens is not None:
        df_resumo["Sensibilidade (%)"] = pd.to_numeric(df_cons[col_sens], errors="coerce")
    if col_t90 is not None:
        df_resumo["t90 (s)"] = pd.to_numeric(df_cons[col_t90], errors="coerce")
    if col_trec is not None:
        df_resumo["t_rec90 (s)"] = pd.to_numeric(df_cons[col_trec], errors="coerce")

    # Exemplo de calculo derivado: razao t90 / t_rec90 e indice simples
    if col_t90 is not None and col_trec is not None:
        df_resumo["Razao t90 / t_rec90"] = _razao_segura(df_cons[col_t90], df_cons[col_trec])

    if col_sens is not None and col_t90 is not None:
        df_resumo["Indice Rapidez-Sensib."] = _razao_segura(
            pd.to_numeric(df_cons[col_sens], errors="coerce") * fator_foco,
            df_cons[col_t90],
        )

    if len(df_resumo.columns) == 1:
        st.info("Colunas esperadas pelo exemplo nao encontradas ainda; exibindo tabela consolidada base.")
        st.dataframe(_formatar_tabela(df_cons, usar_virgula), use_container_width=True, hide_index=True)
        return

    st.dataframe(_formatar_tabela(df_resumo, usar_virgula), use_container_width=True, hide_index=True)
    botao_download_csv(
        df_resumo,
        "Extensao_Resumo_Pesquisa",
        cfg_plot,
        "ext_modelo_csv",
        label="📥 Baixar Tabela da Extensao (.CSV)",
    )
