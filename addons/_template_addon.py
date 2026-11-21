# -*- coding: utf-8 -*-
"""
addons/_template_addon.py
Modelo oficial documentado para criação de novos módulos (Add-ons) na Suíte Analítica.

COMO CRIAR UM NOVO ADD-ON:
  1. Copie este arquivo para a pasta `addons/` com um nome que NÃO comece com `_`
     (exemplo: `addon_06_perfilometria.py` ou `addon_07_afm_rugosidade.py`),
     ou envie o arquivo `.py` diretamente pelo instalador na barra lateral do `app.py`.
  2. Defina as constantes de contrato no topo do módulo:
       - TITLE (str): Nome exibido no cabeçalho da aba.
       - ICON  (str): Emoji ou símbolo curto exibido ao lado do título da aba.
       - ORDER (int | float): Ordem numérica da aba da esquerda para a direita (ex.: 10, 20, 60).
  3. Implemente a função obrigatória `render(cfg_plot: dict) -> None`:
       - `cfg_plot` contém todas as preferências visuais globais escolhidas na barra lateral:
         `font_family`, `usar_virgula`, `tick_dir`, `ticks_top_right`, `minor_ticks`,
         `grid_style`, `font_labels`, `font_ticks`, `font_legend`, `spine_lw`,
         `fig_w`, `fig_h`, `export_dpi`, `export_fmt`, `transparent_bg`, `csv_br`.
  4. Utilize sempre widgets com `key` exclusivo (prefixado pelo nome do seu add-on)
     para evitar colisão de estados entre abas no Streamlit. Os helpers de `core.ui`
     já geram keys no formato `f"{KEY}_<função>"` a partir do prefixo que você informa.
  5. O app.py executa cada aba dentro de um @st.fragment: não use st.sidebar nem st.rerun() no add-on.

HELPERS DISPONÍVEIS EM `core.ui` (reaproveitam core.parsers e core.plotting, sem duplicar regras):
  Entrada ....... painel_entrada()  -> uploader único/múltiplo + override do parser + slot de diagnóstico
  Leitura ....... ler_arquivos()    -> aplica um extrator a cada arquivo, trata erros e mostra o diagnóstico
  Colunas ....... selecionar_colunas_xy(), selecionar_coluna_numerica(), colunas_numericas()
  Visual ........ cabecalho_secao(), seletor_amostra(), seletor_amostras(), linha_metricas(),
                  msg_sem_dados(), msg_aviso(), msg_erro_leitura(),
                  nova_figura_origin(), legenda_origin(), bloco_exportacao()

Extratores prontos de `core.parsers` que funcionam com ler_arquivos():
  ler_arquivo_cientifico (tabela genérica), extrair_keithley_iv, extrair_keithley_it,
  extrair_solartron_eis, extrair_cary_uvvis (use nome_kwarg="nome_arquivo").
"""

import matplotlib.pyplot as plt
import streamlit as st

from core.parsers import ler_arquivo_cientifico
from core.ui import (
    bloco_exportacao,
    cabecalho_secao,
    chave,
    legenda_origin,
    ler_arquivos,
    linha_metricas,
    msg_sem_dados,
    nova_figura_origin,
    painel_entrada,
    selecionar_colunas_xy,
    seletor_amostra,
)

# ==============================================================================
# CONTRATO PADRÃO DO ADD-ON
# ==============================================================================
TITLE = "Template de Novo Módulo"
ICON = "🧩"
ORDER = 999


# ==============================================================================
# FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO DA ABA
# ==============================================================================
def render(cfg_plot: dict) -> None:
    """Ponto de entrada chamado automaticamente pelo app.py ao desenhar esta aba."""
    KEY = "tpl"
    usar_virgula = bool(cfg_plot.get("usar_virgula", True))

    cabecalho_secao(
        "Módulo de Exemplo — Base para Novos Instrumentos",
        "Use este arquivo (`addons/_template_addon.py`) como ponto de partida para adicionar "
        "novas técnicas experimentais mantendo o estilo OriginPro e o parser blindado.",
        icone="🧩",
    )

    # 1) Entrada: uploader + painel retrátil de override do parser + slot de diagnóstico
    entrada = painel_entrada(
        KEY,
        "Carregar arquivo(s) tabular(es) (.dat, .csv, .txt)",
        tipos=("dat", "csv", "txt"),
        multiplos=True,
    )
    if entrada.vazia:
        msg_sem_dados("Carregue um arquivo acima para visualizar o fluxo padrão de leitura e plotagem.")
        return

    # 2) Leitura: troque `ler_arquivo_cientifico` pelo extrator do seu instrumento
    lidos = ler_arquivos(entrada, ler_arquivo_cientifico)
    if not lidos:
        return

    # 3) Escolha da amostra (só aparece se houver mais de um arquivo)
    if len(lidos) > 1:
        nome_sel = seletor_amostra([item.nome for item in lidos], key=chave(KEY, "amostra"), rotulo="Arquivo / Amostra")
        if nome_sel is None:
            return
    else:
        nome_sel = lidos[0].nome
    tabela = next(item.resultado for item in lidos if item.nome == nome_sel)

    # 4) Colunas numéricas dos eixos (mensagens claras se o arquivo não for compatível)
    selecao = selecionar_colunas_xy(tabela.df, tabela.decimal_usado, KEY)
    if selecao is None:
        return

    c1, c2 = st.columns(2)
    legenda = c1.text_input(
        "Nome na Legenda",
        value=nome_sel.rsplit(".", 1)[0],
        key=chave(KEY, "leg", nome_sel),
    )
    cor_linha = c2.color_picker("Cor da Curva", value="#1F618D", key=chave(KEY, "cor"))

    # 5) Gráfico no padrão Origin
    fig, ax = nova_figura_origin(cfg_plot, selecao.col_x, selecao.col_y)
    ax.plot(
        selecao.df[selecao.col_x].values,
        selecao.df[selecao.col_y].values,
        "o-",
        color=cor_linha,
        markersize=4.5,
        lw=1.5,
        label=legenda,
    )
    legenda_origin(ax, cfg_plot)
    fig.tight_layout()

    col_graf, col_info = st.columns([2.2, 1.0])
    with col_graf:
        st.pyplot(fig)
    with col_info:
        st.markdown("#### Resumo Rápido")
        y = selecao.df[selecao.col_y]
        linha_metricas(
            [
                ("Nº de Pontos Válidos", len(selecao.df)),
                ("Máximo em Y", float(y.max())),
                ("Mínimo em Y", float(y.min())),
            ],
            usar_virgula=usar_virgula,
            casas=4,
            container=col_info,
            vertical=True,
        )
        bloco_exportacao(fig, selecao.df, legenda, cfg_plot, KEY)

    plt.close(fig)
