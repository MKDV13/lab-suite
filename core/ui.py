# -*- coding: utf-8 -*-
"""
core/ui.py
Helpers de interface compartilhados pelos add-ons da Suíte Analítica Modular.

Centraliza os padrões que se repetiam em todos os módulos:
  1. Entrada de arquivos  -> painel_entrada()      uploader (único/múltiplo) + override do parser + slot de diagnóstico
  2. Leitura em lote      -> ler_arquivos()         laço try/except por arquivo + diagnóstico do 1º arquivo lido
  3. Colunas numéricas    -> selecionar_colunas_xy() / selecionar_coluna_numerica() com mensagens claras
  4. Visual recorrente    -> cabecalho_secao(), seletor_amostra(s)(), linha_metricas(), mensagens de estado,
                             nova_figura_origin(), legenda_origin(), bloco_exportacao()
  5. Registro de resultados -> sessao_resultados() / ColetorResultados: publica as métricas da aba em core.resultados
                             (espelhando o estado atual da aba) e mostra o painel padrão de resultados registrados

Nada aqui reimplementa parsing ou estilo gráfico: tudo delega a core.parsers e core.plotting.
Não há chamadas ao Streamlit na importação do módulo, e nenhuma função usa st.sidebar nem st.rerun,
então os helpers funcionam dentro dos @st.fragment criados pelo app.py.

Convenção de chaves de widget: `f"{key_prefix}_<função>"` (ex.: `tpl_uploader`, `tpl_col_x`).
Use um key_prefix exclusivo por add-on para evitar colisão de estado entre abas.
"""

import zlib
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core import resultados as registro_central
from core.parsers import (
    OpcoesParser,
    ResultadoLeituraTabela,
    converter_serie_float,
    exibir_diagnostico_leitura,
    render_painel_override_parser,
)
from core.plotting import (
    aplicar_estilo_origin,
    botao_download_csv,
    botao_download_figura,
    fmt_num,
)

# Tipo do objeto devolvido por st.file_uploader (UploadedFile); Any evita depender de caminho interno do Streamlit.
ArquivoUpload = Any

TIPOS_PADRAO: Tuple[str, ...] = ("dat", "txt", "csv")
NOME_AMOSTRA_PADRAO = "Amostra sem nome"

__all__ = [
    "ArquivoLido",
    "ArquivoUpload",
    "ColetorResultados",
    "EntradaArquivos",
    "SelecaoXY",
    "TIPOS_PADRAO",
    "bloco_exportacao",
    "cabecalho_secao",
    "chave",
    "colunas_numericas",
    "ler_arquivos",
    "legenda_origin",
    "linha_metricas",
    "msg_aviso",
    "msg_erro_leitura",
    "msg_sem_dados",
    "nova_figura_origin",
    "painel_entrada",
    "selecionar_coluna_numerica",
    "selecionar_colunas_xy",
    "seletor_amostra",
    "seletor_amostras",
    "sessao_resultados",
    "uploader_arquivos",
]


# ==============================================================================
# 0. UTILITÁRIOS E MENSAGENS DE ESTADO CONSISTENTES
# ==============================================================================
def chave(prefixo: str, *partes: Any) -> str:
    """Monta uma key de widget única: chave('iv', 'leg') -> 'iv_leg'."""
    return "_".join([str(prefixo)] + [str(p) for p in partes])


def cabecalho_secao(titulo: str, descricao: Optional[str] = None, icone: str = "") -> None:
    """Título padrão de uma aba/seção (st.subheader) com legenda opcional logo abaixo."""
    st.subheader(f"{icone} {titulo}".strip())
    if descricao:
        st.caption(descricao)


def msg_sem_dados(texto: str = "Carregue um arquivo acima para visualizar os dados.") -> None:
    """Estado vazio (nenhum arquivo/amostra selecionado): mensagem informativa, não é erro."""
    st.info(texto)


def msg_aviso(texto: str) -> None:
    """Condição que impede o gráfico/cálculo mas pode ser corrigida pelo usuário."""
    st.warning(texto)


def msg_erro_leitura(nome_arquivo: str, exc: BaseException, contexto: str = "") -> None:
    """Falha ao ler um arquivo. `contexto` complementa o texto (ex.: 'arquivo I×t')."""
    prefixo = f"Erro ao ler {contexto}" if contexto else "Erro ao ler"
    st.error(f"{prefixo} `{nome_arquivo}`: {exc}")


# ==============================================================================
# 1. ENTRADA DE ARQUIVOS (UPLOADER + OVERRIDE DO PARSER + SLOT DE DIAGNÓSTICO)
# ==============================================================================
@dataclass
class EntradaArquivos:
    """Resultado de painel_entrada(): sempre traz `arquivos` como lista (vazia se nada foi enviado)."""

    arquivos: List[ArquivoUpload]
    opcoes: OpcoesParser
    slot_diag: Any
    key_prefix: str

    @property
    def vazia(self) -> bool:
        return len(self.arquivos) == 0

    @property
    def primeiro(self) -> Optional[ArquivoUpload]:
        return self.arquivos[0] if self.arquivos else None

    @property
    def nomes(self) -> List[str]:
        return [str(a.name) for a in self.arquivos]


def uploader_arquivos(
    key_prefix: str,
    rotulo: str,
    tipos: Sequence[str] = TIPOS_PADRAO,
    multiplos: bool = False,
    ajuda: Optional[str] = None,
    key_uploader: Optional[str] = None,
) -> List[ArquivoUpload]:
    """
    Só o uploader, já normalizado para LISTA (vazia se nada foi enviado). Use quando o painel de override do
    parser precisa ficar em outra posição da tela (ex.: Add-on 04); caso contrário prefira painel_entrada().
    """
    enviado = st.file_uploader(
        rotulo,
        type=list(tipos) if tipos is not None else None,
        accept_multiple_files=multiplos,
        help=ajuda,
        key=key_uploader or chave(key_prefix, "uploader"),
    )
    if enviado is None:
        return []
    if isinstance(enviado, (list, tuple)):
        return list(enviado)
    return [enviado]


def painel_entrada(
    key_prefix: str,
    rotulo: str,
    tipos: Sequence[str] = TIPOS_PADRAO,
    multiplos: bool = False,
    ajuda: Optional[str] = None,
    mostrar_override: bool = True,
    key_uploader: Optional[str] = None,
) -> EntradaArquivos:
    """
    Renderiza o bloco padrão de entrada de dados:
      uploader (único ou múltiplo) + painel retrátil de override do parser + slot de diagnóstico.

    `key_uploader` permite manter a key já usada por um add-on existente (ex.: 'eis_uploader_multi'),
    preservando o estado do widget ao migrar. Por padrão usa f"{key_prefix}_uploader".
    O `key_prefix` também alimenta as keys do painel de override (p_sep_<prefixo>, p_dec_<prefixo>, ...),
    idênticas às geradas por render_painel_override_parser().
    """
    arquivos = uploader_arquivos(key_prefix, rotulo, tipos, multiplos, ajuda, key_uploader)

    if mostrar_override:
        opcoes, slot_diag = render_painel_override_parser(key_prefix)
    else:
        opcoes, slot_diag = OpcoesParser(), None

    return EntradaArquivos(arquivos=arquivos, opcoes=opcoes, slot_diag=slot_diag, key_prefix=key_prefix)


# ==============================================================================
# 2. LEITURA EM LOTE COM TRATAMENTO DE ERRO E DIAGNÓSTICO
# ==============================================================================
@dataclass
class ArquivoLido:
    """
    Um arquivo lido com sucesso.
    `resultado` é o retorno bruto do extrator (tupla ou objeto), para o add-on desempacotar como já faz hoje;
    `meta` é o ResultadoLeituraTabela localizado nesse retorno (ou None se o extrator não o devolver).
    """

    nome: str
    resultado: Any
    meta: Optional[ResultadoLeituraTabela]


def _parece_meta(obj: Any) -> bool:
    return hasattr(obj, "linhas_brutas_topo") and hasattr(obj, "decimal_usado") and hasattr(obj, "df")


def _extrair_meta(resultado: Any) -> Optional[ResultadoLeituraTabela]:
    if _parece_meta(resultado):
        return resultado
    if isinstance(resultado, (tuple, list)):
        for item in resultado:
            if _parece_meta(item):
                return item
    return None


def ler_arquivos(
    entrada: EntradaArquivos,
    extrator: Callable[..., Any],
    *,
    contexto_erro: str = "",
    nome_kwarg: Optional[str] = None,
    mostrar_diagnostico: bool = True,
    **kwargs_extrator: Any,
) -> List[ArquivoLido]:
    """
    Aplica `extrator(conteudo_bytes, opcoes=<OpcoesParser>, **kwargs_extrator)` a cada arquivo enviado.

    - Falha em um arquivo: mostra st.error (msg_erro_leitura) e segue para os demais.
    - `nome_kwarg`: se informado, passa o nome do arquivo ao extrator com esse nome de parâmetro
      (ex.: nome_kwarg='nome_arquivo' para extrair_cary_uvvis).
    - Diagnóstico: preenche o slot do painel de override com o 1º arquivo lido com sucesso.

    Exemplos:
        lidos = ler_arquivos(entrada, extrair_keithley_iv)                      # (df, meta)
        lidos = ler_arquivos(entrada, extrair_solartron_eis, inverter_sinal_zim=flag)
        lidos = ler_arquivos(entrada, extrair_cary_uvvis, nome_kwarg="nome_arquivo")
        lidos = ler_arquivos(entrada, ler_arquivo_cientifico)                   # ResultadoLeituraTabela
    """
    lidos: List[ArquivoLido] = []
    for arq in entrada.arquivos:
        try:
            kwargs = dict(kwargs_extrator)
            if nome_kwarg:
                kwargs[nome_kwarg] = arq.name
            resultado = extrator(arq.getvalue(), opcoes=entrada.opcoes, **kwargs)
        except Exception as exc:
            msg_erro_leitura(arq.name, exc, contexto_erro)
            continue
        lidos.append(ArquivoLido(nome=str(arq.name), resultado=resultado, meta=_extrair_meta(resultado)))

    if mostrar_diagnostico:
        for item in lidos:
            if item.meta is not None:
                exibir_diagnostico_leitura(entrada.slot_diag, item.meta, item.nome)
                break
    return lidos


# ==============================================================================
# 3. SELEÇÃO E VALIDAÇÃO DE COLUNAS NUMÉRICAS
# ==============================================================================
def colunas_numericas(
    df: pd.DataFrame,
    decimal: str = "auto",
    minimo_validos: int = 2,
) -> Dict[str, pd.Series]:
    """
    Converte cada coluna com converter_serie_float() (mesma regra do parser) e devolve
    {nome: Series float} apenas das colunas com pelo menos `minimo_validos` valores numéricos.
    """
    resultado: Dict[str, pd.Series] = {}
    if df is None or df.empty:
        return resultado
    for col in df.columns:
        serie = converter_serie_float(df[col], decimal=decimal)
        if int(serie.notna().sum()) >= minimo_validos:
            resultado[str(col)] = serie
    return resultado


def _chave_por_opcoes(key_prefix: str, nome: str, opcoes: Sequence[str]) -> str:
    """Key que muda quando a lista de colunas muda (novo arquivo), evitando seleção obsoleta no selectbox."""
    assinatura = zlib.crc32("|".join(opcoes).encode("utf-8")) & 0xFFFFFFFF
    return f"{key_prefix}_{nome}_{assinatura:08x}"


def _aviso_sem_colunas(df_bruto: pd.DataFrame, numericas: Dict[str, pd.Series], minimo: int) -> None:
    rejeitadas = [str(c) for c in df_bruto.columns if str(c) not in numericas]
    detalhe = f" Colunas sem valores numéricos suficientes: {', '.join(rejeitadas)}." if rejeitadas else ""
    plural = "pelo menos 2 colunas numéricas" if minimo >= 2 else "pelo menos 1 coluna numérica"
    msg_aviso(
        f"São necessárias {plural} para escolher os eixos, mas o arquivo tem {len(numericas)}.{detalhe} "
        "Se o separador ou o decimal foram detectados errado, ajuste-os no painel de override do parser."
    )


@dataclass
class SelecaoXY:
    """Par de colunas escolhido; `df` tem exatamente [col_x, col_y], float, sem NaN e índice reiniciado."""

    col_x: str
    col_y: str
    df: pd.DataFrame


def selecionar_colunas_xy(
    df_bruto: pd.DataFrame,
    decimal: str,
    key_prefix: str,
    *,
    rotulo_x: str = "Coluna Eixo X",
    rotulo_y: str = "Coluna Eixo Y",
    padrao_x: Optional[str] = None,
    padrao_y: Optional[str] = None,
    minimo_pontos: int = 2,
    container: Any = None,
) -> Optional[SelecaoXY]:
    """
    Dois selectboxes (X e Y) que listam SOMENTE colunas numéricas. Retorna None, já com mensagem clara
    (st.warning), quando: a tabela está vazia, há menos de 2 colunas numéricas, X e Y são a mesma coluna,
    ou há menos de `minimo_pontos` pares válidos. `decimal` deve vir de ResultadoLeituraTabela.decimal_usado.
    """
    if df_bruto is None or df_bruto.empty:
        msg_aviso("O arquivo não contém linhas de dados reconhecíveis. Confira o painel de override do parser.")
        return None

    numericas = colunas_numericas(df_bruto, decimal)
    if len(numericas) < 2:
        _aviso_sem_colunas(df_bruto, numericas, minimo=2)
        return None

    nomes = list(numericas.keys())
    idx_x = nomes.index(padrao_x) if padrao_x in nomes else 0
    if padrao_y in nomes and padrao_y != nomes[idx_x]:
        idx_y = nomes.index(padrao_y)
    else:
        idx_y = [i for i in range(len(nomes)) if i != idx_x][0]

    alvo = container if container is not None else st
    c1, c2 = alvo.columns(2)
    col_x = c1.selectbox(rotulo_x, nomes, index=idx_x, key=_chave_por_opcoes(key_prefix, "col_x", nomes))
    col_y = c2.selectbox(rotulo_y, nomes, index=idx_y, key=_chave_por_opcoes(key_prefix, "col_y", nomes))

    if col_x == col_y:
        msg_aviso(f"X e Y estão na mesma coluna (`{col_x}`). Escolha colunas diferentes.")
        return None

    df_xy = pd.DataFrame({col_x: numericas[col_x], col_y: numericas[col_y]}).dropna().reset_index(drop=True)
    if len(df_xy) < minimo_pontos:
        msg_aviso(
            f"Apenas {len(df_xy)} par(es) numérico(s) válido(s) entre `{col_x}` e `{col_y}` "
            f"(mínimo: {minimo_pontos}). Escolha outras colunas ou revise o decimal no override do parser."
        )
        return None
    return SelecaoXY(col_x=col_x, col_y=col_y, df=df_xy)


def selecionar_coluna_numerica(
    df_bruto: pd.DataFrame,
    decimal: str,
    key_prefix: str,
    *,
    rotulo: str = "Coluna",
    padrao: Optional[str] = None,
    minimo_pontos: int = 2,
    container: Any = None,
) -> Optional[pd.Series]:
    """Um selectbox de coluna numérica. Retorna a Series float sem NaN (name = coluna) ou None com aviso."""
    if df_bruto is None or df_bruto.empty:
        msg_aviso("O arquivo não contém linhas de dados reconhecíveis. Confira o painel de override do parser.")
        return None

    numericas = colunas_numericas(df_bruto, decimal)
    if len(numericas) < 1:
        _aviso_sem_colunas(df_bruto, numericas, minimo=1)
        return None

    nomes = list(numericas.keys())
    idx = nomes.index(padrao) if padrao in nomes else 0
    alvo = container if container is not None else st
    col = alvo.selectbox(rotulo, nomes, index=idx, key=_chave_por_opcoes(key_prefix, "col", nomes))

    serie = numericas[col].dropna().reset_index(drop=True)
    serie.name = col
    if len(serie) < minimo_pontos:
        msg_aviso(f"A coluna `{col}` tem apenas {len(serie)} valor(es) numérico(s) (mínimo: {minimo_pontos}).")
        return None
    return serie


# ==============================================================================
# 4. HELPERS VISUAIS RECORRENTES
# ==============================================================================
def seletor_amostra(
    nomes: Sequence[str],
    key: str,
    rotulo: str = "Amostra / Curva",
    padrao: Optional[str] = None,
    container: Any = None,
) -> Optional[str]:
    """Selectbox de uma amostra. Retorna None (com mensagem de estado vazio) se a lista estiver vazia."""
    opcoes = [str(n) for n in nomes]
    if not opcoes:
        msg_sem_dados("Nenhuma amostra disponível para seleção.")
        return None
    idx = opcoes.index(padrao) if padrao in opcoes else 0
    alvo = container if container is not None else st
    return alvo.selectbox(rotulo, opcoes, index=idx, key=key)


def seletor_amostras(
    nomes: Sequence[str],
    key: str,
    rotulo: str = "Amostras / Curvas",
    padrao: Optional[Sequence[str]] = None,
    container: Any = None,
) -> List[str]:
    """Multiselect de amostras (padrão: todas). Retorna [] com aviso se a lista estiver vazia ou nada for marcado."""
    opcoes = [str(n) for n in nomes]
    if not opcoes:
        msg_sem_dados("Nenhuma amostra disponível para seleção.")
        return []
    marcadas_padrao = [p for p in (padrao if padrao is not None else opcoes) if p in opcoes]
    alvo = container if container is not None else st
    escolhidas = alvo.multiselect(rotulo, opcoes, default=marcadas_padrao, key=key)
    if not escolhidas:
        msg_aviso("Selecione pelo menos uma amostra.")
    return list(escolhidas)


def linha_metricas(
    itens: Sequence[Tuple[str, Any]],
    usar_virgula: bool = True,
    casas: int = 4,
    container: Any = None,
    vertical: bool = False,
) -> None:
    """
    Conjunto de st.metric (lado a lado, ou empilhados com vertical=True, bom para colunas estreitas).
    Valores float/np.floating passam por fmt_num() (vírgula ABNT ou ponto); inteiros e textos são exibidos como estão.
    """
    if not itens:
        return
    alvo = container if container is not None else st
    destinos = [alvo] * len(itens) if vertical else list(alvo.columns(len(itens)))
    for destino, (rotulo, valor) in zip(destinos, itens):
        if isinstance(valor, (float, np.floating)):
            texto = fmt_num(float(valor), casas, usar_virgula)
        else:
            texto = str(valor)
        destino.metric(rotulo, texto)


def nova_figura_origin(
    cfg_plot: dict,
    xlabel: str = "",
    ylabel: str = "",
    *,
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 150,
    labelpad: int = 6,
) -> Tuple[Figure, Axes]:
    """
    Cria (fig, ax) com as dimensões globais da barra lateral, rótulos nas fontes escolhidas e o estilo Origin
    aplicado por core.plotting.aplicar_estilo_origin. O add-on só plota; use legenda_origin() e
    plt.close(fig) ao final. `labelpad` permite reproduzir o afastamento de rótulo de add-ons já existentes.
    """
    fig, ax = plt.subplots(
        figsize=(float(cfg_plot.get("fig_w", 7.8)), float(cfg_plot.get("fig_h", 5.5))),
        dpi=dpi,
    )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    fonte = int(cfg_plot.get("font_labels", 12))
    ax.set_xlabel(xlabel, fontsize=fonte, labelpad=labelpad)
    ax.set_ylabel(ylabel, fontsize=fonte, labelpad=labelpad)
    aplicar_estilo_origin(ax, cfg_plot, log_x=log_x, log_y=log_y)
    return fig, ax


def legenda_origin(ax: Axes, cfg_plot: dict, loc: str = "best", **kwargs: Any) -> Any:
    """Legenda no padrão Origin (moldura preta fina, sem cantos arredondados). Retorna None se não há rótulos."""
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return None
    params: Dict[str, Any] = dict(
        loc=loc,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=int(cfg_plot.get("font_legend", 10)),
    )
    params.update(kwargs)
    leg = ax.legend(**params)
    leg.get_frame().set_linewidth(0.9)
    return leg


def bloco_exportacao(
    fig: Optional[Figure],
    df: Optional[pd.DataFrame],
    nome_base: str,
    cfg_plot: dict,
    key_prefix: str,
) -> None:
    """Botões de download da figura (formato/DPI globais) e do CSV (padrão Internacional ou Excel BR)."""
    if fig is not None:
        botao_download_figura(fig, f"Grafico_{nome_base}", cfg_plot, key_prefix)
    if df is not None and not df.empty:
        botao_download_csv(df, f"Dados_{nome_base}", cfg_plot, key_prefix)


# ==============================================================================
# 5. REGISTRO DE RESULTADOS (core/resultados.py) PADRONIZADO PARA OS ADD-ONS
# ==============================================================================
@dataclass
class _ItemResultado:
    amostra: str
    metricas: Dict[str, Any]
    unidades: Dict[str, str]
    meta: Dict[str, Any]
    desambiguar: Optional[str] = None


@dataclass
class ColetorResultados:
    """
    Acumula os resultados calculados em UMA execução de render() e os publica no registro central ao final.

    Política "espelho" (a mesma do Add-on 03): o registro de (técnica, origem) passa a conter exatamente as
    amostras coletadas nesta execução. Arquivo removido, seleção desmarcada, legenda renomeada ou retorno
    antecipado do add-on (sem dados) limpam o que ficou obsoleto; nada precisa ser removido à mão.
    Cada amostra é registrada com substituir=True, então o registro sempre reflete a análise exibida
    (sem métricas de um modo de análise anterior). Repetir o mesmo conteúdo é inalterado (idempotente).
    """

    tecnica: str
    origem: str
    cfg_plot: Optional[dict] = None
    key_prefix: str = "reg"
    exibir_painel: bool = True
    publicados: List[str] = field(default_factory=list)
    erro: Optional[str] = None
    _itens: List[_ItemResultado] = field(default_factory=list)

    def adicionar(
        self,
        amostra: str,
        metricas: Dict[str, Any],
        *,
        unidades: Optional[Dict[str, str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        desambiguar: Optional[str] = None,
    ) -> None:
        """
        Coleta o resultado de uma amostra. `unidades` pode listar métricas ausentes neste modo de análise
        (são ignoradas). Se duas amostras coletadas tiverem o mesmo nome, ambas ganham ' [desambiguar]'
        (ex.: o nome do arquivo) para nenhuma sobrescrever a outra.
        """
        nome = str(amostra).strip() or NOME_AMOSTRA_PADRAO
        dados = dict(metricas)
        unid = {k: v for k, v in (unidades or {}).items() if k in dados}
        self._itens.append(_ItemResultado(nome, dados, unid, dict(meta or {}), desambiguar))

    def _nomes_efetivos(self) -> List[str]:
        contagem = Counter(item.amostra for item in self._itens)
        usados: set = set()
        nomes: List[str] = []
        for item in self._itens:
            nome = item.amostra
            if contagem[nome] > 1 and item.desambiguar:
                nome = f"{nome} [{item.desambiguar}]"
            base, k = nome, 2
            while nome in usados:
                nome = f"{base} ({k})"
                k += 1
            usados.add(nome)
            nomes.append(nome)
        return nomes

    def publicar(self) -> int:
        """Grava as amostras coletadas, remove as obsoletas e mostra o painel. Nunca levanta exceção."""
        try:
            nomes = self._nomes_efetivos()
            for nome, item in zip(nomes, self._itens):
                registro_central.registrar(
                    nome,
                    self.tecnica,
                    item.metricas,
                    unidades=item.unidades,
                    meta=item.meta,
                    origem=self.origem,
                    substituir=True,
                )
            registro_central.sincronizar(self.tecnica, self.origem, nomes)
            self.publicados = nomes
        except Exception as exc:
            self.erro = str(exc)
            st.caption(f"⚠️ Registro central de resultados indisponível: {exc}")
            return 0

        if self.publicados and self.exibir_painel:
            self._painel()
        return len(self.publicados)

    def _painel(self) -> None:
        with st.expander(f"📌 Registro Central de Resultados — {self.tecnica} ({len(self.publicados)})", expanded=False):
            df = registro_central.para_dataframe("por_registro", tecnica=self.tecnica, origem=self.origem)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                "Valores em unidades SI/do instrumento, compartilhados com as demais abas e com o relatório "
                "consolidado. O registro espelha a análise exibida e se atualiza sozinho."
            )
            if self.cfg_plot is not None:
                botao_download_csv(
                    df,
                    f"Registro_Resultados_{self.tecnica}",
                    self.cfg_plot,
                    chave(self.key_prefix, "registro"),
                    label="📊 Baixar Registro de Resultados (.CSV)",
                )


@contextmanager
def sessao_resultados(
    tecnica: str,
    origem: str,
    cfg_plot: Optional[dict] = None,
    key_prefix: str = "reg",
    exibir_painel: bool = True,
) -> Iterator[ColetorResultados]:
    """
    Envolve o corpo de render(): o add-on chama `coletor.adicionar(...)` onde calcula as métricas e a publicação
    acontece UMA vez, no fim — inclusive quando o corpo sai cedo por falta de dados (registro espelhado vazio).

        def render(cfg_plot):
            with sessao_resultados(TEC_IV, "addon_01", cfg_plot, "iv") as coletor:
                _render_corpo(cfg_plot, coletor)
    """
    coletor = ColetorResultados(tecnica, origem, cfg_plot, key_prefix, exibir_painel)
    try:
        yield coletor
    finally:
        coletor.publicar()
