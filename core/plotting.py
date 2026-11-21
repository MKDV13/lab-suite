# -*- coding: utf-8 -*-
"""
core/plotting.py
Utilitários gráficos padronizados no estilo OriginPro (eixos, ticks internos,
formatação numérica ABNT/Internacional e exportação vetorial/alta resolução).
"""

import io
import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st


def fmt_num(val: Optional[float], casas: int = 2, usar_virgula: bool = True) -> str:
    """Formata um número em ponto flutuante respeitando vírgula (ABNT) ou ponto decimal."""
    if val is None or not np.isfinite(val):
        return "—"
    if abs(val) < 10 ** (-(casas + 4)):
        val = 0.0
    s = f"{val:.{casas}f}"
    return s.replace(".", ",") if usar_virgula else s


def fmt_cientifico(
    val: Optional[float],
    casas: int = 2,
    exp_fixo: Optional[int] = None,
    usar_virgula: bool = True,
) -> str:
    """Formata número em notação científica compatível com LaTeX/Matplotlib (ex: 1,12 × 10⁻⁵)."""
    if val is None or not np.isfinite(val):
        return "—"
    if val == 0.0:
        return fmt_num(0.0, casas, usar_virgula)
    exp = exp_fixo if exp_fixo is not None else int(np.floor(np.log10(abs(val))))
    mantissa = val / (10.0 ** exp)
    m_str = fmt_num(mantissa, casas, usar_virgula)
    return f"{m_str} × 10$^{{{exp}}}$"


def fmt_eng_zview(val: Optional[float]) -> str:
    """Formata número em notação científica enxuta pronta para colar no ZView 2.9c (sempre ponto)."""
    if val is None or not np.isfinite(val):
        return "0"
    if val == 0.0:
        return "0"
    if 0.01 <= abs(val) < 10000.0:
        return f"{val:.5g}"
    return f"{val:.4e}"


def _formatador_linear_origin(usar_virgula: bool):
    """Evita artefatos como '-0' ou '1e-16' nos ticks lineares e aplica separador decimal."""
    def _fmt(x, pos):
        if abs(x) < 1e-12:
            x = 0.0
        s = f"{x:g}"
        if s == "-0":
            s = "0"
        return s.replace(".", ",") if usar_virgula else s

    return ticker.FuncFormatter(_fmt)


def aplicar_estilo_origin(
    ax: plt.Axes,
    cfg: dict,
    log_x: bool = False,
    log_y: bool = False,
    top_spine: bool = True,
    right_spine: bool = True,
) -> None:
    """Aplica o padrão visual clássico do OriginPro em um eixo Matplotlib."""
    plt.rcParams["font.family"] = cfg.get("font_family", "sans-serif")
    plt.rcParams["mathtext.fontset"] = "dejavusans" if cfg.get("font_family") == "sans-serif" else "stix"

    spine_lw = float(cfg.get("spine_lw", 1.1))
    tick_dir = cfg.get("tick_dir", "in")
    ticks_tr = bool(cfg.get("ticks_top_right", True))
    font_ticks = int(cfg.get("font_ticks", 10))
    usar_virgula = bool(cfg.get("usar_virgula", True))

    ax.tick_params(
        which="both",
        direction=tick_dir,
        top=ticks_tr and top_spine,
        right=ticks_tr and right_spine,
        labelsize=font_ticks,
        colors="black",
    )
    ax.tick_params(which="major", length=6.0, width=spine_lw)
    ax.tick_params(which="minor", length=3.2, width=max(0.7, spine_lw * 0.8))

    if cfg.get("minor_ticks", True):
        if not log_x:
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        if not log_y:
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    for name, spine in ax.spines.items():
        spine.set_linewidth(spine_lw)
        spine.set_color("black")
        if name == "top" and not top_spine:
            spine.set_visible(False)
        if name == "right" and not right_spine:
            spine.set_visible(False)

    grid_style = cfg.get("grid_style", "Desativada (Padrão Origin)")
    if grid_style == "Tracejada suave":
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45, color="#888888")
    elif grid_style == "Sólida fina":
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.30, color="#999999")
    else:
        ax.grid(False)

    if not log_x:
        ax.xaxis.set_major_formatter(_formatador_linear_origin(usar_virgula))
    if not log_y:
        ax.yaxis.set_major_formatter(_formatador_linear_origin(usar_virgula))


def formatar_eixo_freq_bode(
    ax: plt.Axes,
    usar_virgula: bool = True,
    notacao_potencia: bool = False,
) -> None:
    """Formata o eixo logarítmico de frequência para Diagramas de Bode."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=14))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=120)
    )

    if not notacao_potencia:
        def _fmt_freq(val, pos):
            if val <= 0:
                return ""
            exp = np.log10(val)
            if abs(exp - round(exp)) > 1e-6:
                return ""
            if val >= 1.0:
                return f"{int(round(val))}"
            s = f"{val:g}"
            return s.replace(".", ",") if usar_virgula else s

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_freq))
    else:
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))


def botao_download_figura(
    fig: plt.Figure,
    nome_base: str,
    cfg: dict,
    key_suffix: str,
) -> None:
    """Renderiza botão de exportação da figura no formato e DPI globais selecionados."""
    fmt = str(cfg.get("export_fmt", "PNG")).lower()
    dpi = int(cfg.get("export_dpi", 300))
    transp = bool(cfg.get("transparent_bg", False))

    mime_map = {
        "png": "image/png",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
        "tiff": "image/tiff",
    }
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transp,
    )
    nome_limpo = re.sub(r"[^\w\-]+", "_", nome_base).strip("_") or "grafico"
    st.download_button(
        label=f"📥 Baixar Figura ({fmt.upper()} • {dpi} DPI)",
        data=buf.getvalue(),
        file_name=f"{nome_limpo}.{fmt}",
        mime=mime_map.get(fmt, "image/png"),
        key=f"dl_fig_{key_suffix}_{nome_limpo}",
        use_container_width=True,
    )


def botao_download_csv(
    df: pd.DataFrame,
    nome_base: str,
    cfg: dict,
    key_suffix: str,
    label: str = "📊 Baixar Tabela Consolidada (.CSV)",
) -> None:
    """Exporta DataFrame respeitando padrão CSV Internacional (, / .) ou Excel Brasil (; / , com UTF-8 BOM)."""
    csv_br = bool(cfg.get("csv_br", False))
    sep = ";" if csv_br else ","
    dec = "," if csv_br else "."
    enc = "utf-8-sig" if csv_br else "utf-8"

    conteudo = df.to_csv(index=False, sep=sep, decimal=dec).encode(enc)
    nome_limpo = re.sub(r"[^\w\-]+", "_", nome_base).strip("_") or "dados"
    st.download_button(
        label=label,
        data=conteudo,
        file_name=f"{nome_limpo}.csv",
        mime="text/csv",
        key=f"dl_csv_{key_suffix}_{nome_limpo}",
        use_container_width=True,
    )
