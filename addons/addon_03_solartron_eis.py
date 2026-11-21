# -*- coding: utf-8 -*-
"""
addons/addon_03_solartron_eis.py
Módulo 3: Espectroscopia de Impedância Eletroquímica (EIS — Solartron Bode & Nyquist)
  - Integrado ao extrair_solartron_eis e ao painel retrátil render_painel_override_parser
  - Sobreposição de múltiplos espectros (ex.: Antes, Exposto à Umidade, Após Exposição)
  - Diagrama de Bode duplo (Z' cheio e -Z'' vazado ou |Z| e -Fase) e Nyquist customizável
  - Compartilhamento automático em st.session_state para pré-ajuste direto no Preditor ZView
"""

import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import streamlit as st

from core.parsers import (
    exibir_diagnostico_leitura,
    extrair_solartron_eis,
    render_painel_override_parser,
)
from core.plotting import (
    aplicar_estilo_origin,
    botao_download_csv,
    botao_download_figura,
    fmt_num,
    formatar_eixo_freq_bode,
)
from core import resultados as registro_central

TITLE = "3. Gráficos EIS (Bode & Nyquist)"
ICON = "📈"
ORDER = 30


# ==============================================================================
# PUBLICAÇÃO DOS ESPECTROS PARA O ADD-ON 04 (sincronização compatível com @st.fragment)
# ==============================================================================
_CHAVE_MAPA_ABA3 = "eis_carregados_aba3"
_CHAVE_ASSINATURA_ABA3 = "eis_aba3_assinatura"
_CHAVE_VERSAO_ABA3 = "eis_aba3_versao"


def _digest_df(df: pd.DataFrame) -> str:
    """Impressão digital barata do conteúdo de um espectro (detecta troca de dados com mesmo nome/shape)."""
    try:
        bruto = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
        return hashlib.md5(bruto).hexdigest()
    except Exception:
        return f"shape={tuple(df.shape)}"


def _publicar_para_aba4(mapa_ref: Dict[str, pd.DataFrame]) -> None:
    """
    Publica em st.session_state o conjunto de espectros desta aba para o Preditor ZView (Add-on 04).

    Como cada aba roda em seu próprio fragmento, uma alteração aqui NÃO reexecuta a Aba 4.
    Por isso, quando o conjunto publicado muda de fato (arquivo novo/removido, legenda renomeada,
    conteúdo diferente), grava uma nova versão e pede UM rerun global, garantindo que a Aba 4
    leia a lista atualizada. Ajustes puramente visuais (cor, marcador, janela de frequência)
    não alteram a assinatura e não disparam rerun global.
    """
    assin_nova = tuple((k, tuple(df.shape), _digest_df(df)) for k, df in mapa_ref.items())
    assin_ant = st.session_state.get(_CHAVE_ASSINATURA_ABA3)
    if assin_ant == assin_nova:
        return

    st.session_state[_CHAVE_MAPA_ABA3] = {k: df.copy() for k, df in mapa_ref.items()}
    st.session_state[_CHAVE_ASSINATURA_ABA3] = assin_nova
    st.session_state[_CHAVE_VERSAO_ABA3] = int(st.session_state.get(_CHAVE_VERSAO_ABA3, 0)) + 1

    # Primeira publicação da sessão (assin_ant is None): a Aba 4 ainda não renderizou, não precisa de rerun.
    if assin_ant is not None:
        try:
            st.rerun(scope="app")
        except TypeError:  # Streamlit < 1.37 (sem fragmentos): rerun simples
            st.rerun()


# ==============================================================================
# REGISTRO CENTRAL DE RESULTADOS (core/resultados.py)
# ==============================================================================
_ORIGEM_REGISTRO = "addon_03_solartron_eis"
_UNIDADES_EIS = {
    "Z_real_alta_freq": "Ω",
    "Z_real_max": "Ω",
    "minus_Z_imag_pico": "Ω",
    "f_pico": "Hz",
}


def _nome_amostra_registro(curva: Dict[str, Any], contagem_legendas: Counter) -> str:
    """Legenda da curva; se duas curvas ativas tiverem a mesma legenda, acrescenta o arquivo para não sobrescrever."""
    if contagem_legendas[curva["label"]] > 1:
        return f"{curva['label']} [{curva['filename']}]"
    return curva["label"]


def _registrar_resultados_eis(lote: Dict[str, Tuple[Dict[str, float], Dict[str, Any]]]) -> int:
    """
    Publica no registro central as métricas espectrais (em Ω e Hz, sem a escala de exibição) de cada curva ativa
    e remove as curvas desta aba que deixaram de existir (arquivo removido, curva desativada, legenda renomeada).
    Nunca interrompe o add-on: qualquer falha vira um aviso discreto. Retorna quantas curvas foram registradas.
    """
    try:
        for amostra, (metricas, meta) in lote.items():
            registro_central.registrar(
                amostra,
                registro_central.TEC_EIS,
                metricas,
                unidades=_UNIDADES_EIS,
                meta=meta,
                origem=_ORIGEM_REGISTRO,
            )
        registro_central.sincronizar(registro_central.TEC_EIS, _ORIGEM_REGISTRO, list(lote.keys()))
        return len(lote)
    except Exception as exc:
        st.caption(f"⚠️ Registro central de resultados indisponível: {exc}")
        return 0


def render(cfg_plot: dict) -> None:
    usar_virgula = bool(cfg_plot.get("usar_virgula", True))

    st.subheader("Diagramas de Bode e Nyquist Multi-Curvas (Padrão Solartron / Origin)")

    files_eis_plot = st.file_uploader(
        "Carregar um ou mais arquivos de Impedância da Solartron (.z, .txt, .csv, .dat) — ex.: Antes, Exposto em umidade, Após exposição",
        type=["z", "dat", "txt", "csv"],
        accept_multiple_files=True,
        key="eis_uploader_multi",
    )

    opcoes_parser, slot_diag = render_painel_override_parser("eis")

    cores_padrao_eis = ["#8C8C8C", "#29D9EB", "#F79AD3", "#1F618D", "#D35400", "#1E8449", "#7D3C98"]
    marcadores_padrao = ["s", "o", "^", "D", "v", "P", "X"]

    with st.expander("🎛️ Unidades, Formato de Bode/Nyquist, Frequências e Rótulos", expanded=True):
        e1, e2, e3, e4 = st.columns(4)
        unid_z_eis = e1.selectbox("Unidade de Impedância", ["kΩ", "Ω", "MΩ"], index=0, key="eis_unid_z")
        modo_bode_sel = e2.selectbox(
            "Formato do Diagrama de Bode",
            [
                "Z' (esq) e -Z'' (dir) — Padrão Lab",
                "|Z| (esq) e Fase -θ (dir) — Bode Clássico",
                "log₁₀|Z| (esq) e Fase -θ (dir)",
            ],
            index=0,
            key="eis_modo_bode",
        )
        aspecto_nyq = e3.selectbox(
            "Proporção de Eixos no Nyquist",
            ["Livre (Estilo Origin)", "Isométrica 1:1 (Ortonormal)"],
            index=0,
            key="eis_aspect_nyq",
        )
        inverter_zim_manual = e4.selectbox(
            "Convenção de Sinal Z'' no Arquivo",
            ["Detecção Automática", "Forçar inversão (-Z'')", "Manter bruto"],
            index=0,
            key="eis_inv_zim",
        )

        e5, e6, e7, e8 = st.columns(4)
        titulo_bode = e5.text_input("Título do Diagrama de Bode", value="S1 - Diagrama de Bode", key="eis_tit_b")
        titulo_nyq = e6.text_input("Título do Diagrama de Nyquist", value="S1 - Diagrama de Nyquist", key="eis_tit_n")
        anotacao_nyq = e7.text_input(
            "Anotação interna no Nyquist (use \\n p/ quebrar linha)",
            value="",
            placeholder="Ex.: S1 - 342 kΩ\\nS2 - 624 kΩ",
            key="eis_anot_nyq",
        )
        pos_leg_bode = e8.selectbox(
            "Posição da Legenda (Bode)",
            ["center left", "upper right", "upper left", "center right", "best"],
            index=0,
            key="eis_pos_leg_b",
        )

        e9, e10, e11, e12 = st.columns(4)
        notacao_pot_freq = e9.checkbox("Frequência em potências 10ⁿ no Bode", value=False, key="eis_pot10")
        conectar_linhas_nyq = e10.checkbox("Conectar pontos com linha fina no Nyquist", value=False, key="eis_lin_nyq")
        marcar_decadas_nyq = e11.checkbox("Destacar décadas de frequência (10ⁿ Hz) no Nyquist", value=False, key="eis_dec_nyq")
        permitir_neg_zim = e12.checkbox("Exibir cauda indutiva (-Z'' < 0) se existir", value=False, key="eis_show_ind")

        st.markdown("---")
        es1, es2, es3, es4 = st.columns([1.4, 1.0, 1.0, 1.0])
        modo_escala_z = es1.selectbox(
            "Escala dos Eixos de Impedância (Z' e -Z'')",
            [
                "Igualada Automática (Z' e -Z'' na mesma escala)",
                "Manual Única (definir máx. para Z' e -Z'')",
                "Manual Separada (Bode e Nyquist)",
                "Automática Independente",
            ],
            index=0,
            key="eis_modo_escala_z",
        )
        escala_z_min_manual = es2.number_input(
            f"Mínimo Z' / -Z'' ({unid_z_eis})",
            value=0.0,
            step=10.0,
            format="%.2f",
            disabled=modo_escala_z.startswith("Igualada") or modo_escala_z.startswith("Automática"),
            key="eis_z_min_man",
        )
        escala_z_max_manual = es3.number_input(
            f"Máximo Z' / -Z'' ({unid_z_eis})",
            value=800.0,
            min_value=0.0001,
            step=50.0,
            format="%.2f",
            disabled=modo_escala_z.startswith("Igualada") or modo_escala_z.startswith("Automática"),
            key="eis_z_max_man",
        )
        escala_nyq_max_manual = es4.number_input(
            f"Máximo Nyquist ({unid_z_eis}) [Separada]",
            value=800.0,
            min_value=0.0001,
            step=50.0,
            format="%.2f",
            disabled=(modo_escala_z != "Manual Separada (Bode e Nyquist)"),
            key="eis_nyq_max_man",
        )

    if not files_eis_plot:
        _registrar_resultados_eis({})
        _publicar_para_aba4({})
        return

    if inverter_zim_manual.startswith("Detec"):
        inv_flag: Optional[bool] = None
    elif "Forçar" in inverter_zim_manual:
        inv_flag = True
    else:
        inv_flag = False

    curvas_lidas = []
    meta_primeiro = None
    nome_primeiro = ""

    for idx_f, f_obj in enumerate(files_eis_plot):
        try:
            df_s, meta_s = extrair_solartron_eis(
                f_obj.getvalue(),
                inverter_sinal_zim=inv_flag,
                opcoes=opcoes_parser,
            )
            if idx_f == 0:
                meta_primeiro = meta_s
                nome_primeiro = f_obj.name
            curvas_lidas.append({"filename": f_obj.name, "df": df_s})
        except Exception as exc:
            st.error(f"Erro ao ler `{f_obj.name}`: {exc}")

    if meta_primeiro is not None:
        exibir_diagnostico_leitura(slot_diag, meta_primeiro, nome_primeiro)

    if not curvas_lidas:
        _registrar_resultados_eis({})
        _publicar_para_aba4({})
        return

    # Filtro global de faixa de frequência
    f_min_all = float(min(c["df"]["Freq"].min() for c in curvas_lidas))
    f_max_all = float(max(c["df"]["Freq"].max() for c in curvas_lidas))
    log_min = float(np.floor(np.log10(max(1e-6, f_min_all)) * 10.0) / 10.0)
    log_max = float(np.ceil(np.log10(max(1e-5, f_max_all)) * 10.0) / 10.0)

    if log_max > log_min:
        faixa_log = st.slider(
            "Recortar Janela de Frequência Exibida — log₁₀(f / Hz)",
            min_value=log_min,
            max_value=log_max,
            value=(log_min, log_max),
            step=0.1,
            key="eis_freq_slider",
        )
    else:
        faixa_log = (log_min, log_max)

    st.markdown("##### Personalização Individual de Cada Curva Sobreposta")
    cols_cfg_curvas = st.columns(min(len(curvas_lidas), 4))
    curvas_eis: List[Dict] = []

    for idx_f, item in enumerate(curvas_lidas):
        col_ui = cols_cfg_curvas[idx_f % len(cols_cfg_curvas)]
        nome_base_sug = re.sub(r"\.(z|dat|txt|csv)$", "", item["filename"], flags=re.IGNORECASE)
        with col_ui:
            ativa_c = st.checkbox(f"Exibir #{idx_f + 1}", value=True, key=f"eis_on_{idx_f}")
            lbl_c = st.text_input(f"Legenda #{idx_f + 1}", value=nome_base_sug, key=f"eis_lbl_{idx_f}")
            cor_c = st.color_picker(
                f"Cor #{idx_f + 1}",
                value=cores_padrao_eis[idx_f % len(cores_padrao_eis)],
                key=f"eis_cor_{idx_f}",
            )
            mk_c = st.selectbox(
                f"Marcador #{idx_f + 1}",
                options=marcadores_padrao,
                index=idx_f % len(marcadores_padrao),
                key=f"eis_mk_{idx_f}",
            )

        df_f = item["df"]
        mask_f = (np.log10(df_f["Freq"]) >= faixa_log[0] - 1e-6) & (np.log10(df_f["Freq"]) <= faixa_log[1] + 1e-6)
        df_filtrado = df_f.loc[mask_f].reset_index(drop=True)

        if not df_filtrado.empty:
            curvas_eis.append(
                {
                    "ativa": ativa_c,
                    "label": lbl_c.strip() or nome_base_sug,
                    "color": cor_c,
                    "marker": mk_c,
                    "df": df_filtrado,
                    "df_full": item["df"],
                    "filename": item["filename"],
                }
            )

    # Salvar todos os espectros carregados em st.session_state para integração imediata com o Add-on 04 (Preditor ZView)
    mapa_compartilhado = {}
    for c in curvas_eis:
        chave_unica = f"{c['label']} [{c['filename']}]" if c["label"] != c["filename"] else c["filename"]
        mapa_compartilhado[chave_unica] = c["df_full"]
    _publicar_para_aba4(mapa_compartilhado)

    curvas_ativas = [c for c in curvas_eis if c["ativa"]]
    if not curvas_ativas:
        _registrar_resultados_eis({})
        st.info("Marque pelo menos uma curva como ativa para visualizar os diagramas.")
        return

    div_z = {"Ω": 1.0, "kΩ": 1e3, "MΩ": 1e6}[unid_z_eis]

    # Calcular limites compartilhados conforme o modo escolhido
    todos_zre_all = np.concatenate([c["df"]["Z_real"].values / div_z for c in curvas_ativas])
    todos_mzi_all = np.concatenate([c["df"]["minus_Z_imag"].values / div_z for c in curvas_ativas])
    max_auto_comum = max(float(np.max(todos_zre_all)), float(np.max(todos_mzi_all)), 0.0)
    min_auto_comum = min(float(np.min(todos_zre_all)), float(np.min(todos_mzi_all)), 0.0)

    if modo_escala_z.startswith("Igualada"):
        if permitir_neg_zim:
            span_c = max(max_auto_comum - min_auto_comum, 1e-12)
            lim_bode_z = (min_auto_comum - 0.04 * span_c, max_auto_comum + 0.06 * span_c)
        else:
            lim_bode_z = (0.0, max(max_auto_comum * 1.06, 1e-12))
        lim_nyq_z = lim_bode_z
    elif modo_escala_z.startswith("Manual Única"):
        z_hi = max(float(escala_z_max_manual), float(escala_z_min_manual) + 1e-9)
        lim_bode_z = (float(escala_z_min_manual), z_hi)
        lim_nyq_z = lim_bode_z
    elif modo_escala_z == "Manual Separada (Bode e Nyquist)":
        z_hi_b = max(float(escala_z_max_manual), float(escala_z_min_manual) + 1e-9)
        z_hi_n = max(float(escala_nyq_max_manual), float(escala_z_min_manual) + 1e-9)
        lim_bode_z = (float(escala_z_min_manual), z_hi_b)
        lim_nyq_z = (float(escala_z_min_manual), z_hi_n)
    else:
        lim_bode_z = None
        lim_nyq_z = None

    # ==========================================================================
    # 1. CONSTRUÇÃO DO DIAGRAMA DE BODE
    # ==========================================================================
    fig_bode, ax_b1 = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)
    ax_b2 = ax_b1.twinx()

    handles_esq, handles_dir = [], []
    eh_bode_re_im = modo_bode_sel.startswith("Z'")
    eh_log_mod = modo_bode_sel.startswith("log₁₀")

    for c in curvas_ativas:
        f_arr = c["df"]["Freq"].values
        if eh_bode_re_im:
            y1 = c["df"]["Z_real"].values / div_z
            y2 = c["df"]["minus_Z_imag"].values / div_z
            (l1,) = ax_b1.plot(
                f_arr,
                y1,
                marker=c["marker"],
                color=c["color"],
                markerfacecolor=c["color"],
                markeredgecolor=c["color"],
                markersize=5.0,
                linewidth=1.2,
                label=c["label"],
            )
            (l2,) = ax_b2.plot(
                f_arr,
                y2,
                marker=c["marker"],
                color=c["color"],
                markerfacecolor="white",
                markeredgecolor=c["color"],
                markeredgewidth=1.1,
                markersize=5.0,
                linewidth=1.1,
                label=c["label"],
            )
        else:
            y1 = c["df"]["Z_mod"].values / div_z
            y2 = -c["df"]["Phase_deg"].values
            (l1,) = ax_b1.plot(
                f_arr,
                y1,
                marker=c["marker"],
                color=c["color"],
                markerfacecolor=c["color"],
                markeredgecolor=c["color"],
                markersize=4.8,
                linewidth=1.2,
                label=c["label"],
            )
            (l2,) = ax_b2.plot(
                f_arr,
                y2,
                marker=c["marker"],
                color=c["color"],
                markerfacecolor="white",
                markeredgecolor=c["color"],
                markeredgewidth=1.1,
                markersize=4.8,
                linestyle="--",
                linewidth=1.1,
                label=c["label"],
            )
        handles_esq.append(l1)
        handles_dir.append(l2)

    formatar_eixo_freq_bode(ax_b1, usar_virgula=usar_virgula, notacao_potencia=notacao_pot_freq)
    if eh_log_mod:
        ax_b1.set_yscale("log")

    ax_b1.set_xlabel("Frequência (Hz)", fontsize=cfg_plot["font_labels"], labelpad=6)

    if eh_bode_re_im:
        ax_b1.set_ylabel(f"Z' ({unid_z_eis})", fontsize=cfg_plot["font_labels"], labelpad=6)
        ax_b2.set_ylabel(f"-Z'' ({unid_z_eis})", fontsize=cfg_plot["font_labels"], labelpad=8)
        if lim_bode_z is not None:
            ax_b1.set_ylim(lim_bode_z)
            ax_b2.set_ylim(lim_bode_z)
        else:
            ax_b1.set_ylim(bottom=0)
            if not permitir_neg_zim:
                ax_b2.set_ylim(bottom=0)
        titulo_grupo_1, titulo_grupo_2 = "Z'", "-Z''"
    else:
        ax_b1.set_ylabel(f"|Z| ({unid_z_eis})", fontsize=cfg_plot["font_labels"], labelpad=6)
        ax_b2.set_ylabel("-Fase, -θ (°)", fontsize=cfg_plot["font_labels"], labelpad=8)
        if not eh_log_mod:
            if lim_bode_z is not None and not modo_escala_z.startswith("Igualada"):
                ax_b1.set_ylim(lim_bode_z)
            else:
                ax_b1.set_ylim(bottom=0)
        titulo_grupo_1, titulo_grupo_2 = "|Z|", "-Fase (°)"

    if titulo_bode.strip():
        ax_b1.set_title(titulo_bode, fontsize=cfg_plot["font_labels"], pad=8)

    itens_leg = [Line2D([0], [0], color="none", label=titulo_grupo_1)]
    itens_leg.extend(handles_esq)
    itens_leg.append(Line2D([0], [0], color="none", label=titulo_grupo_2))
    itens_leg.extend(handles_dir)

    ax_b1.legend(
        handles=itens_leg,
        loc=pos_leg_bode,
        frameon=False,
        fontsize=max(7, cfg_plot["font_legend"] - 1),
    )

    aplicar_estilo_origin(ax_b1, cfg_plot, log_x=True, log_y=eh_log_mod, right_spine=False)
    aplicar_estilo_origin(ax_b2, cfg_plot, log_x=True, log_y=False, top_spine=False)
    ax_b2.grid(False)
    plt.tight_layout()

    # ==========================================================================
    # 2. CONSTRUÇÃO DO DIAGRAMA DE NYQUIST
    # ==========================================================================
    fig_nyq, ax_nyq = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)

    for c in curvas_ativas:
        zr_p = c["df"]["Z_real"].values / div_z
        mzi_p = c["df"]["minus_Z_imag"].values / div_z
        f_p = c["df"]["Freq"].values

        ax_nyq.plot(
            zr_p,
            mzi_p,
            marker=c["marker"],
            linestyle="-" if conectar_linhas_nyq else "none",
            linewidth=1.0 if conectar_linhas_nyq else 0.0,
            color=c["color"],
            markersize=5.2,
            label=c["label"],
            zorder=3,
        )

        if marcar_decadas_nyq and len(f_p) >= 3:
            decs_int = range(int(np.ceil(np.log10(np.min(f_p)))), int(np.floor(np.log10(np.max(f_p)))) + 1)
            for d_exp in decs_int:
                alvo_f = 10.0 ** d_exp
                idx_d = int(np.argmin(np.abs(np.log10(f_p) - d_exp)))
                if abs(np.log10(f_p[idx_d]) - d_exp) <= 0.15:
                    ax_nyq.plot(
                        zr_p[idx_d],
                        mzi_p[idx_d],
                        marker="o",
                        markersize=6.2,
                        markerfacecolor="none",
                        markeredgecolor="black",
                        markeredgewidth=1.1,
                        zorder=5,
                    )
                    ax_nyq.annotate(
                        f"10$^{{{d_exp}}}$" if d_exp >= 3 or d_exp < 0 else f"{int(alvo_f)}",
                        (zr_p[idx_d], mzi_p[idx_d]),
                        textcoords="offset points",
                        xytext=(4, 5),
                        fontsize=max(7, cfg_plot["font_ticks"] - 2),
                        color="#111111",
                        zorder=6,
                    )

    ax_nyq.set_xlabel(f"Z' ({unid_z_eis})", fontsize=cfg_plot["font_labels"], labelpad=6)
    ax_nyq.set_ylabel(f"-Z'' ({unid_z_eis})", fontsize=cfg_plot["font_labels"], labelpad=6)
    if lim_nyq_z is not None:
        ax_nyq.set_xlim(lim_nyq_z)
        ax_nyq.set_ylim(lim_nyq_z)
        if lim_nyq_z[0] < 0:
            ax_nyq.axhline(0, color="black", lw=0.75, ls="--", zorder=1)
    else:
        ax_nyq.set_xlim(left=0)
        if not permitir_neg_zim:
            ax_nyq.set_ylim(bottom=0)
        else:
            ax_nyq.axhline(0, color="black", lw=0.75, ls="--", zorder=1)

    if aspecto_nyq.startswith("Isométrica"):
        ax_nyq.set_aspect("equal", adjustable="box")

    if titulo_nyq.strip():
        ax_nyq.set_title(titulo_nyq, fontsize=cfg_plot["font_labels"], pad=8)

    if anotacao_nyq.strip():
        ax_nyq.text(
            0.62,
            0.48,
            anotacao_nyq.replace("\\n", "\n"),
            transform=ax_nyq.transAxes,
            fontsize=cfg_plot["font_labels"],
        )

    ax_nyq.legend(loc="upper right", frameon=False, fontsize=cfg_plot["font_legend"])
    aplicar_estilo_origin(ax_nyq, cfg_plot)
    plt.tight_layout()

    col_b, col_n = st.columns(2)
    with col_b:
        st.pyplot(fig_bode)
        botao_download_figura(fig_bode, titulo_bode or "Diagrama_Bode", cfg_plot, "eis_bode")
    with col_n:
        st.pyplot(fig_nyq)
        botao_download_figura(fig_nyq, titulo_nyq or "Diagrama_Nyquist", cfg_plot, "eis_nyq")

    plt.close(fig_bode)
    plt.close(fig_nyq)

    # ==========================================================================
    # 3. RESUMO QUANTITATIVO RÁPIDO E EXPORTAÇÃO CONSOLIDADA (.CSV)
    # ==========================================================================
    resumo_eis = []
    blocos_csv = []
    lote_resultados: Dict[str, Tuple[Dict[str, float], Dict[str, Any]]] = {}
    contagem_legendas = Counter(c["label"] for c in curvas_ativas)

    for c in curvas_ativas:
        df_c = c["df"]
        idx_pk = int(df_c["minus_Z_imag"].idxmax())
        zre_hf = float(df_c["Z_real"].iloc[0]) / div_z
        zre_lf = float(df_c["Z_real"].max()) / div_z
        mzi_max = float(df_c.loc[idx_pk, "minus_Z_imag"]) / div_z
        f_pk = float(df_c.loc[idx_pk, "Freq"])

        lote_resultados[_nome_amostra_registro(c, contagem_legendas)] = (
            {
                "Z_real_alta_freq": float(df_c["Z_real"].iloc[0]),
                "Z_real_max": float(df_c["Z_real"].max()),
                "minus_Z_imag_pico": float(df_c.loc[idx_pk, "minus_Z_imag"]),
                "f_pico": f_pk,
            },
            {
                "arquivo": c["filename"],
                "f_min_Hz": float(df_c["Freq"].min()),
                "f_max_Hz": float(df_c["Freq"].max()),
                "n_pontos": int(len(df_c)),
            },
        )

        resumo_eis.append(
            {
                "Curva": c["label"],
                "Arquivo": c["filename"],
                f"Z' Alta Freq ({unid_z_eis})": fmt_num(zre_hf, 2, usar_virgula),
                f"Z' Máx Baixa Freq ({unid_z_eis})": fmt_num(zre_lf, 2, usar_virgula),
                f"-Z'' Pico ({unid_z_eis})": fmt_num(mzi_max, 2, usar_virgula),
                "f_pico Principal (Hz)": f"{f_pk:.2f}" if f_pk < 1000 else f"{f_pk:.3e}",
            }
        )

        df_exp_c = pd.DataFrame(
            {
                "Amostra": c["label"],
                "Freq_Hz": df_c["Freq"].values,
                f"Z_real_{unid_z_eis}": df_c["Z_real"].values / div_z,
                f"minus_Z_imag_{unid_z_eis}": df_c["minus_Z_imag"].values / div_z,
                f"Z_mod_{unid_z_eis}": df_c["Z_mod"].values / div_z,
                "Phase_deg": df_c["Phase_deg"].values,
            }
        )
        blocos_csv.append(df_exp_c)

    n_registradas = _registrar_resultados_eis(lote_resultados)

    with st.expander("📋 Resumo Espectral Rápido & Exportação Consolidada (.CSV)", expanded=False):
        st.caption(
            "💡 Todos os espectros carregados aqui já estão disponíveis automaticamente na aba "
            "**4. Preditor Circuito ZView** para modelagem de circuitos equivalentes."
        )
        if n_registradas:
            st.caption(f"📌 {n_registradas} curva(s) registrada(s) no registro central de resultados (técnica EIS).")
        st.dataframe(pd.DataFrame(resumo_eis), use_container_width=True, hide_index=True)
        if blocos_csv:
            df_tudo_eis = pd.concat(blocos_csv, ignore_index=True)
            botao_download_csv(df_tudo_eis, "Espectros_EIS_Consolidados", cfg_plot, "eis_all")
