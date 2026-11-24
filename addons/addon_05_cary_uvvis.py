# -*- coding: utf-8 -*-
"""
addons/addon_05_cary_uvvis.py
Módulo 5: Espectroscopia de Absorção UV-Vis e Tauc Plot (Cary 100 WinUV — P3HT & P3DT)
  - Leitura nativa de planilhas multi-amostra pareadas do Cary WinUV, matrizes e arquivos individuais
  - Extração automática de λ_máx, razão vibrônica A0-0 / A0-1 e Largura de Banda de Éxcitons Livres W
    (Modelo de Agregados-H Fracamente Acoplados de Spano para politiofenos regiorregulares)
  - Painel dedicado de Tauc Plot [(αhν)^(1/n) vs. hν] com ajuste automático ou manual da reta tangente
    (e opção de interseção com linha de base sub-gap de Makuła) para extrair o Bandgap Óptico Eg (eV)
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import resultados as registro_central
from core.parsers import extrair_cary_uvvis
from core.plotting import (
    botao_download_csv,
    botao_download_figura,
    fmt_num,
)
from core.ui import (
    ColetorResultados,
    cabecalho_secao,
    legenda_origin,
    ler_arquivos,
    msg_aviso,
    msg_sem_dados,
    nova_figura_origin,
    painel_entrada,
    sessao_resultados,
)

TITLE = "5. Espectroscopia UV-Vis & Tauc"
ICON = "🌈"
ORDER = 50

HC_EV_NM = 1239.841984  # Constante h·c em eV·nm

# Registro central de resultados (core/resultados.py): a amostra é a legenda do espectro (nome editável na tela).
_ORIGEM_REGISTRO = "addon_05_cary_uvvis"
_UNIDADES_UVVIS = {"lambda_max": "nm", "E_max": "eV", "W_Spano": "meV", "Eg": "eV", "lambda_onset": "nm"}


def _calcular_w_spano_mev(razao_00_01: float, ep_ev: float = 0.180) -> Tuple[float, str]:
    """
    Calcula a largura de banda de éxcitons livres W (em meV) pelo modelo de Spano
    (Clark, Silva, Friend, Spano, Phys. Rev. Lett. 98, 206406, 2007; fator de Huang-Rhys S = 1,0):
        A_0-0 / A_0-1 = [ (1 - 0,24 * W / Ep) / (1 + 0,073 * W / Ep) ]^2
    Retorna (W_meV, status_regime).
    """
    if not np.isfinite(razao_00_01) or razao_00_01 <= 0.0:
        return np.nan, "Sinal insuficiente"
    if razao_00_01 >= 1.0:
        return np.nan, "Regime J-dominante (A₀₋₀ ≥ A₀₋₁)"

    r_sqrt = float(np.sqrt(razao_00_01))
    w_ev = ep_ev * (1.0 - r_sqrt) / (0.24 + 0.073 * r_sqrt)
    return float(w_ev * 1000.0), "Agregado-H (Spano)"


def _encontrar_janela_linear_tauc(
    e_ev: np.ndarray,
    y_tauc: np.ndarray,
    e_min_busca: float,
    e_max_busca: float,
    largura_ev: float = 0.10,
) -> Tuple[float, float]:
    """
    Localiza automaticamente o segmento retilíneo mais íngreme (ponto de inflexão da borda de absorção)
    dentro de [e_min_busca, e_max_busca] maximizando inclinação positiva com alta linearidade (R²).
    """
    mask_b = (e_ev >= e_min_busca) & (e_ev <= e_max_busca)
    eb = e_ev[mask_b]
    yb = y_tauc[mask_b]
    if len(eb) < 6:
        return e_min_busca, e_max_busca

    y_amp = float(np.max(yb) - np.min(yb))
    y_piso = float(np.min(yb)) + 0.08 * y_amp
    y_teto = float(np.min(yb)) + 0.90 * y_amp

    melhor_score = -1.0
    melhor_par = (float(eb[0]), float(eb[-1]))

    e_ini_cands = np.linspace(float(eb[0]), max(float(eb[0]), float(eb[-1]) - largura_ev), 45)
    for e_a in e_ini_cands:
        e_b = e_a + largura_ev
        m_seg = (eb >= e_a) & (eb <= e_b)
        if int(np.sum(m_seg)) < 5:
            continue
        es = eb[m_seg]
        ys = yb[m_seg]
        y_med = float(np.mean(ys))
        if not (y_piso <= y_med <= y_teto):
            continue

        slope, intercept = np.polyfit(es, ys, 1)
        if slope <= 0:
            continue

        y_hat = slope * es + intercept
        ss_res = float(np.sum((ys - y_hat) ** 2))
        ss_tot = float(np.sum((ys - y_med) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Priorizar alta linearidade (R² >= 0,985) combinada à máxima inclinação da borda fundamental
        penal_r2 = (r2 ** 8) if r2 > 0 else 0.0
        score = float(slope) * penal_r2
        if score > melhor_score:
            melhor_score = score
            melhor_par = (float(es[0]), float(es[-1]))

    return melhor_par


def _render_corpo(cfg_plot: dict, coletor: ColetorResultados) -> None:
    usar_virgula = bool(cfg_plot.get("usar_virgula", True))

    cabecalho_secao("Espectroscopia Óptica UV-Vis, Agregação de Spano (W) & Tauc Plot (Bandgap Eg)")

    entrada = painel_entrada(
        "uv",
        "Carregar espectros UV-Vis (.csv, .txt, .dat do Cary 100 WinUV — suporta planilha multi-amostra ou arquivos individuais)",
        tipos=("csv", "txt", "dat", "dsp"),
        multiplos=True,
        key_uploader="uv_uploader",
    )

    if entrada.vazia:
        msg_sem_dados("Carregue um ou mais espectros UV-Vis do Cary para iniciar a análise.")
        return

    catalogo_espectros: List[Dict[str, Any]] = []
    for lido in ler_arquivos(entrada, extrair_cary_uvvis, nome_kwarg="nome_arquivo"):
        lista_esp, _meta_uv = lido.resultado
        catalogo_espectros.extend(lista_esp)

    if not catalogo_espectros:
        return

    cores_uv = ["#111111", "#E60000", "#1F618D", "#1E8449", "#8E44AD", "#D35400", "#16A085", "#C0392B"]

    with st.expander("🎛️ Seleção de Amostras, Correção de Base, Bandas Vibrônicas e Modelo de Spano", expanded=True):
        nomes_originais = [
            f"{i + 1}. {e['nome']} ({e['origem']})" for i, e in enumerate(catalogo_espectros)
        ]
        indices_sel = st.multiselect(
            "Espectros ativos para visualização e análise:",
            options=list(range(len(catalogo_espectros))),
            default=list(range(len(catalogo_espectros))),
            format_func=lambda i: nomes_originais[i],
            key="uv_sel_specs",
        )

        u1, u2, u3, u4 = st.columns(4)
        wl_min, wl_max = u1.slider("Janela Espectral λ (nm)", 200, 1100, (350, 900), key="uv_wl_range")
        subtrair_offset = u2.checkbox(
            "Zerar espalhamento em λ longo (800–850 nm)",
            value=False,
            help="Subtrai a absorbância residual média na região transparente acima do bandgap antes de calcular A0-0/A0-1 e Tauc.",
            key="uv_sub_off",
        )
        norm_uv = u3.checkbox("Normalizar espectros pelo pico máximo (0 a 1)", value=False, key="uv_norm")
        marcar_vibronicas = u4.checkbox("Marcar linhas guias A₀₋₀ e A₀₋₁ no gráfico", value=True, key="uv_mk_vib")

        u5, u6, u7, u8 = st.columns(4)
        wl_00 = float(
            u5.number_input(
                "Posição A₀₋₀ (nm)",
                540.0,
                670.0,
                605.0,
                step=1.0,
                help="Transição vibrônica 0–0 associada a cadeias agregadas π-π (~600–610 nm em P3HT/P3DT).",
                key="uv_wl00",
            )
        )
        wl_01 = float(
            u6.number_input(
                "Posição A₀₋₁ (nm)",
                480.0,
                610.0,
                555.0,
                step=1.0,
                help="Transição vibrônica 0–1 (~550–560 nm em politiofenos regiorregulares).",
                key="uv_wl01",
            )
        )
        busca_local_vib = u7.selectbox(
            "Leitura de A₀₋₀ e A₀₋₁",
            ["Valor exato nos λ informados", "Máximo local (±6 nm) em torno de λ"],
            index=0,
            key="uv_modo_vib",
        )
        ep_spano_ev = float(
            u8.number_input(
                "Energia fonônica C=C Ep (eV)",
                0.140,
                0.220,
                0.180,
                step=0.005,
                format="%.3f",
                help="Modo de estiramento simétrico C=C aromático usado no modelo de Spano (padrão politiofenos = 0,180 eV).",
                key="uv_ep_ev",
            )
        )

        # Personalização rápida de rótulos e cores das curvas selecionadas
        if indices_sel:
            st.markdown("**Rótulos na Legenda e Cores Individuais:**")
            cols_pers = st.columns(min(len(indices_sel), 4))
            cfg_curvas_uv: Dict[int, Dict[str, str]] = {}
            for pos_i, idx_esp in enumerate(indices_sel):
                col_c = cols_pers[pos_i % len(cols_pers)]
                esp_item = catalogo_espectros[idx_esp]
                with col_c:
                    lbl_edit = st.text_input(
                        f"Legenda #{pos_i + 1}",
                        value=esp_item["nome"],
                        key=f"uv_lbl_{idx_esp}",
                    )
                    cor_edit = st.color_picker(
                        f"Cor #{pos_i + 1}",
                        value=cores_uv[pos_i % len(cores_uv)],
                        key=f"uv_cor_{idx_esp}",
                    )
                cfg_curvas_uv[idx_esp] = {"label": lbl_edit.strip() or esp_item["nome"], "color": cor_edit}

    if not indices_sel:
        msg_sem_dados("Selecione ao menos um espectro acima.")
        return

    # Processar espectros selecionados
    espectros_proc: List[Dict[str, Any]] = []
    resumo_uv = []

    for idx_esp in indices_sel:
        esp_raw = catalogo_espectros[idx_esp]
        df_full = esp_raw["df"].copy()

        # Subtração opcional de offset de espalhamento na cauda transparente (800-850 nm ou últimos 5% de λ)
        y_full = df_full["abs"].values.astype(float)
        wl_full = df_full["wl"].values.astype(float)
        if subtrair_offset:
            mask_tail = (wl_full >= 780.0) & (wl_full <= 880.0)
            if np.any(mask_tail):
                offset_val = float(np.median(y_full[mask_tail]))
            else:
                offset_val = float(np.min(y_full[-max(3, len(y_full) // 20):]))
            y_full = y_full - offset_val

        df_full["abs_corr"] = y_full
        df_win = df_full[(df_full["wl"] >= wl_min) & (df_full["wl"] <= wl_max)].sort_values("wl").reset_index(drop=True)
        if df_win.empty:
            continue

        x_w = df_win["wl"].values
        y_abs_real = df_win["abs_corr"].values.copy()

        if norm_uv:
            y_min_u = float(np.min(y_abs_real))
            y_max_u = float(np.max(y_abs_real))
            y_plot = (y_abs_real - y_min_u) / (y_max_u - y_min_u) if y_max_u > y_min_u else y_abs_real.copy()
        else:
            y_plot = y_abs_real.copy()

        # Extrair λ_máx, A0-0, A0-1 e largura de banda de éxcitons W de Spano sobre absorbância física real
        idx_max = int(np.argmax(y_abs_real))
        l_max = float(x_w[idx_max])
        e_pico_ev = HC_EV_NM / l_max if l_max > 0 else np.nan

        if busca_local_vib.startswith("Máximo local"):
            m00 = (x_w >= wl_00 - 6.0) & (x_w <= wl_00 + 6.0)
            m01 = (x_w >= wl_01 - 6.0) & (x_w <= wl_01 + 6.0)
            a_00 = float(np.max(y_abs_real[m00])) if np.any(m00) else float(np.interp(wl_00, x_w, y_abs_real))
            a_01 = float(np.max(y_abs_real[m01])) if np.any(m01) else float(np.interp(wl_01, x_w, y_abs_real))
        else:
            a_00 = float(np.interp(wl_00, x_w, y_abs_real))
            a_01 = float(np.interp(wl_01, x_w, y_abs_real))

        razao_vib = (a_00 / a_01) if a_01 > 1e-9 else np.nan
        w_spano_mev, status_spano = _calcular_w_spano_mev(razao_vib, ep_ev=ep_spano_ev)

        lbl_curva = cfg_curvas_uv[idx_esp]["label"]
        cor_curva = cfg_curvas_uv[idx_esp]["color"]

        espectros_proc.append(
            {
                "label": lbl_curva,
                "color": cor_curva,
                "wl": x_w,
                "abs_real": y_abs_real,
                "abs_plot": y_plot,
                "origem": esp_raw["origem"],
                "metricas_registro": {
                    "lambda_max": l_max,
                    "E_max": e_pico_ev,
                    "A00": a_00,
                    "A01": a_01,
                    "razao_A00_A01": razao_vib,
                    "W_Spano": w_spano_mev,
                    "regime": status_spano,
                },
            }
        )

        resumo_uv.append(
            {
                "Amostra": lbl_curva,
                "λ_máx (nm)": round(l_max, 1),
                "E_máx (eV)": round(e_pico_ev, 3) if np.isfinite(e_pico_ev) else np.nan,
                f"A₀₋₀ ({wl_00:.0f} nm)": round(a_00, 4),
                f"A₀₋₁ ({wl_01:.0f} nm)": round(a_01, 4),
                "Razão A₀₋₀ / A₀₋₁": round(razao_vib, 3) if np.isfinite(razao_vib) else np.nan,
                "W Spano (meV)": round(w_spano_mev, 1) if np.isfinite(w_spano_mev) else np.nan,
                "Regime Excitônico": status_spano,
            }
        )

    if not espectros_proc:
        msg_aviso("Nenhum dado numérico válido encontrado na janela de comprimento de onda selecionada.")
        return

    sub_abs, sub_tauc = st.tabs([
        "🌈 Espectros de Absorção & Modelo Excitônico de Spano",
        "📐 Tauc Plot & Extração do Bandgap Óptico (Eg)",
    ])

    # ==========================================================================
    # SUB-ABA 1: ESPECTRO UV-VIS + PARÂMETROS VIBRÔNICOS E W DE SPANO
    # ==========================================================================
    with sub_abs:
        fig_uv, ax_uv = nova_figura_origin(
            cfg_plot,
            "Comprimento de Onda, λ (nm)",
            "Absorbância Normalizada (u.a.)" if norm_uv else "Absorbância (u.a.)",
        )

        for sp in espectros_proc:
            ax_uv.plot(sp["wl"], sp["abs_plot"], linewidth=1.8, color=sp["color"], label=sp["label"], zorder=3)

        if marcar_vibronicas:
            if wl_min <= wl_00 <= wl_max:
                ax_uv.axvline(wl_00, color="#555555", lw=0.9, ls="--", zorder=2)
                ax_uv.text(
                    wl_00 + 3.0,
                    0.92,
                    "A$_{0-0}$",
                    transform=ax_uv.get_xaxis_transform(),
                    fontsize=max(8, cfg_plot["font_ticks"] - 1),
                    color="#333333",
                )
            if wl_min <= wl_01 <= wl_max:
                ax_uv.axvline(wl_01, color="#555555", lw=0.9, ls=":", zorder=2)
                ax_uv.text(
                    wl_01 + 3.0,
                    0.92,
                    "A$_{0-1}$",
                    transform=ax_uv.get_xaxis_transform(),
                    fontsize=max(8, cfg_plot["font_ticks"] - 1),
                    color="#333333",
                )

        ax_uv.set_xlim(wl_min, wl_max)

        legenda_origin(ax_uv, cfg_plot, loc="upper right")
        plt.tight_layout()

        cu1, cu2 = st.columns([1.95, 1.35])
        with cu1:
            st.pyplot(fig_uv)
            botao_download_figura(fig_uv, "Espectro_UV_Vis_Politiofenos", cfg_plot, "uv_spec")
        with cu2:
            st.markdown("#### Ordem Cristalina & Largura de Banda $W$ (Spano)")
            df_res_uv = pd.DataFrame(resumo_uv)
            st.dataframe(df_res_uv, use_container_width=True, hide_index=True)
            st.caption(
                "📌 **Interpretação Física (Spano):** Maior razão $A_{0-0}/A_{0-1}$ corresponde a **menor largura "
                "de banda de éxcitons livres $W$ (meV)**, indicando maior comprimento efetivo de conjugação planar "
                "e empacotamento $\\pi\\text{-}\\pi$ mais ordenado nos domínios cristalinos do filme."
            )
            botao_download_csv(
                df_res_uv,
                "Parametros_Vibronicos_Spano_UVVis",
                cfg_plot,
                "uv_vib_csv",
                label="📊 Baixar Tabela Vibrônica + Spano (.CSV)",
            )
        plt.close(fig_uv)

    # ==========================================================================
    # SUB-ABA 2: TAUC PLOT COMPLETO E AJUSTE DE RETA TANGENTE PARA Eg (eV)
    # ==========================================================================
    with sub_tauc:
        st.markdown("#### Determinação do Bandgap Óptico ($E_g$) via Extrapolação de Tauc")

        tc1, tc2, tc3, tc4 = st.columns(4)
        idx_amostra_tauc = tc1.selectbox(
            "Amostra Principal (Ajuste da Tangente)",
            options=list(range(len(espectros_proc))),
            format_func=lambda i: espectros_proc[i]["label"],
            index=0,
            key="tauc_sample_sel",
        )
        tipo_transicao = tc2.selectbox(
            "Natureza da Transição Eletrônica",
            options=[
                "Direta Permitida — (αhν)² [Padrão P3HT/P3DT]",
                "Indireta Permitida — (αhν)¹/²",
                "Direta Proibida — (αhν)²/³",
                "Onset Linear Direto — Absorbância (γ = 1)",
            ],
            index=0,
            key="tauc_trans_type",
        )
        modo_ajuste_eg = tc3.selectbox(
            "Seleção da Região Linear da Tangente",
            options=[
                "Automático (Máxima Inflexão na Borda)",
                "Janela Manual de Energia [E₁, E₂]",
            ],
            index=0,
            key="tauc_fit_mode",
        )
        modo_base_tauc = tc4.selectbox(
            "Critério de Extrapolação de Eg",
            options=[
                "Interseção com eixo Y = 0 (Tauc Clássico)",
                "Interseção com Reta de Base Sub-gap (Makuła)",
            ],
            index=0,
            help="O método de Makuła corrige caudas de espalhamento/pólarons abaixo do bandgap intersectando a tangente principal com a linha de base pré-borda.",
            key="tauc_base_mode",
        )

        tc5, tc6, tc7 = st.columns([1.4, 1.3, 1.3])
        usar_espessura = tc5.checkbox(
            "Informar espessura do filme d (nm — Perfilometria) para α real em cm⁻¹",
            value=False,
            key="tauc_use_d",
        )
        espessura_nm = (
            float(tc6.number_input("Espessura d (nm)", 5.0, 50000.0, 150.0, step=10.0, key="tauc_d_nm"))
            if usar_espessura
            else None
        )
        sobrepor_todas_tauc = tc7.checkbox(
            "Comparar Eg de todas as amostras ativas em tabela",
            value=len(espectros_proc) > 1,
            key="tauc_all_tbl",
        )

        if "Direta Permitida" in tipo_transicao:
            gamma_exp = 2.0
            rot_y_tauc = r"$(\alpha h\nu)^2$ $(\mathrm{cm}^{-1}\cdot\mathrm{eV})^2$" if usar_espessura else r"$(A \cdot h\nu)^2$ $(\mathrm{u.a.}\cdot\mathrm{eV})^2$"
        elif "Indireta Permitida" in tipo_transicao:
            gamma_exp = 0.5
            rot_y_tauc = r"$(\alpha h\nu)^{1/2}$ $(\mathrm{cm}^{-1}\cdot\mathrm{eV})^{1/2}$" if usar_espessura else r"$(A \cdot h\nu)^{1/2}$ $(\mathrm{u.a.}\cdot\mathrm{eV})^{1/2}$"
        elif "Direta Proibida" in tipo_transicao:
            gamma_exp = 2.0 / 3.0
            rot_y_tauc = r"$(\alpha h\nu)^{2/3}$"
        else:
            gamma_exp = 1.0
            rot_y_tauc = r"Coeficiente $\alpha$ $(\mathrm{cm}^{-1})$" if usar_espessura else "Absorbância, A (u.a.)"

        def _converter_para_tauc(wl_arr: np.ndarray, abs_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            e_arr = HC_EV_NM / wl_arr
            ordem = np.argsort(e_arr)
            e_ord = e_arr[ordem]
            a_ord = np.maximum(0.0, abs_arr[ordem])
            if usar_espessura and espessura_nm is not None and espessura_nm > 0:
                d_cm = espessura_nm * 1e-7
                alpha = (2.302585093 * a_ord) / d_cm
                base_prod = alpha if gamma_exp == 1.0 else (alpha * e_ord)
            else:
                base_prod = a_ord if gamma_exp == 1.0 else (a_ord * e_ord)
            y_t = np.power(base_prod, gamma_exp)
            return e_ord, y_t

        sp_alvo = espectros_proc[idx_amostra_tauc]
        e_alvo, y_alvo = _converter_para_tauc(sp_alvo["wl"], sp_alvo["abs_real"])

        e_min_disp = float(np.round(np.min(e_alvo), 2))
        e_max_disp = float(np.round(np.max(e_alvo), 2))
        e_view_min_def = max(e_min_disp, 1.45)
        e_view_max_def = min(e_max_disp, 3.20)
        if e_view_max_def <= e_view_min_def:
            e_view_min_def, e_view_max_def = e_min_disp, e_max_disp

        sc1, sc2 = st.columns(2)
        e_plot_min, e_plot_max = sc1.slider(
            "Janela de Energia Exibida no Eixo X — hν (eV)",
            min_value=e_min_disp,
            max_value=e_max_disp,
            value=(round(e_view_min_def, 2), round(e_view_max_def, 2)),
            step=0.02,
            key="tauc_e_view",
        )

        if modo_ajuste_eg.startswith("Automático"):
            eb_min_def = max(e_plot_min, 1.75)
            eb_max_def = min(e_plot_max, 2.25)
            if eb_max_def <= eb_min_def + 0.08:
                eb_min_def, eb_max_def = e_plot_min, e_plot_max
            e_busca_min, e_busca_max = sc2.slider(
                "Região de Busca Automática da Borda Fundamental (eV)",
                min_value=e_plot_min,
                max_value=e_plot_max,
                value=(round(eb_min_def, 2), round(eb_max_def, 2)),
                step=0.01,
                key="tauc_auto_win",
            )
            e_fit_1, e_fit_2 = _encontrar_janela_linear_tauc(e_alvo, y_alvo, e_busca_min, e_busca_max)
        else:
            em_1 = max(e_plot_min, 1.92)
            em_2 = min(e_plot_max, 2.06)
            if em_2 <= em_1 + 0.02:
                em_1 = e_plot_min + 0.3 * (e_plot_max - e_plot_min)
                em_2 = e_plot_min + 0.5 * (e_plot_max - e_plot_min)
            e_fit_1, e_fit_2 = sc2.slider(
                "Intervalo Manual Exato da Reta Tangente [E₁, E₂] (eV)",
                min_value=e_plot_min,
                max_value=e_plot_max,
                value=(round(em_1, 3), round(em_2, 3)),
                step=0.005,
                key="tauc_man_win",
            )

        if modo_base_tauc.startswith("Interseção com Reta de Base"):
            b_def1 = max(e_plot_min, min(e_fit_1 - 0.28, e_plot_max - 0.15))
            b_def2 = max(b_def1 + 0.05, e_fit_1 - 0.05)
            e_base_1, e_base_2 = st.slider(
                "Intervalo da Reta de Base Sub-gap (Makuła) antes da borda [E_b1, E_b2] (eV)",
                min_value=e_plot_min,
                max_value=max(e_plot_min + 0.05, round(e_fit_1, 2)),
                value=(round(b_def1, 2), round(b_def2, 2)),
                step=0.01,
                key="tauc_makula_win",
            )
        else:
            e_base_1, e_base_2 = (e_plot_min, e_plot_min)

        def _ajustar_eg_amostra(
            e_arr: np.ndarray,
            y_arr: np.ndarray,
            usar_auto: bool,
        ) -> Dict[str, float]:
            if usar_auto:
                ef1, ef2 = _encontrar_janela_linear_tauc(e_arr, y_arr, e_busca_min, e_busca_max)
            else:
                ef1, ef2 = e_fit_1, e_fit_2

            m_seg = (e_arr >= ef1) & (e_arr <= ef2)
            if int(np.sum(m_seg)) < 2:
                return {"eg": np.nan, "r2": np.nan, "m": np.nan, "b": np.nan, "ef1": ef1, "ef2": ef2, "m0": 0.0, "b0": 0.0, "y_int": 0.0}

            es, ys = e_arr[m_seg], y_arr[m_seg]
            m_tan, b_tan = np.polyfit(es, ys, 1)
            y_hat = m_tan * es + b_tan
            ss_r = float(np.sum((ys - y_hat) ** 2))
            ss_t = float(np.sum((ys - np.mean(ys)) ** 2))
            r2_tan = 1.0 - (ss_r / ss_t) if ss_t > 0 else 1.0

            if m_tan <= 0:
                return {"eg": np.nan, "r2": r2_tan, "m": float(m_tan), "b": float(b_tan), "ef1": ef1, "ef2": ef2, "m0": 0.0, "b0": 0.0, "y_int": 0.0}

            if modo_base_tauc.startswith("Interseção com Reta de Base"):
                m_bs = (e_arr >= e_base_1) & (e_arr <= e_base_2)
                if int(np.sum(m_bs)) >= 2:
                    m0, b0 = np.polyfit(e_arr[m_bs], y_arr[m_bs], 1)
                else:
                    m0, b0 = 0.0, 0.0
                eg_val = float((b0 - b_tan) / (m_tan - m0)) if abs(m_tan - m0) > 1e-12 else np.nan
                y_int = float(m0 * eg_val + b0) if np.isfinite(eg_val) else 0.0
            else:
                m0, b0 = 0.0, 0.0
                eg_val = float(-b_tan / m_tan)
                y_int = 0.0

            return {
                "eg": eg_val,
                "r2": float(r2_tan),
                "m": float(m_tan),
                "b": float(b_tan),
                "ef1": float(ef1),
                "ef2": float(ef2),
                "m0": float(m0),
                "b0": float(b0),
                "y_int": y_int,
            }

        res_eg_alvo = _ajustar_eg_amostra(e_alvo, y_alvo, usar_auto=modo_ajuste_eg.startswith("Automático"))

        # Construção do Gráfico de Tauc
        fig_tauc, ax_tauc = nova_figura_origin(cfg_plot, r"Energia do Fóton, $h\nu$ (eV)", rot_y_tauc)

        mask_vis = (e_alvo >= e_plot_min) & (e_alvo <= e_plot_max)
        e_vis = e_alvo[mask_vis]
        y_vis = y_alvo[mask_vis]

        ax_tauc.axhline(0, color="black", lw=0.8, zorder=1)
        ax_tauc.plot(e_vis, y_vis, "-", color=sp_alvo["color"], linewidth=1.8, label=f"Curva Tauc — {sp_alvo['label']}", zorder=3)

        mask_fit_pts = (e_alvo >= res_eg_alvo["ef1"]) & (e_alvo <= res_eg_alvo["ef2"])
        if np.any(mask_fit_pts):
            ax_tauc.plot(
                e_alvo[mask_fit_pts],
                y_alvo[mask_fit_pts],
                "o",
                color="#E60000",
                markersize=4.5,
                label=f"Trecho Ajustado ({fmt_num(res_eg_alvo['ef1'], 2, usar_virgula)}–{fmt_num(res_eg_alvo['ef2'], 2, usar_virgula)} eV)",
                zorder=4,
            )

        eg_f = res_eg_alvo["eg"]
        if np.isfinite(eg_f) and e_min_disp - 0.5 <= eg_f <= e_max_disp + 0.5:
            e_tan_ini = min(eg_f, res_eg_alvo["ef1"])
            e_tan_fim = res_eg_alvo["ef2"] + 0.05 * (e_plot_max - e_plot_min)
            e_reta = np.linspace(e_tan_ini, e_tan_fim, 100)
            y_reta = res_eg_alvo["m"] * e_reta + res_eg_alvo["b"]

            ax_tauc.plot(
                e_reta,
                y_reta,
                "--",
                color="#E60000",
                linewidth=1.8,
                label=f"Tangente ($E_g$ = {fmt_num(eg_f, 3, usar_virgula)} eV)",
                zorder=5,
            )

            if modo_base_tauc.startswith("Interseção com Reta de Base"):
                e_b_reta = np.linspace(e_plot_min, min(e_plot_max, res_eg_alvo["ef2"]), 100)
                y_b_reta = res_eg_alvo["m0"] * e_b_reta + res_eg_alvo["b0"]
                ax_tauc.plot(e_b_reta, y_b_reta, ":", color="#1F618D", lw=1.5, label="Linha de Base Sub-gap", zorder=4)

            ax_tauc.plot(
                eg_f,
                res_eg_alvo["y_int"],
                marker="D",
                markersize=6.5,
                color="#E60000",
                markeredgecolor="black",
                zorder=6,
            )

            lambda_onset = HC_EV_NM / eg_f if eg_f > 0 else np.nan
            txt_eg_box = (
                f"$E_g$ = {fmt_num(eg_f, 3, usar_virgula)} eV\n"
                f"$\\lambda_{{onset}}$ = {fmt_num(lambda_onset, 1, usar_virgula)} nm\n"
                f"$R^2$ = {fmt_num(res_eg_alvo['r2'], 4, usar_virgula)}"
            )
            ax_tauc.text(
                0.05,
                0.92,
                txt_eg_box,
                transform=ax_tauc.transAxes,
                fontsize=cfg_plot["font_labels"] - 1,
                verticalalignment="top",
            )

        y_max_v = float(np.max(y_vis)) if len(y_vis) > 0 else 1.0
        ax_tauc.set_xlim(e_plot_min, e_plot_max)
        ax_tauc.set_ylim(-0.04 * y_max_v, 1.12 * y_max_v)

        legenda_origin(ax_tauc, cfg_plot, loc="upper left" if eg_f > (e_plot_min + e_plot_max) / 2 else "lower right")
        plt.tight_layout()

        ct_g, ct_m = st.columns([2.05, 1.25])
        with ct_g:
            st.pyplot(fig_tauc)
        with ct_m:
            st.markdown(f"#### Bandgap Extraído — {sp_alvo['label']}")
            if np.isfinite(eg_f) and eg_f > 0:
                st.metric("Bandgap Óptico (Eg)", f"{fmt_num(eg_f, 3, usar_virgula)} eV")
                st.metric("Borda de Absorção (λ_onset)", f"{fmt_num(HC_EV_NM / eg_f, 1, usar_virgula)} nm")
                st.metric("Linearidade da Tangente (R²)", f"{res_eg_alvo['r2']:.5f}")
            else:
                msg_aviso("Ajuste a janela de energia para capturar uma subida linear positiva da borda de absorção.")

            botao_download_figura(fig_tauc, f"Tauc_Plot_{sp_alvo['label']}", cfg_plot, "tauc_fig")

            df_tauc_csv = pd.DataFrame(
                {
                    "Wavelength_nm": HC_EV_NM / e_vis,
                    "Photon_Energy_eV": e_vis,
                    "Tauc_Y": y_vis,
                }
            )
            botao_download_csv(
                df_tauc_csv,
                f"Dados_Tauc_{sp_alvo['label']}",
                cfg_plot,
                "tauc_csv",
                label="📊 Baixar Curva Tauc (.CSV)",
            )

        plt.close(fig_tauc)

        if sobrepor_todas_tauc and len(espectros_proc) >= 1:
            linhas_eg_lote = []
            for sp_item in espectros_proc:
                e_it, y_it = _converter_para_tauc(sp_item["wl"], sp_item["abs_real"])
                r_it = _ajustar_eg_amostra(e_it, y_it, usar_auto=modo_ajuste_eg.startswith("Automático"))
                eg_i = r_it["eg"]
                linhas_eg_lote.append(
                    {
                        "Amostra": sp_item["label"],
                        "Bandgap Eg (eV)": round(eg_i, 3) if (np.isfinite(eg_i) and eg_i > 0) else np.nan,
                        "λ_onset (nm)": round(HC_EV_NM / eg_i, 1) if (np.isfinite(eg_i) and eg_i > 0) else np.nan,
                        "Janela Ajustada (eV)": f"{r_it['ef1']:.2f} – {r_it['ef2']:.2f}",
                        "R² Tangente": round(r_it["r2"], 5) if np.isfinite(r_it["r2"]) else np.nan,
                    }
                )
            st.markdown("##### Comparativo Consolidado de Bandgap Óptico ($E_g$) entre Amostras")
            df_eg_all = pd.DataFrame(linhas_eg_lote)
            st.dataframe(df_eg_all, use_container_width=True, hide_index=True)
            botao_download_csv(
                df_eg_all,
                "Comparativo_Bandgap_Tauc_Politiofenos",
                cfg_plot,
                "tauc_all_csv",
                label="📊 Baixar Tabela Comparativa de Eg (.CSV)",
            )

    # Registro central: métricas vibrônicas/Spano de cada espectro ativo + Eg pelo Tauc na configuração atual da tela
    nan = float("nan")
    for sp_item in espectros_proc:
        eg_i, r2_i = nan, nan
        try:
            e_it, y_it = _converter_para_tauc(sp_item["wl"], sp_item["abs_real"])
            r_it = _ajustar_eg_amostra(e_it, y_it, usar_auto=modo_ajuste_eg.startswith("Automático"))
            eg_i = float(r_it["eg"])
            r2_i = float(r_it["r2"])
        except Exception:
            pass  # o registro nunca derruba a aba: Eg fica indefinido (NaN) para esta amostra

        eg_valido = bool(np.isfinite(eg_i) and eg_i > 0)
        metricas_uv = dict(sp_item["metricas_registro"])
        metricas_uv["Eg"] = eg_i if eg_valido else nan
        metricas_uv["lambda_onset"] = HC_EV_NM / eg_i if eg_valido else nan
        metricas_uv["R2_tauc"] = r2_i if np.isfinite(r2_i) else nan
        coletor.adicionar(
            sp_item["label"],
            metricas_uv,
            unidades=_UNIDADES_UVVIS,
            meta={
                "arquivo": sp_item["origem"],
                "janela_lambda_nm": [float(wl_min), float(wl_max)],
                "posicao_A00_nm": float(wl_00),
                "posicao_A01_nm": float(wl_01),
                "Ep_eV": float(ep_spano_ev),
                "leitura_A0": busca_local_vib,
                "zerar_espalhamento": bool(subtrair_offset),
                "normalizado": bool(norm_uv),
                "transicao_tauc": tipo_transicao,
                "ajuste_tauc": modo_ajuste_eg,
                "extrapolacao_eg": modo_base_tauc,
                "espessura_nm": float(espessura_nm) if espessura_nm is not None else None,
            },
            desambiguar=sp_item["origem"],
        )


def render(cfg_plot: dict) -> None:
    """Ponto de entrada do app.py: o corpo roda dentro da sessão de registro central de resultados."""
    with sessao_resultados(registro_central.TEC_UVVIS, _ORIGEM_REGISTRO, cfg_plot, "uv_reg") as coletor:
        _render_corpo(cfg_plot, coletor)
