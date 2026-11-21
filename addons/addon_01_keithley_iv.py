# -*- coding: utf-8 -*-
"""
addons/addon_01_keithley_iv.py
Módulo 1: Caracterização Elétrica DC (I × V — Keithley LabWorks)
  - Integrado ao parser blindado extrair_keithley_iv e ao painel de override manual
  - Extração de Resistência (Lei de Ohm), Condutividade (σ = k · ΔI/ΔV) e filtro de Outliers
  - Modo de análise consolidada (Média ± Desvio Padrão) OU separação explícita dos ramos de
    Ida (ΔV > 0) e Volta (ΔV < 0) para quantificação de Histerese elétrica (área do laço e ΔI_hist)
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.parsers import (
    exibir_diagnostico_leitura,
    extrair_keithley_iv,
    render_painel_override_parser,
)
from core.plotting import (
    aplicar_estilo_origin,
    botao_download_csv,
    botao_download_figura,
    fmt_cientifico,
    fmt_num,
)

TITLE = "1. Curvas I × V (Keithley)"
ICON = "⚡"
ORDER = 10


def _integrar_trapezio(y: np.ndarray, x: np.ndarray) -> float:
    """Compatível tanto com NumPy 2.x (np.trapezoid) quanto com NumPy 1.x (np.trapz)."""
    func = getattr(np, "trapezoid", None)
    if func is not None:
        return float(func(y, x))
    return float(np.trapz(y, x))


def _formatar_resistencia(r_ohms: float, fmt_sel: str, usar_virgula: bool) -> str:
    if not np.isfinite(r_ohms):
        return "R = —"
    if fmt_sel == "Notação Científica (Ω)":
        return f"R = {fmt_cientifico(r_ohms, 2, None, usar_virgula)} Ω"
    if fmt_sel == "kΩ":
        return f"R = {fmt_num(r_ohms / 1e3, 2, usar_virgula)} kΩ"
    if fmt_sel == "MΩ":
        return f"R = {fmt_num(r_ohms / 1e6, 2, usar_virgula)} MΩ"
    if fmt_sel == "GΩ":
        return f"R = {fmt_num(r_ohms / 1e9, 3, usar_virgula)} GΩ"
    return f"R = {fmt_num(r_ohms, 1, usar_virgula)} Ω"


def _formatar_condutividade(sigma_sm: float, fmt_sel: str, usar_virgula: bool) -> str:
    if not np.isfinite(sigma_sm):
        return "σ = —"
    if fmt_sel == "Notação Científica × 10⁻⁶ S/m":
        return f"σ = {fmt_cientifico(sigma_sm, 2, -6, usar_virgula)} S/m"
    if fmt_sel == "Notação Científica Automática (S/m)":
        return f"σ = {fmt_cientifico(sigma_sm, 2, None, usar_virgula)} S/m"
    if fmt_sel == "S/cm":
        return f"σ = {fmt_cientifico(sigma_sm / 100.0, 2, None, usar_virgula)} S/cm"
    if fmt_sel == "µS/m":
        return f"σ = {fmt_num(sigma_sm * 1e6, 2, usar_virgula)} µS/m"
    if fmt_sel == "mS/m":
        return f"σ = {fmt_num(sigma_sm * 1e3, 3, usar_virgula)} mS/m"
    return f"σ = {sigma_sm:.3e} S/m"


def _interpolar_ramo(
    amostras: List[dict],
    v_grid: np.ndarray,
    v_min: float,
    v_max: float,
    sinal_ramo: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Interpola as amostras ativas sobre v_grid.
    sinal_ramo:
      None -> agrupa todos os pontos (média Ida + Volta por tensão)
      +1   -> filtra apenas Ida (Sweep_Sign > 0)
      -1   -> filtra apenas Volta (Sweep_Sign < 0)
    """
    curvas = []
    for item in amostras:
        df_a = item["df"]
        mask = (df_a["Voltage"] >= v_min) & (df_a["Voltage"] <= v_max)
        if sinal_ramo is not None:
            mask = mask & (df_a["Sweep_Sign"] == sinal_ramo)
        df_sub = df_a.loc[mask, ["Voltage", "Current"]].copy()
        if len(df_sub) < 2:
            continue

        # Arredondar ruído numérico de máquina (ex: 8.60000000000001 V na Keithley) para agrupar degraus exatos
        df_sub["V_round"] = df_sub["Voltage"].round(6)
        df_grp = df_sub.groupby("V_round", as_index=False)["Current"].mean().sort_values("V_round")
        if len(df_grp) >= 2:
            v_vals = df_grp["V_round"].values
            i_vals = df_grp["Current"].values
            # Garantir cobertura razoável da grade antes de extrapolar bordas
            curvas.append(np.interp(v_grid, v_vals, i_vals))

    if not curvas:
        return None
    return np.array(curvas, dtype=float)


def _ajustar_reta_ohm(v_grid: np.ndarray, i_mean: np.ndarray, fator_geom: float) -> Dict[str, float]:
    slope, intercept = np.polyfit(v_grid, i_mean, 1)
    i_fit = intercept + slope * v_grid
    ss_res = float(np.sum((i_mean - i_fit) ** 2))
    ss_tot = float(np.sum((i_mean - np.mean(i_mean)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    r_ohms = (1.0 / slope) if slope != 0 else np.nan
    sigma_sm = fator_geom * slope
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "r_ohms": float(r_ohms),
        "sigma_sm": float(sigma_sm),
    }


def _plotar_serie_iv(
    ax: plt.Axes,
    x_plot: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_amostras: int,
    estilo_erro: str,
    cor: str,
    marcador: str,
    label: str,
    zorder_base: int = 3,
) -> None:
    if estilo_erro == "Barras de Erro (Padrão Origin)":
        ax.errorbar(
            x_plot,
            y_mean,
            yerr=y_std if n_amostras > 1 else None,
            fmt=marcador,
            color=cor,
            ecolor=cor,
            elinewidth=0.8,
            capsize=2.8,
            markersize=4.8,
            label=label,
            zorder=zorder_base,
        )
    elif estilo_erro == "Banda Sombreada":
        ax.plot(
            x_plot,
            y_mean,
            f"{marcador}-",
            color=cor,
            markersize=4.0,
            linewidth=1.4,
            label=label,
            zorder=zorder_base,
        )
        if n_amostras > 1:
            ax.fill_between(
                x_plot,
                y_mean - y_std,
                y_mean + y_std,
                color=cor,
                alpha=0.18,
                zorder=zorder_base - 1,
            )
    else:
        ax.plot(
            x_plot,
            y_mean,
            marcador,
            color=cor,
            markersize=4.8,
            label=label,
            zorder=zorder_base,
        )


def render(cfg_plot: dict) -> None:
    usar_virgula = bool(cfg_plot.get("usar_virgula", True))

    st.subheader("Caracterização Elétrica DC (I × V) — Lei de Ohm, Condutividade e Histerese")

    files_iv = st.file_uploader(
        "Arraste um ou mais arquivos .dat / .csv / .txt da Keithley (mesma condição atmosférica para média)",
        type=["dat", "txt", "csv"],
        accept_multiple_files=True,
        key="iv_uploader",
    )

    opcoes_parser, slot_diag = render_painel_override_parser("iv")

    with st.expander("🎛️ Parâmetros Físicos, Histerese, Unidades e Estilo Visual (I × V)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        legenda_iv = c1.text_input("Nome na Legenda", value="P3HT", key="iv_leg")
        unid_v = c2.selectbox("Unidade de Tensão", ["V", "mV"], index=0, key="iv_unid_v")
        unid_i = c3.selectbox("Unidade de Corrente", ["µA", "mA", "nA", "A"], index=0, key="iv_unid_i")
        fator_geom = float(
            c4.number_input(
                "Fator Geométrico k (m⁻¹)",
                min_value=0.0001,
                max_value=1e7,
                value=5.1000,
                step=0.1,
                format="%.4f",
                help="σ = k × Slope (ΔI/ΔV). Padrão calibrado do IDE = 5,1 m⁻¹.",
                key="iv_k_geom",
            )
        )

        c5, c6, c7, c8 = st.columns(4)
        modo_varredura = c5.selectbox(
            "Tratamento de Varredura / Histerese",
            options=[
                "Curva Média Consolidada (Ida + Volta)",
                "Separar Ida (ΔV > 0) e Volta (ΔV < 0) — Histerese",
                "Apenas Ramo de Ida (ΔV > 0)",
                "Apenas Ramo de Volta (ΔV < 0)",
            ],
            index=0,
            key="iv_modo_hist",
        )
        fmt_res_sel = c6.selectbox(
            "Formato da Resistência (R)",
            ["Notação Científica (Ω)", "kΩ", "MΩ", "Ω", "GΩ"],
            index=0,
            key="iv_fmt_r",
        )
        fmt_cond_sel = c7.selectbox(
            "Formato da Condutividade (σ)",
            [
                "Notação Científica × 10⁻⁶ S/m",
                "Notação Científica Automática (S/m)",
                "S/m",
                "S/cm",
                "µS/m",
                "mS/m",
            ],
            index=0,
            key="iv_fmt_sig",
        )
        estilo_erro_iv = c8.selectbox(
            "Representação de Média/Erro",
            ["Barras de Erro (Padrão Origin)", "Banda Sombreada", "Apenas Pontos Médios"],
            index=0,
            key="iv_est_err",
        )

        c9, c10, c11, c12, c13 = st.columns([1.2, 1.2, 1.0, 0.8, 0.8])
        xlabel_iv = c9.text_input("Rótulo Eixo X", value=f"Voltagem, V ({unid_v})", key="iv_xlab")
        ylabel_iv = c10.text_input("Rótulo Eixo Y", value=f"Corrente, I ({unid_i})", key="iv_ylab")
        n_pts_iv = int(c11.slider("Pontos interpolados", 25, 250, 100, key="iv_npts"))
        cor_pts_iv = c12.color_picker("Cor Principal / Ida", value="#000000", key="iv_cor_ida")
        cor_volta_or_fit = c13.color_picker(
            "Cor Volta" if "Separar" in modo_varredura else "Cor Reta Fit",
            value="#1F618D" if "Separar" in modo_varredura else "#E60000",
            key="iv_cor_sec",
        )

        c14, c15, c16, c17 = st.columns(4)
        eixos_zero_iv = c14.checkbox("Cruzar eixos em (0, 0)", value=True, key="iv_zero_ax")
        mostrar_indiv_iv = c15.checkbox("Mostrar curvas individuais ao fundo", value=False, key="iv_show_ind")
        mostrar_r2_iv = c16.checkbox("Exibir R² na anotação do gráfico", value=False, key="iv_show_r2")
        limites_manuais_iv = c17.checkbox("Definir limites manuais dos eixos", value=False, key="iv_man_lim")

    if not files_iv:
        return

    amostras_lidas = []
    meta_primeiro = None
    nome_primeiro = ""

    for idx_f, f_obj in enumerate(files_iv):
        try:
            df_k, meta_k = extrair_keithley_iv(f_obj.getvalue(), opcoes=opcoes_parser)
            if idx_f == 0:
                meta_primeiro = meta_k
                nome_primeiro = f_obj.name

            fit_ind = _ajustar_reta_ohm(df_k["Voltage"].values, df_k["Current"].values, fator_geom)
            tem_hist_arq = bool(df_k.attrs.get("tem_histerese", False))

            amostras_lidas.append(
                {
                    "nome": f_obj.name,
                    "df": df_k,
                    "slope": fit_ind["slope"],
                    "R_kOhm": fit_ind["r_ohms"] / 1e3,
                    "sigma_Sm": fit_ind["sigma_sm"],
                    "R2": fit_ind["r2"],
                    "tem_histerese": tem_hist_arq,
                }
            )
        except Exception as exc:
            st.error(f"Erro ao ler `{f_obj.name}`: {exc}")

    if meta_primeiro is not None:
        exibir_diagnostico_leitura(slot_diag, meta_primeiro, nome_primeiro)

    if not amostras_lidas:
        return

    with st.expander(
        "🔍 Diagnóstico Individual por Arquivo & Filtro de Amostras Discrepantes",
        expanded=len(amostras_lidas) > 1,
    ):
        df_diag_iv = pd.DataFrame(
            [
                {
                    "Arquivo": a["nome"],
                    "Ciclo Ida/Volta": "Sim" if a["tem_histerese"] else "Rampa Única",
                    "Slope (A/V)": f"{a['slope']:.3e}",
                    "Resistência (kΩ)": round(a["R_kOhm"], 2) if np.isfinite(a["R_kOhm"]) else np.nan,
                    "Condutividade (S/m)": f"{a['sigma_Sm']:.3e}",
                    "R² Linear": round(a["R2"], 5),
                }
                for a in amostras_lidas
            ]
        )
        st.dataframe(df_diag_iv, use_container_width=True, hide_index=True)

        nomes_disp = [a["nome"] for a in amostras_lidas]
        selecionados_iv = st.multiselect(
            "Arquivos incluídos no cálculo da curva consolidada (desmarque eventuais outliers):",
            options=nomes_disp,
            default=nomes_disp,
            key="iv_sel_files",
        )

        res_vals = [
            a["R_kOhm"]
            for a in amostras_lidas
            if a["nome"] in selecionados_iv and np.isfinite(a["R_kOhm"]) and a["R_kOhm"] > 0
        ]
        if len(res_vals) >= 2 and (max(res_vals) / min(res_vals)) > 1.5:
            st.warning(
                f"⚠️ Atenção: A maior resistência selecionada ({max(res_vals):.1f} kΩ) é "
                f"**{max(res_vals) / min(res_vals):.2f}×** maior que a menor ({min(res_vals):.1f} kΩ). "
                "Verifique o contato das ponteiras ou a espessura do filme caso a barra de erro fique muito aberta."
            )

    ativas = [a for a in amostras_lidas if a["nome"] in selecionados_iv]
    if not ativas:
        st.info("Selecione pelo menos um arquivo acima para gerar o gráfico.")
        return

    v_global_min = float(min(a["df"]["Voltage"].min() for a in ativas))
    v_global_max = float(max(a["df"]["Voltage"].max() for a in ativas))
    if v_global_max <= v_global_min:
        st.warning("O intervalo de tensão nos arquivos selecionados é constante (ΔV = 0).")
        return

    v_min_sel, v_max_sel = st.slider(
        "Intervalo de Tensão para Análise e Ajuste Linear (V)",
        min_value=round(v_global_min, 3),
        max_value=round(v_global_max, 3),
        value=(round(v_global_min, 3), round(v_global_max, 3)),
        key="iv_v_range",
    )
    if v_max_sel <= v_min_sel:
        st.warning("Selecione um intervalo onde V_máx > V_mín.")
        return

    v_grid = np.linspace(v_min_sel, v_max_sel, n_pts_iv)
    mult_v = {"V": 1.0, "mV": 1e3}[unid_v]
    mult_i = {"µA": 1e6, "mA": 1e3, "nA": 1e9, "A": 1.0}[unid_i]
    x_plot = v_grid * mult_v

    separar_histerese = modo_varredura.startswith("Separar Ida")

    # ==========================================================================
    # MODO A: SEPARAÇÃO DE RAMOS DE IDA E VOLTA (ANÁLISE DE HISTERESE)
    # ==========================================================================
    if separar_histerese:
        mat_ida = _interpolar_ramo(ativas, v_grid, v_min_sel, v_max_sel, sinal_ramo=1)
        mat_volta = _interpolar_ramo(ativas, v_grid, v_min_sel, v_max_sel, sinal_ramo=-1)

        if mat_ida is None and mat_volta is None:
            st.warning("Não há pontos suficientes no intervalo de tensão selecionado.")
            return

        if mat_ida is None or mat_volta is None:
            st.info(
                "ℹ️ Os arquivos selecionados possuem varredura em apenas uma direção neste intervalo "
                "(sem ciclo fechado de Ida e Volta). Exibindo o ramo disponível."
            )

        cor_fit_hist = "#E60000"
        fig_iv, ax_iv = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)

        if eixos_zero_iv:
            ax_iv.axhline(0, color="black", linewidth=0.85, zorder=1)
            ax_iv.axvline(0, color="black", linewidth=0.85, zorder=1)

        linhas_texto_anotacao = []
        colunas_export: Dict[str, np.ndarray] = {f"Tensao ({unid_v})": x_plot}

        fit_ida = None
        fit_volta = None
        i_mean_ida = None
        i_mean_volta = None
        y_extremos = []

        if mat_ida is not None:
            i_mean_ida = np.mean(mat_ida, axis=0)
            i_std_ida = np.std(mat_ida, axis=0) if len(mat_ida) > 1 else np.zeros_like(i_mean_ida)
            fit_ida = _ajustar_reta_ohm(v_grid, i_mean_ida, fator_geom)

            y_ida = i_mean_ida * mult_i
            y_err_ida = i_std_ida * mult_i
            y_fit_ida = (fit_ida["intercept"] + fit_ida["slope"] * v_grid) * mult_i
            y_extremos.extend([y_ida - y_err_ida, y_ida + y_err_ida])

            if mostrar_indiv_iv:
                for idx_c, arr_i in enumerate(mat_ida):
                    ax_iv.plot(
                        x_plot,
                        arr_i * mult_i,
                        color=cor_pts_iv,
                        alpha=0.22,
                        lw=0.9,
                        label="Individuais (Ida)" if idx_c == 0 else "",
                        zorder=1,
                    )

            _plotar_serie_iv(
                ax_iv,
                x_plot,
                y_ida,
                y_err_ida,
                len(mat_ida),
                estilo_erro_iv,
                cor_pts_iv,
                "o",
                f"{legenda_iv} — Ida (ΔV > 0)",
                zorder_base=3,
            )
            ax_iv.plot(x_plot, y_fit_ida, "-", color=cor_fit_hist, linewidth=1.7, label="Ajuste Linear (Ida)", zorder=5)

            txt_r_ida = _formatar_resistencia(fit_ida["r_ohms"], fmt_res_sel, usar_virgula).replace("R =", "R$_{ida}$ =")
            txt_s_ida = _formatar_condutividade(fit_ida["sigma_sm"], fmt_cond_sel, usar_virgula).replace("σ =", "σ$_{ida}$ =")
            linhas_texto_anotacao.extend([txt_r_ida, txt_s_ida])
            if mostrar_r2_iv:
                linhas_texto_anotacao.append(f"R$^2_{{ida}}$ = {fmt_num(fit_ida['r2'], 4, usar_virgula)}")

            colunas_export[f"Corrente_Ida_Media ({unid_i})"] = y_ida
            colunas_export[f"Corrente_Ida_Desvio ({unid_i})"] = y_err_ida
            colunas_export[f"Ajuste_Ida ({unid_i})"] = y_fit_ida

        if mat_volta is not None:
            i_mean_volta = np.mean(mat_volta, axis=0)
            i_std_volta = np.std(mat_volta, axis=0) if len(mat_volta) > 1 else np.zeros_like(i_mean_volta)
            fit_volta = _ajustar_reta_ohm(v_grid, i_mean_volta, fator_geom)

            y_volta = i_mean_volta * mult_i
            y_err_volta = i_std_volta * mult_i
            y_fit_volta = (fit_volta["intercept"] + fit_volta["slope"] * v_grid) * mult_i
            y_extremos.extend([y_volta - y_err_volta, y_volta + y_err_volta])

            if mostrar_indiv_iv:
                for idx_c, arr_i in enumerate(mat_volta):
                    ax_iv.plot(
                        x_plot,
                        arr_i * mult_i,
                        color=cor_volta_or_fit,
                        alpha=0.22,
                        lw=0.9,
                        ls="--",
                        label="Individuais (Volta)" if idx_c == 0 else "",
                        zorder=1,
                    )

            _plotar_serie_iv(
                ax_iv,
                x_plot,
                y_volta,
                y_err_volta,
                len(mat_volta),
                estilo_erro_iv,
                cor_volta_or_fit,
                "s",
                f"{legenda_iv} — Volta (ΔV < 0)",
                zorder_base=4,
            )
            ax_iv.plot(
                x_plot,
                y_fit_volta,
                "--",
                color=cor_volta_or_fit,
                linewidth=1.6,
                label="Ajuste Linear (Volta)",
                zorder=5,
            )

            txt_r_vol = _formatar_resistencia(fit_volta["r_ohms"], fmt_res_sel, usar_virgula).replace("R =", "R$_{volta}$ =")
            txt_s_vol = _formatar_condutividade(fit_volta["sigma_sm"], fmt_cond_sel, usar_virgula).replace("σ =", "σ$_{volta}$ =")
            linhas_texto_anotacao.extend([txt_r_vol, txt_s_vol])
            if mostrar_r2_iv:
                linhas_texto_anotacao.append(f"R$^2_{{volta}}$ = {fmt_num(fit_volta['r2'], 4, usar_virgula)}")

            colunas_export[f"Corrente_Volta_Media ({unid_i})"] = y_volta
            colunas_export[f"Corrente_Volta_Desvio ({unid_i})"] = y_err_volta
            colunas_export[f"Ajuste_Volta ({unid_i})"] = y_fit_volta

        # Cálculo quantitativo da histerese entre Ida e Volta
        area_hist_av = np.nan
        gap_hist_pct = np.nan
        if i_mean_ida is not None and i_mean_volta is not None:
            diff_abs = np.abs(i_mean_ida - i_mean_volta)
            area_hist_av = _integrar_trapezio(diff_abs, v_grid)  # A·V (Watts)
            pico_corrente = max(1e-15, float(np.max(np.abs(np.concatenate([i_mean_ida, i_mean_volta])))))
            gap_hist_pct = float(np.max(diff_abs) / pico_corrente * 100.0)
            colunas_export[f"Delta_Histerese_Ida_menos_Volta ({unid_i})"] = (i_mean_ida - i_mean_volta) * mult_i

        if limites_manuais_iv:
            y_cat = np.concatenate(y_extremos)
            lc1, lc2, lc3, lc4 = st.columns(4)
            xmin_m = lc1.number_input("X mín", value=float((v_min_sel - 0.8) * mult_v), key="iv_h_xmin")
            xmax_m = lc2.number_input("X máx", value=float((v_max_sel + 0.8) * mult_v), key="iv_h_xmax")
            ymin_m = lc3.number_input("Y mín", value=float(np.min(y_cat) * 1.15), key="iv_h_ymin")
            ymax_m = lc4.number_input("Y máx", value=float(np.max(y_cat) * 1.15), key="iv_h_ymax")
            ax_iv.set_xlim(xmin_m, xmax_m)
            ax_iv.set_ylim(ymin_m, ymax_m)
        else:
            margem_x = 0.04 * (x_plot[-1] - x_plot[0])
            ax_iv.set_xlim(x_plot[0] - margem_x, x_plot[-1] + margem_x)

        if linhas_texto_anotacao:
            ax_iv.text(
                0.04,
                0.95,
                "\n".join(linhas_texto_anotacao),
                transform=ax_iv.transAxes,
                fontsize=max(8, cfg_plot["font_labels"] - 1),
                verticalalignment="top",
            )

        ax_iv.set_xlabel(xlabel_iv, fontsize=cfg_plot["font_labels"], labelpad=7)
        ax_iv.set_ylabel(ylabel_iv, fontsize=cfg_plot["font_labels"], labelpad=7)

        leg = ax_iv.legend(
            loc="lower right",
            frameon=True,
            fancybox=False,
            edgecolor="black",
            fontsize=cfg_plot["font_legend"],
        )
        leg.get_frame().set_linewidth(0.9)

        aplicar_estilo_origin(ax_iv, cfg_plot)
        plt.tight_layout()

        col_g_iv, col_m_iv = st.columns([2.3, 1.0])
        with col_g_iv:
            st.pyplot(fig_iv)
        with col_m_iv:
            st.markdown("#### Métricas de Ida, Volta e Histerese")
            if fit_ida is not None:
                st.metric("Resistência Ida (ΔV > 0)", f"{fit_ida['r_ohms'] / 1e3:.2f} kΩ")
                st.metric("Condutividade σ (Ida)", f"{fit_ida['sigma_sm']:.3e} S/m")
            if fit_volta is not None:
                st.metric("Resistência Volta (ΔV < 0)", f"{fit_volta['r_ohms'] / 1e3:.2f} kΩ")
                st.metric("Condutividade σ (Volta)", f"{fit_volta['sigma_sm']:.3e} S/m")
            if np.isfinite(area_hist_av):
                st.metric("Área do Laço de Histerese", f"{area_hist_av * 1e6:.3f} µW")
                st.metric("Desvio Máx. Ida–Volta", f"{gap_hist_pct:.2f} %")

            botao_download_figura(fig_iv, f"IxV_Histerese_{legenda_iv}", cfg_plot, "iv_hist")
            botao_download_csv(pd.DataFrame(colunas_export), f"Dados_IxV_Histerese_{legenda_iv}", cfg_plot, "iv_hist")

        plt.close(fig_iv)
        return

    # ==========================================================================
    # MODO B: CURVA ÚNICA (CONSOLIDADA IDA+VOLTA, SÓ IDA OU SÓ VOLTA)
    # ==========================================================================
    if modo_varredura.startswith("Apenas Ramo de Ida"):
        sinal_filtro = 1
        sufixo_modo = "Ida"
    elif modo_varredura.startswith("Apenas Ramo de Volta"):
        sinal_filtro = -1
        sufixo_modo = "Volta"
    else:
        sinal_filtro = None
        sufixo_modo = "Consolidado"

    matriz_i = _interpolar_ramo(ativas, v_grid, v_min_sel, v_max_sel, sinal_ramo=sinal_filtro)
    if matriz_i is None:
        st.warning("Nenhum ponto válido encontrado para o ramo/intervalo de tensão selecionado.")
        return

    i_mean = np.mean(matriz_i, axis=0)
    i_std = np.std(matriz_i, axis=0) if len(matriz_i) > 1 else np.zeros_like(i_mean)
    fit_res = _ajustar_reta_ohm(v_grid, i_mean, fator_geom)
    i_fit = fit_res["intercept"] + fit_res["slope"] * v_grid

    txt_r = _formatar_resistencia(fit_res["r_ohms"], fmt_res_sel, usar_virgula)
    txt_sig = _formatar_condutividade(fit_res["sigma_sm"], fmt_cond_sel, usar_virgula)

    y_plot = i_mean * mult_i
    y_err = i_std * mult_i
    y_fit_plot = i_fit * mult_i

    if limites_manuais_iv:
        lc1, lc2, lc3, lc4 = st.columns(4)
        xmin_m = lc1.number_input("X mín", value=float((v_min_sel - 0.8) * mult_v), key="iv_s_xmin")
        xmax_m = lc2.number_input("X máx", value=float((v_max_sel + 0.8) * mult_v), key="iv_s_xmax")
        ymin_m = lc3.number_input("Y mín", value=float(np.min(y_plot - y_err) * 1.15), key="iv_s_ymin")
        ymax_m = lc4.number_input("Y máx", value=float(np.max(y_plot + y_err) * 1.15), key="iv_s_ymax")

    fig_iv, ax_iv = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)

    if eixos_zero_iv:
        ax_iv.axhline(0, color="black", linewidth=0.85, zorder=1)
        ax_iv.axvline(0, color="black", linewidth=0.85, zorder=1)

    if mostrar_indiv_iv:
        for idx_c, arr_i in enumerate(matriz_i):
            ax_iv.plot(
                x_plot,
                arr_i * mult_i,
                color="#888888",
                alpha=0.35,
                linewidth=1.0,
                label="Curvas Individuais" if idx_c == 0 else "",
                zorder=1,
            )

    _plotar_serie_iv(
        ax_iv,
        x_plot,
        y_plot,
        y_err,
        len(matriz_i),
        estilo_erro_iv,
        cor_pts_iv,
        "o",
        legenda_iv,
        zorder_base=3,
    )
    ax_iv.plot(x_plot, y_fit_plot, color=cor_volta_or_fit, linewidth=1.9, zorder=4)

    bloco_texto = f"{txt_r}\n{txt_sig}"
    if mostrar_r2_iv:
        bloco_texto += f"\nR$^2$ = {fmt_num(fit_res['r2'], 4, usar_virgula)}"

    ax_iv.text(
        0.05,
        0.93,
        bloco_texto,
        transform=ax_iv.transAxes,
        fontsize=cfg_plot["font_labels"] - 1,
        verticalalignment="top",
    )

    ax_iv.set_xlabel(xlabel_iv, fontsize=cfg_plot["font_labels"], labelpad=7)
    ax_iv.set_ylabel(ylabel_iv, fontsize=cfg_plot["font_labels"], labelpad=7)

    if limites_manuais_iv:
        ax_iv.set_xlim(xmin_m, xmax_m)
        ax_iv.set_ylim(ymin_m, ymax_m)
    else:
        margem_x = 0.04 * (x_plot[-1] - x_plot[0])
        ax_iv.set_xlim(x_plot[0] - margem_x, x_plot[-1] + margem_x)

    leg = ax_iv.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=cfg_plot["font_legend"],
    )
    leg.get_frame().set_linewidth(0.9)

    aplicar_estilo_origin(ax_iv, cfg_plot)
    plt.tight_layout()

    col_g_iv, col_m_iv = st.columns([2.3, 1.0])
    with col_g_iv:
        st.pyplot(fig_iv)
    with col_m_iv:
        st.markdown(f"#### Resultados ({sufixo_modo})")
        st.metric("Resistência Média (R)", f"{fit_res['r_ohms'] / 1e3:.2f} kΩ")
        st.metric("Condutividade (σ)", f"{fit_res['sigma_sm']:.3e} S/m")
        st.metric("Slope Linear (ΔI/ΔV)", f"{fit_res['slope']:.4e} A/V")
        st.metric("Qualidade do Ajuste (R²)", f"{fit_res['r2']:.5f}")

        botao_download_figura(fig_iv, f"IxV_{legenda_iv}_{sufixo_modo}", cfg_plot, "iv_std")

        df_export_iv = pd.DataFrame(
            {
                f"Tensao ({unid_v})": x_plot,
                f"Corrente_Media ({unid_i})": y_plot,
                f"Desvio_Padrao ({unid_i})": y_err,
                f"Ajuste_Linear ({unid_i})": y_fit_plot,
            }
        )
        botao_download_csv(df_export_iv, f"Dados_IxV_{legenda_iv}_{sufixo_modo}", cfg_plot, "iv_std")

    plt.close(fig_iv)