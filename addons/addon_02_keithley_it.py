# -*- coding: utf-8 -*-
"""
addons/addon_02_keithley_it.py
Módulo 2: Resposta Dinâmica de Sensoriamento (I × t — Ciclos de Gás, Umidade e COVs)
  - Integrado ao extrair_keithley_it sem escalonamento oculto: seletor explícito da unidade
    de corrente bruta usando info_escala["unidade_sugerida"] exclusivamente como recomendação
  - Correção opcional de deriva de linha de base (Drift Linear / Polinomial / Âncoras Pré-Pulso)
  - Extração cinética completa por ciclo: Sensibilidade (ΔI/I₀ %), Tempo de Resposta (t90),
    Tempo de Recuperação (t_rec90) com interpolação sub-ponto e Taxa de Recuperação (%)
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.parsers import (
    exibir_diagnostico_leitura,
    extrair_keithley_it,
    render_painel_override_parser,
)
from core.plotting import (
    aplicar_estilo_origin,
    botao_download_csv,
    botao_download_figura,
    fmt_num,
)

TITLE = "2. Sensoriamento I × t (Ciclos)"
ICON = "⏱️"
ORDER = 20


def _interpolar_tempo_90(t_arr: np.ndarray, progresso: np.ndarray, nivel: float = 0.90) -> float:
    """
    Calcula o instante exato (em relação a t_arr[0]) em que a fração normalizada `progresso`
    atinge `nivel` (padrão 0,90), realizando interpolação linear entre os pontos adjacentes.
    """
    if len(t_arr) < 2 or len(progresso) < 2:
        return np.nan
    idxs = np.where(progresso >= nivel)[0]
    if len(idxs) == 0:
        return np.nan

    k = int(idxs[0])
    if k == 0:
        return 0.0

    p0, p1 = float(progresso[k - 1]), float(progresso[k])
    t0, t1 = float(t_arr[k - 1]), float(t_arr[k])
    if p1 > p0:
        frac = (nivel - p0) / (p1 - p0)
        t_cruz = t0 + np.clip(frac, 0.0, 1.0) * (t1 - t0)
    else:
        t_cruz = t1
    return max(0.0, float(t_cruz - t_arr[0]))


def _estimar_curva_drift(
    t_arr: np.ndarray,
    i_si: np.ndarray,
    janelas_gas: List[Tuple[float, float]],
    modo_drift: str,
) -> Optional[np.ndarray]:
    """
    Estima a linha de base lenta B(t) em Amperes para remoção opcional de deriva:
      - 'Âncoras Pré-Pulso (Interpolação Linear)': conecta os pontos de base imediatamente anteriores a cada pulso
      - 'Regressão Linear nas Zonas Estabilizadas de Purga': reta ajustada sobre a purga inicial e o terço final de cada purga
      - 'Polinomial Grau 2 nas Zonas Estabilizadas de Purga': curva quadrática suave nas regiões estabilizadas de purga
      - 'Reta Simples Início–Fim da Janela': conecta a mediana inicial à mediana final da série
    """
    if modo_drift.startswith("Desativada") or len(t_arr) < 4:
        return None

    t_min, t_max = float(t_arr[0]), float(t_arr[-1])

    if modo_drift.startswith("Reta Simples"):
        n_borda = max(2, min(10, len(t_arr) // 20))
        t_a, i_a = float(np.mean(t_arr[:n_borda])), float(np.median(i_si[:n_borda]))
        t_b, i_b = float(np.mean(t_arr[-n_borda:])), float(np.median(i_si[-n_borda:]))
        if abs(t_b - t_a) < 1e-12:
            return np.full_like(i_si, i_a)
        slope = (i_b - i_a) / (t_b - t_a)
        return i_a + slope * (t_arr - t_a)

    janelas_ord = sorted([(a, b) for (a, b) in janelas_gas if b > t_min and a < t_max], key=lambda x: x[0])

    if modo_drift.startswith("Âncoras Pré-Pulso") and len(janelas_ord) >= 2:
        ts_anc = []
        is_anc = []
        for j_ini, j_fim in janelas_ord:
            larg = max(1e-6, j_fim - j_ini)
            mask_pre = (t_arr >= j_ini - 0.30 * larg) & (t_arr <= j_ini)
            if np.any(mask_pre):
                ts_anc.append(float(np.median(t_arr[mask_pre])))
                is_anc.append(float(np.median(i_si[mask_pre])))
        # Incluir final da última purga se houver margem após o último pulso
        ult_fim = janelas_ord[-1][1]
        if t_max - ult_fim > 0.25 * (janelas_ord[-1][1] - janelas_ord[-1][0]):
            mask_fim = t_arr >= (t_max - 0.20 * (t_max - ult_fim))
            if np.any(mask_fim):
                ts_anc.append(float(np.median(t_arr[mask_fim])))
                is_anc.append(float(np.median(i_si[mask_fim])))

        if len(ts_anc) >= 2:
            # Extrapolação linear suave nas pontas além da 1ª e última âncora
            ts_np = np.array(ts_anc, dtype=float)
            is_np = np.array(is_anc, dtype=float)
            base_interp = np.interp(t_arr, ts_np, is_np)
            # Ajustar inclinação esquerda antes de ts_np[0]
            s_esq = (is_np[1] - is_np[0]) / max(1e-12, ts_np[1] - ts_np[0])
            mask_esq = t_arr < ts_np[0]
            base_interp[mask_esq] = is_np[0] + s_esq * (t_arr[mask_esq] - ts_np[0])
            # Ajustar inclinação direita após ts_np[-1]
            s_dir = (is_np[-1] - is_np[-2]) / max(1e-12, ts_np[-1] - ts_np[-2])
            mask_dir = t_arr > ts_np[-1]
            base_interp[mask_dir] = is_np[-1] + s_dir * (t_arr[mask_dir] - ts_np[-1])
            return base_interp

    # Construir máscara de zonas estabilizadas de purga (antes do 1º pulso + 35% finais de cada intervalo de purga)
    mask_estavel = np.zeros(len(t_arr), dtype=bool)
    if not janelas_ord:
        mask_estavel[:] = True
    else:
        # Purga inicial completa antes do 1º pulso
        mask_estavel |= t_arr <= janelas_ord[0][0]
        # Trecho final estabilizado de cada purga entre pulsos (evita ajustar a cauda exponencial logo após desligar o gás)
        for idx_j in range(len(janelas_ord)):
            p_ini = janelas_ord[idx_j][1]
            p_fim = janelas_ord[idx_j + 1][0] if (idx_j + 1) < len(janelas_ord) else t_max
            if p_fim > p_ini:
                t_estab = p_ini + 0.60 * (p_fim - p_ini)
                mask_estavel |= (t_arr >= t_estab) & (t_arr <= p_fim)

    if np.sum(mask_estavel) < 4:
        mask_estavel[:] = True
        for j_ini, j_fim in janelas_ord:
            mask_estavel &= ~((t_arr >= j_ini) & (t_arr <= j_fim))

    if np.sum(mask_estavel) < 3:
        return None

    grau = 2 if "Grau 2" in modo_drift else 1
    t_cent = t_arr - t_min
    coefs = np.polyfit(t_cent[mask_estavel], i_si[mask_estavel], deg=grau)
    return np.polyval(coefs, t_cent)


def render(cfg_plot: dict) -> None:
    usar_virgula = bool(cfg_plot.get("usar_virgula", True))

    st.subheader("Resposta Dinâmica de Sensoriamento (I × t) — Ciclos, Drift, t90 e Recuperação")

    file_it = st.file_uploader(
        "Carregar arquivo .dat / .csv / .txt de corrente versus tempo (Keithley)",
        type=["dat", "txt", "csv"],
        key="it_uploader",
    )

    opcoes_parser, slot_diag = render_painel_override_parser("it")

    # Ler previamente o arquivo (se presente) para disponibilizar info_escala como sugestão informativa
    df_it = None
    info_escala = None
    if file_it is not None:
        try:
            df_it, meta_it, info_escala = extrair_keithley_it(file_it.getvalue(), opcoes=opcoes_parser)
            exibir_diagnostico_leitura(slot_diag, meta_it, file_it.name)
        except Exception as exc:
            st.error(f"Erro ao ler arquivo I×t `{file_it.name}`: {exc}")
            return

    with st.expander("🎛️ Escala Bruta, Drift de Linha de Base, Ciclos e Unidades", expanded=True):
        if info_escala is not None:
            st.caption(
                f"🔎 **Inspeção da coluna `{info_escala['col_i']}`:** "
                f"mediana numérica bruta = `{info_escala['mediana_abs_bruta']:.4g}` | "
                f"Unidade sugerida pelo cabeçalho/magnitude: **`{info_escala['unidade_sugerida']}`** "
                "(confirme explicitamente no seletor abaixo)."
            )

        t1, t2, t3, t4 = st.columns(4)
        titulo_it = t1.text_input(
            "Título Superior do Gráfico",
            value="S1 - P3HT (Drop-casting) – Exposição à Umidade",
            key="it_titulo",
        )
        legenda_it = t2.text_input("Legenda da Curva", value="P3HT", key="it_leg")
        unid_i_bruta = t3.selectbox(
            "Unidade de Corrente Bruta no Arquivo",
            options=[
                "A (Ampere — Padrão Keithley)",
                "µA (Microampere)",
                "mA (Miliampere)",
                "nA (Nanoampere)",
            ],
            index=0,
            help="Define a unidade real em que os números foram gravados no arquivo de entrada.",
            key="it_unid_bruta",
        )
        modo_y_it = t4.selectbox(
            "Grandeza Exibida no Eixo Y",
            options=[
                "Corrente Absoluta (µA)",
                "Corrente Absoluta (nA)",
                "Corrente Absoluta (mA)",
                "Corrente Absoluta (A)",
                "Resposta Relativa ΔI/I₀ (%)",
                "Corrente Normalizada (I/I₀)",
            ],
            index=0,
            key="it_modo_y",
        )

        # Alerta amigável (sem jamais sobrescrever a escolha explícita do usuário!) se a sugestão divergir
        if info_escala is not None:
            sigla_escolhida = unid_i_bruta.split(" ")[0]
            sigla_sugerida = info_escala["unidade_sugerida"].split(" ")[0]
            if sigla_escolhida != sigla_sugerida:
                st.info(
                    f"💡 **Aviso de Escala:** O seletor está definido como **{sigla_escolhida}**, enquanto a "
                    f"heurística de leitura sugeriu **{info_escala['unidade_sugerida']}** (mediana bruta = "
                    f"`{info_escala['mediana_abs_bruta']:.4g}`). Se o arquivo já tiver sido exportado convertido, "
                    "ajuste o seletor 'Unidade de Corrente Bruta no Arquivo' acima."
                )

        t5, t6, t7, t8 = st.columns(4)
        unid_tempo_entrada = t5.selectbox(
            "Unidade de Tempo no Arquivo",
            ["Segundos (s)", "Minutos (min)", "Horas (h)"],
            index=0,
            key="it_t_in",
        )
        unid_tempo_plot = t6.selectbox(
            "Unidade de Tempo no Eixo X",
            ["Minutos (min)", "Segundos (s)", "Horas (h)"],
            index=0,
            key="it_t_out",
        )
        modo_drift = t7.selectbox(
            "Correção de Drift de Linha de Base",
            options=[
                "Desativada (Sinal Original)",
                "Âncoras Pré-Pulso (Interpolação Linear)",
                "Regressão Linear nas Zonas de Purga",
                "Polinomial Grau 2 nas Zonas de Purga",
                "Reta Simples Início–Fim da Janela",
            ],
            index=0,
            help="Remove a deriva lenta da linha de base preservando o patamar inicial I₀.",
            key="it_drift_mode",
        )
        criterio_trec = t8.selectbox(
            "Critério de Recuperação (t_rec90)",
            options=[
                "90% de retorno à base pré-pulso (Padrão)",
                "90% da amplitude recuperada na purga",
            ],
            index=0,
            help="No padrão clássico, exige retorno de 90% em direção ao I_base pré-pulso; no modo relativo à purga, mede 90% da relaxação efetivamente ocorrida no intervalo de purga.",
            key="it_crit_trec",
        )

        t9, t10, t11, t12 = st.columns(4)
        zerar_t0 = t9.checkbox("Zerar tempo inicial (t₀ = 0)", value=True, key="it_z_t0")
        marcar_pontos_cinetica = t10.checkbox(
            "Destacar pontos t90 e t_rec90 no gráfico",
            value=False,
            key="it_mk_t90",
        )
        cor_curva_it = t11.color_picker("Cor da Curva I×t", value="#000000", key="it_cor_c")
        cor_faixa_it = t12.color_picker("Cor das Faixas de Gás", value="#8CA2FF", key="it_cor_f")

        st.markdown("**Configuração das Janelas de Exposição ao Analito (Faixas Verticais):**")
        f1, f2, f3 = st.columns([1.25, 2.15, 1.0])
        modo_ciclos = f1.radio(
            "Definição dos Ciclos",
            ["Lista Manual de Intervalos", "Periódico Automático", "Sem Faixas"],
            index=0,
            key="it_modo_ciclos",
        )

        janelas_gas: List[Tuple[float, float]] = []
        sigla_t_plot = unid_tempo_plot.split(" ")[0]
        if modo_ciclos == "Lista Manual de Intervalos":
            str_janelas = f2.text_input(
                f"Intervalos de exposição ({sigla_t_plot}) separados por vírgula:",
                value="2-3, 5-6, 8-9, 11-12, 14-15",
                help="Exemplo: 5 pulsos de 1 min intercalados por 2 min de purga de N2.",
                key="it_str_jan",
            )
            for bloco in str_janelas.split(","):
                if "-" in bloco:
                    partes = bloco.strip().split("-")
                    if len(partes) == 2:
                        try:
                            ini, fim = float(partes[0]), float(partes[1])
                            if fim > ini:
                                janelas_gas.append((ini, fim))
                        except ValueError:
                            pass
        elif modo_ciclos == "Periódico Automático":
            pc1, pc2, pc3, pc4 = f2.columns(4)
            t_ini_1 = float(pc1.number_input("1º pulso", 0.0, 1e5, 2.0, step=0.5, key="it_p_ini"))
            dur_gas = float(pc2.number_input("Dur. Gás", 0.01, 1e4, 1.0, step=0.5, key="it_p_gas"))
            dur_purga = float(pc3.number_input("Dur. Purga", 0.01, 1e4, 2.0, step=0.5, key="it_p_pur"))
            n_ciclos = int(pc4.number_input("Nº Ciclos", 1, 50, 5, key="it_p_n"))
            for k_c in range(n_ciclos):
                ini = t_ini_1 + k_c * (dur_gas + dur_purga)
                janelas_gas.append((ini, ini + dur_gas))

        rotulo_purga = f3.text_input("Rótulo nas regiões de purga", value="N$_2$", key="it_rot_purga")

    if df_it is None:
        return

    # 1. Conversão explícita de tempo para segundos e depois para a unidade do eixo X
    fator_t_in_sec = {"Segundos (s)": 1.0, "Minutos (min)": 60.0, "Horas (h)": 3600.0}[unid_tempo_entrada]
    t_sec_full = df_it["Time_raw"].values.astype(float) * fator_t_in_sec
    if zerar_t0 and len(t_sec_full) > 0:
        t_sec_full = t_sec_full - t_sec_full[0]

    if unid_tempo_plot.startswith("Minutos"):
        t_plot_full = t_sec_full / 60.0
        fator_plot_para_seg = 60.0
        lbl_x_it = "Tempo em minutos"
    elif unid_tempo_plot.startswith("Horas"):
        t_plot_full = t_sec_full / 3600.0
        fator_plot_para_seg = 3600.0
        lbl_x_it = "Tempo (h)"
    else:
        t_plot_full = t_sec_full
        fator_plot_para_seg = 1.0
        lbl_x_it = "Tempo (s)"

    if float(np.max(t_plot_full)) <= float(np.min(t_plot_full)):
        st.warning("O vetor de tempo possui valores constantes.")
        return

    t_min_corte, t_max_corte = st.slider(
        f"Recortar janela temporal analisada ({lbl_x_it})",
        min_value=float(np.min(t_plot_full)),
        max_value=float(np.max(t_plot_full)),
        value=(float(np.min(t_plot_full)), float(np.max(t_plot_full))),
        key="it_slider_corte",
    )
    mask_t = (t_plot_full >= t_min_corte) & (t_plot_full <= t_max_corte)
    t_plot = t_plot_full[mask_t]

    if len(t_plot) < 3:
        st.warning("Selecione uma janela temporal com pelo menos 3 pontos.")
        return

    # 2. Conversão explícita da corrente bruta para Amperes (SI)
    fator_i_bruta_si = {
        "A (Ampere — Padrão Keithley)": 1.0,
        "µA (Microampere)": 1e-6,
        "mA (Miliampere)": 1e-3,
        "nA (Nanoampere)": 1e-9,
    }[unid_i_bruta]
    i_amp_orig = df_it["Current_raw"].values[mask_t].astype(float) * fator_i_bruta_si

    janelas_validas = sorted(
        [(a, b) for (a, b) in janelas_gas if b > t_min_corte and a < t_max_corte],
        key=lambda par: par[0],
    )

    # 3. Estimativa e correção opcional de Drift de Linha de Base
    curva_base_si = _estimar_curva_drift(t_plot, i_amp_orig, janelas_validas, modo_drift)
    mostrar_base_no_grafico = False
    if curva_base_si is not None:
        mostrar_base_no_grafico = st.checkbox(
            "Exibir também o sinal original + curva de deriva estimada ao fundo",
            value=False,
            key="it_show_drift_ref",
        )
        i_amp_si = i_amp_orig - (curva_base_si - curva_base_si[0])
    else:
        i_amp_si = i_amp_orig.copy()

    # Corrente inicial de referência I0
    if janelas_validas and np.any(t_plot <= janelas_validas[0][0]):
        pts_pre0 = i_amp_si[t_plot <= janelas_validas[0][0]]
        i0_base = float(np.median(pts_pre0[-min(len(pts_pre0), 10):]))
    else:
        i0_base = float(np.median(i_amp_si[: max(3, min(15, len(i_amp_si)))]))
    if abs(i0_base) < 1e-18:
        i0_base = 1e-12

    def _converter_si_para_eixo_y(arr_si: np.ndarray) -> Tuple[np.ndarray, str]:
        if modo_y_it == "Corrente Absoluta (µA)":
            return arr_si * 1e6, "Corrente, I (µA)"
        if modo_y_it == "Corrente Absoluta (nA)":
            return arr_si * 1e9, "Corrente, I (nA)"
        if modo_y_it == "Corrente Absoluta (mA)":
            return arr_si * 1e3, "Corrente, I (mA)"
        if modo_y_it == "Corrente Absoluta (A)":
            return arr_si, "Corrente, I (A)"
        if modo_y_it == "Resposta Relativa ΔI/I₀ (%)":
            return ((arr_si - i0_base) / abs(i0_base)) * 100.0, "Resposta Relativa, ΔI/I$_0$ (%)"
        return arr_si / i0_base, "Corrente Normalizada, I/I$_0$"

    y_it, lbl_y_it = _converter_si_para_eixo_y(i_amp_si)

    # 4. Extração de Sensibilidade, t90 e t_rec90 por Ciclo
    metricas_ciclos = []
    pontos_t90_plot = []
    pontos_trec_plot = []

    for idx_j, (j_ini, j_fim) in enumerate(janelas_validas):
        dur_pulso = max(1e-6, j_fim - j_ini)
        prox_ini = janelas_validas[idx_j + 1][0] if (idx_j + 1) < len(janelas_validas) else t_max_corte

        mask_pre = (t_plot >= j_ini - 0.35 * dur_pulso) & (t_plot <= j_ini)
        mask_pulso = (t_plot >= j_ini) & (t_plot <= j_fim)
        mask_purga = (t_plot >= j_fim) & (t_plot <= prox_ini)

        if not (np.any(mask_pre) and np.any(mask_pulso)):
            continue

        # Linha de base imediatamente anterior ao pulso
        pts_pre_c = i_amp_si[mask_pre]
        i_base_c = float(np.median(pts_pre_c[-min(len(pts_pre_c), 5):]))

        t_pulso = t_plot[mask_pulso]
        i_pulso = i_amp_si[mask_pulso]

        idx_ext = int(np.argmax(np.abs(i_pulso - i_base_c)))
        i_pico_c = float(i_pulso[idx_ext])
        delta_i = i_pico_c - i_base_c
        sens_pct = (delta_i / abs(i_base_c)) * 100.0 if abs(i_base_c) > 1e-18 else np.nan

        # Cálculo interpolado de t90 (subida/descida até 90% de delta_i)
        t90_plot_unid = np.nan
        if abs(delta_i) > 1e-18:
            progresso_resp = (i_pulso - i_base_c) / delta_i
            t90_plot_unid = _interpolar_tempo_90(t_pulso, progresso_resp, nivel=0.90)
            if np.isfinite(t90_plot_unid):
                t_abs_90 = float(t_pulso[0]) + t90_plot_unid
                y_abs_90 = float(np.interp(t_abs_90, t_plot, y_it))
                pontos_t90_plot.append((t_abs_90, y_abs_90))

        # Cálculo interpolado de t_rec90 e Taxa de Recuperação (%) na janela de purga subsequente
        trec90_plot_unid = np.nan
        taxa_rec_pct = np.nan
        if np.any(mask_purga) and np.sum(mask_purga) >= 2:
            t_purga = t_plot[mask_purga]
            i_purga = i_amp_si[mask_purga]
            i_off = float(i_purga[0])
            i_fim_purga = float(np.median(i_purga[-min(len(i_purga), 4):]))

            desvio_off = i_off - i_base_c
            if abs(desvio_off) > 1e-18:
                taxa_rec_pct = float(((i_off - i_fim_purga) / desvio_off) * 100.0)

            if criterio_trec.startswith("90% de retorno à base"):
                denom_rec = i_base_c - i_off
            else:
                denom_rec = i_fim_purga - i_off

            if abs(denom_rec) > 1e-18:
                progresso_rec = (i_purga - i_off) / denom_rec
                trec90_plot_unid = _interpolar_tempo_90(t_purga, progresso_rec, nivel=0.90)
                if np.isfinite(trec90_plot_unid):
                    t_abs_rec = float(t_purga[0]) + trec90_plot_unid
                    y_abs_rec = float(np.interp(t_abs_rec, t_plot, y_it))
                    pontos_trec_plot.append((t_abs_rec, y_abs_rec))

        metricas_ciclos.append(
            {
                "Ciclo": idx_j + 1,
                f"Pulso ({sigla_t_plot})": f"{j_ini:g} – {j_fim:g}",
                "I_base (µA)": round(i_base_c * 1e6, 3),
                "I_pico (µA)": round(i_pico_c * 1e6, 3),
                "Sensib. ΔI/I₀ (%)": round(sens_pct, 2) if np.isfinite(sens_pct) else np.nan,
                "t90 (s)": round(t90_plot_unid * fator_plot_para_seg, 2) if np.isfinite(t90_plot_unid) else np.nan,
                "t_rec90 (s)": round(trec90_plot_unid * fator_plot_para_seg, 2) if np.isfinite(trec90_plot_unid) else np.nan,
                "Recuperação (%)": round(taxa_rec_pct, 1) if np.isfinite(taxa_rec_pct) else np.nan,
            }
        )

    # 5. Plotagem Padrão OriginPro
    fig_it, ax_it = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)

    y_todos_lim = [y_it]
    if curva_base_si is not None and mostrar_base_no_grafico:
        y_orig_plot, _ = _converter_si_para_eixo_y(i_amp_orig)
        y_base_plot, _ = _converter_si_para_eixo_y(curva_base_si)
        ax_it.plot(t_plot, y_orig_plot, color="#888888", alpha=0.55, lw=1.0, label="Sinal Bruto (Com Drift)", zorder=2)
        ax_it.plot(t_plot, y_base_plot, "--", color="#C0392B", lw=1.2, label="Deriva de Base Estimada", zorder=2)
        y_todos_lim.extend([y_orig_plot, y_base_plot])

    lbl_curva_prin = f"{legenda_it} (Drift Corrigido)" if curva_base_si is not None else legenda_it
    ax_it.plot(t_plot, y_it, color=cor_curva_it, linewidth=1.4, label=lbl_curva_prin, zorder=3)

    if marcar_pontos_cinetica:
        if pontos_t90_plot:
            tx90, ty90 = zip(*pontos_t90_plot)
            ax_it.plot(tx90, ty90, "o", color="#D35400", markersize=5.5, label="Ponto t$_{90}$", zorder=5)
        if pontos_trec_plot:
            txr, tyr = zip(*pontos_trec_plot)
            ax_it.plot(txr, tyr, "s", color="#1E8449", markersize=5.2, label="Ponto t$_{rec90}$", zorder=5)

    y_concat = np.concatenate(y_todos_lim)
    y_span_it = max(1e-12, float(np.max(y_concat) - np.min(y_concat)))
    ax_it.set_ylim(float(np.min(y_concat)) - 0.10 * y_span_it, float(np.max(y_concat)) + 0.16 * y_span_it)
    ax_it.set_xlim(t_min_corte, t_max_corte)

    for j_ini, j_fim in janelas_validas:
        ax_it.axvspan(j_ini, j_fim, color=cor_faixa_it, alpha=0.55, lw=0, zorder=1)

    y_topo_txt = float(np.max(y_concat)) + 0.08 * y_span_it
    if rotulo_purga.strip() and janelas_validas:
        bordas_purga = [t_min_corte]
        for j_ini, j_fim in janelas_validas:
            bordas_purga.append(j_ini)
            bordas_purga.append(j_fim)
        bordas_purga.append(t_max_corte)
        for k_b in range(0, len(bordas_purga) - 1, 2):
            p_a = max(t_min_corte, bordas_purga[k_b])
            p_b = min(t_max_corte, bordas_purga[k_b + 1])
            if (p_b - p_a) > 0.03 * (t_max_corte - t_min_corte):
                ax_it.text(
                    (p_a + p_b) / 2.0,
                    y_topo_txt,
                    rotulo_purga,
                    ha="center",
                    va="center",
                    fontsize=max(7, cfg_plot["font_ticks"] - 1),
                )

    if titulo_it.strip():
        ax_it.set_title(titulo_it, fontsize=cfg_plot["font_labels"], pad=8)
    ax_it.set_xlabel(lbl_x_it, fontsize=cfg_plot["font_labels"], labelpad=7)
    ax_it.set_ylabel(lbl_y_it, fontsize=cfg_plot["font_labels"], labelpad=7)

    leg_it = ax_it.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=cfg_plot["font_legend"],
    )
    leg_it.get_frame().set_linewidth(0.85)

    aplicar_estilo_origin(ax_it, cfg_plot)
    plt.tight_layout()

    cg_it, cm_it = st.columns([2.15, 1.15])
    with cg_it:
        st.pyplot(fig_it)
    with cm_it:
        botao_download_figura(fig_it, f"Ixt_{legenda_it}", cfg_plot, "it_fig")

        df_serie_export = pd.DataFrame(
            {
                f"Tempo ({sigla_t_plot})": t_plot,
                "Corrente_Original (uA)": i_amp_orig * 1e6,
                "Corrente_Processada (uA)": i_amp_si * 1e6,
                lbl_y_it: y_it,
            }
        )
        if curva_base_si is not None:
            df_serie_export["Linha_Base_Estimada (uA)"] = curva_base_si * 1e6

        botao_download_csv(
            df_serie_export,
            f"Serie_Temporal_Ixt_{legenda_it}",
            cfg_plot,
            "it_serie",
            label="📈 Baixar Série Temporal Processada (.CSV)",
        )

        if metricas_ciclos:
            st.markdown("#### Cinética por Ciclo (`t90` & `t_rec90`)")
            df_mc = pd.DataFrame(metricas_ciclos)
            st.dataframe(df_mc, use_container_width=True, hide_index=True)

            mc1, mc2 = st.columns(2)
            if df_mc["Sensib. ΔI/I₀ (%)"].notna().any():
                mc1.metric("Sensib. Média", f"{fmt_num(float(df_mc['Sensib. ΔI/I₀ (%)'].mean()), 2, usar_virgula)} %")
            if df_mc["Recuperação (%)"].notna().any():
                mc2.metric("Recuperação Média", f"{fmt_num(float(df_mc['Recuperação (%)'].mean()), 1, usar_virgula)} %")

            mc3, mc4 = st.columns(2)
            if df_mc["t90 (s)"].notna().any():
                mc3.metric("t90 Médio", f"{fmt_num(float(df_mc['t90 (s)'].mean()), 1, usar_virgula)} s")
            if df_mc["t_rec90 (s)"].notna().any():
                mc4.metric("t_rec90 Médio", f"{fmt_num(float(df_mc['t_rec90 (s)'].mean()), 1, usar_virgula)} s")
            else:
                mc4.metric("t_rec90 Médio", "< 90% na purga")

            botao_download_csv(
                df_mc,
                f"Metricas_Ciclos_Ixt_{legenda_it}",
                cfg_plot,
                "it_ciclos",
                label="📊 Baixar Métricas dos Ciclos (.CSV)",
            )

    plt.close(fig_it)