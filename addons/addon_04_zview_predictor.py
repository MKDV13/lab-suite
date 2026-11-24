# -*- coding: utf-8 -*-
"""
addons/addon_04_zview_predictor.py
Módulo 4: Preditor e Pré-Ajuste de Circuito Equivalente para o ZView 2.9c (Solartron)
  - Cache reativo (@st.cache_data) no teste causal Boukamp Lin-KK e na otimização CNLS,
    tornando instantânea qualquer troca de unidades, cores, modelos do ranking ou formato decimal
  - Suporte automático ou manual ao indutor parasita L1 em série (j·ω·L1) quando há indutância
    de cabos/ponteiras em alta frequência (Z'' > 0 / -Z'' < 0)
  - Seleção estatística AIC/BIC com penalização contra sobre-parametrização
  - Diagrama elétrico modular em estágios + exportação limpa de parâmetros e arquivo .z para o ZView
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy.optimize import least_squares
    from scipy.signal import find_peaks, savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from core import resultados as registro_central
from core.parsers import (
    extrair_solartron_eis,
    render_painel_override_parser,
)
from core.plotting import (
    aplicar_estilo_origin,
    botao_download_csv,
    botao_download_figura,
    fmt_eng_zview,
    fmt_num,
    formatar_eixo_freq_bode,
)
from core.ui import (
    ColetorResultados,
    EntradaArquivos,
    cabecalho_secao,
    ler_arquivos,
    msg_aviso,
    msg_sem_dados,
    sessao_resultados,
    uploader_arquivos,
)

TITLE = "4. Preditor Circuito ZView"
ICON = "🧠"
ORDER = 40

# Registro central de resultados (core/resultados.py). A amostra usa a MESMA regra de nome da Aba 3
# (legenda = nome do arquivo sem extensão), para o EIS e o ajuste ZView da mesma amostra se encontrarem.
_ORIGEM_REGISTRO = "addon_04_zview_predictor"


# ==============================================================================
# ELEMENTOS ELETROQUÍMICOS ELEMENTARES E ESTRUTURA DE RESULTADO
# ==============================================================================
def z_rc(w: np.ndarray, r: float, c: float) -> np.ndarray:
    return r / (1.0 + 1j * w * r * c)


def z_rcpe(w: np.ndarray, r: float, t: float, p: float) -> np.ndarray:
    return r / (1.0 + r * t * ((1j * w) ** p))


def z_warburg_semi(w: np.ndarray, aw: float) -> np.ndarray:
    """Impedância de Warburg semi-infinita: Aw * (1 - j) / sqrt(w)."""
    return aw * (1.0 - 1j) / np.sqrt(w)


def calc_c_efetiva_brug(r: float, t: float, p: float, rs: float = 0.0) -> Tuple[float, float]:
    """
    Converte parâmetros CPE (T, P) em capacitância efetiva física real (Farads):
      - Hsu-Mansfeld (relaxação volumétrica / contornos de grão): C_hm = T^(1/P) * R^((1-P)/P)
      - Brug (interface bloqueante/faradaica acoplada a Rs): C_brug = T^(1/P) * (Rs^-1 + R^-1)^((P-1)/P)
    """
    if r <= 0 or t <= 0 or p <= 0:
        return np.nan, np.nan
    c_hm = (t ** (1.0 / p)) * (r ** ((1.0 - p) / p))
    if rs > 1.0:
        inv_r = (1.0 / rs) + (1.0 / r)
        c_brug = (t ** (1.0 / p)) * (inv_r ** ((p - 1.0) / p))
    else:
        c_brug = c_hm
    return float(c_hm), float(c_brug)


@dataclass
class ResultadoCircuito:
    id_modelo: str
    tem_l1: bool
    codigo_zview: str
    nome_curto: str
    interpretacao_fisica: str
    param_nomes: List[str]
    param_valores: Dict[str, float]
    param_erros_pct: Dict[str, float]
    derivados: Dict[str, float]
    z_calc_real: np.ndarray
    z_calc_imag: np.ndarray
    chi2_red: float
    erro_medio_pct: float
    aic: float
    bic: float
    score_selecao: float
    penalidade_motivo: str
    passos_zview: List[str]


# ==============================================================================
# NÚCLEO MATEMÁTICO CACHEADO (@st.cache_data) — LIN-KK E OTIMIZAÇÃO CNLS
# ==============================================================================
@st.cache_data(show_spinner=False, max_entries=64)
def calcular_residuo_linkk_cached(
    f_tup: Tuple[float, ...],
    zr_tup: Tuple[float, ...],
    zi_tup: Tuple[float, ...],
) -> float:
    """
    Teste causal de Kramers-Kronig (Boukamp Lin-KK normalizado via SVD).
    Cacheado sobre as tuplas numéricas para resposta instantânea na interface.
    """
    try:
        f = np.asarray(f_tup, dtype=float)
        zr = np.asarray(zr_tup, dtype=float)
        zi = np.asarray(zi_tup, dtype=float)

        w = 2.0 * np.pi * f
        z_scale = max(1.0, float(np.max(np.hypot(zr, zi))))
        zr_n = zr / z_scale
        zi_n = zi / z_scale
        z_mod_n = np.maximum(np.hypot(zr_n, zi_n), 1e-4)

        w_min, w_max = float(np.min(w)), float(np.max(w))
        n_taus = min(max(12, int(len(f) * 0.55)), 34)
        taus = np.logspace(np.log10(0.2 / w_max), np.log10(5.0 / w_min), n_taus)

        n = len(w)
        A_re = np.zeros((n, n_taus + 3))
        A_im = np.zeros((n, n_taus + 3))
        A_re[:, 0] = 1.0 / z_mod_n
        A_im[:, 1] = (w / w_max) / z_mod_n
        A_im[:, 2] = -(w_min / w) / z_mod_n

        for k, tau in enumerate(taus):
            wt = w * tau
            denom = 1.0 + wt * wt
            A_re[:, k + 3] = (1.0 / denom) / z_mod_n
            A_im[:, k + 3] = (-wt / denom) / z_mod_n

        A = np.vstack([A_re, A_im])
        b = np.concatenate([zr_n / z_mod_n, zi_n / z_mod_n])
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-10)
        res = A @ x - b
        return float(np.sqrt(np.mean(res ** 2)) * 100.0)
    except Exception:
        return np.nan


def _diagnosticar_topologia_espectro(
    f: np.ndarray,
    zr: np.ndarray,
    zi: np.ndarray,
) -> Dict[str, Any]:
    """Inspeciona picos de relaxação, Rs, indutância de alta frequência e cauda de Warburg."""
    mzi = -zi
    n_pts = len(f)
    phase_deg = np.rad2deg(np.arctan2(zi, zr))

    idx_asc = np.argsort(f)
    f_asc = f[idx_asc]
    mzi_asc = mzi[idx_asc]
    zr_asc = zr[idx_asc]
    ph_asc = -phase_deg[idx_asc]

    # Descartar cauda puramente indutiva (mzi < 0) ao procurar picos capacitivos
    mzi_pos_asc = np.maximum(0.0, mzi_asc)
    win = min(11, n_pts if n_pts % 2 == 1 else n_pts - 1)
    if SCIPY_AVAILABLE and win >= 5:
        mzi_smooth = savgol_filter(mzi_pos_asc, window_length=win, polyorder=2)
        ph_smooth = savgol_filter(np.maximum(0.0, ph_asc), window_length=win, polyorder=2)
    else:
        mzi_smooth = mzi_pos_asc
        ph_smooth = np.maximum(0.0, ph_asc)

    max_mzi = max(1e-9, float(np.max(mzi_smooth)))
    zr_span = max(1.0, float(np.max(zr) - np.min(zr)))

    picos_freq: List[float] = []
    if SCIPY_AVAILABLE and n_pts >= 8:
        dist_min = max(3, int(n_pts * 0.10))
        pk_idx, _ = find_peaks(mzi_smooth, prominence=0.035 * max_mzi, distance=dist_min)
        pk_ph, _ = find_peaks(ph_smooth, prominence=3.5, distance=dist_min)
        if len(pk_idx) == 1 and len(pk_ph) >= 2:
            todos = sorted(list(set(pk_idx.tolist() + pk_ph.tolist())))
            filtrados = []
            for idx_p in todos:
                if not filtrados or abs(np.log10(f_asc[idx_p]) - np.log10(f_asc[filtrados[-1]])) >= 0.65:
                    if mzi_smooth[idx_p] > 0.08 * max_mzi:
                        filtrados.append(idx_p)
            pk_idx = np.array(filtrados, dtype=int)

        for i in pk_idx:
            picos_freq.append(float(f_asc[i]))

    if not picos_freq:
        idx_max = int(np.argmax(mzi_pos_asc))
        picos_freq = [float(f_asc[idx_max])]

    # Verificar Rs e artefato indutivo nas maiores frequências (f está ordenado decrescente)
    zr_hf = float(np.median(zr[: min(3, n_pts)]))
    zr_lf = float(np.max(zr))
    razao_rs = zr_hf / max(1.0, zr_lf)
    rs_relevante = razao_rs > 0.008 and zr_hf > 15.0

    n_hf = max(2, int(n_pts * 0.15))
    zi_hf = zi[:n_hf]
    f_hf = f[:n_hf]
    tem_indutancia_hf = bool(np.any(zi_hf > max(2.0, 0.012 * max_mzi)))
    if tem_indutancia_hf:
        idx_l_max = int(np.argmax(zi_hf))
        l1_seed = max(1e-9, float(zi_hf[idx_l_max] / (2.0 * np.pi * f_hf[idx_l_max])))
    else:
        l1_seed = 1e-6

    # Verificar cauda difusional de Warburg nas menores frequências
    n_lf = max(4, int(n_pts * 0.15))
    zr_tail = zr_asc[:n_lf]
    mzi_tail = mzi_asc[:n_lf]
    tem_warburg = False
    slope_tail = 0.0
    if len(zr_tail) >= 4:
        dzr = float(zr_tail[0] - zr_tail[-1])
        dmzi = float(mzi_tail[0] - mzi_tail[-1])
        if dzr > 0.06 * zr_span and dmzi > 0.05 * max_mzi:
            coefs = np.polyfit(zr_tail, mzi_tail, 1)
            slope_tail = float(coefs[0])
            tem_minimo_interm = mzi_tail[0] > mzi_tail[-1] and float(np.max(mzi_asc[n_lf:])) > mzi_tail[-1] * 1.15
            if 0.35 <= slope_tail <= 1.85 and tem_minimo_interm:
                tem_warburg = True

    kk_erro_pct = calcular_residuo_linkk_cached(
        tuple(float(x) for x in f),
        tuple(float(x) for x in zr),
        tuple(float(x) for x in zi),
    )

    return {
        "n_picos": len(picos_freq),
        "picos_freq": sorted(picos_freq, reverse=True),
        "zr_hf": max(0.0, zr_hf),
        "zr_lf": zr_lf,
        "zr_span": zr_span,
        "rs_relevante": rs_relevante,
        "tem_indutancia_hf": tem_indutancia_hf,
        "l1_seed": float(np.clip(l1_seed, 1e-9, 5e-2)),
        "tem_warburg": tem_warburg,
        "slope_tail": slope_tail,
        "kk_erro_pct": kk_erro_pct,
    }


@st.cache_data(show_spinner=False, max_entries=32)
def executar_predicao_cnls_cached(
    f_tup: Tuple[float, ...],
    zr_tup: Tuple[float, ...],
    zi_tup: Tuple[float, ...],
    modo_peso: str,
    modo_indutor: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ajusta a biblioteca de circuitos ZView via CNLS (Complex Non-Linear Least Squares).
    Retorna dicionários puros serializáveis pelo cache do Streamlit (zero risco de PicklingError em reload dinâmico).
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("A biblioteca 'scipy' é necessária para o ajuste CNLS.")

    f = np.asarray(f_tup, dtype=float)
    zr_exp = np.asarray(zr_tup, dtype=float)
    zi_exp = np.asarray(zi_tup, dtype=float)
    w = 2.0 * np.pi * f
    z_exp = zr_exp + 1j * zi_exp
    n_pts = len(f)

    diag = _diagnosticar_topologia_espectro(f, zr_exp, zi_exp)
    z_max = float(np.max(np.abs(z_exp)))
    if modo_peso.startswith("Modulus"):
        escala_peso = np.maximum(np.abs(z_exp), 0.015 * z_max)
    else:
        escala_peso = np.full(n_pts, z_max * 0.35)

    r_span = max(10.0, diag["zr_span"])
    rs_seed = max(1.0, diag["zr_hf"] * 0.85) if diag["zr_hf"] > 5.0 else 10.0
    l1_seed = diag["l1_seed"]

    f_picos = diag["picos_freq"]
    f_med = float(np.sqrt(np.min(f) * np.max(f)))
    f_p1 = f_picos[0] if len(f_picos) >= 1 else f_med
    if len(f_picos) >= 2:
        f_hi, f_lo = max(f_picos[0], f_picos[1]), min(f_picos[0], f_picos[1])
    else:
        f_hi = min(float(np.max(f)) * 0.35, f_p1 * 4.0)
        f_lo = max(float(np.min(f)) * 2.5, f_p1 * 0.15)

    p_seed = 0.88
    c1_seed = max(1e-13, 1.0 / (2.0 * np.pi * f_p1 * r_span))
    t1_seed = (c1_seed ** p_seed) * (r_span ** (p_seed - 1.0))

    r_hi_seed = 0.45 * r_span
    r_lo_seed = 0.55 * r_span
    c_hi_seed = max(1e-13, 1.0 / (2.0 * np.pi * f_hi * r_hi_seed))
    c_lo_seed = max(1e-13, 1.0 / (2.0 * np.pi * f_lo * r_lo_seed))
    t_hi_seed = (c_hi_seed ** p_seed) * (r_hi_seed ** (p_seed - 1.0))
    t_lo_seed = (c_lo_seed ** p_seed) * (r_lo_seed ** (p_seed - 1.0))
    aw_seed = max(1.0, 0.15 * r_span * np.sqrt(2.0 * np.pi * float(np.min(f))))

    modelos_base = [
        {
            "id": "R_CPE_NO_RS",
            "codigo": "(R1 || CPE1)",
            "nome": "1 Semicírculo sem Rs (Filme Homogêneo em IDE)",
            "fisica": (
                "Transporte dominado por uma única constante de tempo (volume do filme homogêneo, típico do P3HT em pó), "
                "onde a resistência ôhmica dos contatos metálicos de ouro (Rs) é desprezível frente à resistência do polímero."
            ),
            "params": [
                ("R1", True, r_span, 1.0, 1e11),
                ("CPE1-T", True, t1_seed, 1e-15, 1e-2),
                ("CPE1-P", False, p_seed, 0.40, 1.00),
            ],
            "func": lambda w_, p: z_rcpe(w_, p["R1"], p["CPE1-T"], p["CPE1-P"]),
            "passos": [
                "No ZView 2.9c, abra Tools -> Equivalent Circuits.",
                "Clique com o botão direito na linha -> New Parallel -> adicione o resistor R1 e, em paralelo, o elemento CPE1.",
                "Omitir Rs evita incertezas acima de 500% quando o arco parte praticamente da origem em alta frequência.",
                "Cole os chutes iniciais abaixo com Mode = 'Calc-Modulus', rode 1 ajuste com CPE1-P em 'Fixed' e depois libere tudo em 'Free (+)'.",
            ],
        },
        {
            "id": "RS_R_CPE",
            "codigo": "Rs - (R1 || CPE1)",
            "nome": "Randles Simples com CPE (Rs + 1 Constante de Tempo)",
            "fisica": (
                "Circuito clássico de 1 constante de tempo com resistência ôhmica em série (Rs de contatos/cabos) "
                "seguida do paralelo entre a resistência do filme (R1) e a capacitância distribuída (CPE1)."
            ),
            "params": [
                ("Rs", True, rs_seed, 1e-2, 1e9),
                ("R1", True, r_span, 1.0, 1e11),
                ("CPE1-T", True, t1_seed, 1e-15, 1e-2),
                ("CPE1-P", False, p_seed, 0.40, 1.00),
            ],
            "func": lambda w_, p: p["Rs"] + z_rcpe(w_, p["R1"], p["CPE1-T"], p["CPE1-P"]),
            "passos": [
                "No ZView 2.9c, abra Tools -> Equivalent Circuits.",
                "Adicione um resistor em série (Rs) -> à direita crie um bloco paralelo com R1 e CPE1.",
                "Cole os chutes iniciais abaixo. Se o erro % de Rs ficar >35%, marque Rs como 'Fixed' ou use o modelo sem Rs.",
            ],
        },
        {
            "id": "RS_R_C",
            "codigo": "Rs - (R1 || C1)",
            "nome": "Debye Ideal (Rs + Resistor/Capacitor Puro)",
            "fisica": (
                "Semicírculo ideal sem dispersão de frequência (expoente P = 1 exato). "
                "Adequado apenas para filmes sem rugosidade superficial ou heterogeneidade energética."
            ),
            "params": [
                ("Rs", True, rs_seed, 1e-2, 1e9),
                ("R1", True, r_span, 1.0, 1e11),
                ("C1", True, c1_seed, 1e-15, 1e-2),
            ],
            "func": lambda w_, p: p["Rs"] + z_rc(w_, p["R1"], p["C1"]),
            "passos": [
                "No ZView 2.9c, abra Tools -> Equivalent Circuits.",
                "Adicione Rs em série seguido de um paralelo entre o resistor R1 e um capacitor ideal C1.",
                "Cole os valores iniciais abaixo em 'Free (+)' e execute Fit.",
            ],
        },
        {
            "id": "VOIGT_2_NO_RS",
            "codigo": "(R1 || CPE1) - (R2 || CPE2)",
            "nome": "Duplo Voigt sem Rs (2 Semicírculos: Bulk + Interface/Grão)",
            "fisica": (
                "Dois processos físicos distintos em série (2 picos no Bode / 2 semicírculos no Nyquist, típico do P3HT em 'palha'): "
                "(R1 || CPE1) em alta frequência modela o transporte intrínseco nos domínios poliméricos (bulk) e "
                "(R2 || CPE2) em baixa frequência modela contornos de agregados ou a interface filme/eletrodo (Rs desprezível)."
            ),
            "params": [
                ("R1", True, r_hi_seed, 1.0, 1e11),
                ("CPE1-T", True, t_hi_seed, 1e-15, 1e-2),
                ("CPE1-P", False, p_seed, 0.40, 1.00),
                ("R2", True, r_lo_seed, 1.0, 1e11),
                ("CPE2-T", True, t_lo_seed, 1e-15, 1e-2),
                ("CPE2-P", False, p_seed, 0.40, 1.00),
            ],
            "func": lambda w_, p: (
                z_rcpe(w_, p["R1"], p["CPE1-T"], p["CPE1-P"])
                + z_rcpe(w_, p["R2"], p["CPE2-T"], p["CPE2-P"])
            ),
            "passos": [
                "No ZView 2.9c, abra Tools -> Equivalent Circuits.",
                "Crie o 1º bloco paralelo (R1 || CPE1) para a região de alta frequência.",
                "Em série logo à direita, adicione o 2º bloco paralelo (R2 || CPE2) para a região de baixa frequência.",
                "Cole todos os chutes iniciais abaixo, deixe CPE1-P e CPE2-P como 'Fixed' na 1ª rodada de Fit e depois passe tudo para 'Free (+)'.",
            ],
        },
        {
            "id": "RS_VOIGT_2",
            "codigo": "Rs - (R1 || CPE1) - (R2 || CPE2)",
            "nome": "Duplo Voigt Completo (Rs + 2 Semicírculos em Série)",
            "fisica": (
                "Modelo completo de duas constantes de tempo com resistência ôhmica série (Rs): "
                "separa Rs, transporte volumétrico (R1 || CPE1, alta frequência) e barreira interfacial/grãos (R2 || CPE2, baixa frequência)."
            ),
            "params": [
                ("Rs", True, rs_seed, 1e-2, 1e9),
                ("R1", True, r_hi_seed, 1.0, 1e11),
                ("CPE1-T", True, t_hi_seed, 1e-15, 1e-2),
                ("CPE1-P", False, p_seed, 0.40, 1.00),
                ("R2", True, r_lo_seed, 1.0, 1e11),
                ("CPE2-T", True, t_lo_seed, 1e-15, 1e-2),
                ("CPE2-P", False, p_seed, 0.40, 1.00),
            ],
            "func": lambda w_, p: (
                p["Rs"]
                + z_rcpe(w_, p["R1"], p["CPE1-T"], p["CPE1-P"])
                + z_rcpe(w_, p["R2"], p["CPE2-T"], p["CPE2-P"])
            ),
            "passos": [
                "No ZView 2.9c, abra Tools -> Equivalent Circuits.",
                "Monte em série: Rs -> 1º paralelo (R1 || CPE1) -> 2º paralelo (R2 || CPE2).",
                "Cole os chutes iniciais abaixo. Se Rs apresentar incerteza alta, fixe Rs ou utilize o Duplo Voigt sem Rs.",
            ],
        },
        {
            "id": "LADDER_NESTED_2",
            "codigo": "Rs - (CPE1 || [R1 - (R2 || CPE2)])",
            "nome": "Circuito Escada / Aninhado (Filme Poroso sobre Eletrodo)",
            "fisica": (
                "Arquitetura aninhada para filmes porosos permeáveis a vapor de água/analito: a capacitância geométrica do filme (CPE1) "
                "envolve o caminho resistivo nos poros (R1) até a interface eletroativa com os dedos de ouro (R2 || CPE2)."
            ),
            "params": [
                ("Rs", True, rs_seed, 1e-2, 1e9),
                ("CPE1-T", True, t_hi_seed, 1e-15, 1e-2),
                ("CPE1-P", False, p_seed, 0.45, 1.00),
                ("R1", True, r_hi_seed, 1.0, 1e11),
                ("R2", True, r_lo_seed, 1.0, 1e11),
                ("CPE2-T", True, t_lo_seed, 1e-15, 1e-2),
                ("CPE2-P", False, p_seed, 0.40, 1.00),
            ],
            "func": lambda w_, p: (
                p["Rs"]
                + 1.0
                / (
                    p["CPE1-T"] * ((1j * w_) ** p["CPE1-P"])
                    + 1.0 / (p["R1"] + z_rcpe(w_, p["R2"], p["CPE2-T"], p["CPE2-P"]))
                )
            ),
            "passos": [
                "No ZView 2.9c, adicione Rs em série.",
                "Crie um bloco paralelo externo: no ramo superior coloque CPE1; no ramo inferior coloque R1 em série com um sub-paralelo (R2 || CPE2).",
                "Cole os chutes iniciais da tabela abaixo e ajuste inicialmente com CPE1-P e CPE2-P fixos.",
            ],
        },
        {
            "id": "RANDLES_WARBURG",
            "codigo": "Rs - (CPE1 || [R1 - W1])",
            "nome": "Randles com Difusão de Warburg (1 Arco + Cauda 45°)",
            "fisica": (
                "Semicírculo em média/alta frequência acoplado a transporte difusional semi-infinito de moléculas de umidade "
                "ou portadores iônicos em baixa frequência (reta inclinada próxima a 45° no Nyquist)."
            ),
            "params": [
                ("Rs", True, rs_seed, 1e-2, 1e9),
                ("CPE1-T", True, t1_seed, 1e-15, 1e-2),
                ("CPE1-P", False, p_seed, 0.45, 1.00),
                ("R1", True, r_span * 0.75, 1.0, 1e11),
                ("W1-Aw", True, aw_seed, 1e-2, 1e11),
            ],
            "func": lambda w_, p: (
                p["Rs"]
                + 1.0
                / (
                    p["CPE1-T"] * ((1j * w_) ** p["CPE1-P"])
                    + 1.0 / (p["R1"] + z_warburg_semi(w_, p["W1-Aw"]))
                )
            ),
            "passos": [
                "No ZView 2.9c, adicione Rs em série.",
                "Crie um bloco paralelo: no ramo superior adicione CPE1; no ramo inferior coloque R1 em série com um elemento Warburg (Wo1 ou Ws1 com P = 0.5 fixo).",
                "Ajuste primeiro o semicírculo mantendo Warburg fixo e em seguida libere todos os parâmetros.",
            ],
        },
        {
            "id": "VOIGT2_WARBURG",
            "codigo": "Rs - (R1 || CPE1) - (CPE2 || [R2 - W1])",
            "nome": "2 Semicírculos + Difusão de Warburg em Baixa Frequência",
            "fisica": (
                "Combina transporte no volume do filme (R1 || CPE1), transferência interfacial (R2 || CPE2) "
                "e difusão lenta de espécies em baixíssima frequência (W1)."
            ),
            "params": [
                ("Rs", True, rs_seed, 1e-2, 1e9),
                ("R1", True, r_hi_seed, 1.0, 1e11),
                ("CPE1-T", True, t_hi_seed, 1e-15, 1e-2),
                ("CPE1-P", False, p_seed, 0.45, 1.00),
                ("CPE2-T", True, t_lo_seed, 1e-15, 1e-2),
                ("CPE2-P", False, p_seed, 0.45, 1.00),
                ("R2", True, r_lo_seed * 0.8, 1.0, 1e11),
                ("W1-Aw", True, aw_seed, 1e-2, 1e11),
            ],
            "func": lambda w_, p: (
                p["Rs"]
                + z_rcpe(w_, p["R1"], p["CPE1-T"], p["CPE1-P"])
                + 1.0
                / (
                    p["CPE2-T"] * ((1j * w_) ** p["CPE2-P"])
                    + 1.0 / (p["R2"] + z_warburg_semi(w_, p["W1-Aw"]))
                )
            ),
            "passos": [
                "No ZView 2.9c, monte Rs -> paralelo (R1 || CPE1) -> paralelo onde o ramo superior é CPE2 e o inferior tem R2 em série com Warburg.",
                "Ajuste em etapas: mantenha o Warburg fixo na 1ª rodada e libere ao final.",
            ],
        },
    ]

    # Construir lista efetiva de especificações conforme o modo de indutor L1 em série
    def _criar_variante_com_l1(spec_orig: dict) -> dict:
        fn_orig = spec_orig["func"]
        passo_l1 = (
            "Indutor de Alta Frequência: Adicione um indutor em série no início da linha "
            "(New Series Element -> L -> renomeie para L1) para compensar a indutância parasita dos cabos acima de ~100 kHz."
        )
        return {
            "id": spec_orig["id"],
            "tem_l1": True,
            "codigo": f"L1 - {spec_orig['codigo']}",
            "nome": f"{spec_orig['nome']} + Indutor L1",
            "fisica": spec_orig["fisica"] + " Inclui indutor série L1 para modelar a reatância indutiva (Z'' > 0) dos cabos em alta frequência.",
            "params": [("L1", True, l1_seed, 1e-10, 1e-1)] + list(spec_orig["params"]),
            "func": lambda w_, p, _f=fn_orig: 1j * w_ * p["L1"] + _f(w_, p),
            "passos": [passo_l1] + list(spec_orig["passos"]),
        }

    modelos_para_testar = []
    if modo_indutor.startswith("Forçar"):
        for mb in modelos_base:
            modelos_para_testar.append(_criar_variante_com_l1(mb))
    elif modo_indutor.startswith("Sem indutor"):
        for mb in modelos_base:
            m_copy = dict(mb)
            m_copy["tem_l1"] = False
            modelos_para_testar.append(m_copy)
    else:
        # Modo Automático: avalia sem L1 e, se houver indutância detectada em alta frequência, compete também com L1
        for mb in modelos_base:
            m_copy = dict(mb)
            m_copy["tem_l1"] = False
            modelos_para_testar.append(m_copy)
            if diag["tem_indutancia_hf"]:
                modelos_para_testar.append(_criar_variante_com_l1(mb))

    resultados_dict: List[Dict[str, Any]] = []
    n_obs = 2 * n_pts

    for spec in modelos_para_testar:
        p_defs = spec["params"]
        k_params = len(p_defs)
        if n_obs <= k_params + 2:
            continue

        x0, lb, ub = [], [], []
        for _, is_log, seed_v, min_v, max_v in p_defs:
            val_c = float(np.clip(seed_v, min_v * 1.001, max_v * 0.999))
            if is_log:
                x0.append(np.log10(val_c))
                lb.append(np.log10(min_v))
                ub.append(np.log10(max_v))
            else:
                x0.append(val_c)
                lb.append(min_v)
                ub.append(max_v)

        def _unpack(vec, _defs=p_defs):
            d = {}
            for idx_p, (nome_p, is_log, _, _, _) in enumerate(_defs):
                d[nome_p] = float(10.0 ** vec[idx_p]) if is_log else float(vec[idx_p])
            return d

        def _residuos(vec, _spec=spec):
            d = _unpack(vec)
            zc = _spec["func"](w, d)
            r_re = (zc.real - zr_exp) / escala_peso
            r_im = (zc.imag - zi_exp) / escala_peso
            return np.concatenate([r_re, r_im])

        try:
            opt = least_squares(
                _residuos,
                x0=np.array(x0, dtype=float),
                bounds=(np.array(lb, dtype=float), np.array(ub, dtype=float)),
                method="trf",
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=1200,
            )
        except Exception:
            continue

        p_vals = _unpack(opt.x)

        # Garantir ordenação R1/CPE1 = alta frequência e R2/CPE2 = baixa frequência nos modelos Duplo Voigt
        if spec["id"] in ("VOIGT_2_NO_RS", "RS_VOIGT_2"):
            _, c_eff1 = calc_c_efetiva_brug(p_vals["R1"], p_vals["CPE1-T"], p_vals["CPE1-P"])
            _, c_eff2 = calc_c_efetiva_brug(p_vals["R2"], p_vals["CPE2-T"], p_vals["CPE2-P"])
            tau1 = p_vals["R1"] * c_eff1
            tau2 = p_vals["R2"] * c_eff2
            if np.isfinite(tau1) and np.isfinite(tau2) and tau1 > tau2:
                p_vals["R1"], p_vals["R2"] = p_vals["R2"], p_vals["R1"]
                p_vals["CPE1-T"], p_vals["CPE2-T"] = p_vals["CPE2-T"], p_vals["CPE1-T"]
                p_vals["CPE1-P"], p_vals["CPE2-P"] = p_vals["CPE2-P"], p_vals["CPE1-P"]

        z_calc = spec["func"](w, p_vals)
        rss_w = float(np.sum(opt.fun ** 2))
        dof = max(1, n_obs - k_params)
        chi2_red = rss_w / dof
        erro_mod_pct = float(np.mean(np.abs(z_calc - z_exp) / np.maximum(np.abs(z_exp), 1.0)) * 100.0)

        erros_pct: Dict[str, float] = {}
        try:
            jac = opt.jac
            cov = chi2_red * np.linalg.pinv(jac.T @ jac + 1e-13 * np.eye(k_params))
            diag_cov = np.maximum(0.0, np.diag(cov))
            for idx_p, (nome_p, is_log, _, _, _) in enumerate(p_defs):
                sigma_theta = float(np.sqrt(diag_cov[idx_p]))
                if is_log:
                    erros_pct[nome_p] = float(min(999.9, 100.0 * np.log(10.0) * sigma_theta))
                else:
                    erros_pct[nome_p] = float(min(999.9, 100.0 * sigma_theta / max(1e-9, abs(opt.x[idx_p]))))
        except Exception:
            for nome_p, _, _, _, _ in p_defs:
                erros_pct[nome_p] = np.nan

        rss_safe = max(rss_w, 1e-24)
        aic = float(n_obs * np.log(rss_safe / n_obs) + 2.0 * k_params)
        bic = float(n_obs * np.log(rss_safe / n_obs) + k_params * np.log(n_obs))

        derivados: Dict[str, float] = {}
        rs_val = p_vals.get("Rs", 0.0)
        if "C1" in p_vals:
            derivados["C1_eff (F)"] = float(p_vals["C1"])
            derivados["C1_brug (F)"] = float(p_vals["C1"])
            derivados["tau1 (s)"] = float(p_vals["R1"] * p_vals["C1"])
            derivados["f_pico1 (Hz)"] = float(1.0 / max(1e-18, 2.0 * np.pi * derivados["tau1 (s)"]))
        if "CPE1-T" in p_vals and "R1" in p_vals:
            c_hm1, c_br1 = calc_c_efetiva_brug(p_vals["R1"], p_vals["CPE1-T"], p_vals["CPE1-P"], rs_val)
            derivados["C1_eff (F)"] = c_hm1
            derivados["C1_brug (F)"] = c_br1
            derivados["tau1 (s)"] = float(p_vals["R1"] * c_hm1)
            derivados["f_pico1 (Hz)"] = float(1.0 / max(1e-18, 2.0 * np.pi * derivados["tau1 (s)"]))
        if "CPE2-T" in p_vals and "R2" in p_vals:
            c_hm2, c_br2 = calc_c_efetiva_brug(p_vals["R2"], p_vals["CPE2-T"], p_vals["CPE2-P"], rs_val)
            derivados["C2_eff (F)"] = c_hm2
            derivados["C2_brug (F)"] = c_br2
            derivados["tau2 (s)"] = float(p_vals["R2"] * c_hm2)
            derivados["f_pico2 (Hz)"] = float(1.0 / max(1e-18, 2.0 * np.pi * derivados["tau2 (s)"]))

        # Penalizações físicas contra parâmetros fantasmas ou mal condicionados
        penalidade = 0.0
        motivos = []

        if "Rs" in p_vals:
            r_tot_mod = p_vals.get("R1", 0.0) + p_vals.get("R2", 0.0)
            if p_vals["Rs"] < 0.0015 * r_tot_mod or erros_pct.get("Rs", 0.0) > 55.0:
                penalidade += 18.0
                motivos.append("Rs desprezível / erro alto")

        if "L1" in p_vals:
            if erros_pct.get("L1", 0.0) > 55.0:
                penalidade += 22.0
                motivos.append("L1 mal condicionado (erro >55%)")

        if "tau1 (s)" in derivados and "tau2 (s)" in derivados:
            ratio_tau = max(derivados["tau1 (s)"], derivados["tau2 (s)"]) / max(
                1e-18, min(derivados["tau1 (s)"], derivados["tau2 (s)"])
            )
            r_min_frac = min(p_vals["R1"], p_vals["R2"]) / max(1.0, p_vals["R1"] + p_vals["R2"])
            if ratio_tau < 2.2:
                penalidade += 45.0
                motivos.append("Constantes τ1 e τ2 sobrepostas/redundantes")
            if r_min_frac < 0.025:
                penalidade += 40.0
                motivos.append("2º arco <2,5% da resistência total")

        if spec["id"] == "LADDER_NESTED_2":
            penalidade += 4.0

        if "W1-Aw" in p_vals and not diag["tem_warburg"] and erros_pct.get("W1-Aw", 0.0) > 25.0:
            penalidade += 35.0
            motivos.append("Sem cauda difusional clara")

        erros_principais = [
            v for k_n, v in erros_pct.items() if k_n not in ("Rs", "L1") and np.isfinite(v)
        ]
        if any(e > 65.0 for e in erros_principais):
            penalidade += 30.0
            motivos.append("Parâmetros mal condicionados (erro >65%)")

        resultados_dict.append(
            {
                "id_modelo": spec["id"],
                "tem_l1": bool(spec["tem_l1"]),
                "codigo_zview": spec["codigo"],
                "nome_curto": spec["nome"],
                "interpretacao_fisica": spec["fisica"],
                "param_nomes": [p[0] for p in p_defs],
                "param_valores": p_vals,
                "param_erros_pct": erros_pct,
                "derivados": derivados,
                "z_calc_real": z_calc.real,
                "z_calc_imag": z_calc.imag,
                "chi2_red": chi2_red,
                "erro_medio_pct": erro_mod_pct,
                "aic": aic,
                "bic": bic,
                "score_selecao": float(bic + penalidade),
                "penalidade_motivo": "; ".join(motivos) if motivos else "Ajuste bem condicionado",
                "passos_zview": spec["passos"],
            }
        )

    resultados_dict.sort(key=lambda d: d["score_selecao"])
    return resultados_dict, diag


# ==============================================================================
# MOTOR MODULAR DE DESENHO DO DIAGRAMA ELÉTRICO (COM OU SEM L1 E Rs)
# ==============================================================================
def _draw_box_elem(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    label_top: str,
    label_val: str,
    cor_borda: str = "#1B4F72",
    cor_fundo: str = "#EBF5FB",
) -> None:
    box = FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=cor_borda,
        facecolor=cor_fundo,
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy + 0.03,
        label_top,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#111111",
        zorder=5,
    )
    if label_val:
        ax.text(
            cx,
            cy - h / 2.0 - 0.18,
            label_val,
            ha="center",
            va="top",
            fontsize=8.3,
            color="#333333",
            zorder=5,
        )


def desenhar_esquema_zview(res: ResultadoCircuito, usar_virgula: bool = True) -> plt.Figure:
    """
    Desenha dinamicamente a cadeia de estágios em série/paralelo/escada do circuito recomendado,
    adaptando automaticamente o comprimento caso L1 e/ou Rs estejam presentes.
    """
    pv = res.param_valores

    def _f_ohm(v: float) -> str:
        if v >= 1e6:
            return f"{fmt_num(v / 1e6, 2, usar_virgula)} MΩ"
        if v >= 1e3:
            return f"{fmt_num(v / 1e3, 2, usar_virgula)} kΩ"
        return f"{fmt_num(v, 1, usar_virgula)} Ω"

    def _f_ind(v: float) -> str:
        if v >= 1e-3:
            return f"{fmt_num(v * 1e3, 2, usar_virgula)} mH"
        if v >= 1e-6:
            return f"{fmt_num(v * 1e6, 2, usar_virgula)} µH"
        return f"{v:.2e} H"

    def _f_cpe(t_key: str, p_key: str) -> str:
        return f"T={pv[t_key]:.2e}\nP={fmt_num(pv[p_key], 3, usar_virgula)}"

    # Montar lista sequencial de estágios topológicos
    estagios: List[Tuple] = []
    if res.tem_l1 and "L1" in pv:
        estagios.append(("series", ("L1", _f_ind(pv["L1"]), "#0E6251", "#E8F8F5")))
    if "Rs" in pv:
        estagios.append(("series", ("Rs", _f_ohm(pv["Rs"]), "#1E8449", "#EAFAF1")))

    mid = res.id_modelo
    if mid in ("R_CPE_NO_RS", "RS_R_CPE"):
        estagios.append(
            (
                "parallel",
                ("R1", _f_ohm(pv["R1"]), "#1B4F72", "#EBF5FB"),
                ("CPE1", _f_cpe("CPE1-T", "CPE1-P"), "#7D3C98", "#F5EEF8"),
            )
        )
    elif mid == "RS_R_C":
        estagios.append(
            (
                "parallel",
                ("R1", _f_ohm(pv["R1"]), "#1B4F72", "#EBF5FB"),
                ("C1", f"C={pv['C1']:.2e} F", "#B9770E", "#FEF9E7"),
            )
        )
    elif mid in ("VOIGT_2_NO_RS", "RS_VOIGT_2"):
        estagios.append(
            (
                "parallel",
                ("R1 (Alta f)", _f_ohm(pv["R1"]), "#1B4F72", "#EBF5FB"),
                ("CPE1", _f_cpe("CPE1-T", "CPE1-P"), "#7D3C98", "#F5EEF8"),
            )
        )
        estagios.append(
            (
                "parallel",
                ("R2 (Baixa f)", _f_ohm(pv["R2"]), "#922B21", "#FDEDEC"),
                ("CPE2", _f_cpe("CPE2-T", "CPE2-P"), "#7D3C98", "#F5EEF8"),
            )
        )
    elif mid == "LADDER_NESTED_2":
        estagios.append(
            (
                "ladder",
                ("CPE1 (Filme)", _f_cpe("CPE1-T", "CPE1-P"), "#7D3C98", "#F5EEF8"),
                ("R1", _f_ohm(pv["R1"]), "#1B4F72", "#EBF5FB"),
                ("R2", _f_ohm(pv["R2"]), "#922B21", "#FDEDEC"),
                ("CPE2", _f_cpe("CPE2-T", "CPE2-P"), "#7D3C98", "#F5EEF8"),
            )
        )
    elif mid == "RANDLES_WARBURG":
        estagios.append(
            (
                "randles_w",
                ("CPE1", _f_cpe("CPE1-T", "CPE1-P"), "#7D3C98", "#F5EEF8"),
                ("R1", _f_ohm(pv["R1"]), "#1B4F72", "#EBF5FB"),
                ("W1 (Warburg)", f"Aw={pv['W1-Aw']:.2e}", "#B9770E", "#FEF9E7"),
            )
        )
    else:  # VOIGT2_WARBURG
        estagios.append(
            (
                "parallel",
                ("R1 (Alta f)", _f_ohm(pv["R1"]), "#1B4F72", "#EBF5FB"),
                ("CPE1", _f_cpe("CPE1-T", "CPE1-P"), "#7D3C98", "#F5EEF8"),
            )
        )
        estagios.append(
            (
                "randles_w",
                ("CPE2", _f_cpe("CPE2-T", "CPE2-P"), "#7D3C98", "#F5EEF8"),
                ("R2", _f_ohm(pv["R2"]), "#922B21", "#FDEDEC"),
                ("W1 (Warburg)", f"Aw={pv['W1-Aw']:.2e}", "#B9770E", "#FEF9E7"),
            )
        )

    larguras_map = {"series": 1.85, "parallel": 2.90, "randles_w": 4.20, "ladder": 5.10}
    largura_total = 0.90 + sum(larguras_map[st_item[0]] for st_item in estagios)

    fig, ax = plt.subplots(figsize=(max(9.2, min(13.0, largura_total * 1.15)), 3.2), dpi=150)
    ax.set_xlim(0.0, largura_total)
    ax.set_ylim(-1.85, 1.95)
    ax.axis("off")

    lw = 2.0
    col_wire = "#222222"

    x_cur = 0.45
    ax.add_patch(Circle((x_cur - 0.08, 0.0), 0.08, facecolor="white", edgecolor=col_wire, linewidth=2.0, zorder=5))

    for item in estagios:
        tipo = item[0]
        if tipo == "series":
            lbl, val_s, cb, cf = item[1]
            ax.plot([x_cur, x_cur + 0.30], [0, 0], color=col_wire, lw=lw)
            _draw_box_elem(ax, x_cur + 0.925, 0.0, 1.25, 0.48, lbl, val_s, cb, cf)
            ax.plot([x_cur + 1.55, x_cur + 1.85], [0, 0], color=col_wire, lw=lw)
            x_cur += 1.85

        elif tipo == "parallel":
            top_el, bot_el = item[1], item[2]
            ax.plot([x_cur, x_cur + 0.35], [0, 0], color=col_wire, lw=lw)
            x_l, x_r = x_cur + 0.35, x_cur + 2.55
            ax.plot([x_l, x_l], [-0.75, 0.75], color=col_wire, lw=lw)
            ax.plot([x_r, x_r], [-0.75, 0.75], color=col_wire, lw=lw)
            ax.plot([x_l, x_r], [0.75, 0.75], color=col_wire, lw=lw)
            ax.plot([x_l, x_r], [-0.75, -0.75], color=col_wire, lw=lw)
            _draw_box_elem(ax, (x_l + x_r) / 2.0, 0.75, 1.45, 0.48, top_el[0], top_el[1], top_el[2], top_el[3])
            _draw_box_elem(ax, (x_l + x_r) / 2.0, -0.75, 1.45, 0.48, bot_el[0], bot_el[1], bot_el[2], bot_el[3])
            ax.plot([x_r, x_cur + 2.90], [0, 0], color=col_wire, lw=lw)
            x_cur += 2.90

        elif tipo == "randles_w":
            cpe_el, r_el, w_el = item[1], item[2], item[3]
            ax.plot([x_cur, x_cur + 0.35], [0, 0], color=col_wire, lw=lw)
            x_l, x_r = x_cur + 0.35, x_cur + 3.85
            ax.plot([x_l, x_l], [-0.75, 0.75], color=col_wire, lw=lw)
            ax.plot([x_r, x_r], [-0.75, 0.75], color=col_wire, lw=lw)
            ax.plot([x_l, x_r], [0.75, 0.75], color=col_wire, lw=lw)
            ax.plot([x_l, x_r], [-0.75, -0.75], color=col_wire, lw=lw)
            _draw_box_elem(ax, (x_l + x_r) / 2.0, 0.75, 1.50, 0.48, cpe_el[0], cpe_el[1], cpe_el[2], cpe_el[3])
            _draw_box_elem(ax, x_l + 0.95, -0.75, 1.30, 0.48, r_el[0], r_el[1], r_el[2], r_el[3])
            _draw_box_elem(ax, x_l + 2.55, -0.75, 1.45, 0.48, w_el[0], w_el[1], w_el[2], w_el[3])
            ax.plot([x_r, x_cur + 4.20], [0, 0], color=col_wire, lw=lw)
            x_cur += 4.20

        elif tipo == "ladder":
            cpe1_el, r1_el, r2_el, cpe2_el = item[1], item[2], item[3], item[4]
            ax.plot([x_cur, x_cur + 0.35], [0, 0], color=col_wire, lw=lw)
            x_l, x_r = x_cur + 0.35, x_cur + 4.75
            ax.plot([x_l, x_l], [-0.65, 1.05], color=col_wire, lw=lw)
            ax.plot([x_r, x_r], [-0.65, 1.05], color=col_wire, lw=lw)
            ax.plot([x_l, x_r], [1.05, 1.05], color=col_wire, lw=lw)
            _draw_box_elem(ax, (x_l + x_r) / 2.0, 1.05, 1.55, 0.46, cpe1_el[0], cpe1_el[1], cpe1_el[2], cpe1_el[3])

            # Ramo inferior com R1 em série com sub-bloco (R2 || CPE2)
            ax.plot([x_l, x_l + 0.35], [-0.65, -0.65], color=col_wire, lw=lw)
            _draw_box_elem(ax, x_l + 0.95, -0.65, 1.20, 0.46, r1_el[0], r1_el[1], r1_el[2], r1_el[3])
            ax.plot([x_l + 1.55, x_l + 1.95], [-0.65, -0.65], color=col_wire, lw=lw)
            xs_l, xs_r = x_l + 1.95, x_r - 0.35
            ax.plot([xs_l, xs_l], [-1.25, -0.05], color=col_wire, lw=lw)
            ax.plot([xs_r, xs_r], [-1.25, -0.05], color=col_wire, lw=lw)
            ax.plot([xs_l, xs_r], [-0.05, -0.05], color=col_wire, lw=lw)
            ax.plot([xs_l, xs_r], [-1.25, -1.25], color=col_wire, lw=lw)
            _draw_box_elem(ax, (xs_l + xs_r) / 2.0, -0.05, 1.30, 0.42, r2_el[0], r2_el[1], r2_el[2], r2_el[3])
            _draw_box_elem(ax, (xs_l + xs_r) / 2.0, -1.25, 1.30, 0.42, cpe2_el[0], cpe2_el[1], cpe2_el[2], cpe2_el[3])
            ax.plot([xs_r, x_r], [-0.65, -0.65], color=col_wire, lw=lw)
            ax.plot([x_r, x_cur + 5.10], [0, 0], color=col_wire, lw=lw)
            x_cur += 5.10

    ax.add_patch(Circle((x_cur + 0.08, 0.0), 0.08, facecolor="white", edgecolor=col_wire, linewidth=2.0, zorder=5))
    ax.set_title(
        f"Topologia Recomendada ZView:  {res.codigo_zview}",
        fontsize=12.5,
        fontweight="bold",
        pad=8,
        color="#111111",
    )
    plt.tight_layout()
    return fig


def gerar_espectro_sintetico_solartron(tipo: str) -> bytes:
    """Gera arquivo .z no padrão Solartron ZPlot para calibração imediata da interface."""
    f = np.logspace(6, -1, 65)
    w = 2.0 * np.pi * f
    rng = np.random.default_rng(42)

    if tipo == "1_arco_po":
        z = 45.0 + z_rcpe(w, 130e3, 1.6e-10, 0.94)
    elif tipo == "2_arcos_palha":
        z = 60.0 + z_rcpe(w, 480e3, 1.3e-10, 0.91) + z_rcpe(w, 360e3, 1.4e-8, 0.84)
    elif tipo == "2_arcos_indutor":
        z = 1j * w * 4.5e-6 + 85.0 + z_rcpe(w, 18e3, 2.2e-9, 0.91) + z_rcpe(w, 24e3, 1.8e-7, 0.86)
    else:
        z = 120.0 + 1.0 / (2.5e-10 * ((1j * w) ** 0.90) + 1.0 / (95e3 + z_warburg_semi(w, 42e3)))

    ruido = rng.normal(0, 0.004 * np.abs(z), size=(2, len(f)))
    z_noisy = (z.real + ruido[0]) + 1j * (z.imag + ruido[1])

    linhas = [
        "ZPLOT2 ASCII",
        "Instrument: Solartron 1260 Impedance Gain-Phase Analyzer",
        "End Comments",
        "Freq(Hz)\tAmpl\tBias\tTime(Sec)\tZ'(a)\tZ''(b)",
    ]
    for idx, freq in enumerate(f):
        linhas.append(
            f"{freq:.5e}\t0.1000\t0.0000\t{idx * 1.5:.2f}\t{z_noisy[idx].real:.5e}\t{z_noisy[idx].imag:.5e}"
        )
    return ("\n".join(linhas)).encode("utf-8")


def _exportar_formato_zplot_ascii(df_eis: pd.DataFrame) -> bytes:
    """Exporta o espectro filtrado em formato .z nativo compatível com abertura direta no ZView 2.9c."""
    linhas = [
        "ZPLOT2 ASCII",
        "Exported by Suite Analitica Modular (Filtered Spectrum for ZView 2.9c)",
        "End Comments",
        "Freq(Hz)\tAmpl\tBias\tTime(Sec)\tZ'(a)\tZ''(b)",
    ]
    for idx, row in df_eis.iterrows():
        linhas.append(
            f"{row['Freq']:.6e}\t0.1000\t0.0000\t{float(idx):.2f}\t{row['Z_real']:.6e}\t{row['Z_imag']:.6e}"
        )
    return ("\n".join(linhas)).encode("utf-8")


# ==============================================================================
# INTERFACE PRINCIPAL DO ADD-ON
# ==============================================================================
def _unidade_param_zview(nome_param: str) -> str:
    """Unidade do elemento ZView pelo nome (mesma convenção da tabela de chutes iniciais; '' = adimensional)."""
    if nome_param == "L1":
        return "H"
    if nome_param.startswith("R"):
        return "Ω"
    if nome_param.startswith("C1"):
        return "F"
    if nome_param.endswith("-T"):
        return "S·s^P"
    if nome_param.endswith("-P"):
        return ""
    return "Ω·s^-0.5"


def _amostra_da_chave_aba3(chave: str) -> str:
    """A Aba 3 publica 'legenda [arquivo]' (ou só o arquivo, se a legenda for igual a ele): devolve a legenda."""
    if chave.endswith("]") and " [" in chave:
        return chave[: chave.rindex(" [")]
    return chave


def _registrar_resultado_zview(
    coletor: ColetorResultados,
    amostra: str,
    ranking: List[ResultadoCircuito],
    diag_info: Dict[str, Any],
    meta_extra: Dict[str, Any],
) -> None:
    """Registra o modelo RECOMENDADO (1º do ranking): qualidade do ajuste, parâmetros ZView, derivados e diagnóstico."""
    melhor = ranking[0]
    metricas: Dict[str, Any] = {
        "modelo": melhor.codigo_zview,
        "n_parametros": int(len(melhor.param_nomes)),
        "erro_medio": float(melhor.erro_medio_pct),
        "chi2_red": float(melhor.chi2_red),
        "AIC": float(melhor.aic),
        "BIC": float(melhor.bic),
        "Z_real_alta_freq": float(diag_info["zr_hf"]),
        "KK_erro": float(diag_info["kk_erro_pct"]),
        "n_picos": int(diag_info["n_picos"]),
    }
    unidades: Dict[str, str] = {"erro_medio": "%", "Z_real_alta_freq": "Ω", "KK_erro": "%"}

    for nome_param in melhor.param_nomes:
        metricas[nome_param] = float(melhor.param_valores[nome_param])
        unidades[nome_param] = _unidade_param_zview(nome_param)
    for chave_deriv, valor_deriv in melhor.derivados.items():
        nome_deriv = chave_deriv.split(" (")[0]
        metricas[nome_deriv] = float(valor_deriv)
        if "(" in chave_deriv:
            unidades[nome_deriv] = chave_deriv.split("(")[-1].rstrip(")")

    meta = dict(meta_extra)
    meta["ranking_top3"] = [m.codigo_zview for m in ranking[:3]]
    meta["incertezas_pct"] = {k: float(v) for k, v in melhor.param_erros_pct.items()}
    coletor.adicionar(amostra, metricas, unidades={k: v for k, v in unidades.items() if v}, meta=meta)


def _render_corpo(cfg_plot: dict, coletor: ColetorResultados) -> None:
    usar_virgula = bool(cfg_plot.get("usar_virgula", True))

    cabecalho_secao(
        "🧠 Preditor Automático de Circuito Equivalente para o ZView 2.9c",
        "Analisa a topologia espectral da Solartron, valida Kramers-Kronig (Lin-KK), ajusta todos os modelos "
        "candidatos via CNLS com cache instantâneo e entrega o diagrama + parâmetros prontos para o ZView.",
    )

    if not SCIPY_AVAILABLE:
        st.error("A biblioteca `scipy` não está instalada. Execute `pip install scipy` no terminal e reinicie.")
        return

    col_src1, col_src2 = st.columns([1.9, 1.3])
    with col_src1:
        arquivos_zv = uploader_arquivos(
            "zv",
            "Carregar espectro individual da Solartron (.z, .txt, .csv, .dat)",
            tipos=("z", "dat", "txt", "csv"),
            key_uploader="zv_uploader",
        )
    with col_src2:
        opcoes_demo = [
            "Nenhum (usar arquivo enviado acima)",
            "🧪 Exemplo 1: P3HT Pó (1 Semicírculo Homogêneo)",
            "🧪 Exemplo 2: P3HT 'Palha' (2 Semicírculos / 2 Picos no Bode)",
            "🧪 Exemplo 3: Filme com Difusão de Umidade (Warburg)",
            "🧪 Exemplo 4: Alta Frequência com Indutância de Cabo (L1)",
        ]
        arqs_aba3 = list(st.session_state.get("eis_carregados_aba3", {}).keys())
        for k_a in reversed(arqs_aba3):
            opcoes_demo.insert(1, f"📂 Reutilizar da Aba 3: {k_a}")

        fonte_aux = st.selectbox(
            "Ou reutilize um espectro da Aba 3 / exemplo de calibração:",
            options=opcoes_demo,
            index=0,
            key="zv_src_sel",
        )

        # Estado explícito da ponte Aba 3 → Aba 4 (a Aba 3 pede rerun global ao mudar; o botão é um reforço manual).
        versao_aba3 = int(st.session_state.get("eis_aba3_versao", 0))
        st.caption(
            f"🔗 Aba 3: {len(arqs_aba3)} espectro(s) disponível(is) · sincronização nº {versao_aba3}."
        )
        st.button(
            "🔄 Sincronizar com a Aba 3",
            key="zv_btn_sync_aba3",
            use_container_width=True,
            help="Relê os espectros publicados pela Aba 3. Normalmente isso já ocorre sozinho "
                 "quando a Aba 3 muda; use se a lista parecer desatualizada.",
        )

    opcoes_parser, slot_diag = render_painel_override_parser("zv")
    entrada = EntradaArquivos(arquivos_zv, opcoes_parser, slot_diag, "zv")

    df_alvo_zv: Optional[pd.DataFrame] = None
    nome_alvo_zv = ""
    amostra_registro: Optional[str] = None  # None = não registrar (exemplos sintéticos não são medições)

    if not entrada.vazia:
        lidos = ler_arquivos(entrada, extrair_solartron_eis, contexto_erro="arquivo Solartron")
        if not lidos:
            return
        df_alvo_zv, _meta_zv = lidos[0].resultado
        nome_alvo_zv = lidos[0].nome
        amostra_registro = registro_central.amostra_de_arquivo(nome_alvo_zv)
    elif fonte_aux.startswith("📂 Reutilizar da Aba 3: "):
        chave = fonte_aux.replace("📂 Reutilizar da Aba 3: ", "")
        mapa_aba3 = st.session_state.get("eis_carregados_aba3", {})
        if chave in mapa_aba3:
            df_alvo_zv = mapa_aba3[chave].copy()
            nome_alvo_zv = chave
            amostra_registro = _amostra_da_chave_aba3(chave)
    elif "Exemplo 1" in fonte_aux:
        df_alvo_zv, _ = extrair_solartron_eis(gerar_espectro_sintetico_solartron("1_arco_po"))
        nome_alvo_zv = "Simulacao_P3HT_Po_1Arco.z"
    elif "Exemplo 2" in fonte_aux:
        df_alvo_zv, _ = extrair_solartron_eis(gerar_espectro_sintetico_solartron("2_arcos_palha"))
        nome_alvo_zv = "Simulacao_P3HT_Palha_2Arcos.z"
    elif "Exemplo 3" in fonte_aux:
        df_alvo_zv, _ = extrair_solartron_eis(gerar_espectro_sintetico_solartron("warburg"))
        nome_alvo_zv = "Simulacao_Randles_Warburg.z"
    elif "Exemplo 4" in fonte_aux:
        df_alvo_zv, _ = extrair_solartron_eis(gerar_espectro_sintetico_solartron("2_arcos_indutor"))
        nome_alvo_zv = "Simulacao_Indutor_L1_AltaFreq.z"

    if df_alvo_zv is None or df_alvo_zv.empty:
        msg_sem_dados(
            "Envie um espectro individual da Solartron ou selecione um espectro da Aba 3 / exemplo de calibração."
        )
        return

    zc1, zc2, zc3, zc4 = st.columns([1.6, 1.2, 1.3, 0.9])
    log_f_min = float(np.floor(np.log10(df_alvo_zv["Freq"].min()) * 10.0) / 10.0)
    log_f_max = float(np.ceil(np.log10(df_alvo_zv["Freq"].max()) * 10.0) / 10.0)
    faixa_log_f = zc1.slider(
        "Janela de Frequência — log₁₀(f / Hz)",
        min_value=log_f_min,
        max_value=log_f_max,
        value=(log_f_min, log_f_max),
        step=0.1,
        help="Permite cortar ruído extremo de alta frequência ou deriva em mHz antes do ajuste.",
        key="zv_f_slider",
    )
    modo_peso_zv = zc2.selectbox(
        "Ponderação CNLS",
        ["Modulus (Padrão ZView)", "Proporcional à Escala Máxima"],
        index=0,
        key="zv_peso",
    )
    modo_indutor_zv = zc3.selectbox(
        "Indutor Série L1 (Alta Freq.)",
        [
            "Automático (Se Z'' > 0 em alta f)",
            "Forçar inclusão de L1 em série",
            "Sem indutor L1 (Apenas R/C/CPE/W)",
        ],
        index=0,
        help="Compensa a reatância indutiva dos cabos/ponteiras da Solartron em altas frequências.",
        key="zv_l1_mode",
    )
    unid_zv_plot = zc4.selectbox("Unidade Gráfica", ["kΩ", "Ω", "MΩ"], index=0, key="zv_unid_p")

    df_fit = df_alvo_zv[
        (np.log10(df_alvo_zv["Freq"]) >= faixa_log_f[0] - 1e-6)
        & (np.log10(df_alvo_zv["Freq"]) <= faixa_log_f[1] + 1e-6)
    ].reset_index(drop=True)

    if len(df_fit) < 8:
        msg_aviso("Selecione uma janela de frequência com pelo menos 8 pontos experimentais.")
        return

    f_tup = tuple(float(x) for x in df_fit["Freq"].values)
    zr_tup = tuple(float(x) for x in df_fit["Z_real"].values)
    zi_tup = tuple(float(x) for x in df_fit["Z_imag"].values)

    with st.spinner("Ajustando biblioteca de circuitos ZView (resultado em cache para troca instantânea)..."):
        ranking_raw, diag_info = executar_predicao_cnls_cached(
            f_tup,
            zr_tup,
            zi_tup,
            modo_peso=modo_peso_zv,
            modo_indutor=modo_indutor_zv,
        )

    ranking_modelos = [ResultadoCircuito(**d) for d in ranking_raw]
    if not ranking_modelos:
        st.error("Não foi possível convergir os modelos matemáticos sobre os dados fornecidos.")
        return

    if amostra_registro is not None:
        _registrar_resultado_zview(
            coletor,
            amostra_registro,
            ranking_modelos,
            diag_info,
            {
                "arquivo": nome_alvo_zv,
                "janela_log_f": [float(faixa_log_f[0]), float(faixa_log_f[1])],
                "n_pontos": int(len(df_fit)),
                "ponderacao": modo_peso_zv,
                "indutor": modo_indutor_zv,
            },
        )

    # 1. DIAGNÓSTICO AUTOMÁTICO DO ESPECTRO
    st.markdown("#### 1. Diagnóstico Físico Automatizado do Espectro Solartron")
    d1, d2, d3, d4 = st.columns(4)
    picos_txt = ", ".join(
        [f"{fp:.1f} Hz" if fp < 1000 else f"{fp / 1e3:.1f} kHz" for fp in diag_info["picos_freq"]]
    )
    d1.metric("Constantes Capacitivas", f"{diag_info['n_picos']} pico(s)", f"f_pico: {picos_txt}")
    d2.metric(
        "Resistência Série (Rs em Alta f)",
        f"{diag_info['zr_hf']:.1f} Ω",
        "Relevante no circuito" if diag_info["rs_relevante"] else "Desprezível (<0,8% de Z'máx)",
    )
    d3.metric(
        "Indutância HF / Warburg BF",
        f"L1: {'Sim' if diag_info['tem_indutancia_hf'] else 'Não'} | W: {'Sim' if diag_info['tem_warburg'] else 'Não'}",
        f"L_est ≈ {diag_info['l1_seed'] * 1e6:.2f} µH" if diag_info["tem_indutancia_hf"] else "Sem cauda indutiva",
    )
    kk_val = diag_info["kk_erro_pct"]
    kk_status = "Causal e estacionário (<1,5%)" if (np.isfinite(kk_val) and kk_val < 1.5) else "Verificar deriva/ruído"
    d4.metric("Validação Kramers-Kronig (Lin-KK)", f"{kk_val:.2f} %" if np.isfinite(kk_val) else "—", kk_status)

    if diag_info["tem_indutancia_hf"]:
        st.info(
            "💡 **Indutância em Alta Frequência Detectada ($Z'' > 0$ / $-Z'' < 0$):** O motor incluiu automaticamente "
            "modelos com indutor série **`L1`** na competição. Alternativamente, você pode cortar o extremo de alta "
            "frequência no seletor `log₁₀(f)` acima e exportar o espectro limpo `.z` direto para o ZView."
        )

    # 2. RANKING ESTATÍSTICO
    st.markdown("#### 2. Ranking Estatístico de Modelos Equivalentes (AIC / BIC / χ²)")
    linhas_rank = []
    for idx_r, r_mod in enumerate(ranking_modelos):
        status_str = (
            "🏆 1º RECOMENDADO"
            if idx_r == 0
            else ("✅ Alternativa Viável" if idx_r == 1 else "⚪ Secundário")
        )
        linhas_rank.append(
            {
                "Posição": status_str,
                "Circuito ZView": r_mod.codigo_zview,
                "Modelo Físico": r_mod.nome_curto,
                "Nº Parâm.": len(r_mod.param_nomes),
                "Erro Médio |ΔZ| (%)": round(r_mod.erro_medio_pct, 2),
                "χ² Reduzido": f"{r_mod.chi2_red:.3e}",
                "BIC (Menor = Melhor)": round(r_mod.bic, 1),
                "Diagnóstico Numérico": r_mod.penalidade_motivo,
            }
        )
    st.dataframe(pd.DataFrame(linhas_rank), use_container_width=True, hide_index=True)

    opcoes_inspecao = [
        f"{'🏆 ' if i == 0 else ''}{m.codigo_zview} — {m.nome_curto} (Erro: {m.erro_medio_pct:.2f}%)"
        for i, m in enumerate(ranking_modelos)
    ]
    idx_escolhido = st.selectbox(
        "Inspecionar Diagrama, Chutes Iniciais e Curva Ajustada do Modelo:",
        options=range(len(opcoes_inspecao)),
        format_func=lambda i: opcoes_inspecao[i],
        index=0,
        key="zv_sel_model",
    )
    modelo_ativo = ranking_modelos[idx_escolhido]

    # 3. DIAGRAMA ELÉTRICO + TABELA E EXPORTAÇÃO LIMPA PARA O ZVIEW
    st.markdown("#### 3. Diagrama Esquemático e Parâmetros Limpos para o ZView 2.9c")
    fig_circ = desenhar_esquema_zview(modelo_ativo, usar_virgula=usar_virgula)
    st.pyplot(fig_circ)
    botao_download_figura(fig_circ, f"Esquema_ZView_{modelo_ativo.id_modelo}", cfg_plot, "zv_schem")
    plt.close(fig_circ)

    col_guia, col_chutes = st.columns([1.1, 1.4])
    with col_guia:
        st.success(f"**Interpretação Física:** {modelo_ativo.interpretacao_fisica}")
        st.markdown("**Passo a passo na janela `Equivalent Circuits` do ZView:**")
        for num_p, texto_passo in enumerate(modelo_ativo.passos_zview, start=1):
            st.markdown(f"{num_p}. {texto_passo}")

        # Bloco copiável direto (sempre com ponto decimal padrão ZView)
        linhas_clip = [f"# Circuito ZView: {modelo_ativo.codigo_zview}"]
        for p_nome in modelo_ativo.param_nomes:
            linhas_clip.append(f"{p_nome:<10} = {fmt_eng_zview(modelo_ativo.param_valores[p_nome])}")
        st.markdown("**Cola Rápida (Valores em Padrão Ponto Decimal do ZView):**")
        st.code("\n".join(linhas_clip), language="ini")

    with col_chutes:
        st.markdown("**Tabela de Chutes Iniciais Pré-Convergidos (CNLS):**")
        linhas_params = []
        for p_nome in modelo_ativo.param_nomes:
            val_p = modelo_ativo.param_valores[p_nome]
            err_p = modelo_ativo.param_erros_pct.get(p_nome, np.nan)
            if p_nome == "L1":
                unid_p = "H (Henry)"
            elif p_nome.startswith("R"):
                unid_p = "Ω"
            elif p_nome.startswith("C1"):
                unid_p = "F"
            elif p_nome.endswith("-T"):
                unid_p = "S·s^P"
            elif p_nome.endswith("-P"):
                unid_p = "adim. (0–1)"
            else:
                unid_p = "Ω·s^-0.5"

            modo_sug = "Fixed (1ª rodada) -> Free (+)" if p_nome.endswith("-P") else "Free (+)"
            linhas_params.append(
                {
                    "Elemento ZView": p_nome,
                    "Valor Inicial (ZView)": fmt_eng_zview(val_p),
                    "Unidade": unid_p,
                    "Modo Sugerido": modo_sug,
                    "Incerteza (%)": round(err_p, 2) if np.isfinite(err_p) else np.nan,
                }
            )
        df_chutes = pd.DataFrame(linhas_params)
        st.dataframe(df_chutes, use_container_width=True, hide_index=True)

        if modelo_ativo.derivados:
            st.markdown("**Capacitâncias Efetivas Reais Extraídas (Hsu-Mansfeld & Brug):**")
            partes_deriv = []
            if "C1_eff (F)" in modelo_ativo.derivados:
                partes_deriv.append(
                    f"- **Arco 1 (Alta f):** $C_{{1,\\text{{HM}}}} = {modelo_ativo.derivados['C1_eff (F)']:.3e}\\text{{ F}}$ "
                    f"| $C_{{1,\\text{{Brug}}}} = {modelo_ativo.derivados['C1_brug (F)']:.3e}\\text{{ F}}$ "
                    f"| $\\tau_1 = {modelo_ativo.derivados['tau1 (s)']:.3e}\\text{{ s}}$ "
                    f"($f_{{pico,1}} \\approx {modelo_ativo.derivados['f_pico1 (Hz)']:.1f}\\text{{ Hz}}$)"
                )
            if "C2_eff (F)" in modelo_ativo.derivados:
                partes_deriv.append(
                    f"- **Arco 2 (Baixa f):** $C_{{2,\\text{{HM}}}} = {modelo_ativo.derivados['C2_eff (F)']:.3e}\\text{{ F}}$ "
                    f"| $\\tau_2 = {modelo_ativo.derivados['tau2 (s)']:.3e}\\text{{ s}}$ "
                    f"($f_{{pico,2}} \\approx {modelo_ativo.derivados['f_pico2 (Hz)']:.2f}\\text{{ Hz}}$)"
                )
            st.markdown("\n".join(partes_deriv))

        # Adicionar grandezas derivadas como linhas informativas extras na exportação CSV de parâmetros
        df_params_export = df_chutes.copy()
        for k_d, v_d in modelo_ativo.derivados.items():
            df_params_export = pd.concat(
                [
                    df_params_export,
                    pd.DataFrame(
                        [
                            {
                                "Elemento ZView": f"[Derivado] {k_d}",
                                "Valor Inicial (ZView)": fmt_eng_zview(v_d),
                                "Unidade": k_d.split("(")[-1].rstrip(")") if "(" in k_d else "",
                                "Modo Sugerido": "Cálculo Físico",
                                "Incerteza (%)": np.nan,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            botao_download_csv(
                df_params_export,
                f"Parametros_ZView_{nome_alvo_zv}",
                cfg_plot,
                "zv_params",
                label="📋 Baixar Parâmetros ZView (.CSV)",
            )
        with c_exp2:
            nome_z_limpo = re.sub(r"[^\w\-]+", "_", nome_alvo_zv.rsplit(".", 1)[0]).strip("_") or "espectro"
            st.download_button(
                label="💾 Exportar Espectro Filtrado (.z p/ ZView)",
                data=_exportar_formato_zplot_ascii(df_fit),
                file_name=f"{nome_z_limpo}_filtrado.z",
                mime="text/plain",
                key="zv_dl_zplot",
                use_container_width=True,
            )

    # 4. GRÁFICOS DE CONFERÊNCIA DO AJUSTE (NYQUIST, BODE E RESÍDUOS)
    st.markdown("#### 4. Conferência Visual do Ajuste (Experimental Solartron vs. Circuito Previsto)")
    c_es1, c_es2, c_es3, c_es4, c_es5 = st.columns([1.45, 0.85, 0.85, 1.0, 0.95])
    modo_esc_v4 = c_es1.selectbox(
        "Escala de Z' e -Z''",
        [
            "Igualada Automática (Z' e -Z'' na mesma escala)",
            "Manual (definir mín. e máx.)",
            "Automática Independente",
        ],
        index=0,
        key="zv_v4_modo_escala",
    )
    min_esc_v4 = c_es2.number_input(
        f"Mín. ({unid_zv_plot})",
        value=0.0,
        step=10.0,
        format="%.2f",
        disabled=(modo_esc_v4 != "Manual (definir mín. e máx.)"),
        key="zv_v4_min_escala",
    )
    max_esc_v4 = c_es3.number_input(
        f"Máx. ({unid_zv_plot})",
        value=1400.0,
        min_value=0.0001,
        step=50.0,
        format="%.2f",
        disabled=(modo_esc_v4 != "Manual (definir mín. e máx.)"),
        key="zv_v4_max_escala",
    )
    notacao_pot_v4 = c_es4.checkbox(
        "Freq. em 10ⁿ",
        value=True,
        key="zv_v4_pot10",
    )
    layout_v4 = c_es5.selectbox(
        "Layout",
        ["Separados (2 linhas)", "Lado a lado (3 painéis)"],
        index=0,
        key="zv_v4_layout",
    )

    div_zv = {"Ω": 1.0, "kΩ": 1e3, "MΩ": 1e6}[unid_zv_plot]
    f_fit = df_fit["Freq"].values
    zr_e = df_fit["Z_real"].values / div_zv
    mzi_e = df_fit["minus_Z_imag"].values / div_zv
    zr_c = modelo_ativo.z_calc_real / div_zv
    mzi_c = -modelo_ativo.z_calc_imag / div_zv

    mod_e = np.maximum(np.hypot(zr_e, mzi_e), 1e-9)
    res_re_pct = ((zr_c - zr_e) / mod_e) * 100.0
    res_im_pct = ((mzi_c - mzi_e) / mod_e) * 100.0

    def _aplicar_escala_v4(ax_n, ax_b_left, ax_b_right) -> None:
        if modo_esc_v4.startswith("Igualada"):
            min_v4 = min(
                float(np.min(zr_e)),
                float(np.min(mzi_e)),
                float(np.min(zr_c)),
                float(np.min(mzi_c)),
                0.0,
            )
            max_v4 = max(
                float(np.max(zr_e)),
                float(np.max(mzi_e)),
                float(np.max(zr_c)),
                float(np.max(mzi_c)),
                0.0,
            )
            if min_v4 < 0.0:
                span_v4 = max(max_v4 - min_v4, 1e-12)
                lim_v4 = (min_v4 - 0.04 * span_v4, max_v4 + 0.06 * span_v4)
            else:
                lim_v4 = (0.0, max(max_v4 * 1.06, 1e-12))
            ax_n.set_xlim(lim_v4)
            ax_n.set_ylim(lim_v4)
            ax_b_left.set_ylim(lim_v4)
            ax_b_right.set_ylim(lim_v4)
        elif modo_esc_v4.startswith("Manual"):
            lo_v4 = float(min_esc_v4)
            hi_v4 = max(float(max_esc_v4), lo_v4 + 1e-9)
            ax_n.set_xlim(lo_v4, hi_v4)
            ax_n.set_ylim(lo_v4, hi_v4)
            ax_b_left.set_ylim(lo_v4, hi_v4)
            ax_b_right.set_ylim(lo_v4, hi_v4)

    def _popular_nyquist(ax_n) -> None:
        ax_n.axhline(0, color="#666666", lw=0.7, ls=":")
        ax_n.plot(zr_e, mzi_e, "o", color="#222222", markersize=5.0, label="Solartron (Exp.)")
        ax_n.plot(zr_c, mzi_c, "-", color="#E60000", linewidth=2.0, label=f"Fit: {modelo_ativo.codigo_zview}")
        ax_n.set_xlabel(f"Z' ({unid_zv_plot})", fontsize=cfg_plot["font_labels"])
        ax_n.set_ylabel(f"-Z'' ({unid_zv_plot})", fontsize=cfg_plot["font_labels"])
        ax_n.set_title("Nyquist (Exp. vs Fit)", fontsize=cfg_plot["font_labels"])
        ax_n.legend(fontsize=max(8, cfg_plot["font_legend"] - 1), frameon=True, edgecolor="black")
        aplicar_estilo_origin(ax_n, cfg_plot)

    def _popular_bode(ax_b_left):
        ax_b_right = ax_b_left.twinx()
        ax_b_left.plot(f_fit, zr_e, "s", color="#333333", markersize=4.5, label="Z' Exp.")
        ax_b_left.plot(f_fit, zr_c, "-", color="#1F618D", linewidth=1.8, label="Z' Fit")
        ax_b_right.plot(f_fit, mzi_e, "o", markerfacecolor="white", markeredgecolor="#C0392B", markersize=4.8, label="-Z'' Exp.")
        ax_b_right.plot(f_fit, mzi_c, "--", color="#E60000", linewidth=1.8, label="-Z'' Fit")
        formatar_eixo_freq_bode(ax_b_left, usar_virgula=usar_virgula, notacao_potencia=notacao_pot_v4)
        ax_b_left.set_xlabel("Frequência (Hz)", fontsize=cfg_plot["font_labels"])
        ax_b_left.set_ylabel(f"Z' ({unid_zv_plot})", fontsize=cfg_plot["font_labels"], color="#1F618D")
        ax_b_right.set_ylabel(f"-Z'' ({unid_zv_plot})", fontsize=cfg_plot["font_labels"], color="#C0392B")
        ax_b_left.set_title("Bode (Exp. vs Fit)", fontsize=cfg_plot["font_labels"])
        aplicar_estilo_origin(ax_b_left, cfg_plot, log_x=True, right_spine=False)
        aplicar_estilo_origin(ax_b_right, cfg_plot, log_x=True, top_spine=False)
        ax_b_right.grid(False)
        return ax_b_right

    def _popular_residuos(ax_r) -> None:
        ax_r.axhline(0, color="black", lw=0.9, ls="--")
        ax_r.plot(f_fit, res_re_pct, "o-", color="#1F618D", markersize=3.8, lw=1.0, label="Resíduo ΔZ' / |Z| (%)")
        ax_r.plot(f_fit, res_im_pct, "s-", color="#E60000", markersize=3.8, lw=1.0, label="Resíduo Δ(-Z'') / |Z| (%)")
        formatar_eixo_freq_bode(ax_r, usar_virgula=usar_virgula, notacao_potencia=notacao_pot_v4)
        ax_r.set_xlabel("Frequência (Hz)", fontsize=cfg_plot["font_labels"])
        ax_r.set_ylabel("Resíduo Relativo (%)", fontsize=cfg_plot["font_labels"])
        ax_r.set_title("Qualidade dos Resíduos", fontsize=cfg_plot["font_labels"])
        ax_r.legend(fontsize=max(8, cfg_plot["font_legend"] - 1), frameon=True, edgecolor="black")
        aplicar_estilo_origin(ax_r, cfg_plot, log_x=True)

    fig_ver_dl: plt.Figure
    if layout_v4.startswith("Separados"):
        fig_nyq_v4, ax_vn_s = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)
        _popular_nyquist(ax_vn_s)

        fig_bode_v4, ax_vb_s = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)
        ax_vb2_s = _popular_bode(ax_vb_s)
        _aplicar_escala_v4(ax_vn_s, ax_vb_s, ax_vb2_s)
        fig_nyq_v4.tight_layout()
        fig_bode_v4.tight_layout()

        fig_res_v4, ax_vr_s = plt.subplots(figsize=(cfg_plot["fig_w"], cfg_plot["fig_h"]), dpi=150)
        _popular_residuos(ax_vr_s)
        fig_res_v4.tight_layout()

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.pyplot(fig_nyq_v4)
            botao_download_figura(fig_nyq_v4, f"Nyquist_Fit_ZView_{nome_alvo_zv}", cfg_plot, "zv_nyq_s")
        with c_s2:
            st.pyplot(fig_bode_v4)
            botao_download_figura(fig_bode_v4, f"Bode_Fit_ZView_{nome_alvo_zv}", cfg_plot, "zv_bode_s")

        st.pyplot(fig_res_v4)
        botao_download_figura(fig_res_v4, f"Residuos_Fit_ZView_{nome_alvo_zv}", cfg_plot, "zv_res_s")

        fig_ver_dl = fig_bode_v4
        plt.close(fig_nyq_v4)
        plt.close(fig_bode_v4)
        plt.close(fig_res_v4)
    else:
        fig_ver, (ax_vn, ax_vb, ax_vr) = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=140)
        _popular_nyquist(ax_vn)
        ax_vb2 = _popular_bode(ax_vb)
        _popular_residuos(ax_vr)
        _aplicar_escala_v4(ax_vn, ax_vb, ax_vb2)
        plt.tight_layout()
        st.pyplot(fig_ver)
        fig_ver_dl = fig_ver

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        botao_download_figura(fig_ver_dl, f"Conferencia_Fit_ZView_{nome_alvo_zv}", cfg_plot, "zv_ver")
    with col_dl2:
        df_fit_csv = pd.DataFrame(
            {
                "Freq_Hz": f_fit,
                "Z_real_exp_Ohm": df_fit["Z_real"].values,
                "Z_imag_exp_Ohm": df_fit["Z_imag"].values,
                "Z_real_fit_Ohm": modelo_ativo.z_calc_real,
                "Z_imag_fit_Ohm": modelo_ativo.z_calc_imag,
                "Residuo_Re_pct": res_re_pct,
                "Residuo_Im_pct": res_im_pct,
            }
        )
        botao_download_csv(
            df_fit_csv,
            f"Curvas_CNLS_{nome_alvo_zv}",
            cfg_plot,
            "zv_curves",
            label="📊 Baixar Curva Experimental + Ajustada (.CSV)",
        )
    if layout_v4.startswith("Lado a lado"):
        plt.close(fig_ver)


def render(cfg_plot: dict) -> None:
    """Ponto de entrada do app.py: o corpo roda dentro da sessão de registro central de resultados."""
    with sessao_resultados(registro_central.TEC_ZVIEW, _ORIGEM_REGISTRO, cfg_plot, "zv_reg") as coletor:
        _render_corpo(cfg_plot, coletor)
