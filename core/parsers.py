# -*- coding: utf-8 -*-
"""
core/parsers.py
Motor universal e blindado de leitura de arquivos científicos de laboratório:
  - Detecção segura de codificação (UTF-8, UTF-8-BOM, UTF-16 LE/BE, CP1252/Latin-1 preservando µ, Ω, °)
  - Resolução rigorosa da colisão entre vírgula decimal vs. vírgula delimitadora em múltiplas linhas
  - Conversão float elemento-a-elemento imune a corrupção de pontos decimais e notação científica (1,23E-05 / 1.23E-05)
  - Suporte nativo a Keithley (I×V com separação de varredura Ida/Volta e I×t), Solartron (ZPlot .z / SMaRT CSV)
    e Cary WinUV UV-Vis (arquivos simples, matriz compartilhada e CSV multi-amostra pareado do Cary 100)
  - Painel retrátil reutilizável para inspeção bruta e override manual em qualquer Add-on
"""

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================================
# ESTRUTURAS DE CONFIGURAÇÃO E METADADOS DE LEITURA
# ==============================================================================
@dataclass
class OpcoesParser:
    sep: str = "auto"        # "auto", ",", ";", "\t", r"\s+"
    decimal: str = "auto"    # "auto", ".", ","
    encoding: str = "auto"   # "auto", "utf-8", "cp1252", "latin-1", "utf-16"
    skip_extra: int = 0      # Linhas adicionais a ignorar no topo

    def to_cache_tuple(self) -> Tuple[str, str, str, int]:
        return (self.sep, self.decimal, self.encoding, int(self.skip_extra))


@dataclass
class ResultadoLeituraTabela:
    df: pd.DataFrame
    encoding_usado: str = "utf-8"
    sep_usado: str = ","
    decimal_usado: str = "."
    linha_inicio_dados: int = 0
    tem_cabecalho: bool = True
    linhas_brutas_topo: List[str] = field(default_factory=list)


# ==============================================================================
# EXPRESSÕES REGULARES PRÉ-COMPILADAS PARA NÚMEROS CIENTÍFICOS
# ==============================================================================
_RE_NUM_DOT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$")
_RE_NUM_COMMA = re.compile(r"^[+-]?(?:\d+(?:,\d+)?|,\d+)(?:[eEdD][+-]?\d+)?$")
_RE_THOUSAND_BR = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})+,\d+(?:[eEdD][+-]?\d+)?$")
_RE_THOUSAND_US = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+\.\d+(?:[eEdD][+-]?\d+)?$")
_RE_INT_SCI = re.compile(r"^[+-]?\d+(?:[eEdD][+-]?\d+)?$")

_TOKENS_NULOS = {"", "nan", "null", "none", "na", "n/a", "---", "--", "-", "ovld", "overload", "inf", "-inf"}


# ==============================================================================
# 1. DECODIFICAÇÃO SEGURA DE BYTES (UTF-8 / UTF-16 / CP1252 / LATIN-1)
# ==============================================================================
def decodificar_bytes_seguro(conteudo_bytes: bytes, encoding_forcado: str = "auto") -> Tuple[str, str]:
    if not conteudo_bytes:
        return "", "utf-8"

    if encoding_forcado and encoding_forcado != "auto":
        texto = conteudo_bytes.decode(encoding_forcado, errors="replace")
        return texto.replace("\r\n", "\n").replace("\r", "\n"), encoding_forcado

    if conteudo_bytes.startswith(b"\xff\xfe") or conteudo_bytes.startswith(b"\xfe\xff"):
        texto = conteudo_bytes.decode("utf-16", errors="replace")
        return texto.replace("\r\n", "\n").replace("\r", "\n"), "utf-16"

    if conteudo_bytes.startswith(b"\xef\xbb\xbf"):
        texto = conteudo_bytes.decode("utf-8-sig", errors="replace")
        return texto.replace("\r\n", "\n").replace("\r", "\n"), "utf-8-sig"

    amostra = conteudo_bytes[: min(2048, len(conteudo_bytes))]
    taxa_nulos = amostra.count(b"\x00") / max(1, len(amostra))
    if taxa_nulos > 0.20:
        enc_cand = "utf-16-le" if amostra[1::2].count(b"\x00") >= amostra[0::2].count(b"\x00") else "utf-16-be"
        try:
            texto = conteudo_bytes.decode(enc_cand)
            return texto.replace("\r\n", "\n").replace("\r", "\n"), enc_cand
        except UnicodeDecodeError:
            pass

    try:
        texto = conteudo_bytes.decode("utf-8")
        return texto.replace("\r\n", "\n").replace("\r", "\n"), "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        texto = conteudo_bytes.decode("cp1252")
        return texto.replace("\r\n", "\n").replace("\r", "\n"), "cp1252"
    except UnicodeDecodeError:
        texto = conteudo_bytes.decode("latin-1", errors="replace")
        return texto.replace("\r\n", "\n").replace("\r", "\n"), "latin-1"


# ==============================================================================
# 2. CONVERSÃO SEGURA DE FLOAT (ELEMENTO A ELEMENTO COM CACHE VETORIAL)
# ==============================================================================
def converter_token_float(val: Any, decimal: str = "auto") -> float:
    if val is None:
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val) if np.isfinite(val) else np.nan

    s = str(val).strip().strip('"').strip("'")
    if not s:
        return np.nan

    s = s.replace("\ufeff", "").replace("\xa0", "").replace("\u200b", "").replace("−", "-").strip()
    if s.lower() in _TOKENS_NULOS:
        return np.nan

    if ("d" in s or "D" in s) and re.search(r"\d[dD][+-]?\d+$", s):
        s = re.sub(r"([0-9])[dD]([+-]?\d+)$", r"\1e\2", s)

    if decimal == ".":
        if "," not in s:
            try:
                return float(s)
            except ValueError:
                return np.nan
        if _RE_THOUSAND_US.match(s):
            try:
                return float(s.replace(",", ""))
            except ValueError:
                return np.nan
        if "." not in s and _RE_NUM_COMMA.match(s):
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return np.nan
        return np.nan

    if decimal == ",":
        if _RE_THOUSAND_BR.match(s):
            s_clean = s.replace(".", "").replace(",", ".")
            try:
                return float(s_clean)
            except ValueError:
                return np.nan
        if "." not in s:
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return np.nan
        if "," not in s and _RE_NUM_DOT.match(s):
            try:
                return float(s)
            except ValueError:
                return np.nan
        return np.nan

    tem_virg = "," in s
    tem_ponto = "." in s

    if not tem_virg:
        try:
            return float(s)
        except ValueError:
            return np.nan

    if tem_virg and not tem_ponto:
        if s.count(",") > 1 and re.match(r"^[+-]?\d{1,3}(?:,\d{3})+$", s):
            try:
                return float(s.replace(",", ""))
            except ValueError:
                return np.nan
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return np.nan

    base_mantissa = re.split(r"[eE]", s)[0]
    ult_virg = base_mantissa.rfind(",")
    ult_ponto = base_mantissa.rfind(".")

    if ult_virg > ult_ponto:
        if _RE_THOUSAND_BR.match(s):
            try:
                return float(s.replace(".", "").replace(",", "."))
            except ValueError:
                return np.nan
    else:
        if _RE_THOUSAND_US.match(s):
            try:
                return float(s.replace(",", ""))
            except ValueError:
                return np.nan

    return np.nan


def converter_serie_float(serie: pd.Series, decimal: str = "auto") -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").astype(float)

    valores_unicos = serie.dropna().unique()
    mapa = {v: converter_token_float(v, decimal=decimal) for v in valores_unicos}
    return serie.map(mapa).astype(float)


# ==============================================================================
# 3. DETECTOR DE DIALETO CIENTÍFICO (SEPARADOR VS. DECIMAL EM 15+ LINHAS)
# ==============================================================================
def _split_linha_candidato(linha: str, sep: str) -> List[str]:
    if sep == r"\s+":
        return [t.strip() for t in re.split(r"\s+", linha.strip()) if t.strip() != ""]
    try:
        leitor = csv.reader([linha], delimiter=sep, quotechar='"', skipinitialspace=True)
        tokens = next(leitor)
        return [t.strip() for t in tokens]
    except Exception:
        return [t.strip() for t in linha.split(sep)]


def _fundir_unidades_cabecalho(tokens_hdr: List[str], n_cols_dados: int) -> Optional[List[str]]:
    """
    Quando um cabeçalho separado por espaços contém tokens de unidade isolados como
    '(s)', '(A)', '(V)', '(µA)', junta-os ao nome anterior para manter o mesmo número de colunas dos dados.
    """
    if len(tokens_hdr) <= n_cols_dados or n_cols_dados <= 0:
        return None

    merged: List[str] = []
    for tok in tokens_hdr:
        t_clean = tok.strip()
        es_unidade = bool(re.match(r"^[\(\[][^\]\)]{1,12}[\)\]]$", t_clean))
        if es_unidade and merged:
            merged[-1] = f"{merged[-1]} {t_clean}".strip()
        else:
            merged.append(t_clean)

    if len(merged) == n_cols_dados:
        return merged
    return None


_RE_TIME_STAMP = re.compile(r"^\d{1,4}:\d{2}(?::\d{2}(?:[.,]\d+)?)?$")
_RE_DATE_STAMP = re.compile(r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")


def _eh_token_neutro(t: str) -> bool:
    return (not t) or (t.lower() in _TOKENS_NULOS) or bool(_RE_TIME_STAMP.match(t)) or bool(_RE_DATE_STAMP.match(t))


def _token_eh_compativel(tok: str, dec: str) -> Tuple[bool, bool]:
    t = tok.strip().strip('"').strip("'").replace("−", "-")
    if _eh_token_neutro(t):
        return True, False
    if _RE_INT_SCI.match(t):
        return True, False
    if dec == ".":
        if _RE_NUM_DOT.match(t) or _RE_THOUSAND_US.match(t):
            return True, ("." in t)
        return False, False
    else:
        if _RE_NUM_COMMA.match(t) or _RE_THOUSAND_BR.match(t):
            return True, ("," in t)
        return False, False


def _avaliar_linha_numerica(linha: str, sep: str, dec: str) -> Tuple[bool, int, int, int]:
    stripped = linha.strip()
    if not stripped or stripped.startswith(("#", "!", "//")):
        return False, 0, 0, 0

    tokens = _split_linha_candidato(stripped, sep)
    tokens_uteis = tokens[:-1] if (len(tokens) > 3 and tokens[-1] == "") else tokens
    n_cols = len(tokens_uteis)
    if n_cols < 2:
        return False, n_cols, 0, 0

    n_num = 0
    n_evid_dec = 0
    for tok in tokens_uteis:
        ok, evid = _token_eh_compativel(tok, dec)
        if not ok:
            return False, n_cols, 0, 0
        t_clean = tok.strip().strip('"').strip("'").replace("−", "-")
        if not _eh_token_neutro(t_clean):
            n_num += 1
        if evid:
            n_evid_dec += 1

    if n_num < 2:
        return False, n_cols, n_num, n_evid_dec

    return True, n_cols, n_num, n_evid_dec


def detectar_bloco_e_dialeto(
    linhas: List[str],
    sep_override: str = "auto",
    dec_override: str = "auto",
) -> Tuple[str, str, int, bool, Optional[int]]:
    n_total = len(linhas)
    if n_total == 0:
        return ",", ".", 0, False, None

    marcador_idx = None
    for idx, line in enumerate(linhas):
        l_low = line.strip().lower()
        if l_low in ("[data]", "[dados]", "data:") or l_low.startswith("end comments"):
            marcador_idx = idx + 1
            break

    seps_cand = [sep_override] if sep_override != "auto" else ["\t", ";", ",", r"\s+"]
    decs_cand = [dec_override] if dec_override != "auto" else [".", ","]
    pares_cand = [(s, d) for s in seps_cand for d in decs_cand]

    melhor_score = -1e9
    melhor_cfg = (",", ".", marcador_idx or 0, True, None)

    inicio_busca = marcador_idx if marcador_idx is not None else 0

    for sep_c, dec_c in pares_cand:
        first_num_idx = None
        for i in range(inicio_busca, min(n_total, inicio_busca + 250)):
            ok1, c1, _, _ = _avaliar_linha_numerica(linhas[i], sep_c, dec_c)
            if not ok1:
                continue
            confirmacoes = 1
            j = i + 1
            vistas = 0
            while j < n_total and vistas < 5:
                if linhas[j].strip():
                    vistas += 1
                    ok_j, c_j, _, _ = _avaliar_linha_numerica(linhas[j], sep_c, dec_c)
                    if ok_j and abs(c_j - c1) <= 1:
                        confirmacoes += 1
                j += 1
            if confirmacoes >= min(3, 1 + vistas):
                first_num_idx = i
                break

        if first_num_idx is None:
            continue

        amostra_cols = []
        linhas_validas = 0
        linhas_testadas = 0
        total_nums = 0
        total_evid_dec = 0
        fim_bloco_idx = None

        for k in range(first_num_idx, n_total):
            s_line = linhas[k].strip()
            if not s_line:
                continue
            ok_k, ck, nk, evk = _avaliar_linha_numerica(s_line, sep_c, dec_c)
            if linhas_testadas < 25:
                linhas_testadas += 1
                if ok_k:
                    linhas_validas += 1
                    amostra_cols.append(ck)
                    total_nums += nk
                    total_evid_dec += evk
            if not ok_k and linhas_validas >= 3:
                proximas_ok = False
                for m in range(k + 1, min(n_total, k + 4)):
                    if linhas[m].strip():
                        ok_m, _, _, _ = _avaliar_linha_numerica(linhas[m], sep_c, dec_c)
                        if ok_m:
                            proximas_ok = True
                            break
                if not proximas_ok:
                    fim_bloco_idx = k
                    break

        if linhas_validas == 0:
            continue

        taxa_validade = linhas_validas / max(1, linhas_testadas)
        col_modal = int(pd.Series(amostra_cols).mode().iloc[0])
        consistencia_cols = sum(1 for c in amostra_cols if c == col_modal) / len(amostra_cols)

        score = (
            taxa_validade * 100.0
            + consistencia_cols * 40.0
            + min(col_modal, 8) * 4.0
            + total_evid_dec * 2.5
        )
        if sep_c in ("\t", ";", ","):
            score += 1.5
        if total_evid_dec == 0 and dec_c == ".":
            score += 0.2

        tem_header = False
        header_idx = first_num_idx
        for h in range(first_num_idx - 1, max(-1, inicio_busca - 1), -1):
            if linhas[h].strip():
                ok_h, _, _, _ = _avaliar_linha_numerica(linhas[h], sep_c, dec_c)
                if not ok_h:
                    for sep_h in (sep_c, ",", ";", "\t", r"\s+"):
                        toks_h = _split_linha_candidato(linhas[h], sep_h)
                        if toks_h and toks_h[-1] == "":
                            toks_h = toks_h[:-1]
                        toks_h_adj = _fundir_unidades_cabecalho(toks_h, col_modal) or toks_h
                        if len(toks_h_adj) >= 2 and (sep_h == sep_c or abs(len(toks_h_adj) - col_modal) <= 2):
                            tem_header = True
                            header_idx = h
                            break
                break

        if score > melhor_score:
            melhor_score = score
            melhor_cfg = (sep_c, dec_c, header_idx, tem_header, fim_bloco_idx)

    return melhor_cfg


# ==============================================================================
# 4. LEITOR GERAL DE TABELA CIENTÍFICA
# ==============================================================================
def ler_arquivo_cientifico(
    conteudo_bytes: bytes,
    opcoes: Optional[OpcoesParser] = None,
) -> ResultadoLeituraTabela:
    opts = opcoes or OpcoesParser()
    texto, enc_usado = decodificar_bytes_seguro(conteudo_bytes, opts.encoding)
    linhas = texto.splitlines()

    if opts.skip_extra > 0 and opts.skip_extra < len(linhas):
        linhas_proc = linhas[opts.skip_extra:]
    else:
        linhas_proc = linhas

    if not linhas_proc:
        return ResultadoLeituraTabela(df=pd.DataFrame(), encoding_usado=enc_usado)

    sep_det, dec_det, start_idx, tem_header, fim_idx = detectar_bloco_e_dialeto(
        linhas_proc,
        sep_override=opts.sep,
        dec_override=opts.decimal,
    )

    fatia_linhas = [l for l in (linhas_proc[start_idx:fim_idx] if fim_idx is not None else linhas_proc[start_idx:]) if l.strip()]

    if not fatia_linhas:
        return ResultadoLeituraTabela(
            df=pd.DataFrame(),
            encoding_usado=enc_usado,
            sep_usado=sep_det,
            decimal_usado=dec_det,
            linha_inicio_dados=start_idx + opts.skip_extra,
            tem_cabecalho=False,
            linhas_brutas_topo=linhas[:15],
        )

    nomes_header_normalizado: Optional[List[str]] = None
    if tem_header and len(fatia_linhas) >= 2:
        linha_hdr = fatia_linhas[0]
        toks_dados_1 = _split_linha_candidato(fatia_linhas[1], sep_det)
        if toks_dados_1 and toks_dados_1[-1] == "":
            toks_dados_1 = toks_dados_1[:-1]
        n_dados_uteis = len(toks_dados_1)

        toks_hdr_mesmo_sep = _split_linha_candidato(linha_hdr, sep_det)
        if toks_hdr_mesmo_sep and toks_hdr_mesmo_sep[-1] == "":
            toks_hdr_mesmo_sep = toks_hdr_mesmo_sep[:-1]

        fundido = _fundir_unidades_cabecalho(toks_hdr_mesmo_sep, n_dados_uteis)
        if fundido is not None:
            nomes_header_normalizado = fundido
        elif len(toks_hdr_mesmo_sep) < 2 or abs(len(toks_hdr_mesmo_sep) - n_dados_uteis) >= 2:
            for sh in (",", ";", "\t", r"\s+"):
                if sh == sep_det:
                    continue
                th = _split_linha_candidato(linha_hdr, sh)
                if th and th[-1] == "":
                    th = th[:-1]
                th_adj = _fundir_unidades_cabecalho(th, n_dados_uteis) or th
                if len(th_adj) >= 2 and abs(len(th_adj) - n_dados_uteis) <= 2:
                    nomes_header_normalizado = th_adj
                    break

    if nomes_header_normalizado is not None:
        bloco_dados_sem_hdr = "\n".join(fatia_linhas[1:])
        df_raw = pd.read_csv(
            io.StringIO(bloco_dados_sem_hdr),
            sep=sep_det,
            header=None,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
            keep_default_na=False,
        )
        cols_atrib = []
        for idx_c in range(df_raw.shape[1]):
            if idx_c < len(nomes_header_normalizado) and nomes_header_normalizado[idx_c].strip():
                cols_atrib.append(nomes_header_normalizado[idx_c].strip())
            else:
                cols_atrib.append(f"Col_{idx_c + 1}")
        df_raw.columns = cols_atrib
    else:
        bloco_util = "\n".join(fatia_linhas)
        df_raw = pd.read_csv(
            io.StringIO(bloco_util),
            sep=sep_det,
            header=0 if tem_header else None,
            index_col=False,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
            keep_default_na=False,
        )

    novas_cols = []
    vistos: Dict[str, int] = {}
    for idx_c, c in enumerate(df_raw.columns):
        nome = str(c).strip().strip('"').strip("'") if tem_header else f"Col_{idx_c + 1}"
        if not nome or nome.lower().startswith("unnamed"):
            nome = f"Col_{idx_c + 1}"
        if nome in vistos:
            vistos[nome] += 1
            nome = f"{nome}_{vistos[nome]}"
        else:
            vistos[nome] = 1
        novas_cols.append(nome)
    df_raw.columns = novas_cols

    cols_manter = [c for c in df_raw.columns if (df_raw[c].astype(str).str.strip() != "").any()]
    df_raw = df_raw[cols_manter]

    return ResultadoLeituraTabela(
        df=df_raw,
        encoding_usado=enc_usado,
        sep_usado=sep_det,
        decimal_usado=dec_det,
        linha_inicio_dados=start_idx + opts.skip_extra,
        tem_cabecalho=tem_header,
        linhas_brutas_topo=linhas[:15],
    )


# ==============================================================================
# 5. EXTRATORES ESPECIALIZADOS POR EQUIPAMENTO
# ==============================================================================
def extrair_keithley_iv(
    conteudo_bytes: bytes,
    opcoes: Optional[OpcoesParser] = None,
) -> Tuple[pd.DataFrame, ResultadoLeituraTabela]:
    res = ler_arquivo_cientifico(conteudo_bytes, opcoes)
    df = res.df
    if df.empty:
        raise ValueError("Arquivo vazio ou sem tabela numérica reconhecível.")

    dec = res.decimal_usado
    cols_lower = {c.lower(): c for c in df.columns}
    col_v, col_i, col_t = None, None, None

    for k, orig in cols_lower.items():
        if any(w in k for w in ["source", "volt", "tens", "potencial", "v (v)"]) or k == "v":
            col_v = orig
            break
    for k, orig in cols_lower.items():
        if any(w in k for w in ["volt", "tens", "v ("]) or k == "v":
            continue
        if any(w in k for w in ["measure", "curr", "corr", "i (a)", "i (ua)", "i (µa)"]) or k == "i":
            col_i = orig
            break
    for k, orig in cols_lower.items():
        if any(w in k for w in ["time", "tempo", "sec", "min", "stamp"]):
            col_t = orig
            break

    cols = list(df.columns)
    if col_v is None or col_i is None:
        if len(cols) >= 3:
            col_t = col_t or cols[0]
            col_v = col_v or cols[1]
            col_i = col_i or cols[2]
        elif len(cols) == 2:
            col_v = col_v or cols[0]
            col_i = col_i or cols[1]
        else:
            raise ValueError("Não foi possível identificar as colunas de Tensão e Corrente.")

    out = pd.DataFrame()
    out["Order"] = np.arange(len(df), dtype=int)
    if col_t and col_t in df.columns:
        out["Time"] = converter_serie_float(df[col_t], decimal=dec)
    else:
        out["Time"] = np.arange(len(df), dtype=float)

    out["Voltage"] = converter_serie_float(df[col_v], decimal=dec)
    out["Current"] = converter_serie_float(df[col_i], decimal=dec)
    out = out.dropna(subset=["Voltage", "Current"]).reset_index(drop=True)

    if len(out) < 2:
        raise ValueError("Menos de 2 pares válidos (Tensão, Corrente) encontrados após limpeza.")

    v_arr = out["Voltage"].values
    dv = np.diff(v_arr, prepend=v_arr[0])
    if len(dv) > 1 and dv[0] == 0:
        dv[0] = dv[1]

    dirs = np.where(dv > 1e-12, 1, np.where(dv < -1e-12, -1, 0))
    ultimo_sinal = 1
    for idx_d in range(len(dirs)):
        if dirs[idx_d] == 0:
            dirs[idx_d] = ultimo_sinal
        else:
            ultimo_sinal = dirs[idx_d]

    out["Sweep_Sign"] = dirs
    out["Sweep_Branch"] = np.where(dirs >= 0, "Ida (ΔV > 0)", "Volta (ΔV < 0)")
    tem_ida = bool(np.any(dirs > 0))
    tem_volta = bool(np.any(dirs < 0))
    out.attrs["tem_histerese"] = tem_ida and tem_volta

    return out, res


def extrair_keithley_it(
    conteudo_bytes: bytes,
    opcoes: Optional[OpcoesParser] = None,
) -> Tuple[pd.DataFrame, ResultadoLeituraTabela, dict]:
    res = ler_arquivo_cientifico(conteudo_bytes, opcoes)
    df = res.df
    if df.empty:
        raise ValueError("Arquivo I×t vazio ou ilegível.")

    dec = res.decimal_usado
    cols_lower = {c.lower(): c for c in df.columns}
    col_t, col_i = None, None

    for k, orig in cols_lower.items():
        if any(w in k for w in ["time", "tempo", "sec", "min", "stamp"]) or k == "t":
            col_t = orig
            break
    for k, orig in cols_lower.items():
        if any(w in k for w in ["volt", "tens", "v ("]) or k == "v":
            continue
        if any(w in k for w in ["current", "curr", "corrente", "corr", "i (", "amp"]) or k == "i":
            col_i = orig
            break

    cols = list(df.columns)
    if col_t is None:
        col_t = cols[0]
    if col_i is None:
        candidatas_sem_v = [c for c in cols if c != col_t and not any(w in c.lower() for w in ["volt", "tens", "v ("])]
        if candidatas_sem_v:
            col_i = candidatas_sem_v[0]
        else:
            col_i = cols[1] if len(cols) >= 2 else cols[0]

    out = pd.DataFrame()
    out["Time_raw"] = converter_serie_float(df[col_t], decimal=dec)
    out["Current_raw"] = converter_serie_float(df[col_i], decimal=dec)
    out = out.dropna(subset=["Time_raw", "Current_raw"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("Nenhum ponto numérico válido encontrado nas colunas de Tempo e Corrente.")

    header_i_low = col_i.lower()
    med_abs = float(np.nanmedian(np.abs(out["Current_raw"].values)))
    if "ua" in header_i_low or "µa" in header_i_low or "micro" in header_i_low:
        unid_sugerida = "µA"
    elif "na" in header_i_low or "nano" in header_i_low:
        unid_sugerida = "nA"
    elif "ma" in header_i_low or "mili" in header_i_low:
        unid_sugerida = "mA"
    elif med_abs > 0.1:
        unid_sugerida = "µA (verificar se o arquivo já veio convertido)"
    else:
        unid_sugerida = "A"

    info_escala = {
        "col_t": col_t,
        "col_i": col_i,
        "mediana_abs_bruta": med_abs,
        "unidade_sugerida": unid_sugerida,
    }
    return out, res, info_escala


def extrair_solartron_eis(
    conteudo_bytes: bytes,
    inverter_sinal_zim: Optional[bool] = None,
    opcoes: Optional[OpcoesParser] = None,
) -> Tuple[pd.DataFrame, ResultadoLeituraTabela]:
    res = ler_arquivo_cientifico(conteudo_bytes, opcoes)
    df = res.df
    if df.empty:
        raise ValueError("Arquivo Solartron vazio ou não reconhecido.")

    dec = res.decimal_usado
    df_num = pd.DataFrame({c: converter_serie_float(df[c], decimal=dec) for c in df.columns})

    cols_lower = {c.lower(): c for c in df.columns}
    col_f, col_zr, col_zi, col_mod, col_phase = None, None, None, None, None

    for k, orig in cols_lower.items():
        if any(w in k for w in ["freq", "hz", "frequ"]):
            col_f = orig
            break
    for k, orig in cols_lower.items():
        if "cap" in k:
            continue
        if any(w in k for w in ["z'(a)", "z'", "zre", "z_re", "impedance real", "real", "re(z)", "z1"]):
            col_zr = orig
            break
    for k, orig in cols_lower.items():
        if "cap" in k:
            continue
        if any(w in k for w in ["z''(b)", "-z''", "z''", "zim", "z_im", "impedance imag", "imag", "im(z)", "z2"]):
            col_zi = orig
            break
    for k, orig in cols_lower.items():
        if any(w in k for w in ["|z|", "zmod", "mod", "mag"]):
            col_mod = orig
            break
    for k, orig in cols_lower.items():
        if any(w in k for w in ["phase", "theta", "fase", "deg", "ang"]):
            col_phase = orig
            break

    cols = list(df.columns)
    if col_f is None and len(cols) >= 3:
        if len(cols) >= 12:
            col_f = cols[4]
        else:
            for c_cand in cols:
                s_c = df_num[c_cand].dropna().values
                if len(s_c) >= 5 and np.all(s_c > 0):
                    diffs = np.diff(np.log10(s_c))
                    monotona = (np.mean(diffs <= 1e-6) >= 0.90) or (np.mean(diffs >= -1e-6) >= 0.90)
                    span_dec = np.log10(np.max(s_c) / np.min(s_c))
                    if monotona and span_dec >= 1.5:
                        col_f = c_cand
                        break
            if col_f is None:
                col_f = cols[0]

    if col_zr is None and col_zi is None and col_mod is None:
        if len(cols) >= 12:
            col_zr, col_zi = cols[10], cols[11]
        elif len(cols) >= 6:
            col_zr, col_zi = cols[4], cols[5]
        elif len(cols) >= 3:
            restantes = [c for c in cols if c != col_f]
            col_zr, col_zi = restantes[0], restantes[1]

    if col_f is None:
        raise ValueError("Coluna de Frequência (Hz) não encontrada no arquivo EIS.")

    out = pd.DataFrame()
    out["Freq"] = df_num[col_f]

    if col_zr is not None and col_zi is not None:
        zr = df_num[col_zr].values
        zi = df_num[col_zi].values
        header_negativo = col_zi.strip().startswith("-")
        maioria_positiva = np.nanmedian(zi) > 0
        if inverter_sinal_zim is True:
            zi = -zi
        elif inverter_sinal_zim is None and (header_negativo or maioria_positiva):
            zi = -zi
        out["Z_real"] = zr
        out["Z_imag"] = zi
    elif col_mod is not None and col_phase is not None:
        mod = df_num[col_mod].values
        rad = np.deg2rad(df_num[col_phase].values)
        out["Z_real"] = mod * np.cos(rad)
        zi = mod * np.sin(rad)
        if np.nanmedian(zi) > 0 and inverter_sinal_zim is not False:
            zi = -zi
        out["Z_imag"] = zi
    else:
        raise ValueError("Colunas de impedância (Z', Z'' ou |Z|, Fase) não identificadas.")

    fator_escala = 1.0
    ref_h = (col_zr or col_mod or "").lower()
    if "kohm" in ref_h or "kω" in ref_h:
        fator_escala = 1e3
    elif "mohm" in ref_h or "mω" in ref_h:
        fator_escala = 1e6

    out["Z_real"] = out["Z_real"] * fator_escala
    out["Z_imag"] = out["Z_imag"] * fator_escala
    out = out.dropna(subset=["Freq", "Z_real", "Z_imag"])
    out = out[out["Freq"] > 0].sort_values("Freq", ascending=False).reset_index(drop=True)

    if out.empty:
        raise ValueError("Nenhum ponto válido de impedância com Freq > 0 encontrado.")

    out["minus_Z_imag"] = -out["Z_imag"]
    out["Z_mod"] = np.hypot(out["Z_real"], out["Z_imag"])
    out["Phase_deg"] = np.rad2deg(np.arctan2(out["Z_imag"], out["Z_real"]))

    return out, res


def extrair_cary_uvvis(
    conteudo_bytes: bytes,
    nome_arquivo: str = "Espectro",
    opcoes: Optional[OpcoesParser] = None,
) -> Tuple[List[Dict[str, Any]], ResultadoLeituraTabela]:
    opts = opcoes or OpcoesParser()
    texto, enc_usado = decodificar_bytes_seguro(conteudo_bytes, opts.encoding)
    linhas = texto.splitlines()
    if opts.skip_extra > 0 and opts.skip_extra < len(linhas):
        linhas = linhas[opts.skip_extra:]

    nome_limpo_arq = re.sub(r"\.(csv|txt|dat|dsp)$", "", nome_arquivo, flags=re.IGNORECASE)

    linhas_nao_vazias = [(i, l) for i, l in enumerate(linhas[:8]) if l.strip()]
    if len(linhas_nao_vazias) >= 3:
        idx_l0, l0 = linhas_nao_vazias[0]
        idx_l1, l1 = linhas_nao_vazias[1]
        sep_cary = opts.sep if opts.sep != "auto" else (
            ";" if l1.count(";") > l1.count(",") and l1.count(";") >= 2 else (
                "\t" if l1.count("\t") >= 2 else ","
            )
        )
        toks_l0 = _split_linha_candidato(l0, sep_cary)
        toks_l1 = _split_linha_candidato(l1, sep_cary)

        n_wl_headers = sum(
            1 for t in toks_l1
            if any(w in t.lower() for w in ["wave", "nm", "compr", "lambda", "λ"])
        )
        if n_wl_headers >= 2 and len(toks_l1) >= 4:
            espectros_cary: List[Dict[str, Any]] = []
            linhas_dados_cary = []
            for linha_d in linhas[idx_l1 + 1:]:
                s_d = linha_d.strip()
                if not s_d:
                    if len(linhas_dados_cary) >= 5:
                        break
                    continue
                toks_d = _split_linha_candidato(s_d, sep_cary)
                v0 = converter_token_float(toks_d[0] if len(toks_d) > 0 else "", opts.decimal)
                v1 = converter_token_float(toks_d[1] if len(toks_d) > 1 else "", opts.decimal)
                if np.isfinite(v0) and np.isfinite(v1):
                    linhas_dados_cary.append(toks_d)
                elif len(linhas_dados_cary) >= 5:
                    break

            if linhas_dados_cary:
                max_cols = max(len(r) for r in linhas_dados_cary)
                matriz_str = [r + [""] * (max_cols - len(r)) for r in linhas_dados_cary]
                df_cary_raw = pd.DataFrame(matriz_str)

                n_pares = len(toks_l1) // 2
                for p in range(n_pares):
                    c_wl = 2 * p
                    c_ab = 2 * p + 1
                    if c_ab >= df_cary_raw.shape[1]:
                        break
                    nome_amostra = toks_l0[c_wl].strip() if c_wl < len(toks_l0) and toks_l0[c_wl].strip() else f"{nome_limpo_arq}_S{p+1}"
                    s_wl = converter_serie_float(df_cary_raw[c_wl], decimal=opts.decimal)
                    s_ab = converter_serie_float(df_cary_raw[c_ab], decimal=opts.decimal)
                    df_par = pd.DataFrame({"wl": s_wl, "abs": s_ab}).dropna().sort_values("wl").reset_index(drop=True)
                    if len(df_par) >= 3:
                        espectros_cary.append({
                            "nome": nome_amostra,
                            "df": df_par,
                            "origem": nome_arquivo,
                        })

                if espectros_cary:
                    meta_cary = ResultadoLeituraTabela(
                        df=df_cary_raw.head(30),
                        encoding_usado=enc_usado,
                        sep_usado=sep_cary,
                        decimal_usado=opts.decimal,
                        linha_inicio_dados=idx_l1 + 1,
                        tem_cabecalho=True,
                        linhas_brutas_topo=linhas[:15],
                    )
                    return espectros_cary, meta_cary

    res = ler_arquivo_cientifico(conteudo_bytes, opts)
    df = res.df
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"Não foi possível extrair pares (Comprimento de Onda, Absorbância) de '{nome_arquivo}'.")

    dec = res.decimal_usado
    cols = list(df.columns)
    wl_base = converter_serie_float(df[cols[0]], decimal=dec)

    espectros: List[Dict[str, Any]] = []
    if len(cols) == 2:
        ab_s = converter_serie_float(df[cols[1]], decimal=dec)
        df_clean = pd.DataFrame({"wl": wl_base, "abs": ab_s}).dropna().sort_values("wl").reset_index(drop=True)
        if not df_clean.empty:
            espectros.append({"nome": nome_limpo_arq, "df": df_clean, "origem": nome_arquivo})
    else:
        for c_extra in cols[1:]:
            ab_s = converter_serie_float(df[c_extra], decimal=dec)
            df_clean = pd.DataFrame({"wl": wl_base, "abs": ab_s}).dropna().sort_values("wl").reset_index(drop=True)
            if len(df_clean) >= 3:
                rotulo = c_extra if not c_extra.startswith("Col_") else f"{nome_limpo_arq}_{c_extra}"
                espectros.append({"nome": rotulo, "df": df_clean, "origem": nome_arquivo})

    if not espectros:
        raise ValueError(f"Dados numéricos UV-Vis inválidos ou vazios em '{nome_arquivo}'.")

    return espectros, res


# ==============================================================================
# 6. MINI-PAINEL RETRÁTIL REUTILIZÁVEL DE INSPEÇÃO E OVERRIDE MANUAL
# ==============================================================================
def render_painel_override_parser(key_prefix: str) -> Tuple[OpcoesParser, Any]:
    exp = st.expander("🛠️ Inspecionar Leitura Bruta / Override Manual de Separador e Decimal", expanded=False)
    with exp:
        c1, c2, c3, c4 = st.columns(4)
        sep_map = {
            "Automático ( Multi-linha Regex )": "auto",
            "Vírgula ( , )": ",",
            "Ponto-e-vírgula ( ; )": ";",
            "Tabulação ( \\t )": "\t",
            "Espaços múltiplos": r"\s+",
        }
        dec_map = {
            "Automático": "auto",
            "Ponto ( . )": ".",
            "Vírgula ( , )": ",",
        }
        enc_map = {
            "Automático (UTF-8 / CP1252 / UTF-16)": "auto",
            "UTF-8": "utf-8",
            "Windows-1252 (Latin / Lab PC)": "cp1252",
            "Latin-1 (ISO-8859-1)": "latin-1",
            "UTF-16": "utf-16",
        }

        sep_lbl = c1.selectbox("Separador de Colunas", list(sep_map.keys()), index=0, key=f"p_sep_{key_prefix}")
        dec_lbl = c2.selectbox("Separador Decimal", list(dec_map.keys()), index=0, key=f"p_dec_{key_prefix}")
        enc_lbl = c3.selectbox("Codificação de Texto", list(enc_map.keys()), index=0, key=f"p_enc_{key_prefix}")
        skip_ex = c4.number_input("Pular linhas extras (topo)", 0, 500, 0, step=1, key=f"p_skp_{key_prefix}")

        slot_diag = st.container()

    opcoes = OpcoesParser(
        sep=sep_map[sep_lbl],
        decimal=dec_map[dec_lbl],
        encoding=enc_map[enc_lbl],
        skip_extra=int(skip_ex),
    )
    return opcoes, slot_diag


def exibir_diagnostico_leitura(slot_container: Any, meta: ResultadoLeituraTabela, nome_arquivo: str = "") -> None:
    if slot_container is None or meta is None:
        return
    sep_legivel = {"\t": "TAB (\\t)", r"\s+": "Espaços (\\s+)", ",": "Vírgula (,)", ";": "Ponto-e-vírgula (;)"}.get(
        meta.sep_usado, meta.sep_usado
    )
    with slot_container:
        st.caption(
            f"**Diagnóstico `{nome_arquivo or '1º arquivo'}`:** "
            f"Encoding = `{meta.encoding_usado}` | "
            f"Separador = `{sep_legivel}` | "
            f"Decimal = `'{meta.decimal_usado}'` | "
            f"Início da tabela = `linha {meta.linha_inicio_dados + 1}` | "
            f"Cabeçalho detectado = `{'Sim' if meta.tem_cabecalho else 'Não'}`"
        )
        col_raw, col_df = st.columns([1.1, 1.3])
        with col_raw:
            st.markdown("**Primeiras linhas brutas do arquivo:**")
            st.code("\n".join(meta.linhas_brutas_topo[:12]), language="text")
        with col_df:
            st.markdown("**Prévia da tabela bruta extraída:**")
            st.dataframe(meta.df.head(6), use_container_width=True, hide_index=True)