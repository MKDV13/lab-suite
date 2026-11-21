# -*- coding: utf-8 -*-
"""
tests/test_parsers.py
Testes automatizados de core/parsers.py (leitura de arquivos de instrumentos).

Baseados exclusivamente no código real de core/parsers.py: só usam nomes que existem nele
(decodificar_bytes_seguro, converter_token_float, converter_serie_float, detectar_bloco_e_dialeto,
ler_arquivo_cientifico, extrair_keithley_iv, extrair_keithley_it, extrair_solartron_eis,
extrair_cary_uvvis, OpcoesParser, ResultadoLeituraTabela, render_painel_override_parser,
exibir_diagnostico_leitura). Todas as amostras são sintéticas e geradas em memória.

Como rodar (Windows, na raiz do projeto, usando o ambiente virtual .venv):
    .venv\\Scripts\\python.exe -m pytest tests -v

Limitações reais do parser atual ficam documentadas com xfail(strict=True): se um dia o parser
passar a tratar o caso, o teste "passa por engano" (XPASS) e a suíte falha, avisando para remover o marcador.
"""

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from core import parsers as P  # noqa: E402
from core.parsers import (  # noqa: E402
    OpcoesParser,
    ResultadoLeituraTabela,
    converter_serie_float,
    converter_token_float,
    decodificar_bytes_seguro,
    detectar_bloco_e_dialeto,
    exibir_diagnostico_leitura,
    extrair_cary_uvvis,
    extrair_keithley_it,
    extrair_keithley_iv,
    extrair_solartron_eis,
    ler_arquivo_cientifico,
    render_painel_override_parser,
)

# O pandas pode avisar sobre linhas mal formadas em arquivos propositalmente quebrados.
pytestmark = pytest.mark.filterwarnings("ignore::pandas.errors.ParserWarning")


def quase(esperado, rel: float = 1e-9):
    """pytest.approx só com tolerância relativa: o padrão (abs=1e-12) é grande demais para correntes de nA."""
    return pytest.approx(esperado, rel=rel, abs=0.0)


# ==============================================================================
# CONSTRUTORES DE AMOSTRAS SINTÉTICAS
# ==============================================================================
def tabela(
    colunas: Sequence[Sequence[float]],
    cabecalho: Optional[Sequence[str]] = None,
    sep: str = ",",
    dec: str = ".",
    sep_final: bool = False,
) -> str:
    """Monta o texto de uma tabela numérica: `colunas` são listas de mesmo tamanho; `dec` troca o ponto decimal."""
    linhas: List[str] = []
    if cabecalho is not None:
        linhas.append(sep.join(cabecalho) + (sep if sep_final else ""))
    for k in range(len(colunas[0])):
        celulas = [format(float(col[k]), ".10g").replace(".", dec) for col in colunas]
        linhas.append(sep.join(celulas) + (sep if sep_final else ""))
    return "\n".join(linhas) + "\n"


# (separador do arquivo, decimal do arquivo) -> id legível
DIALETOS = [
    pytest.param(",", ".", id="virgula_ponto"),
    pytest.param(";", ",", id="pontoevirgula_virgula"),
    pytest.param(";", ".", id="pontoevirgula_ponto"),
    pytest.param("\t", ".", id="tab_ponto"),
    pytest.param("\t", ",", id="tab_virgula"),
    pytest.param(" ", ".", id="espaco_ponto"),
]


def sep_esperado(sep: str) -> str:
    """O parser reporta separador de espaços como a regex \\s+."""
    return r"\s+" if sep == " " else sep


V_LISTA = [0.1, 0.2, 0.3, 0.4, 0.5]
I_LISTA = [1.5e-9, 2.5e-9, 3.5e-9, 4.5e-9, 5.5e-9]

# Varredura Keithley completa: ida 0 -> 1 V e volta 1 -> 0 V (9 pontos, histerese presente)
V_CICLO = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]
I_CICLO = [2e-6 * v for v in V_CICLO]
T_CICLO = [float(k) for k in range(len(V_CICLO))]

# Circuito de Randles: Rs + Rct || Cdl (Rs = 100 Ω, Rct = 5 kΩ, Cdl = 100 nF), 25 frequências de 100 kHz a 0,1 Hz
RS, RCT, CDL = 100.0, 5000.0, 1e-7
FREQ = np.logspace(5, -1, 25)
Z_RANDLES = RS + RCT / (1 + 1j * 2 * np.pi * FREQ * RCT * CDL)

# Cary UV-Vis: varredura decrescente de 800 a 400 nm
WL = [800.0 - 50.0 * k for k in range(9)]
ABS_A = [round(0.05 + 0.1 * k, 4) for k in range(9)]
ABS_B = [round(0.02 + 0.07 * k, 4) for k in range(9)]

TEXTO_SEM_NUMEROS = "isto é\num texto\nsem números\nnenhum\n"
BINARIO = bytes(range(256)) * 4


def iv_bytes(sep=",", dec=".", cabecalho=("Time", "Voltage", "Current"), com_tempo=True, v=None, i=None) -> bytes:
    v = V_CICLO if v is None else v
    i = [2e-6 * x for x in v] if i is None else i
    colunas = [T_CICLO[: len(v)], v, i] if com_tempo else [v, i]
    cab = list(cabecalho) if cabecalho is not None else None
    if cab is not None and not com_tempo and len(cab) == 3:
        cab = cab[1:]
    return tabela(colunas, cab, sep, dec).encode("utf-8")


def it_bytes(cab_i="Current (A)", valor=1.5e-9, cab_t="Time (s)", sep=",", dec=".", n=10) -> bytes:
    t = [float(k) for k in range(n)]
    return tabela([t, [valor] * n], [cab_t, cab_i] if cab_i else None, sep, dec).encode("utf-8")


def eis_bytes(
    cab=("Freq(Hz)", "Z'(Ohm)", "Z''(Ohm)"),
    sep=",",
    dec=".",
    sinal_zim=-1.0,
    escala=1.0,
    prefixo="",
) -> bytes:
    """Espectro de Randles; sinal_zim < 0 grava Z'' na convenção capacitiva (negativo), senão grava Z'' positivo."""
    zim = Z_RANDLES.imag if sinal_zim < 0 else -Z_RANDLES.imag
    colunas = [FREQ, Z_RANDLES.real / escala, zim / escala]
    return (prefixo + tabela(colunas, list(cab) if cab else None, sep, dec)).encode("utf-8")


def eis_polar_bytes(fase_negativa=True) -> bytes:
    fase = np.degrees(np.angle(Z_RANDLES))
    if not fase_negativa:
        fase = -fase
    return tabela([FREQ, np.abs(Z_RANDLES), fase], ["Freq(Hz)", "|Z|(Ohm)", "Phase(deg)"]).encode("utf-8")


def cary_multi_bytes(sep=",", dec=".", rodape=True, nome_a="P3HT_A") -> bytes:
    """CSV multi-amostra do Cary WinUV: linha 0 = nomes, linha 1 = pares 'Wavelength (nm), Abs', dados e rodapé."""
    linhas = [
        sep.join([nome_a, "", "P3DT_B", ""]),
        sep.join(["Wavelength (nm)", "Abs", "Wavelength (nm)", "Abs"]),
    ]
    for w, a, b in zip(WL, ABS_A, ABS_B):
        celulas = [f"{w:.1f}", f"{a:.4f}", f"{w:.1f}", f"{b:.4f}"]
        linhas.append(sep.join(c.replace(".", dec) for c in celulas))
    if rodape:
        linhas += ["", "Instrument Parameters", "Scan rate: 600 nm/min", "Slit: 2 nm"]
    return ("\n".join(linhas) + "\n").encode("utf-8")


def cary_simples_bytes(sep=",", dec=".", cabecalho=True) -> bytes:
    cab = ["Wavelength (nm)", "Abs"] if cabecalho else None
    return tabela([WL, ABS_A], cab, sep, dec).encode("utf-8")


def numeros(res: ResultadoLeituraTabela, coluna: str) -> List[float]:
    """Converte uma coluna do DataFrame bruto com o decimal detectado — o que os add-ons fazem."""
    return converter_serie_float(res.df[coluna], decimal=res.decimal_usado).tolist()


# ==============================================================================
# 1. DECODIFICAÇÃO DE BYTES
# ==============================================================================
class TestDecodificacao:
    def test_bytes_vazios(self):
        assert decodificar_bytes_seguro(b"") == ("", "utf-8")

    def test_utf8_preserva_simbolos_de_instrumento(self):
        texto, enc = decodificar_bytes_seguro("Corrente (µA) Z (Ω) 25 °C".encode("utf-8"))
        assert enc == "utf-8"
        assert texto == "Corrente (µA) Z (Ω) 25 °C"

    def test_utf8_com_bom(self):
        texto, enc = decodificar_bytes_seguro(b"\xef\xbb\xbf" + "µA\n1,5".encode("utf-8"))
        assert enc == "utf-8-sig"
        assert texto == "µA\n1,5"  # o BOM não pode vazar para o texto

    def test_utf16_com_bom(self):
        texto, enc = decodificar_bytes_seguro("µA Ω\n1,5\n".encode("utf-16"))
        assert enc == "utf-16"
        assert texto == "µA Ω\n1,5\n"

    def test_utf16_sem_bom_detectado_pelos_bytes_nulos(self):
        texto, enc = decodificar_bytes_seguro("Freq;Zre\n1;2\n3;4\n".encode("utf-16-le"))
        assert enc == "utf-16-le"
        assert texto == "Freq;Zre\n1;2\n3;4\n"

    def test_cp1252_quando_nao_e_utf8_valido(self):
        texto, enc = decodificar_bytes_seguro("Corrente (µA) 25°C\n1,5".encode("cp1252"))
        assert enc == "cp1252"
        assert texto == "Corrente (µA) 25°C\n1,5"

    def test_latin1_decodifica_via_cp1252_sem_perder_caracteres(self):
        texto, enc = decodificar_bytes_seguro("Tensão (V)".encode("latin-1"))
        assert texto == "Tensão (V)"
        assert enc in ("cp1252", "latin-1")

    def test_quebras_de_linha_normalizadas(self):
        texto, _ = decodificar_bytes_seguro(b"a\r\nb\rc\nd")
        assert texto == "a\nb\nc\nd"

    def test_encoding_forcado_e_respeitado(self):
        texto, enc = decodificar_bytes_seguro("µ".encode("latin-1"), "latin-1")
        assert (texto, enc) == ("µ", "latin-1")

    def test_encoding_forcado_errado_nao_descarta_bytes_silenciosamente(self):
        texto, enc = decodificar_bytes_seguro("µ".encode("cp1252"), "utf-8")
        assert enc == "utf-8"
        assert texto == "\ufffd"  # caractere de substituição visível, nunca errors='ignore'


# ==============================================================================
# 2. CONVERSÃO DE NÚMEROS
# ==============================================================================
CASOS_TOKEN = [
    # (token, decimal, esperado)
    ("1,23E-05", "auto", 1.23e-05),
    ("1.23E-05", "auto", 1.23e-05),
    ("+0.57244E-06", "auto", 5.7244e-07),
    ("1e5", "auto", 1e5),
    ("1.234,56", "auto", 1234.56),
    ("1,234.56", "auto", 1234.56),
    ("1,234,567", "auto", 1234567.0),
    ("1.23D-05", "auto", 1.23e-05),
    ("−3,2", "auto", -3.2),  # sinal de menos Unicode
    ("  7 ", "auto", 7.0),
    ('"4,5"', "auto", 4.5),
    ("\xa01,5", "auto", 1.5),  # espaço não separável
    ("1,5", ".", 1.5),  # equipamento mandou vírgula mesmo com override '.'
    ("1,234.56", ".", 1234.56),
    ("1,5", ",", 1.5),
    ("1.5", ",", 1.5),  # ponto decimal puro não é destruído com override ','
    ("1.234,56", ",", 1234.56),
    (3, "auto", 3.0),
    (np.float64(2.5), "auto", 2.5),
]


class TestConverterToken:
    @pytest.mark.parametrize("token,decimal,esperado", CASOS_TOKEN)
    def test_valores_validos(self, token, decimal, esperado):
        assert converter_token_float(token, decimal) == quase(esperado, rel=1e-12)

    @pytest.mark.parametrize(
        "token",
        ["OVLD", "overload", "nan", "NaN", "null", "N/A", "-", "---", "", "   ", "abc", "1,2,3", "1.234.567", "12abc"],
    )
    def test_tokens_nulos_ou_invalidos_viram_nan(self, token):
        assert np.isnan(converter_token_float(token))

    @pytest.mark.parametrize("valor", [None, np.nan, np.inf, -np.inf])
    def test_none_e_nao_finitos_viram_nan(self, valor):
        assert np.isnan(converter_token_float(valor))

    def test_nunca_levanta_excecao_com_lixo(self):
        for lixo in ["\x00", "1..2", "--5", "1e", "e5", ",", ".", "1,,2", "0x10"]:
            converter_token_float(lixo)  # não pode levantar


class TestConverterSerie:
    def test_series_de_texto_com_virgula_decimal(self):
        s = pd.Series(["1,5", "2,5E-3", "x", None, "1.234,5"])
        out = converter_serie_float(s, ",")
        assert out.iloc[[0, 1, 4]].tolist() == quase([1.5, 2.5e-3, 1234.5])
        assert out.iloc[[2, 3]].isna().all()

    def test_retorna_float64(self):
        assert converter_serie_float(pd.Series(["1", "2", "3"])).dtype == np.float64

    def test_serie_numerica_passa_direto(self):
        assert converter_serie_float(pd.Series([1, 2, 3])).tolist() == [1.0, 2.0, 3.0]
        out = converter_serie_float(pd.Series([1.5, None]))
        assert out.iloc[0] == 1.5 and np.isnan(out.iloc[1])

    def test_preserva_o_indice(self):
        s = pd.Series(["1,5", "2,5"], index=[10, 20])
        assert converter_serie_float(s, ",").index.tolist() == [10, 20]

    def test_valores_repetidos_convertem_igual(self):
        out = converter_serie_float(pd.Series(["0,1"] * 5 + ["0,2"]), ",")
        assert out.tolist() == pytest.approx([0.1] * 5 + [0.2])


# ==============================================================================
# 3. DETECÇÃO DE DIALETO E BLOCO DE DADOS
# ==============================================================================
class TestDetectarDialeto:
    def test_sem_linhas_devolve_padrao(self):
        assert detectar_bloco_e_dialeto([]) == (",", ".", 0, False, None)

    def test_ponto_e_virgula_com_virgula_decimal_e_cabecalho(self):
        linhas = "V;I\n0,1;1,5\n0,2;2,5\n0,3;3,5\n0,4;4,5".splitlines()
        assert detectar_bloco_e_dialeto(linhas) == (";", ",", 0, True, None)

    def test_marcador_data_define_o_inicio(self):
        linhas = ["Instrument: X", "Date: 1", "[Data]", "V,I", "0.1,1", "0.2,2", "0.3,3", "0.4,4"]
        sep, dec, inicio, tem_cab, fim = detectar_bloco_e_dialeto(linhas)
        assert (sep, dec, inicio, tem_cab, fim) == (",", ".", 3, True, None)

    def test_rodape_textual_marca_o_fim_do_bloco(self):
        linhas = ["V,I", "0.1,1", "0.2,2", "0.3,3", "0.4,4", "", "Instrument Parameters", "Slit=2", "Lamp=ON"]
        sep, dec, inicio, tem_cab, fim = detectar_bloco_e_dialeto(linhas)
        assert (sep, dec, tem_cab) == (",", ".", True)
        assert fim == 6  # índice da primeira linha de rodapé útil

    def test_override_de_separador_e_decimal_restringe_os_candidatos(self):
        linhas = "V;I\n0,1;1,5\n0,2;2,5\n0,3;3,5\n0,4;4,5".splitlines()
        sep, dec, *_ = detectar_bloco_e_dialeto(linhas, sep_override=";", dec_override=",")
        assert (sep, dec) == (";", ",")

    def test_texto_puro_cai_no_padrao_sem_levantar(self):
        assert detectar_bloco_e_dialeto(["alpha beta", "gamma delta", "epsilon zeta"])[:2] == (",", ".")


# ==============================================================================
# 4. LEITOR GERAL DE TABELA CIENTÍFICA
# ==============================================================================
class TestLerArquivoCientifico:
    @pytest.mark.parametrize("sep,dec", DIALETOS)
    def test_dialetos_com_cabecalho(self, sep, dec):
        conteudo = tabela([V_LISTA, I_LISTA], ["Tensao", "Corrente"], sep, dec).encode("utf-8")
        res = ler_arquivo_cientifico(conteudo)
        assert isinstance(res, ResultadoLeituraTabela)
        assert res.sep_usado == sep_esperado(sep)
        assert res.decimal_usado == dec
        assert res.tem_cabecalho is True
        assert list(res.df.columns) == ["Tensao", "Corrente"]
        assert numeros(res, "Tensao") == quase(V_LISTA)
        assert numeros(res, "Corrente") == quase(I_LISTA)

    @pytest.mark.parametrize("sep,dec", DIALETOS)
    def test_dialetos_sem_cabecalho_nomeiam_col_n(self, sep, dec):
        conteudo = tabela([V_LISTA, I_LISTA], None, sep, dec).encode("utf-8")
        res = ler_arquivo_cientifico(conteudo)
        assert res.tem_cabecalho is False
        assert list(res.df.columns) == ["Col_1", "Col_2"]
        assert len(res.df) == 5
        assert numeros(res, "Col_2") == quase(I_LISTA)

    def test_inteiros_puros_preferem_ponto(self):
        res = ler_arquivo_cientifico(b"a;b\n1;2\n3;4\n5;6\n7;8\n")
        assert (res.sep_usado, res.decimal_usado) == (";", ".")
        assert numeros(res, "b") == [2.0, 4.0, 6.0, 8.0]

    def test_milhar_brasileiro(self):
        res = ler_arquivo_cientifico(b"x;y\n1.234,5;2.000,5\n3.234,5;4.000,5\n5.234,5;6.000,5\n7.234,5;8.000,5\n")
        assert (res.sep_usado, res.decimal_usado) == (";", ",")
        assert numeros(res, "x") == pytest.approx([1234.5, 3234.5, 5234.5, 7234.5])

    def test_metadados_antes_da_tabela(self):
        conteudo = (
            "Instrument: Keithley 2450\nDate: 2026-11-20\nOperator: X\n\n"
            + tabela([V_LISTA, I_LISTA], ["V", "I"])
        ).encode("utf-8")
        res = ler_arquivo_cientifico(conteudo)
        assert res.linha_inicio_dados == 4
        assert list(res.df.columns) == ["V", "I"]
        assert len(res.df) == 5
        assert res.linhas_brutas_topo[0] == "Instrument: Keithley 2450"

    def test_marcador_data_pula_o_cabecalho_de_metadados(self):
        conteudo = ("[Header]\nSample=P3HT\n[Data]\n" + tabela([V_LISTA, I_LISTA], ["V", "I"], ";", ",")).encode("utf-8")
        res = ler_arquivo_cientifico(conteudo)
        assert (res.sep_usado, res.decimal_usado, res.linha_inicio_dados) == (";", ",", 3)
        assert len(res.df) == 5

    def test_rodape_textual_e_descartado(self):
        conteudo = (tabela([V_LISTA, I_LISTA], ["V", "I"]) + "\nInstrument Parameters\nSlit=2\nLamp=ON\n").encode("utf-8")
        res = ler_arquivo_cientifico(conteudo)
        assert len(res.df) == 5
        assert numeros(res, "I") == quase(I_LISTA)

    def test_linhas_de_comentario_no_topo(self):
        conteudo = ("# gerado pelo instrumento\n# 2026-11-20\n" + tabela([V_LISTA, I_LISTA], ["V", "I"])).encode("utf-8")
        res = ler_arquivo_cientifico(conteudo)
        assert res.linha_inicio_dados == 2
        assert list(res.df.columns) == ["V", "I"]

    def test_separador_extra_no_fim_nao_desloca_colunas(self):
        conteudo = tabela([V_LISTA, I_LISTA], ["V", "I"], ";", ",", sep_final=True).encode("utf-8")
        res = ler_arquivo_cientifico(conteudo)
        assert list(res.df.columns) == ["V", "I"]
        assert numeros(res, "V") == quase(V_LISTA)

    def test_colunas_repetidas_ficam_unicas(self):
        res = ler_arquivo_cientifico(b"x,x,x\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n")
        assert len(res.df.columns) == 3
        assert len(set(res.df.columns)) == 3

    def test_tokens_nulos_viram_nan_na_conversao(self):
        res = ler_arquivo_cientifico(b"V,I\n0.1,1.0\n0.2,-\n0.3,OVLD\n0.4,4.0\n0.5,5.0\n")
        assert len(res.df) == 5
        i = numeros(res, "I")
        assert np.isnan(i[1]) and np.isnan(i[2])
        assert [i[0], i[3], i[4]] == [1.0, 4.0, 5.0]

    def test_quebras_de_linha_crlf(self):
        res = ler_arquivo_cientifico(b"V,I\r\n0.1,1\r\n0.2,2\r\n0.3,3\r\n0.4,4\r\n")
        assert list(res.df.columns) == ["V", "I"] and len(res.df) == 4

    def test_formato_hibrido_solartron_smart(self):
        # Cabeçalho com ',' e dados com ';' + ';' final + relógio (hh:mm:ss) na 1ª coluna
        pontos = [(1e5, 100.2, -5.1), (1e4, 110.5, -30.2), (1e3, 300.1, -900.5), (1e2, 2500.3, -2400.1)]
        linhas = ["Time,Freq,Zre,Zim"] + [
            f"00:00:{14 + k:02d};{f:.3f};{zr:.3f};{zi:.3f};".replace(".", ",") for k, (f, zr, zi) in enumerate(pontos)
        ]
        res = ler_arquivo_cientifico(("\n".join(linhas) + "\n").encode("utf-8"))
        assert (res.sep_usado, res.decimal_usado, res.tem_cabecalho) == (";", ",", True)
        assert list(res.df.columns) == ["Time", "Freq", "Zre", "Zim"]
        assert numeros(res, "Zim") == quase([-5.1, -30.2, -900.5, -2400.1])

    def test_linhas_brutas_topo_limitadas_a_15(self):
        assert len(ler_arquivo_cientifico(("a\n" * 30).encode()).linhas_brutas_topo) == 15


class TestEncodingsNaLeitura:
    @pytest.mark.parametrize(
        "codec,enc_esperado",
        [("utf-8", "utf-8"), ("utf-8-sig", "utf-8-sig"), ("utf-16", "utf-16"), ("cp1252", "cp1252"), ("latin-1", "cp1252")],
    )
    def test_cabecalho_com_acentos_e_micro_sobrevive(self, codec, enc_esperado):
        texto = "Tensão (V);Corrente (µA)\n" + tabela([V_LISTA, [1.5, 2.5, 3.5, 4.5, 5.5]], None, ";", ",")
        res = ler_arquivo_cientifico(texto.encode(codec))
        assert res.encoding_usado == enc_esperado
        assert list(res.df.columns) == ["Tensão (V)", "Corrente (µA)"]
        assert (res.sep_usado, res.decimal_usado) == (";", ",")
        assert numeros(res, "Corrente (µA)") == quase([1.5, 2.5, 3.5, 4.5, 5.5])

    def test_arquivo_gravado_em_disco_cp1252(self, tmp_path):
        caminho = tmp_path / "amostra_lab.dat"
        caminho.write_bytes(("Corrente (µA);Tensão (V)\n" + tabela([[1.5, 2.5, 3.5, 4.5], [0.1, 0.2, 0.3, 0.4]], None, ";", ",")).encode("cp1252"))
        res = ler_arquivo_cientifico(caminho.read_bytes())
        assert res.encoding_usado == "cp1252"
        assert list(res.df.columns) == ["Corrente (µA)", "Tensão (V)"]


# ==============================================================================
# 5. OpcoesParser (OVERRIDE MANUAL)
# ==============================================================================
class TestOpcoesParser:
    def test_padroes(self):
        o = OpcoesParser()
        assert (o.sep, o.decimal, o.encoding, o.skip_extra) == ("auto", "auto", "auto", 0)

    def test_tupla_de_cache_reflete_todos_os_campos(self):
        assert OpcoesParser().to_cache_tuple() == ("auto", "auto", "auto", 0)
        assert OpcoesParser(sep=";", decimal=",", encoding="cp1252", skip_extra=3).to_cache_tuple() == (";", ",", "cp1252", 3)

    def test_tupla_de_cache_e_hashavel_e_distingue_opcoes(self):
        a, b = OpcoesParser(sep=";").to_cache_tuple(), OpcoesParser(sep=",").to_cache_tuple()
        assert a != b
        assert {a: 1}[OpcoesParser(sep=";").to_cache_tuple()] == 1  # usável como chave de cache

    def test_forcar_separador_e_decimal(self):
        conteudo = tabela([V_LISTA, I_LISTA], ["V", "I"], ";", ",").encode("utf-8")
        res = ler_arquivo_cientifico(conteudo, OpcoesParser(sep=";", decimal=","))
        assert (res.sep_usado, res.decimal_usado) == (";", ",")
        assert numeros(res, "I") == quase(I_LISTA)

    def test_forcar_separador_de_espacos(self):
        res = ler_arquivo_cientifico(b"V I\n0.1 1.5\n0.2 2.5\n0.3 3.5\n0.4 4.5\n", OpcoesParser(sep=r"\s+"))
        assert res.sep_usado == r"\s+" and len(res.df) == 4

    def test_pular_linhas_extras_do_topo(self):
        conteudo = b"lixo 1\nlixo 2\nV,I\n0.1,1\n0.2,2\n0.3,3\n0.4,4\n"
        res = ler_arquivo_cientifico(conteudo, OpcoesParser(skip_extra=2))
        assert res.linha_inicio_dados == 2
        assert list(res.df.columns) == ["V", "I"] and len(res.df) == 4

    def test_skip_extra_maior_que_o_arquivo_e_ignorado(self):
        res = ler_arquivo_cientifico(b"V,I\n0.1,1\n0.2,2\n0.3,3\n0.4,4\n", OpcoesParser(skip_extra=100))
        assert list(res.df.columns) == ["V", "I"] and len(res.df) == 4

    def test_encoding_forcado_correto(self):
        conteudo = ("Corrente (µA);V\n" + tabela([[1.5, 2.5, 3.5, 4.5], [0.1, 0.2, 0.3, 0.4]], None, ";", ",")).encode("cp1252")
        res = ler_arquivo_cientifico(conteudo, OpcoesParser(encoding="cp1252"))
        assert res.encoding_usado == "cp1252"
        assert res.df.columns[0] == "Corrente (µA)"

    def test_encoding_forcado_errado_mostra_caractere_de_substituicao(self):
        conteudo = ("Corrente (µA);V\n" + tabela([[1.5, 2.5, 3.5, 4.5], [0.1, 0.2, 0.3, 0.4]], None, ";", ",")).encode("cp1252")
        res = ler_arquivo_cientifico(conteudo, OpcoesParser(encoding="utf-8"))
        assert res.encoding_usado == "utf-8"
        assert "\ufffd" in res.df.columns[0]
        assert len(res.df) == 4  # a tabela segue legível

    def test_opcoes_chegam_aos_extratores(self):
        conteudo = b"lixo 1\nlixo 2\n" + iv_bytes()
        df, meta = extrair_keithley_iv(conteudo, OpcoesParser(skip_extra=2))
        assert len(df) == 9 and meta.linha_inicio_dados == 2


# ==============================================================================
# 6. KEITHLEY I×V
# ==============================================================================
class TestKeithleyIV:
    @pytest.mark.parametrize("sep,dec", DIALETOS)
    def test_ciclo_completo_em_todos_os_dialetos(self, sep, dec):
        df, meta = extrair_keithley_iv(iv_bytes(sep, dec))
        assert isinstance(meta, ResultadoLeituraTabela)
        assert list(df.columns) == ["Order", "Time", "Voltage", "Current", "Sweep_Sign", "Sweep_Branch"]
        assert df["Voltage"].tolist() == quase(V_CICLO)
        assert df["Current"].tolist() == quase(I_CICLO)
        assert df["Time"].tolist() == quase(T_CICLO)
        assert df["Order"].tolist() == list(range(9))
        assert (meta.sep_usado, meta.decimal_usado) == (sep_esperado(sep), dec)

    def test_ramos_de_ida_e_volta_e_histerese(self):
        df, _ = extrair_keithley_iv(iv_bytes())
        assert df["Sweep_Sign"].tolist() == [1] * 5 + [-1] * 4
        assert df["Sweep_Branch"].str.startswith("Ida").sum() == 5
        assert df["Sweep_Branch"].str.startswith("Volta").sum() == 4
        assert df.attrs["tem_histerese"] is True

    def test_somente_ida_nao_tem_histerese(self):
        df, _ = extrair_keithley_iv(iv_bytes(v=[0.0, 0.25, 0.5, 0.75, 1.0]))
        assert df.attrs["tem_histerese"] is False
        assert set(df["Sweep_Sign"]) == {1}

    def test_somente_volta_nao_tem_histerese(self):
        df, _ = extrair_keithley_iv(iv_bytes(v=[1.0, 0.75, 0.5, 0.25, 0.0]))
        assert df.attrs["tem_histerese"] is False
        assert set(df["Sweep_Sign"]) == {-1}  # o 1º ponto herda a direção do 2º
        assert df["Sweep_Branch"].str.startswith("Volta").all()

    def test_tensao_repetida_herda_a_ultima_direcao(self):
        df, _ = extrair_keithley_iv(b"V,I\n0.0,1\n0.5,2\n0.5,3\n1.0,4\n")
        assert df["Sweep_Sign"].tolist() == [1, 1, 1, 1]

    def test_sem_coluna_de_tempo_usa_a_ordem_das_linhas(self):
        df, _ = extrair_keithley_iv(iv_bytes(com_tempo=False, cabecalho=("Voltage", "Current")))
        assert df["Time"].tolist() == [float(k) for k in range(9)]

    def test_coluna_de_tempo_do_arquivo_e_preservada(self):
        df, _ = extrair_keithley_iv(b"Time,Voltage,Current\n10,0.0,1\n20,0.5,2\n30,1.0,3\n40,0.5,2\n")
        assert df["Time"].tolist() == [10.0, 20.0, 30.0, 40.0]
        assert df["Order"].tolist() == [0, 1, 2, 3]

    @pytest.mark.parametrize(
        "cabecalho",
        [("Time (s)", "Source Voltage (V)", "Measure Current (A)"), ("Tempo", "Tensão (V)", "Corrente (A)"), ("t", "V", "I")],
    )
    def test_nomes_de_colunas_reconhecidos(self, cabecalho):
        df, _ = extrair_keithley_iv(iv_bytes(cabecalho=cabecalho))
        assert df["Voltage"].tolist() == quase(V_CICLO)
        assert df["Current"].tolist() == quase(I_CICLO)

    def test_ordem_trocada_corrente_antes_de_tensao(self):
        conteudo = tabela([I_CICLO, V_CICLO], ["I", "V"]).encode("utf-8")
        df, _ = extrair_keithley_iv(conteudo)
        assert df["Voltage"].tolist() == quase(V_CICLO)
        assert df["Current"].tolist() == quase(I_CICLO)

    def test_tres_colunas_sem_cabecalho_assumem_tempo_tensao_corrente(self):
        df, _ = extrair_keithley_iv(iv_bytes("\t", ".", cabecalho=None))
        assert df["Voltage"].tolist() == quase(V_CICLO)
        assert df["Current"].tolist() == quase(I_CICLO)

    def test_pontos_com_overload_sao_descartados(self):
        df, _ = extrair_keithley_iv(b"V,I\n0.0,1\n0.5,OVLD\n1.0,4\n1.5,5\n")
        assert df["Voltage"].tolist() == [0.0, 1.0, 1.5]
        assert df["Current"].tolist() == [1.0, 4.0, 5.0]
        assert df["Order"].tolist() == [0, 2, 3]  # "Order" guarda a posição ORIGINAL da linha no arquivo

    def test_arquivo_em_disco_com_metadados(self, tmp_path):
        caminho = tmp_path / "iv_amostra1.dat"
        caminho.write_bytes(b"Instrument: Keithley 2450\nSample: P3HT-01\n\n" + iv_bytes())
        df, meta = extrair_keithley_iv(caminho.read_bytes())
        assert meta.linha_inicio_dados == 3
        assert len(df) == 9

    @pytest.mark.parametrize(
        "conteudo,trecho",
        [
            (b"", "vazio"),
            (TEXTO_SEM_NUMEROS.encode("utf-8"), "identificar as colunas"),
            (b"1\n2\n3\n4\n5\n", "identificar as colunas"),
            (b"V,I\n1,2\n", "Menos de 2 pares"),
            (b"V,I\n1,OVLD\n2,OVLD\n3,OVLD\n4,OVLD\n", "Menos de 2 pares"),
        ],
        ids=["vazio", "so_texto", "uma_coluna", "um_par", "tudo_overload"],
    )
    def test_entradas_invalidas_levantam_valueerror_claro(self, conteudo, trecho):
        with pytest.raises(ValueError, match=trecho):
            extrair_keithley_iv(conteudo)

    def test_binario_corrompido_levanta_valueerror(self):
        with pytest.raises(ValueError):
            extrair_keithley_iv(BINARIO)


# ==============================================================================
# 7. KEITHLEY I×t
# ==============================================================================
class TestKeithleyIT:
    @pytest.mark.parametrize("sep,dec", DIALETOS)
    def test_dialetos(self, sep, dec):
        df, meta, info = extrair_keithley_it(it_bytes("Corrente", 1.5e-9, "Tempo", sep, dec))
        assert list(df.columns) == ["Time_raw", "Current_raw"]
        assert df["Time_raw"].tolist() == [float(k) for k in range(10)]
        assert df["Current_raw"].tolist() == quase([1.5e-9] * 10)
        assert info["col_t"] == "Tempo" and info["col_i"] == "Corrente"
        assert (meta.sep_usado, meta.decimal_usado) == (sep_esperado(sep), dec)

    @pytest.mark.parametrize(
        "cab_i,valor,unidade",
        [
            ("Current (A)", 1.5e-9, "A"),
            ("Current (µA)", 2.5, "µA"),
            ("Corrente (nA)", 25.0, "nA"),
            ("Current (mA)", 0.5, "mA"),
            ("Measure", 3e-7, "A"),
        ],
    )
    def test_unidade_sugerida_pelo_cabecalho_ou_ordem_de_grandeza(self, cab_i, valor, unidade):
        _, _, info = extrair_keithley_it(it_bytes(cab_i, valor))
        assert info["unidade_sugerida"] == unidade

    def test_sem_unidade_e_mediana_alta_pede_verificacao(self):
        _, _, info = extrair_keithley_it(it_bytes("Measure", 3.0))
        assert info["unidade_sugerida"].startswith("µA")
        assert "verificar" in info["unidade_sugerida"]

    def test_nao_altera_a_escala_da_corrente(self):
        df, _, info = extrair_keithley_it(it_bytes("Current (nA)", 25.0))
        assert df["Current_raw"].tolist() == [25.0] * 10
        assert info["mediana_abs_bruta"] == pytest.approx(25.0)

    def test_sem_cabecalho_usa_primeira_e_ultima_colunas(self):
        df, _, info = extrair_keithley_it(it_bytes(None, 1.5e-9))
        assert (info["col_t"], info["col_i"]) == ("Col_1", "Col_2")
        assert len(df) == 10

    def test_arquivo_com_tensao_no_meio_pega_tempo_e_corrente_pelos_nomes(self):
        linhas = ["Time,Voltage,Current"] + [f"{k},0.5,{1e-9 * (k + 1):.3e}" for k in range(10)]
        df, _, info = extrair_keithley_it(("\n".join(linhas) + "\n").encode("utf-8"))
        assert (info["col_t"], info["col_i"]) == ("Time", "Current")
        assert df["Current_raw"].iloc[0] == quase(1e-9)

    def test_pontos_nao_numericos_sao_descartados(self):
        df, _, _ = extrair_keithley_it(b"Time,Current\n1,0.1\n2,OVLD\n3,0.3\n4,0.4\n")
        assert df["Time_raw"].tolist() == [1.0, 3.0, 4.0]

    @pytest.mark.parametrize(
        "conteudo,trecho",
        [
            (b"", "vazio"),
            (TEXTO_SEM_NUMEROS.encode("utf-8"), "Nenhum ponto"),
            (b"Time,Current\n1,OVLD\n2,OVLD\n3,OVLD\n4,OVLD\n", "Nenhum ponto"),
        ],
        ids=["vazio", "so_texto", "tudo_overload"],
    )
    def test_entradas_invalidas_levantam_valueerror_claro(self, conteudo, trecho):
        with pytest.raises(ValueError, match=trecho):
            extrair_keithley_it(conteudo)


# ==============================================================================
# 8. SOLARTRON EIS
# ==============================================================================
SINAIS_ZIM = [
    # (Z'' gravado no arquivo, cabeçalho de Z'', inverter_sinal_zim, sinal esperado de Z_imag)
    pytest.param("neg", "Z''(Ohm)", None, -1, id="convencao_negativa_auto"),
    pytest.param("pos", "Z''(Ohm)", None, -1, id="positivo_auto_inverte"),
    pytest.param("pos", "Z''(Ohm)", True, -1, id="positivo_forcar_inversao"),
    pytest.param("pos", "Z''(Ohm)", False, +1, id="positivo_manter_bruto"),
    pytest.param("neg", "Z''(Ohm)", True, +1, id="negativo_forcar_inversao"),
    pytest.param("neg", "Z''(Ohm)", False, -1, id="negativo_manter_bruto"),
    pytest.param("pos", "-Z''(Ohm)", None, -1, id="cabecalho_menos_zim_dados_positivos"),
    pytest.param("neg", "-Z''(Ohm)", None, +1, id="cabecalho_menos_zim_inverte_sempre"),
]


class TestSolartronEIS:
    @pytest.mark.parametrize("gravado,cab_zim,flag,sinal", SINAIS_ZIM)
    def test_convencao_de_sinal_de_zim(self, gravado, cab_zim, flag, sinal):
        conteudo = eis_bytes(cab=("Freq(Hz)", "Z'(Ohm)", cab_zim), sinal_zim=-1.0 if gravado == "neg" else +1.0)
        df, _ = extrair_solartron_eis(conteudo, inverter_sinal_zim=flag)
        assert np.sign(df["Z_imag"].median()) == sinal
        # -Z'' e Z'' sempre são espelhos, qualquer que seja a convenção escolhida
        assert df["minus_Z_imag"].tolist() == pytest.approx((-df["Z_imag"]).tolist())

    def test_arco_capacitivo_recuperado_do_circuito_de_randles(self):
        df, _ = extrair_solartron_eis(eis_bytes())
        assert df["Z_real"].iloc[0] == pytest.approx(RS, rel=0.01)  # alta frequência ≈ Rs
        assert df["Z_real"].max() == pytest.approx(RS + RCT, rel=1e-4)  # baixa frequência ≈ Rs + Rct
        assert df["minus_Z_imag"].max() == pytest.approx(RCT / 2, rel=0.05)  # topo do semicírculo ≈ Rct/2
        assert (df["Z_imag"] <= 0).all()

    def test_colunas_derivadas_sao_consistentes(self):
        df, _ = extrair_solartron_eis(eis_bytes())
        assert list(df.columns) == ["Freq", "Z_real", "Z_imag", "minus_Z_imag", "Z_mod", "Phase_deg"]
        assert df["Z_mod"].tolist() == pytest.approx(np.hypot(df["Z_real"], df["Z_imag"]).tolist())
        assert df["Phase_deg"].tolist() == pytest.approx(np.degrees(np.arctan2(df["Z_imag"], df["Z_real"])).tolist())
        assert df["Z_mod"].tolist() == pytest.approx(np.abs(Z_RANDLES).tolist(), rel=1e-6)

    def test_frequencias_em_ordem_decrescente(self):
        df, _ = extrair_solartron_eis(eis_bytes())
        assert df["Freq"].is_monotonic_decreasing
        assert df["Freq"].iloc[0] == pytest.approx(1e5) and df["Freq"].iloc[-1] == pytest.approx(0.1)

    def test_arquivo_em_ordem_crescente_e_reordenado(self):
        linhas = eis_bytes().decode("utf-8").splitlines()
        invertido = "\n".join([linhas[0]] + linhas[1:][::-1]) + "\n"
        df, _ = extrair_solartron_eis(invertido.encode("utf-8"))
        assert df["Freq"].is_monotonic_decreasing
        assert df["Z_real"].iloc[0] == pytest.approx(RS, rel=0.01)

    def test_frequencias_nao_positivas_sao_descartadas(self):
        linhas = eis_bytes().decode("utf-8").splitlines()
        linhas[3:3] = ["0.0,1.0,-1.0", "-5.0,1.0,-1.0"]
        df, _ = extrair_solartron_eis(("\n".join(linhas) + "\n").encode("utf-8"))
        assert len(df) == 25
        assert (df["Freq"] > 0).all()

    @pytest.mark.parametrize(
        "sep,dec,cab",
        [
            (",", ".", ("Freq(Hz)", "Z'(Ohm)", "Z''(Ohm)")),
            (";", ",", ("Freq", "Z'(a)", "Z''(b)")),
            ("\t", ".", ("Freq(Hz)", "Z'(Ohm)", "Z''(Ohm)")),
            (",", ".", ("Frequency [Hz]", "Zre", "Zim")),
            (",", ".", None),
        ],
        ids=["csv", "zplot_ponto_virgula", "tab", "zre_zim", "sem_cabecalho"],
    )
    def test_dialetos_e_nomes_de_colunas(self, sep, dec, cab):
        df, _ = extrair_solartron_eis(eis_bytes(cab=cab, sep=sep, dec=dec))
        assert len(df) == 25
        assert df["Z_real"].max() == pytest.approx(RS + RCT, rel=1e-4)
        assert df["Z_imag"].median() < 0

    def test_coordenadas_polares_modulo_e_fase(self):
        df, _ = extrair_solartron_eis(eis_polar_bytes(fase_negativa=True))
        assert df["Z_real"].tolist() == pytest.approx(Z_RANDLES.real.tolist(), rel=1e-6)
        assert df["Z_imag"].tolist() == pytest.approx(Z_RANDLES.imag.tolist(), rel=1e-6, abs=1e-6)

    def test_polar_com_fase_positiva_e_invertida_no_modo_automatico(self):
        df, _ = extrair_solartron_eis(eis_polar_bytes(fase_negativa=False))
        assert (df["Z_imag"] <= 0).all()

    def test_polar_com_fase_positiva_pode_ser_mantida_bruta(self):
        df, _ = extrair_solartron_eis(eis_polar_bytes(fase_negativa=False), inverter_sinal_zim=False)
        assert (df["Z_imag"] >= 0).all()

    @pytest.mark.parametrize("unidade,escala", [("kOhm", 1e3), ("MOhm", 1e6)])
    def test_escala_pelo_cabecalho_devolve_ohms(self, unidade, escala):
        conteudo = eis_bytes(cab=("Freq(Hz)", f"Z'({unidade})", f"Z''({unidade})"), escala=escala)
        df, _ = extrair_solartron_eis(conteudo)
        assert df["Z_real"].max() == pytest.approx(RS + RCT, rel=1e-4)
        assert df["minus_Z_imag"].max() == pytest.approx(RCT / 2, rel=0.05)

    def test_cabecalho_de_metadados_estilo_zplot(self):
        prefixo = "ZPLOT2 ASCII\nSample: P3HT\nDate: 2026-11-20\nEnd Comments\n"
        df, meta = extrair_solartron_eis(eis_bytes(prefixo=prefixo))
        assert len(df) == 25
        assert meta.linhas_brutas_topo[0] == "ZPLOT2 ASCII"

    def test_formato_smart_com_relogio_e_separador_final(self):
        pontos = [(1e5, 100.2, -5.1), (1e4, 110.5, -30.2), (1e3, 300.1, -900.5), (1e2, 2500.3, -2400.1)]
        linhas = ["Time,Freq,Zre,Zim"] + [
            f"00:00:{14 + k:02d};{f:.3f};{zr:.3f};{zi:.3f};".replace(".", ",") for k, (f, zr, zi) in enumerate(pontos)
        ]
        df, meta = extrair_solartron_eis(("\n".join(linhas) + "\n").encode("utf-8"))
        assert (meta.sep_usado, meta.decimal_usado) == (";", ",")
        assert df["Freq"].tolist() == quase([1e5, 1e4, 1e3, 1e2])
        assert df["Z_real"].tolist() == quase([100.2, 110.5, 300.1, 2500.3])
        assert df["Z_imag"].tolist() == quase([-5.1, -30.2, -900.5, -2400.1])

    def test_simbolo_omega_no_cabecalho_utf8(self):
        conteudo = "Freq (Hz);Z' (Ω);Z'' (Ω)\n1000;100;-10\n100;110;-30\n10;300;-100\n1;500;-60\n".encode("utf-8")
        df, meta = extrair_solartron_eis(conteudo)
        assert meta.encoding_usado == "utf-8"
        assert df["Z_real"].tolist() == [100.0, 110.0, 300.0, 500.0]

    def test_sem_nomes_reconhecidos_usa_as_duas_colunas_apos_a_frequencia(self):
        conteudo = b"Freq,A,B,C\n1000,1,-2,3\n100,2,-3,4\n10,3,-4,5\n1,4,-5,6\n"
        df, _ = extrair_solartron_eis(conteudo)
        assert df["Z_real"].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert df["Z_imag"].tolist() == [-2.0, -3.0, -4.0, -5.0]

    @pytest.mark.parametrize(
        "conteudo,trecho",
        [
            (b"", "vazio"),
            (TEXTO_SEM_NUMEROS.encode("utf-8"), "Frequência"),
            (b"Freq,Zre\n1000,1\n100,2\n10,3\n1,4\n", "impedância"),
            (b"Freq,Zre,Zim\n0,1,-1\n0,2,-2\n0,3,-3\n0,4,-4\n", "Freq > 0"),
        ],
        ids=["vazio", "so_texto", "duas_colunas", "todas_freq_zero"],
    )
    def test_entradas_invalidas_levantam_valueerror_claro(self, conteudo, trecho):
        with pytest.raises(ValueError, match=trecho):
            extrair_solartron_eis(conteudo)

    def test_binario_corrompido_levanta_valueerror(self):
        with pytest.raises(ValueError):
            extrair_solartron_eis(BINARIO)


# ==============================================================================
# 9. CARY UV-VIS
# ==============================================================================
class TestCaryUVVis:
    @pytest.mark.parametrize(
        "sep,dec,nome_arquivo",
        [(",", ".", "amostras.csv"), (";", ",", "amostras_br.csv"), ("\t", ".", "amostras_tab.txt")],
        ids=["virgula_ponto", "pontoevirgula_virgula", "tab"],
    )
    def test_csv_multi_amostra_pareado(self, sep, dec, nome_arquivo):
        espectros, meta = extrair_cary_uvvis(cary_multi_bytes(sep, dec), nome_arquivo)
        assert [e["nome"] for e in espectros] == ["P3HT_A", "P3DT_B"]
        assert all(e["origem"] == nome_arquivo for e in espectros)
        assert espectros[0]["df"]["abs"].tolist() == quase(ABS_A[::-1])
        assert espectros[1]["df"]["abs"].tolist() == quase(ABS_B[::-1])
        assert meta.sep_usado == sep and meta.tem_cabecalho is True and meta.linha_inicio_dados == 2

    def test_pares_ordenados_por_comprimento_de_onda_crescente(self):
        espectros, _ = extrair_cary_uvvis(cary_multi_bytes(), "x.csv")
        for e in espectros:
            assert list(e["df"].columns) == ["wl", "abs"]
            assert e["df"]["wl"].is_monotonic_increasing
            assert e["df"]["wl"].iloc[0] == 400.0 and e["df"]["wl"].iloc[-1] == 800.0

    def test_rodape_instrumental_e_descartado(self):
        com, _ = extrair_cary_uvvis(cary_multi_bytes(rodape=True), "x.csv")
        sem, _ = extrair_cary_uvvis(cary_multi_bytes(rodape=False), "x.csv")
        assert [len(e["df"]) for e in com] == [len(e["df"]) for e in sem] == [9, 9]

    def test_nome_de_amostra_vazio_usa_nome_do_arquivo(self):
        espectros, _ = extrair_cary_uvvis(cary_multi_bytes(nome_a=""), "meu.arquivo.csv")
        assert espectros[0]["nome"] == "meu.arquivo_S1"
        assert espectros[1]["nome"] == "P3DT_B"

    def test_multi_amostra_em_cp1252(self):
        conteudo = cary_multi_bytes().decode("utf-8").replace("Abs", "Abs (µ)").encode("cp1252")
        espectros, meta = extrair_cary_uvvis(conteudo, "x.csv")
        assert len(espectros) == 2 and meta.encoding_usado == "cp1252"

    @pytest.mark.parametrize(
        "sep,dec",
        [(",", "."), (";", ","), ("\t", ".")],
        ids=["virgula_ponto", "pontoevirgula_virgula", "tab"],
    )
    def test_arquivo_simples_de_duas_colunas(self, sep, dec):
        espectros, meta = extrair_cary_uvvis(cary_simples_bytes(sep, dec), "filme_P3HT.csv")
        assert len(espectros) == 1
        assert espectros[0]["nome"] == "filme_P3HT"
        assert espectros[0]["df"]["wl"].tolist() == quase(WL[::-1])
        assert espectros[0]["df"]["abs"].tolist() == quase(ABS_A[::-1])
        assert (meta.sep_usado, meta.decimal_usado) == (sep, dec)

    def test_duas_colunas_sem_cabecalho(self):
        espectros, _ = extrair_cary_uvvis(cary_simples_bytes(cabecalho=False), "semcab.dat")
        assert espectros[0]["nome"] == "semcab"
        assert len(espectros[0]["df"]) == 9

    @pytest.mark.parametrize(
        "nome_arquivo,esperado",
        [("filme.csv", "filme"), ("filme.CSV", "filme"), ("filme.txt", "filme"), ("filme.dat", "filme"), ("filme.dsp", "filme"), ("filme.xyz", "filme.xyz")],
    )
    def test_extensao_removida_do_nome_da_amostra(self, nome_arquivo, esperado):
        espectros, _ = extrair_cary_uvvis(cary_simples_bytes(), nome_arquivo)
        assert espectros[0]["nome"] == esperado

    def test_nome_padrao_quando_nao_informado(self):
        espectros, _ = extrair_cary_uvvis(cary_simples_bytes())
        assert espectros[0]["nome"] == "Espectro"

    def test_matriz_com_eixo_compartilhado_e_cabecalho(self):
        conteudo = tabela([WL, ABS_A, ABS_B], ["Wavelength (nm)", "Amostra1", "Amostra2"]).encode("utf-8")
        espectros, _ = extrair_cary_uvvis(conteudo, "mat.csv")
        assert [e["nome"] for e in espectros] == ["Amostra1", "Amostra2"]
        assert espectros[1]["df"]["abs"].tolist() == quase(ABS_B[::-1])

    def test_matriz_sem_cabecalho_nomeia_pelas_colunas(self):
        conteudo = tabela([WL, ABS_A, ABS_B]).encode("utf-8")
        espectros, _ = extrair_cary_uvvis(conteudo, "mat2.csv")
        assert [e["nome"] for e in espectros] == ["mat2_Col_2", "mat2_Col_3"]

    def test_opcoes_de_separador_e_decimal_no_multi_amostra(self):
        espectros, meta = extrair_cary_uvvis(cary_multi_bytes(";", ","), "x.csv", OpcoesParser(sep=";", decimal=","))
        assert len(espectros) == 2
        assert (meta.sep_usado, meta.decimal_usado) == (";", ",")

    def test_pular_linhas_extras_no_arquivo_simples(self):
        espectros, _ = extrair_cary_uvvis(b"lixo1\nlixo2\n" + cary_simples_bytes(), "x.csv", OpcoesParser(skip_extra=2))
        assert len(espectros[0]["df"]) == 9

    def test_multi_amostra_aceita_o_minimo_de_3_pontos_por_espectro(self):
        conteudo = b"A,,B,\nWavelength (nm),Abs,Wavelength (nm),Abs\n500,0.1,500,0.2\n501,0.2,501,0.3\n502,0.3,502,0.4\n"
        espectros, _ = extrair_cary_uvvis(conteudo, "x.csv")
        assert [len(e["df"]) for e in espectros] == [3, 3]

    def test_arquivo_simples_aceita_2_pontos(self):
        espectros, _ = extrair_cary_uvvis(b"wl,a\n500,0.1\n501,0.2\n", "x.csv")
        assert len(espectros[0]["df"]) == 2

    def test_arquivo_em_disco(self, tmp_path):
        caminho = tmp_path / "P3HT_filme_01.csv"
        caminho.write_bytes(cary_simples_bytes(";", ","))
        espectros, _ = extrair_cary_uvvis(caminho.read_bytes(), caminho.name)
        assert espectros[0]["nome"] == "P3HT_filme_01"

    @pytest.mark.parametrize(
        "conteudo,trecho",
        [
            (b"", "Não foi possível extrair"),
            (TEXTO_SEM_NUMEROS.encode("utf-8"), "Não foi possível extrair"),
            (b"1\n2\n3\n4\n5\n", "Não foi possível extrair"),
            (b"wl,a,b\n500,0.1,0.2\n501,0.2,0.3\n", "inválidos ou vazios"),
            (b"A,,B,\nWavelength (nm),Abs,Wavelength (nm),Abs\n500,0.1,500,0.2\n501,0.2,501,0.3\n", "inválidos ou vazios"),
        ],
        ids=["vazio", "so_texto", "uma_coluna", "matriz_com_2_pontos", "multi_com_2_pontos"],
    )
    def test_entradas_invalidas_levantam_valueerror_claro(self, conteudo, trecho):
        with pytest.raises(ValueError, match=trecho):
            extrair_cary_uvvis(conteudo, "x.csv")


# ==============================================================================
# 10. ENTRADAS CORROMPIDAS NO LEITOR GERAL
# ==============================================================================
class TestEntradasInvalidasNoLeitorGeral:
    @pytest.mark.parametrize("conteudo", [b"", b"   \n\n  \n", b"V,I\n"], ids=["vazio", "so_espacos", "so_cabecalho"])
    def test_sem_dados_devolve_dataframe_vazio(self, conteudo):
        res = ler_arquivo_cientifico(conteudo)
        assert isinstance(res, ResultadoLeituraTabela)
        assert res.df.empty

    @pytest.mark.parametrize(
        "conteudo",
        [TEXTO_SEM_NUMEROS.encode("utf-8"), BINARIO, b"1\n2\n3\n4\n5\n"],
        ids=["so_texto", "binario", "uma_coluna"],
    )
    def test_sem_tabela_numerica_nao_produz_duas_colunas(self, conteudo):
        res = ler_arquivo_cientifico(conteudo)
        assert res.df.shape[1] < 2  # nenhum par de colunas utilizável

    def test_nunca_levanta_excecao_com_entradas_estranhas(self):
        for conteudo in [b"\x00" * 50, b"\xff\xfe", b",,,,\n,,,,\n,,,,\n", b";\n;\n;\n", "µµµ\nΩΩΩ\n".encode("utf-8")]:
            ler_arquivo_cientifico(conteudo)  # não pode levantar


# ==============================================================================
# 11. LIMITAÇÕES CONHECIDAS DO PARSER ATUAL (xfail estrito)
# ==============================================================================
CSV_PT_BR_COM_TEXTO = (
    "Tensao;Corrente;Rotulo\n0,0;1,0E-9;a\n0,1;2,1E-9;b\n0,2;3,0E-9;c\n0,3;4,2E-9;d\n0,4;5,1E-9;e\n"
).encode("utf-8")


class TestLimitacoesConhecidas:
    """
    Linhas de dados exigem que TODAS as células sejam numéricas (ou nulas/data/hora). Uma coluna de texto
    no bloco de dados invalida todas as linhas; o detector não acha tabela e volta ao padrão (',', '.').
    """

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Limitação atual: coluna de texto no bloco de dados impede detectar ';' + vírgula decimal.",
    )
    def test_coluna_de_texto_no_bloco_pt_br_automatico(self):
        res = ler_arquivo_cientifico(CSV_PT_BR_COM_TEXTO)
        assert (res.sep_usado, res.decimal_usado) == (";", ",")
        assert list(res.df.columns) == ["Tensao", "Corrente", "Rotulo"]

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Limitação atual: nem o override sep=';' / decimal=',' resolve; o fallback ignora a coluna de texto.",
    )
    def test_coluna_de_texto_no_bloco_pt_br_com_override(self):
        res = ler_arquivo_cientifico(CSV_PT_BR_COM_TEXTO, OpcoesParser(sep=";", decimal=","))
        assert (res.sep_usado, res.decimal_usado) == (";", ",")
        assert numeros(res, "Corrente") == quase([1.0e-9, 2.1e-9, 3.0e-9, 4.2e-9, 5.1e-9])

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason="Limitação atual: extrair_keithley_iv herda a falha do leitor geral e levanta ValueError nesse arquivo.",
    )
    def test_keithley_iv_com_coluna_de_texto_pt_br(self):
        df, _ = extrair_keithley_iv(CSV_PT_BR_COM_TEXTO)
        assert df["Current"].tolist() == quase([1.0e-9, 2.1e-9, 3.0e-9, 4.2e-9, 5.1e-9])

    def test_coluna_de_texto_so_funciona_por_coincidencia_no_padrao_virgula_ponto(self):
        # Guarda de regressão: no dialeto padrão (',' + '.') a coluna de texto passa porque o FALLBACK
        # do detector coincide com o dialeto real do arquivo.
        conteudo = b"Tensao,Corrente,Rotulo\n0.0,1.0E-9,a\n0.1,2.1E-9,b\n0.2,3.0E-9,c\n0.3,4.2E-9,d\n0.4,5.1E-9,e\n"
        res = ler_arquivo_cientifico(conteudo)
        assert list(res.df.columns) == ["Tensao", "Corrente", "Rotulo"]
        assert res.df["Rotulo"].tolist() == list("abcde")
        assert numeros(res, "Corrente") == quase([1.0e-9, 2.1e-9, 3.0e-9, 4.2e-9, 5.1e-9])


# ==============================================================================
# 12. PAINEL DE OVERRIDE E DIAGNÓSTICO (UI) — com um "st" falso, sem servidor Streamlit
# ==============================================================================
class _Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, nome):
        return getattr(self._st, nome)


class StFalso:
    """Imita só o que parsers.py usa de streamlit; `indices` escolhe a opção (por key) dos selectbox."""

    def __init__(self, indices=None, skip=0):
        self.indices = indices or {}
        self.skip = skip
        self.chamadas: List[tuple] = []
        self.opcoes_por_key = {}

    def expander(self, rotulo, expanded=False):
        self.chamadas.append(("expander", rotulo))
        return _Ctx(self)

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(self) for _ in range(n)]

    def container(self):
        return _Ctx(self)

    def selectbox(self, rotulo, opcoes, index=0, key=None):
        opcoes = list(opcoes)
        self.opcoes_por_key[key] = opcoes
        self.chamadas.append(("selectbox", key))
        return opcoes[self.indices.get(key, index)]

    def number_input(self, rotulo, minimo, maximo, valor, step=1, key=None):
        self.chamadas.append(("number_input", key))
        return self.skip

    def caption(self, texto):
        self.chamadas.append(("caption", texto))

    def markdown(self, texto):
        self.chamadas.append(("markdown", texto))

    def code(self, texto, language=None):
        self.chamadas.append(("code", texto))

    def dataframe(self, df, **kwargs):
        self.chamadas.append(("dataframe", len(df)))

    def textos(self, tipo):
        return [c[1] for c in self.chamadas if c[0] == tipo]


class TestPainelOverrideEDiagnostico:
    def test_padrao_devolve_opcoes_automaticas(self, monkeypatch):
        falso = StFalso()
        monkeypatch.setattr(P, "st", falso)
        opcoes, slot = render_painel_override_parser("meu")
        assert opcoes == OpcoesParser()
        assert slot is not None
        assert [c[1] for c in falso.chamadas if c[0] in ("selectbox", "number_input")] == [
            "p_sep_meu",
            "p_dec_meu",
            "p_enc_meu",
            "p_skp_meu",
        ]

    def test_selecoes_viram_opcoes_do_parser(self, monkeypatch):
        # índices das listas: sep = auto, ',', ';', TAB, espaços | dec = auto, '.', ',' | enc = auto, utf-8, cp1252, latin-1, utf-16
        falso = StFalso(indices={"p_sep_x": 2, "p_dec_x": 2, "p_enc_x": 2}, skip=3)
        monkeypatch.setattr(P, "st", falso)
        opcoes, _ = render_painel_override_parser("x")
        assert opcoes == OpcoesParser(sep=";", decimal=",", encoding="cp1252", skip_extra=3)

    @pytest.mark.parametrize(
        "indice,sep",
        [(0, "auto"), (1, ","), (2, ";"), (3, "\t"), (4, r"\s+")],
    )
    def test_todas_as_opcoes_de_separador(self, monkeypatch, indice, sep):
        monkeypatch.setattr(P, "st", StFalso(indices={"p_sep_s": indice}))
        opcoes, _ = render_painel_override_parser("s")
        assert opcoes.sep == sep

    @pytest.mark.parametrize("indice,enc", [(0, "auto"), (1, "utf-8"), (2, "cp1252"), (3, "latin-1"), (4, "utf-16")])
    def test_todas_as_opcoes_de_encoding(self, monkeypatch, indice, enc):
        monkeypatch.setattr(P, "st", StFalso(indices={"p_enc_e": indice}))
        opcoes, _ = render_painel_override_parser("e")
        assert opcoes.encoding == enc

    def test_opcoes_do_painel_alimentam_o_leitor(self, monkeypatch):
        monkeypatch.setattr(P, "st", StFalso(indices={"p_sep_k": 2, "p_dec_k": 2}))
        opcoes, _ = render_painel_override_parser("k")
        res = ler_arquivo_cientifico(tabela([V_LISTA, I_LISTA], ["V", "I"], ";", ",").encode("utf-8"), opcoes)
        assert (res.sep_usado, res.decimal_usado) == (";", ",")

    def test_diagnostico_sem_container_ou_sem_meta_nao_faz_nada(self, monkeypatch):
        falso = StFalso()
        monkeypatch.setattr(P, "st", falso)
        meta = ler_arquivo_cientifico(tabela([V_LISTA, I_LISTA], ["V", "I"]).encode("utf-8"))
        exibir_diagnostico_leitura(None, meta, "a.dat")
        exibir_diagnostico_leitura(falso.container(), None, "a.dat")
        assert falso.chamadas == []

    @pytest.mark.parametrize(
        "sep,dec,sep_legivel",
        [(",", ".", "Vírgula (,)"), (";", ",", "Ponto-e-vírgula (;)"), ("\t", ".", "TAB"), (" ", ".", "Espaços")],
    )
    def test_diagnostico_descreve_o_dialeto_detectado(self, monkeypatch, sep, dec, sep_legivel):
        falso = StFalso()
        monkeypatch.setattr(P, "st", falso)
        conteudo = tabela([V_LISTA, I_LISTA], ["V", "I"], sep, dec).encode("utf-8")
        meta = ler_arquivo_cientifico(conteudo)
        exibir_diagnostico_leitura(falso.container(), meta, "amostra.dat")
        legenda = falso.textos("caption")[0]
        assert "amostra.dat" in legenda
        assert sep_legivel in legenda
        assert f"Decimal = `'{dec}'`" in legenda
        assert "Encoding = `utf-8`" in legenda
        assert "Cabeçalho detectado = `Sim`" in legenda

    def test_diagnostico_mostra_linhas_brutas_e_previa_da_tabela(self, monkeypatch):
        falso = StFalso()
        monkeypatch.setattr(P, "st", falso)
        conteudo = ("Instrument: Keithley\nDate: 2026-11-20\n" + tabela([V_LISTA, I_LISTA], ["V", "I"])).encode("utf-8")
        meta = ler_arquivo_cientifico(conteudo)
        exibir_diagnostico_leitura(falso.container(), meta)
        assert "Instrument: Keithley" in falso.textos("code")[0]
        assert falso.textos("dataframe") == [5]
        assert "linha 3" in falso.textos("caption")[0]  # linha_inicio_dados (índice 2) exibida como linha 3
        assert "1º arquivo" in falso.textos("caption")[0]  # nome padrão quando não informado
