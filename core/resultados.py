# -*- coding: utf-8 -*-
"""
core/resultados.py
Registro central de resultados por AMOSTRA e TÉCNICA, guardado em st.session_state.

Generaliza o padrão "add-on 03 publica -> add-on 04 lê" para qualquer par de abas e alimenta o futuro
relatório consolidado. Cada add-on registra métricas escalares (σ, histerese, t90, Rct, Eg, W ...) com unidades
e metadados; qualquer aba consulta, filtra e exporta (DataFrame / JSON).

Princípios de projeto
  - Identidade do registro = (técnica, amostra). Registrar de novo a mesma dupla ATUALIZA o registro existente:
    nunca duplica linhas. Por padrão mescla as métricas; com substituir=True troca o conjunto inteiro.
  - Idempotente: repetir exatamente o mesmo conteúdo (o que acontece a cada rerun do Streamlit) devolve
    "inalterado" e não muda revisão, data nem versão global. Mudanças reais guardam, em `alteracoes`,
    o valor antigo e o novo de cada métrica, então nada é sobrescrito sem rastro.
  - Armazenamento em dicts/listas/escalares simples (nada de instâncias de classes do projeto no session_state),
    para sobreviver ao hot reload de core/ feito pelo app.py. As classes públicas (Resultado) são só "visões" de leitura.
  - Seguro com @st.fragment: nenhuma chamada ao Streamlit na importação; sem st.sidebar e sem st.rerun.
    Como fragmentos não se reexecutam entre si, quem LÊ (ex.: relatório) deve consultar versao() ao renderizar
    e oferecer um botão de atualizar quando mudou_desde(versao_vista) for True.
  - O session_state some ao recarregar a página: use exportar_json()/importar_json() para guardar os resultados.

Convenção de nomes (sugestão para os add-ons concordarem entre si)
  Técnicas ... TEC_IV, TEC_IT, TEC_EIS, TEC_ZVIEW, TEC_UVVIS
  Métricas ... nomes curtos sem espaços e unidade à parte, p. ex. sigma (S/m), histerese (bool), t90 (s),
               Rct (Ω), Eg (eV), W (meV). Valores sempre em unidades SI/do instrumento, não na escala de exibição.

Exemplo
    from core import resultados as res

    status = res.registrar("P3HT_A1", res.TEC_IV, {"sigma": 1.2e-3, "R2": 0.998},
                           unidades={"sigma": "S/m"}, meta={"arquivo": "iv_a1.dat"}, origem="addon_01")
    df = res.para_dataframe("consolidado")          # uma linha por amostra, colunas "<técnica> · <métrica>"
"""

import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

CHAVE_ESTADO = "suite_resultados"
SCHEMA = 1

# Nomes de técnica padronizados (use estas constantes para as abas "conversarem" entre si)
TEC_IV = "I×V"
TEC_IT = "I×t"
TEC_EIS = "EIS"
TEC_ZVIEW = "ZView"
TEC_UVVIS = "UV-Vis"

# Status devolvidos por registrar()
NOVO = "novo"
ATUALIZADO = "atualizado"
INALTERADO = "inalterado"

Filtro = Union[str, Sequence[str], None]

__all__ = [
    "ATUALIZADO",
    "CHAVE_ESTADO",
    "INALTERADO",
    "NOVO",
    "Resultado",
    "SCHEMA",
    "TEC_EIS",
    "TEC_IT",
    "TEC_IV",
    "TEC_UVVIS",
    "TEC_ZVIEW",
    "amostras",
    "existe",
    "exportar_json",
    "importar_json",
    "limpar",
    "listar",
    "mudou_desde",
    "obter",
    "para_dataframe",
    "registrar",
    "registrar_lote",
    "remover",
    "remover_metricas",
    "renomear_amostra",
    "sincronizar",
    "tecnicas",
    "total",
    "versao",
]


# ==============================================================================
# VISÃO DE LEITURA DE UM REGISTRO
# ==============================================================================
@dataclass(frozen=True)
class Resultado:
    """
    Cópia de leitura de um registro (alterá-la NÃO altera o registro central).
    `alteracoes` mapeia métrica -> (valor_antigo, valor_novo) da última atualização real
    (None = não existia / foi removida).
    """

    amostra: str
    tecnica: str
    metricas: Dict[str, Any]
    unidades: Dict[str, str]
    meta: Dict[str, Any]
    origem: str
    criado_em: str
    atualizado_em: str
    revisao: int
    alteracoes: Dict[str, Tuple[Any, Any]]

    def valor(self, nome: str, padrao: Any = None) -> Any:
        return self.metricas.get(nome, padrao)

    def rotulo(self, nome: str, incluir_unidade: bool = True) -> str:
        unidade = self.unidades.get(nome, "")
        return f"{nome} ({unidade})" if (incluir_unidade and unidade) else nome

    def para_dict(self) -> Dict[str, Any]:
        return {
            "amostra": self.amostra,
            "tecnica": self.tecnica,
            "metricas": copy.deepcopy(self.metricas),
            "unidades": dict(self.unidades),
            "meta": copy.deepcopy(self.meta),
            "origem": self.origem,
            "criado_em": self.criado_em,
            "atualizado_em": self.atualizado_em,
            "revisao": self.revisao,
            "alteracoes": {k: [v[0], v[1]] for k, v in self.alteracoes.items()},
        }


def _visao(rec: Dict[str, Any]) -> Resultado:
    return Resultado(
        amostra=rec["amostra"],
        tecnica=rec["tecnica"],
        metricas=copy.deepcopy(rec["metricas"]),
        unidades=dict(rec["unidades"]),
        meta=copy.deepcopy(rec["meta"]),
        origem=rec["origem"],
        criado_em=rec["criado_em"],
        atualizado_em=rec["atualizado_em"],
        revisao=int(rec["revisao"]),
        alteracoes={k: (v[0], v[1]) for k, v in rec["alteracoes"].items()},
    )


# ==============================================================================
# ARMAZENAMENTO E NORMALIZAÇÃO (interno)
# ==============================================================================
def _estado() -> Dict[str, Any]:
    """Dicionário-raiz em st.session_state: {'schema', 'versao', 'registros': {técnica: {amostra: registro}}}."""
    est = st.session_state.get(CHAVE_ESTADO)
    if est is None:
        est = {"schema": SCHEMA, "versao": 0, "registros": {}}
        st.session_state[CHAVE_ESTADO] = est
        return est
    if not (isinstance(est, dict) and est.get("schema") == SCHEMA and isinstance(est.get("registros"), dict)):
        raise RuntimeError(
            f"st.session_state['{CHAVE_ESTADO}'] tem formato incompatível com o schema {SCHEMA}; "
            "o registro central não foi alterado."
        )
    return est


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _nome(valor: Any, rotulo: str) -> str:
    texto = "" if valor is None else str(valor).strip()
    if not texto:
        raise ValueError(f"O nome da {rotulo} não pode ser vazio.")
    return texto


def _escalar(valor: Any, nome: str) -> Any:
    """Métricas aceitam apenas escalares (None, bool, int, float, str); numpy vira tipo nativo."""
    if isinstance(valor, np.generic):
        valor = valor.item()
    if valor is None or isinstance(valor, (bool, int, float, str)):
        return valor
    raise TypeError(
        f"A métrica '{nome}' tem valor do tipo {type(valor).__name__}; use escalares "
        "(número, texto, booleano ou None). Séries/tabelas pertencem ao add-on, não ao registro."
    )


def _normalizar_metricas(metricas: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(metricas, Mapping) or not metricas:
        raise ValueError("Informe ao menos uma métrica em `metricas` (dict nome -> valor).")
    saida: Dict[str, Any] = {}
    for nome, valor in metricas.items():
        saida[_nome(nome, "métrica")] = _escalar(valor, str(nome))
    return saida


def _json_simples(valor: Any) -> Any:
    """Normaliza metadados para tipos JSON (dict/list/str/número/bool/None). Tipos exóticos viram texto."""
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, np.ndarray):
        valor = valor.tolist()
    if valor is None or isinstance(valor, (bool, int, float, str)):
        return valor
    if isinstance(valor, Mapping):
        return {str(k): _json_simples(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [_json_simples(v) for v in valor]
    return str(valor)


def _iguais(a: Any, b: Any) -> bool:
    """Igualdade que trata NaN == NaN e ignora ruído de ponto flutuante (rel 1e-12); recursiva em dict/list."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return math.isclose(fa, fb, rel_tol=1e-12, abs_tol=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_iguais(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_iguais(x, y) for x, y in zip(a, b))
    return bool(a == b)


def _conjunto(filtro: Filtro) -> Optional[set]:
    if filtro is None:
        return None
    if isinstance(filtro, str):
        return {filtro.strip()}
    return {str(x).strip() for x in filtro}


def _registros_filtrados(
    amostra: Filtro = None,
    tecnica: Filtro = None,
    origem: Filtro = None,
) -> List[Dict[str, Any]]:
    f_amostra, f_tecnica, f_origem = _conjunto(amostra), _conjunto(tecnica), _conjunto(origem)
    saida: List[Dict[str, Any]] = []
    for nome_tec, por_amostra in _estado()["registros"].items():
        if f_tecnica is not None and nome_tec not in f_tecnica:
            continue
        for nome_am, rec in por_amostra.items():
            if f_amostra is not None and nome_am not in f_amostra:
                continue
            if f_origem is not None and rec["origem"] not in f_origem:
                continue
            saida.append(rec)
    return saida


def _marcar_mudanca(est: Dict[str, Any]) -> None:
    est["versao"] = int(est["versao"]) + 1


def _podar_tecnica_vazia(est: Dict[str, Any], tecnica: str) -> None:
    if tecnica in est["registros"] and not est["registros"][tecnica]:
        del est["registros"][tecnica]


# ==============================================================================
# REGISTRO / ATUALIZAÇÃO
# ==============================================================================
def registrar(
    amostra: str,
    tecnica: str,
    metricas: Mapping[str, Any],
    *,
    unidades: Optional[Mapping[str, str]] = None,
    meta: Optional[Mapping[str, Any]] = None,
    origem: str = "",
    substituir: bool = False,
) -> str:
    """
    Registra (ou atualiza) as métricas de uma amostra em uma técnica. Retorna NOVO, ATUALIZADO ou INALTERADO.

    - Mesma (técnica, amostra): atualiza o registro existente. `substituir=False` (padrão) mescla métricas,
      unidades e metadados; `substituir=True` troca métricas, unidades e metadados pelos informados.
    - INALTERADO (conteúdo idêntico, NaN-aware): nada muda — seguro de chamar a cada rerun.
    - Atualização real: revisão +1, atualizado_em novo e `alteracoes` com (antigo, novo) por métrica.
    - `origem` identifica quem registrou (ex.: nome do add-on); vazio mantém a origem anterior.
    - Levanta ValueError/TypeError para nomes vazios, métricas vazias, valores não escalares ou
      unidades de métricas inexistentes — sem alterar o registro.
    """
    amostra = _nome(amostra, "amostra")
    tecnica = _nome(tecnica, "técnica")
    novas = _normalizar_metricas(metricas)
    unid_novas = {_nome(k, "métrica"): str(v) for k, v in (unidades or {}).items()}
    meta_nova = _json_simples(dict(meta)) if meta else {}

    est = _estado()
    atual = est["registros"].get(tecnica, {}).get(amostra)

    if atual is None:
        metricas_fin, unid_fin, meta_fin = novas, unid_novas, meta_nova
    elif substituir:
        metricas_fin, unid_fin, meta_fin = novas, unid_novas, meta_nova
    else:
        metricas_fin = {**atual["metricas"], **novas}
        unid_fin = {**atual["unidades"], **unid_novas}
        meta_fin = {**atual["meta"], **meta_nova}

    desconhecidas = [k for k in unid_fin if k not in metricas_fin]
    if desconhecidas:
        raise ValueError(f"Unidade informada para métrica inexistente: {', '.join(desconhecidas)}.")

    agora = _agora()
    if atual is None:
        est["registros"].setdefault(tecnica, {})[amostra] = {
            "amostra": amostra,
            "tecnica": tecnica,
            "metricas": copy.deepcopy(metricas_fin),
            "unidades": dict(unid_fin),
            "meta": copy.deepcopy(meta_fin),
            "origem": origem or "",
            "criado_em": agora,
            "atualizado_em": agora,
            "revisao": 1,
            "alteracoes": {},
        }
        _marcar_mudanca(est)
        return NOVO

    origem_fin = origem or atual["origem"]
    if (
        _iguais(metricas_fin, atual["metricas"])
        and unid_fin == atual["unidades"]
        and _iguais(meta_fin, atual["meta"])
        and origem_fin == atual["origem"]
    ):
        return INALTERADO

    alteracoes: Dict[str, List[Any]] = {}
    for nome in list(atual["metricas"].keys()) + [n for n in metricas_fin if n not in atual["metricas"]]:
        antigo = atual["metricas"].get(nome)
        novo = metricas_fin.get(nome)
        existia, existe_agora = nome in atual["metricas"], nome in metricas_fin
        if existia != existe_agora or (existia and not _iguais(antigo, novo)):
            alteracoes[nome] = [antigo, novo]

    atual["metricas"] = copy.deepcopy(metricas_fin)
    atual["unidades"] = dict(unid_fin)
    atual["meta"] = copy.deepcopy(meta_fin)
    atual["origem"] = origem_fin
    atual["atualizado_em"] = agora
    atual["revisao"] = int(atual["revisao"]) + 1
    atual["alteracoes"] = alteracoes
    _marcar_mudanca(est)
    return ATUALIZADO


def registrar_lote(
    tecnica: str,
    por_amostra: Mapping[str, Mapping[str, Any]],
    *,
    unidades: Optional[Mapping[str, str]] = None,
    meta: Optional[Mapping[str, Any]] = None,
    origem: str = "",
    substituir: bool = False,
) -> Dict[str, str]:
    """Registra várias amostras de uma técnica: {amostra: {métrica: valor}}. Retorna {amostra: status}."""
    return {
        amostra: registrar(
            amostra,
            tecnica,
            metricas,
            unidades=unidades,
            meta=meta,
            origem=origem,
            substituir=substituir,
        )
        for amostra, metricas in por_amostra.items()
    }


def sincronizar(tecnica: str, origem: str, amostras_vigentes: Sequence[str]) -> List[str]:
    """
    Faz o registro de (técnica, origem) espelhar o estado atual da aba: remove os registros dessa técnica
    registrados por essa origem cuja amostra NÃO está em `amostras_vigentes` (arquivo removido, curva
    desativada, legenda renomeada). Registros de outras origens ou técnicas não são tocados.
    Retorna os nomes removidos.
    """
    tecnica = _nome(tecnica, "técnica")
    vigentes = {str(a).strip() for a in amostras_vigentes}
    est = _estado()
    por_amostra = est["registros"].get(tecnica, {})
    obsoletas = [nome for nome, rec in por_amostra.items() if rec["origem"] == origem and nome not in vigentes]
    for nome in obsoletas:
        del por_amostra[nome]
    if obsoletas:
        _podar_tecnica_vazia(est, tecnica)
        _marcar_mudanca(est)
    return obsoletas


# ==============================================================================
# CONSULTA
# ==============================================================================
def obter(amostra: str, tecnica: str) -> Optional[Resultado]:
    rec = _estado()["registros"].get(str(tecnica).strip(), {}).get(str(amostra).strip())
    return _visao(rec) if rec is not None else None


def existe(amostra: str, tecnica: str) -> bool:
    return str(amostra).strip() in _estado()["registros"].get(str(tecnica).strip(), {})


def listar(amostra: Filtro = None, tecnica: Filtro = None, origem: Filtro = None) -> List[Resultado]:
    """Registros (cópias) filtrados por amostra, técnica e/ou origem (str ou lista), na ordem de registro."""
    return [_visao(rec) for rec in _registros_filtrados(amostra, tecnica, origem)]


def amostras(tecnica: Filtro = None) -> List[str]:
    """Nomes de amostra (sem repetição, na ordem em que apareceram), opcionalmente só de certas técnicas."""
    vistos: Dict[str, None] = {}
    for rec in _registros_filtrados(tecnica=tecnica):
        vistos.setdefault(rec["amostra"], None)
    return list(vistos)


def tecnicas(amostra: Filtro = None) -> List[str]:
    """Técnicas com registros, opcionalmente só as que têm certas amostras."""
    vistos: Dict[str, None] = {}
    for rec in _registros_filtrados(amostra=amostra):
        vistos.setdefault(rec["tecnica"], None)
    return list(vistos)


def total() -> int:
    """Número de registros (pares técnica/amostra)."""
    return sum(len(v) for v in _estado()["registros"].values())


def versao() -> int:
    """Contador global: só aumenta quando algo REALMENTE muda (registro novo/atualizado, remoção, importação)."""
    return int(_estado()["versao"])


def mudou_desde(versao_anterior: int) -> bool:
    """True se o registro mudou depois de `versao_anterior` (valor guardado pelo consumidor, ex.: relatório)."""
    return versao() != int(versao_anterior)


# ==============================================================================
# REMOÇÃO E EDIÇÃO
# ==============================================================================
def remover(amostra: str, tecnica: str) -> bool:
    """Remove um registro exato. Retorna False (sem alterar a versão) se não existia."""
    amostra, tecnica = str(amostra).strip(), str(tecnica).strip()
    est = _estado()
    por_amostra = est["registros"].get(tecnica)
    if not por_amostra or amostra not in por_amostra:
        return False
    del por_amostra[amostra]
    _podar_tecnica_vazia(est, tecnica)
    _marcar_mudanca(est)
    return True


def remover_metricas(amostra: str, tecnica: str, nomes: Sequence[str]) -> int:
    """Remove métricas específicas de um registro (e o registro, se ficar sem métricas). Retorna quantas saíram."""
    amostra, tecnica = str(amostra).strip(), str(tecnica).strip()
    est = _estado()
    rec = est["registros"].get(tecnica, {}).get(amostra)
    if rec is None:
        return 0
    alvo = [n for n in dict.fromkeys(str(n).strip() for n in nomes) if n in rec["metricas"]]
    if not alvo:
        return 0
    rec["alteracoes"] = {n: [rec["metricas"][n], None] for n in alvo}
    for n in alvo:
        del rec["metricas"][n]
        rec["unidades"].pop(n, None)
    if rec["metricas"]:
        rec["atualizado_em"] = _agora()
        rec["revisao"] = int(rec["revisao"]) + 1
    else:
        del est["registros"][tecnica][amostra]
        _podar_tecnica_vazia(est, tecnica)
    _marcar_mudanca(est)
    return len(alvo)


def limpar(
    *,
    amostra: Filtro = None,
    tecnica: Filtro = None,
    origem: Filtro = None,
    tudo: bool = False,
) -> int:
    """
    Remove os registros que casam com TODOS os filtros informados. Sem nenhum filtro exige tudo=True
    (evita apagar o registro inteiro por engano). Retorna quantos registros foram removidos.
    """
    sem_filtro = amostra is None and tecnica is None and origem is None
    if sem_filtro and not tudo:
        raise ValueError("limpar() sem filtros exige tudo=True; informe amostra, tecnica ou origem.")
    est = _estado()
    alvos = [(r["tecnica"], r["amostra"]) for r in _registros_filtrados(amostra, tecnica, origem)]
    for tec, am in alvos:
        del est["registros"][tec][am]
    for tec in {t for t, _ in alvos}:
        _podar_tecnica_vazia(est, tec)
    if alvos:
        _marcar_mudanca(est)
    return len(alvos)


def renomear_amostra(
    nome_atual: str,
    novo_nome: str,
    tecnica: Filtro = None,
    *,
    sobrescrever: bool = False,
) -> int:
    """
    Renomeia uma amostra (em todas as técnicas ou só nas indicadas). Se já existir registro com o novo
    nome na mesma técnica, levanta ValueError (nada é alterado), a menos que sobrescrever=True.
    Retorna quantos registros foram renomeados.
    """
    atual, novo = _nome(nome_atual, "amostra"), _nome(novo_nome, "amostra")
    if atual == novo:
        return 0
    est = _estado()
    alvo_tec = _conjunto(tecnica)
    movimentos = []
    for nome_tec, por_amostra in est["registros"].items():
        if alvo_tec is not None and nome_tec not in alvo_tec:
            continue
        if atual in por_amostra:
            if novo in por_amostra and not sobrescrever:
                raise ValueError(f"Já existe '{novo}' na técnica '{nome_tec}'. Use sobrescrever=True para substituir.")
            movimentos.append(nome_tec)
    for nome_tec in movimentos:
        por_amostra = est["registros"][nome_tec]
        rec = por_amostra.pop(atual)
        rec["amostra"] = novo
        rec["atualizado_em"] = _agora()
        rec["revisao"] = int(rec["revisao"]) + 1
        por_amostra[novo] = rec
    if movimentos:
        _marcar_mudanca(est)
    return len(movimentos)


# ==============================================================================
# EXPORTAÇÃO: DATAFRAME E JSON
# ==============================================================================
_RESERVADAS = {"Amostra", "Técnica", "Origem", "Atualizado em", "Revisão"}


def _col_metrica(rec: Dict[str, Any], nome: str, incluir_unidades: bool) -> str:
    unidade = rec["unidades"].get(nome, "")
    return f"{nome} ({unidade})" if (incluir_unidades and unidade) else nome


def para_dataframe(
    formato: str = "longo",
    *,
    amostra: Filtro = None,
    tecnica: Filtro = None,
    origem: Filtro = None,
    incluir_unidades: bool = True,
) -> pd.DataFrame:
    """
    Converte o registro (filtrado) em tabela.

      "longo"        uma linha por métrica: Amostra, Técnica, Métrica, Valor, Unidade, Origem, Atualizado em, Revisão
      "por_registro" uma linha por (amostra, técnica); uma coluna por métrica "nome (unidade)" + Origem, Atualizado em, Revisão
      "consolidado"  uma linha por amostra; colunas "<técnica> · nome (unidade)" — base do relatório consolidado

    Sem registros devolve um DataFrame vazio com as colunas-base do formato.
    """
    regs = _registros_filtrados(amostra, tecnica, origem)

    if formato == "longo":
        colunas = ["Amostra", "Técnica", "Métrica", "Valor", "Unidade", "Origem", "Atualizado em", "Revisão"]
        linhas = [
            [r["amostra"], r["tecnica"], nome, valor, r["unidades"].get(nome, ""), r["origem"], r["atualizado_em"], r["revisao"]]
            for r in regs
            for nome, valor in r["metricas"].items()
        ]
        return pd.DataFrame(linhas, columns=colunas)

    if formato == "por_registro":
        ordem: Dict[str, None] = {}
        linhas_d: List[Dict[str, Any]] = []
        for r in regs:
            linha: Dict[str, Any] = {"Amostra": r["amostra"], "Técnica": r["tecnica"]}
            for nome, valor in r["metricas"].items():
                col = _col_metrica(r, nome, incluir_unidades)
                if col in _RESERVADAS:
                    col = f"{col} (métrica)"
                ordem.setdefault(col, None)
                linha[col] = valor
            linha.update({"Origem": r["origem"], "Atualizado em": r["atualizado_em"], "Revisão": r["revisao"]})
            linhas_d.append(linha)
        colunas = ["Amostra", "Técnica"] + list(ordem) + ["Origem", "Atualizado em", "Revisão"]
        return pd.DataFrame(linhas_d, columns=colunas)

    if formato == "consolidado":
        ordem_c: Dict[str, None] = {}
        por_amostra: Dict[str, Dict[str, Any]] = {}
        for r in regs:
            linha_a = por_amostra.setdefault(r["amostra"], {"Amostra": r["amostra"]})
            for nome, valor in r["metricas"].items():
                col = f"{r['tecnica']} · {_col_metrica(r, nome, incluir_unidades)}"
                ordem_c.setdefault(col, None)
                linha_a[col] = valor
        return pd.DataFrame(list(por_amostra.values()), columns=["Amostra"] + list(ordem_c))

    raise ValueError("formato deve ser 'longo', 'por_registro' ou 'consolidado'.")


def exportar_json(indent: Optional[int] = 2) -> str:
    """
    Serializa todo o registro em JSON (UTF-8). Aceita NaN/Infinity (extensão do módulo json do Python).
    Guarde o texto (ex.: st.download_button) para recuperar os resultados depois de recarregar a página.
    """
    dados = {
        "schema": SCHEMA,
        "exportado_em": _agora(),
        "registros": [_visao(r).para_dict() for r in _registros_filtrados()],
    }
    return json.dumps(dados, ensure_ascii=False, indent=indent)


def importar_json(texto: Union[str, bytes], *, substituir_tudo: bool = False) -> Dict[str, int]:
    """
    Carrega um JSON gerado por exportar_json(). Todo o conteúdo é validado ANTES de qualquer alteração.
    Por padrão mescla com o registro atual (mesma regra de registrar()); substituir_tudo=True esvazia antes.
    Retorna {'novo': n, 'atualizado': n, 'inalterado': n}.
    """
    try:
        dados = json.loads(texto)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"JSON inválido: {exc}") from exc
    if not isinstance(dados, dict) or dados.get("schema") != SCHEMA or not isinstance(dados.get("registros"), list):
        raise ValueError(f"Arquivo não é um registro de resultados no schema {SCHEMA}.")

    itens = []
    for i, rec in enumerate(dados["registros"], start=1):
        if not isinstance(rec, dict) or not all(k in rec for k in ("amostra", "tecnica", "metricas")):
            raise ValueError(f"Registro {i} inválido: faltam 'amostra', 'tecnica' ou 'metricas'.")
        try:
            _nome(rec["amostra"], "amostra")
            _nome(rec["tecnica"], "técnica")
            _normalizar_metricas(rec["metricas"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Registro {i} inválido: {exc}") from exc
        itens.append(rec)

    if substituir_tudo:
        limpar(tudo=True)
    contagem = {NOVO: 0, ATUALIZADO: 0, INALTERADO: 0}
    for rec in itens:
        status = registrar(
            rec["amostra"],
            rec["tecnica"],
            rec["metricas"],
            unidades=rec.get("unidades") or None,
            meta=rec.get("meta") or None,
            origem=rec.get("origem") or "",
        )
        contagem[status] += 1
    return contagem
