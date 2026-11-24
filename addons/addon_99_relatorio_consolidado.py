# -*- coding: utf-8 -*-
"""
addons/addon_99_relatorio_consolidado.py
Módulo 99: Relatório Consolidado e Extensões de Pesquisa (core/resultados.py)
  - Agrega automaticamente todas as métricas escalares registradas por amostra e técnica
    (I×V, I×t, EIS, ZView, UV-Vis e qualquer add-on futuro)
  - Compatível com @st.fragment: monitora `res.versao()` / `res.mudou_desde(...)` e oferece
    atualização sob demanda sem depender de rerun global
  - Permite persistir e restaurar sessões completas via JSON (`exportar_json` / `importar_json`)
  - Motor modular interno de "Extensões de Pesquisa" (.py em `report_extensions/` ou via upload):
    permite acoplar análises específicas da pesquisa do usuário (cálculos derivados, rankings,
    gráficos customizados, tabelas prontas para artigo) sem alterar o núcleo genérico
"""

import ast
import importlib.util
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from core import resultados as res
from core.plotting import botao_download_csv, fmt_num
from core.ui import cabecalho_secao, msg_aviso, msg_sem_dados

TITLE = "99. Relatório Consolidado"
ICON = "📋"
ORDER = 99

BASE_DIR = Path(__file__).resolve().parent.parent
EXTENSIONS_DIR = BASE_DIR / "report_extensions"
EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# CONTRATO E CARREGAMENTO LEVE DE EXTENSÕES DE PESQUISA (.py)
# ==============================================================================
TEMPLATE_EXTENSAO = "\n".join(
    [
        "# -*- coding: utf-8 -*-",
        '"""',
        "report_extensions/exemplo_extensao_pesquisa.py",
        "Modelo de Extensão Modular para o Relatório Consolidado do lab-suite.",
        "",
        "Qualquer arquivo .py colocado em `report_extensions/` (sem iniciar com '_') que defina",
        "`render_extensao(contexto, cfg_plot)` aparece automaticamente dentro da aba",
        '"Relatório Consolidado" como uma seção acoplável específica da sua pesquisa.',
        "",
        "Conteúdo de `contexto`:",
        '  - "df_consolidado": pd.DataFrame (1 linha por amostra, colunas "<técnica> · <métrica>")',
        '  - "df_por_registro": pd.DataFrame (1 linha por par amostra/técnica)',
        '  - "df_longo": pd.DataFrame (1 linha por métrica)',
        '  - "registros": List[core.resultados.Resultado]',
        '  - "amostras": List[str]',
        '  - "tecnicas": List[str]',
        '"""',
        "",
        "import pandas as pd",
        "import streamlit as st",
        "",
        'TITLE = "Resumo Comparativo da Pesquisa (Exemplo)"',
        'DESCRIPTION = "Calcula razões e comparações personalizadas sobre as métricas registradas."',
        "ORDER = 10",
        "",
        "",
        "def render_extensao(contexto: dict, cfg_plot: dict) -> None:",
        '    df_cons: pd.DataFrame = contexto["df_consolidado"]',
        "    if df_cons.empty:",
        '        st.info("Sem amostras registradas para processar nesta extensão.")',
        "        return",
        "",
        '    st.write(f"Amostras disponíveis na extensão: **{len(df_cons)}**")',
        "    st.dataframe(df_cons, use_container_width=True, hide_index=True)",
        "",
    ]
)


def _validar_codigo_extensao(conteudo_bytes: bytes, nome_arquivo: str) -> Tuple[bool, str]:
    try:
        codigo_str = conteudo_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            codigo_str = conteudo_bytes.decode("cp1252")
        except UnicodeDecodeError:
            return False, f"Codificação inválida em '{nome_arquivo}'. Salve o arquivo em UTF-8."

    try:
        arvore = ast.parse(codigo_str, filename=nome_arquivo)
    except SyntaxError as exc:
        return False, f"Erro de sintaxe na linha {exc.lineno}: {exc.msg}"

    tem_render = any(
        isinstance(node, ast.FunctionDef) and node.name == "render_extensao"
        for node in arvore.body
    )
    if not tem_render:
        return (
            False,
            f"O arquivo '{nome_arquivo}' não define a função obrigatória "
            "`def render_extensao(contexto, cfg_plot):`.",
        )

    return True, codigo_str


def _carregar_extensoes(pasta: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    extensoes: List[Dict[str, Any]] = []
    erros: List[Dict[str, str]] = []
    if not pasta.exists():
        return extensoes, erros

    for caminho_py in sorted(pasta.glob("*.py"), key=lambda p: p.name.lower()):
        if caminho_py.name.startswith("_"):
            continue
        nome_mod = f"report_extensions.{caminho_py.stem}"
        try:
            spec = importlib.util.spec_from_file_location(nome_mod, str(caminho_py))
            if spec is None or spec.loader is None:
                erros.append({"arquivo": caminho_py.name, "erro": "Spec importlib inválida."})
                continue
            modulo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo)

            fn = getattr(modulo, "render_extensao", None)
            if not callable(fn):
                erros.append(
                    {
                        "arquivo": caminho_py.name,
                        "erro": "Função `render_extensao(contexto, cfg_plot)` ausente.",
                    }
                )
                continue

            titulo = str(getattr(modulo, "TITLE", caminho_py.stem)).strip() or caminho_py.stem
            desc = str(getattr(modulo, "DESCRIPTION", "")).strip()
            try:
                ordem = float(getattr(modulo, "ORDER", 999))
            except (TypeError, ValueError):
                ordem = 999.0

            extensoes.append(
                {
                    "filename": caminho_py.name,
                    "stem": caminho_py.stem,
                    "title": titulo,
                    "description": desc,
                    "order": ordem,
                    "render_fn": fn,
                }
            )
        except Exception:
            erros.append({"arquivo": caminho_py.name, "erro": traceback.format_exc()})

    extensoes.sort(key=lambda item: (item["order"], item["title"].lower(), item["filename"]))
    return extensoes, erros


# ==============================================================================
# FORMATAÇÃO VISUAL DE DATAFRAME RESPEITANDO cfg_plot
# ==============================================================================
def _formatar_df_exibicao(df: pd.DataFrame, usar_virgula: bool) -> pd.DataFrame:
    if df.empty:
        return df
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_bool_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda v: "Sim" if v else "Não")
        elif pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(
                lambda v: "—" if pd.isna(v) else fmt_num(float(v), 4, usar_virgula)
            )
    return df_fmt


# ==============================================================================
# INTERFACE PRINCIPAL DO RELATÓRIO CONSOLIDADO
# ==============================================================================
def render(cfg_plot: dict) -> None:
    usar_virgula = bool(cfg_plot.get("usar_virgula", True))
    cabecalho_secao("Relatório Consolidado por Amostra & Extensões de Pesquisa")

    # 1. Sincronização com @st.fragment (detecta alterações feitas em outras abas)
    ver_atual = res.versao()
    ver_vista = st.session_state.get("rel_versao_vista", ver_atual)

    c_top1, c_top2, c_top3, c_top4 = st.columns([1.1, 1.1, 1.1, 1.4])
    c_top1.metric("Amostras Registradas", len(res.amostras()))
    c_top2.metric("Técnicas Ativas", len(res.tecnicas()))
    c_top3.metric("Registros Totais", res.total())
    with c_top4:
        st.write("")
        if st.button("🔄 Sincronizar Dados das Abas", use_container_width=True, key="rel_btn_sync"):
            st.session_state["rel_versao_vista"] = res.versao()
            st.rerun()

    if res.mudou_desde(ver_vista):
        st.info(
            "🔔 Novos resultados foram registrados ou atualizados em outras abas desde a última "
            "visualização. Clique em **Sincronizar Dados das Abas** acima para atualizar."
        )
    else:
        st.session_state["rel_versao_vista"] = ver_atual

    # 2. Persistência de Sessão (Backup / Restauração JSON) e Gerenciamento
    with st.expander("💾 Sessão Analítica (Salvar / Restaurar JSON & Gerenciar Registros)", expanded=False):
        cj1, cj2, cj3 = st.columns([1.2, 1.4, 1.0])
        with cj1:
            json_str = res.exportar_json(indent=2)
            st.download_button(
                label="📥 Exportar Sessão Completa (.JSON)",
                data=json_str.encode("utf-8"),
                file_name="sessao_lab_suite.json",
                mime="application/json",
                use_container_width=True,
                disabled=(res.total() == 0),
                key="rel_dl_json",
            )
            st.caption(
                "Como o Streamlit limpa o `session_state` ao recarregar a página, salve o `.JSON` "
                "para retomar seus resultados depois."
            )

        with cj2:
            arq_json = st.file_uploader(
                "Restaurar ou mesclar sessão (.JSON)",
                type=["json"],
                accept_multiple_files=False,
                key="rel_up_json",
            )
            modo_imp = st.radio(
                "Modo de importação",
                ["Mesclar com registros atuais", "Substituir sessão inteira"],
                horizontal=True,
                key="rel_modo_imp",
            )
            if arq_json is not None and st.button("📤 Importar JSON", use_container_width=True, key="rel_btn_imp"):
                try:
                    cont = res.importar_json(
                        arq_json.getvalue(),
                        substituir_tudo=modo_imp.startswith("Substituir"),
                    )
                    st.session_state["rel_versao_vista"] = res.versao()
                    st.success(
                        f"Importação concluída: {cont['novo']} novo(s), "
                        f"{cont['atualizado']} atualizado(s), {cont['inalterado']} inalterado(s)."
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(f"Falha ao importar JSON: {exc}")

        with cj3:
            if res.amostras():
                st.markdown("**Renomear Amostra:**")
                am_orig = st.selectbox("Amostra atual", res.amostras(), key="rel_ren_orig")
                am_novo = st.text_input("Novo nome (unifica técnicas)", value=am_orig, key="rel_ren_novo")
                if st.button("✏️ Renomear", use_container_width=True, key="rel_btn_ren"):
                    try:
                        n_ren = res.renomear_amostra(am_orig, am_novo)
                        if n_ren > 0:
                            st.session_state["rel_versao_vista"] = res.versao()
                            st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

            st.markdown("---")
            if st.button("🗑️ Limpar Todos os Registros", use_container_width=True, disabled=(res.total() == 0), key="rel_btn_clear"):
                res.limpar(tudo=True)
                st.session_state["rel_versao_vista"] = res.versao()
                st.rerun()

    if res.total() == 0:
        msg_sem_dados(
            "Nenhum resultado registrado ainda. Execute análises nas outras abas (I×V, I×t, EIS, ZView, UV-Vis) "
            "ou importe um arquivo `.JSON` de sessão salva acima."
        )
        return

    # 3. Filtros Globais do Relatório
    st.markdown("### 🔎 Filtros do Relatório")
    f1, f2, f3 = st.columns([1.4, 1.4, 0.8])
    todas_amostras = res.amostras()
    todas_tecnicas = res.tecnicas()

    sel_amostras = f1.multiselect(
        "Filtrar Amostras (vazio = todas)",
        options=todas_amostras,
        default=[],
        key="rel_filt_amostras",
    )
    sel_tecnicas = f2.multiselect(
        "Filtrar Técnicas (vazio = todas)",
        options=todas_tecnicas,
        default=[],
        key="rel_filt_tecnicas",
    )
    incluir_unidades = f3.checkbox("Unidades nos cabeçalhos", value=True, key="rel_inc_unid")

    filtro_am = sel_amostras if sel_amostras else None
    filtro_tec = sel_tecnicas if sel_tecnicas else None

    df_consolidado = res.para_dataframe(
        "consolidado",
        amostra=filtro_am,
        tecnica=filtro_tec,
        incluir_unidades=incluir_unidades,
    )
    df_por_registro = res.para_dataframe(
        "por_registro",
        amostra=filtro_am,
        tecnica=filtro_tec,
        incluir_unidades=incluir_unidades,
    )
    df_longo = res.para_dataframe(
        "longo",
        amostra=filtro_am,
        tecnica=filtro_tec,
        incluir_unidades=incluir_unidades,
    )
    registros_filtrados = res.listar(amostra=filtro_am, tecnica=filtro_tec)

    if df_consolidado.empty:
        msg_aviso("Nenhum registro corresponde aos filtros selecionados.")
        return

    # 4. Visão Principal: Tabela Consolidada por Amostra
    st.markdown("### 📊 Tabela Consolidada (1 Linha por Amostra)")
    st.dataframe(
        _formatar_df_exibicao(df_consolidado, usar_virgula),
        use_container_width=True,
        hide_index=True,
    )

    ce1, ce2, ce3 = st.columns(3)
    with ce1:
        botao_download_csv(
            df_consolidado,
            "Relatorio_Consolidado_Amostras",
            cfg_plot,
            "rel_csv_cons",
            label="📥 Baixar Consolidado (.CSV)",
        )
    with ce2:
        botao_download_csv(
            df_por_registro,
            "Relatorio_Por_Registro",
            cfg_plot,
            "rel_csv_reg",
            label="📥 Baixar Por Registro (.CSV)",
        )
    with ce3:
        botao_download_csv(
            df_longo,
            "Relatorio_Metricas_Longo",
            cfg_plot,
            "rel_csv_longo",
            label="📥 Baixar Formato Longo (.CSV)",
        )

    # 5. Visões Detalhadas e Auditoria / Rastreabilidade
    with st.expander("🧾 Detalhamento por Técnica, Metadados e Histórico de Revisões", expanded=False):
        tab_reg, tab_longo, tab_aud = st.tabs(
            ["Por Registro (Amostra × Técnica)", "Lista de Métricas (Longo)", "Auditoria & Metadados"]
        )
        with tab_reg:
            st.dataframe(
                _formatar_df_exibicao(df_por_registro, usar_virgula),
                use_container_width=True,
                hide_index=True,
            )
        with tab_longo:
            st.dataframe(
                _formatar_df_exibicao(df_longo, usar_virgula),
                use_container_width=True,
                hide_index=True,
            )
        with tab_aud:
            for item in registros_filtrados:
                rotulo_exp = (
                    f"**{item.amostra}** — `{item.tecnica}` "
                    f"(rev. {item.revisao} | {item.atualizado_em} | origem: `{item.origem or 'n/d'}`)"
                )
                with st.expander(rotulo_exp, expanded=False):
                    ca1, ca2 = st.columns(2)
                    with ca1:
                        st.markdown("**Métricas e Unidades:**")
                        st.json(
                            {
                                k: {"valor": v, "unidade": item.unidades.get(k, "")}
                                for k, v in item.metricas.items()
                            }
                        )
                    with ca2:
                        st.markdown("**Metadados (`meta`) e Últimas Alterações:**")
                        st.json(
                            {
                                "meta": item.meta,
                                "alteracoes_ultima_revisao": item.alteracoes,
                            }
                        )

    st.divider()

    # 6. Camada Modular Interna: Extensões Específicas da Pesquisa do Usuário
    st.markdown("### 🧩 Extensões Modulares de Pesquisa (`report_extensions/`)")
    st.caption(
        "Acoplamento opcional para análises específicas da sua pesquisa (cálculos derivados, "
        "correlações entre técnicas, tabelas formatadas para artigo) sem alterar o relatório genérico."
    )

    with st.expander("➕ Instalar ou Criar Modelo de Extensão (.py)", expanded=False):
        cm1, cm2 = st.columns([1.1, 1.4])
        with cm1:
            st.download_button(
                label="📄 Baixar Modelo `exemplo_extensao_pesquisa.py`",
                data=TEMPLATE_EXTENSAO.encode("utf-8"),
                file_name="exemplo_extensao_pesquisa.py",
                mime="text/x-python",
                use_container_width=True,
                key="rel_dl_tpl_ext",
            )
            st.caption(
                "Salve seus scripts na pasta `report_extensions/` do projeto ou envie pelo seletor ao lado."
            )
        with cm2:
            arqs_ext = st.file_uploader(
                "Enviar extensão(ões) de pesquisa (.py)",
                type=["py"],
                accept_multiple_files=True,
                key="rel_up_ext",
            )
            if arqs_ext and st.button("📥 Instalar Extensão(ões)", use_container_width=True, key="rel_btn_inst_ext"):
                ok_list, err_list = [], []
                for arq in arqs_ext:
                    stem = re.sub(r"[^\w\-]", "_", Path(arq.name).stem).lstrip("_") or "extensao_custom"
                    nome_dest = f"{stem}.py"
                    valido, res_cod = _validar_codigo_extensao(arq.getvalue(), nome_dest)
                    if not valido:
                        err_list.append(res_cod)
                        continue
                    (EXTENSIONS_DIR / nome_dest).write_text(res_cod, encoding="utf-8")
                    ok_list.append(nome_dest)
                if ok_list:
                    st.success(f"Extensão(ões) instalada(s): {', '.join(ok_list)}")
                if err_list:
                    st.error("\n\n".join(err_list))
                if ok_list:
                    st.rerun()

    extensoes_ativas, erros_ext = _carregar_extensoes(EXTENSIONS_DIR)

    if erros_ext:
        st.warning(f"{len(erros_ext)} extensão(ões) com falha ao carregar:")
        for err in erros_ext:
            with st.expander(f"❌ Erro em `report_extensions/{err['arquivo']}`", expanded=False):
                st.code(err["erro"], language="python")

    if not extensoes_ativas:
        st.info(
            "Nenhuma extensão específica instalada em `report_extensions/`. "
            "Baixe o modelo acima se quiser plugar análises próprias da sua pesquisa."
        )
        return

    contexto_extensao = {
        "df_consolidado": df_consolidado,
        "df_por_registro": df_por_registro,
        "df_longo": df_longo,
        "registros": registros_filtrados,
        "amostras": list(df_consolidado["Amostra"].tolist()),
        "tecnicas": [t for t in todas_tecnicas if not filtro_tec or t in filtro_tec],
    }

    for ext in extensoes_ativas:
        with st.expander(f"🔬 {ext['title']} (`{ext['filename']}`)", expanded=True):
            if ext["description"]:
                st.caption(ext["description"])
            try:
                ext["render_fn"](contexto_extensao, cfg_plot)
            except Exception as exc:
                st.error(f"Erro na execução da extensão `{ext['filename']}`: {exc}")
                with st.expander("Ver traceback", expanded=False):
                    st.code(traceback.format_exc(), language="python")
