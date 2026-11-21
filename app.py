# -*- coding: utf-8 -*-
"""
app.py
Aplicação principal da Suíte Analítica Modular (P3HT / P3DT & Preditor ZView).
Responsável por:
  - Carregar dinamicamente todos os módulos válidos da pasta addons/ via importlib,
    com CACHE baseado no mtime dos arquivos de addons/ e core/ (só reimporta quando algo é editado)
  - Ordenar as abas pelo atributo numérico ORDER
  - Isolar cada aba em um @st.fragment: mexer num controle de uma aba não recalcula as demais
  - Gerenciar a barra lateral global no padrão OriginPro + opção de exportação CSV (Internacional / Excel BR)
  - Instalar novos arquivos .py diretamente na pasta addons/ com validação sintática AST
"""

import ast
import importlib.util
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

ADDONS_DIR = BASE_DIR / "addons"
CORE_DIR = BASE_DIR / "core"
ADDONS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Suíte Analítica Modular — P3HT/P3DT & ZView",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.fragment (Streamlit >= 1.37); em versões 1.33–1.36 usa a variante experimental;
# em versões anteriores o app continua funcionando, apenas sem isolamento por aba.
_FRAGMENT = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def validar_codigo_addon(conteudo_bytes: bytes, nome_arquivo: str) -> Tuple[bool, str]:
    """Valida sintaxe Python e presença da função obrigatória render(cfg_plot) antes de instalar."""
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
        isinstance(node, ast.FunctionDef) and node.name == "render"
        for node in arvore.body
    )
    if not tem_render:
        return (
            False,
            f"O arquivo '{nome_arquivo}' não define a função obrigatória `def render(cfg_plot):`.",
        )

    return True, codigo_str


# ==============================================================================
# CARREGAMENTO DE ADD-ONS COM CACHE POR MTIME
# ==============================================================================
def _listar_arquivos_addons(pasta_addons: Path) -> List[Path]:
    """Arquivos .py de addons/ que participam do carregamento (ignora os iniciados por '_')."""
    if not pasta_addons.exists():
        return []
    return sorted(
        (p for p in pasta_addons.glob("*.py") if not p.name.startswith("_")),
        key=lambda p: p.name.lower(),
    )


def _assinatura_projeto(pasta_addons: Path, pasta_core: Path) -> Tuple[Tuple[str, int, int], ...]:
    """
    Assinatura barata (caminho, mtime_ns, tamanho) dos arquivos de addons/ e core/.
    Qualquer edição, inclusão ou remoção muda a assinatura e invalida o cache de importação.
    """
    arquivos = _listar_arquivos_addons(pasta_addons)
    if pasta_core.exists():
        arquivos += sorted(pasta_core.glob("*.py"), key=lambda p: p.name.lower())

    assinatura: List[Tuple[str, int, int]] = []
    for caminho in arquivos:
        try:
            info = caminho.stat()
        except OSError:
            continue
        assinatura.append((str(caminho), info.st_mtime_ns, info.st_size))
    return tuple(assinatura)


def _importar_addons(pasta_addons: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Descobre e importa dinamicamente todos os arquivos .py de addons/ (ignorando iniciados por '_').
    Ordena pelo atributo ORDER (padrão 999) seguido de TITLE.
    Antes de importar, descarta os submódulos de `core` já carregados para que os add-ons
    sempre se liguem à versão mais recente de core/parsers.py e core/plotting.py.
    """
    modulos_carregados: List[Dict[str, Any]] = []
    erros_importacao: List[Dict[str, str]] = []

    if not pasta_addons.exists():
        return modulos_carregados, erros_importacao

    for nome_mod in [n for n in sys.modules if n == "core" or n.startswith("core.")]:
        del sys.modules[nome_mod]

    for caminho_py in _listar_arquivos_addons(pasta_addons):
        nome_modulo = f"addons.{caminho_py.stem}"
        try:
            spec = importlib.util.spec_from_file_location(nome_modulo, str(caminho_py))
            if spec is None or spec.loader is None:
                erros_importacao.append(
                    {
                        "arquivo": caminho_py.name,
                        "erro": "Não foi possível criar a especificação importlib do módulo.",
                    }
                )
                continue

            modulo = importlib.util.module_from_spec(spec)
            sys.modules[nome_modulo] = modulo
            spec.loader.exec_module(modulo)

            func_render = getattr(modulo, "render", None)
            if not callable(func_render):
                erros_importacao.append(
                    {
                        "arquivo": caminho_py.name,
                        "erro": "Função `render(cfg_plot)` ausente ou não chamável.",
                    }
                )
                continue

            titulo = str(getattr(modulo, "TITLE", caminho_py.stem)).strip() or caminho_py.stem
            icone = str(getattr(modulo, "ICON", "📊")).strip() or "📊"
            try:
                ordem = float(getattr(modulo, "ORDER", 999))
            except (TypeError, ValueError):
                ordem = 999.0

            modulos_carregados.append(
                {
                    "filename": caminho_py.name,
                    "stem": caminho_py.stem,
                    "title": titulo,
                    "icon": icone,
                    "order": ordem,
                    "module": modulo,
                }
            )
        except Exception:
            erros_importacao.append(
                {
                    "arquivo": caminho_py.name,
                    "erro": traceback.format_exc(),
                }
            )

    modulos_carregados.sort(key=lambda item: (item["order"], item["title"].lower(), item["filename"]))
    return modulos_carregados, erros_importacao


@st.cache_resource(show_spinner=False, max_entries=2)
def _carregar_addons_cacheado(
    caminho_addons: str,
    assinatura: Tuple[Tuple[str, int, int], ...],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Importa os add-ons uma única vez por 'assinatura' de mtime.
    Os dois argumentos entram na chave de cache; `assinatura` não é usada no corpo de propósito.
    """
    del assinatura
    return _importar_addons(Path(caminho_addons))


def carregar_addons(pasta_addons: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Retorna (módulos, erros) reaproveitando o cache enquanto nenhum arquivo for editado."""
    assinatura = _assinatura_projeto(pasta_addons, CORE_DIR)
    modulos, erros = _carregar_addons_cacheado(str(pasta_addons), assinatura)

    # O file watcher do Streamlit pode remover módulos de sys.modules; mantém o registro consistente
    # (necessário, por exemplo, para o unpickle de dataclasses retornadas por funções @st.cache_data).
    for item in modulos:
        sys.modules.setdefault(f"addons.{item['stem']}", item["module"])
    return modulos, erros


# ==============================================================================
# FRAGMENTOS POR ABA
# ==============================================================================
def _criar_fragmento_aba(info_addon: Dict[str, Any]) -> Callable[[Dict[str, Any]], None]:
    """
    Envolve `render(cfg_plot)` do add-on em um @st.fragment com nome exclusivo.
    Interações dentro da aba reexecutam apenas esse fragmento; alterações na barra lateral
    (que alteram cfg_plot) continuam disparando o rerun completo e re-registram o fragmento.
    """
    render_addon = info_addon["module"].render
    titulo = info_addon["title"]
    arquivo = info_addon["filename"]

    def _corpo(cfg_plot: Dict[str, Any]) -> None:
        try:
            render_addon(cfg_plot)
        except Exception as exc:
            st.error(
                f"Ocorreu um erro durante a execução do módulo **{titulo}** "
                f"(`{arquivo}`): {exc}"
            )
            with st.expander("🔍 Ver rastreamento técnico completo (Traceback)", expanded=False):
                st.code(traceback.format_exc(), language="python")

    # Nome/qualname únicos => ID de fragmento estável e distinto por add-on.
    sufixo = re.sub(r"\W", "_", info_addon["stem"])
    _corpo.__name__ = f"fragmento_{sufixo}"
    _corpo.__qualname__ = f"fragmento_{sufixo}"

    return _FRAGMENT(_corpo) if _FRAGMENT is not None else _corpo


def render_sidebar_global(
    addons_ativos: List[Dict[str, Any]],
    erros_addons: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Constrói a barra lateral global de padronização OriginPro, exportação CSV e instalador de Add-ons."""
    st.sidebar.title("⚙️ Padrão Gráfico (Origin)")

    with st.sidebar.expander("🖋️ Fontes, Eixos e Ticks", expanded=True):
        font_family_sel = st.selectbox(
            "Família da Fonte",
            options=[
                "sans-serif (Arial / Padrão Origin)",
                "serif (Times New Roman)",
                "monospace",
            ],
            index=0,
            key="glb_font_family",
        )
        font_family = font_family_sel.split(" ")[0]

        usar_virgula = st.radio(
            "Separador Decimal nos Gráficos",
            options=[
                "Vírgula (12,34 — Padrão BR/ABNT)",
                "Ponto (12.34 — Artigos Internacionais)",
            ],
            index=0,
            key="glb_usar_virgula",
        ).startswith("Vírgula")

        tick_dir_label = st.selectbox(
            "Direção dos Ticks",
            options=[
                "Para dentro (in — Padrão Origin)",
                "Para fora (out)",
                "Ambos (inout)",
            ],
            index=0,
            key="glb_tick_dir",
        )
        if "in —" in tick_dir_label:
            tick_dir = "in"
        elif "out)" in tick_dir_label:
            tick_dir = "out"
        else:
            tick_dir = "inout"

        ticks_top_right = st.checkbox("Mostrar ticks no topo e direita", value=True, key="glb_ticks_tr")
        minor_ticks = st.checkbox("Ativar sub-divisões (Minor Ticks)", value=True, key="glb_minor_ticks")
        grid_style = st.selectbox(
            "Grade de fundo (Grid)",
            options=[
                "Desativada (Padrão Origin)",
                "Tracejada suave",
                "Sólida fina",
            ],
            index=0,
            key="glb_grid_style",
        )

        c_f1, c_f2 = st.columns(2)
        font_labels = int(c_f1.number_input("Fonte Eixos", 8, 24, 12, key="glb_flab"))
        font_ticks = int(c_f2.number_input("Fonte Ticks", 7, 22, 10, key="glb_ftck"))

        c_f3, c_f4 = st.columns(2)
        font_legend = int(c_f3.number_input("Fonte Legenda", 7, 22, 10, key="glb_fleg"))
        spine_lw = float(c_f4.number_input("Espessura Borda", 0.5, 3.5, 1.1, step=0.1, key="glb_slw"))

    with st.sidebar.expander("🖼️ Dimensões, Imagem e CSV", expanded=True):
        c_w, c_h = st.columns(2)
        fig_w = float(c_w.number_input("Largura (pol)", 4.0, 18.0, 7.8, step=0.2, key="glb_figw"))
        fig_h = float(c_h.number_input("Altura (pol)", 3.0, 14.0, 5.5, step=0.2, key="glb_figh"))

        c_d, c_fmt = st.columns(2)
        export_dpi = int(c_d.selectbox("Resolução (DPI)", [300, 600, 150], index=0, key="glb_dpi"))
        export_fmt = str(c_fmt.selectbox("Formato Figura", ["PNG", "PDF", "SVG", "TIFF"], index=0, key="glb_fmt"))

        transparent_bg = st.checkbox("Fundo transparente na imagem", value=False, key="glb_transp")
        modo_csv = st.radio(
            "Padrão de Exportação CSV",
            options=[
                "Internacional ( , e . | UTF-8 )",
                "Excel Brasil ( ; e , | UTF-8 BOM )",
            ],
            index=0,
            key="glb_csv_br",
        )
        csv_br = modo_csv.startswith("Excel Brasil")

    with st.sidebar.expander("🧩 Instalar / Gerenciar Add-ons", expanded=False):
        st.caption(
            "Envie um arquivo `.py` seguindo o contrato de `addons/_template_addon.py` "
            "para adicioná-lo automaticamente como uma nova aba."
        )
        novos_arquivos = st.file_uploader(
            "Selecionar módulo(s) .py",
            type=["py"],
            accept_multiple_files=True,
            key="glb_addon_uploader",
        )
        sobrescrever = st.checkbox(
            "Permitir atualizar/sobrescrever add-on existente",
            value=True,
            key="glb_addon_overwrite",
        )

        if novos_arquivos:
            if st.button("📥 Validar e Instalar Add-on(s)", use_container_width=True, key="btn_instalar_addon"):
                instalados = []
                falhas = []
                for arq in novos_arquivos:
                    stem_limpo = re.sub(r"[^\w\-]", "_", Path(arq.name).stem).lstrip("_")
                    if not stem_limpo:
                        stem_limpo = "addon_custom"
                    nome_final = f"{stem_limpo}.py"
                    destino = ADDONS_DIR / nome_final

                    if destino.exists() and not sobrescrever:
                        falhas.append(f"`{nome_final}` já existe (marque a opção de sobrescrever).")
                        continue

                    valido, resultado = validar_codigo_addon(arq.getvalue(), nome_final)
                    if not valido:
                        falhas.append(resultado)
                        continue

                    destino.write_text(resultado, encoding="utf-8")
                    instalados.append(nome_final)

                if instalados:
                    st.session_state["msg_addon_sucesso"] = (
                        f"Add-on(s) instalado(s) com sucesso: {', '.join(instalados)}"
                    )
                if falhas:
                    st.session_state["msg_addon_erro"] = "\n\n".join(falhas)
                if instalados:
                    st.rerun()

        if "msg_addon_sucesso" in st.session_state:
            st.success(st.session_state.pop("msg_addon_sucesso"))
        if "msg_addon_erro" in st.session_state:
            st.error(st.session_state.pop("msg_addon_erro"))

        st.caption(
            "Recarga automática: os módulos só são reimportados quando algum arquivo de "
            "`addons/` ou `core/` é editado. Use o botão abaixo para forçar a recarga."
        )
        if st.button("🔄 Recarregar Pasta addons/", use_container_width=True, key="btn_reload_addons"):
            _carregar_addons_cacheado.clear()
            st.rerun()

        if addons_ativos:
            st.markdown("**Módulos ativos (por `ORDER`):**")
            for a in addons_ativos:
                st.caption(f"• `{int(a['order']) if a['order'].is_integer() else a['order']}` — {a['icon']} **{a['title']}** (`{a['filename']}`)")

        if erros_addons:
            st.warning(f"{len(erros_addons)} módulo(s) com falha de importação:")
            for err in erros_addons:
                st.caption(f"❌ `{err['arquivo']}`")
                st.code(err["erro"], language="text")

    return {
        "font_family": font_family,
        "usar_virgula": usar_virgula,
        "tick_dir": tick_dir,
        "ticks_top_right": ticks_top_right,
        "minor_ticks": minor_ticks,
        "grid_style": grid_style,
        "font_labels": font_labels,
        "font_ticks": font_ticks,
        "font_legend": font_legend,
        "spine_lw": spine_lw,
        "fig_w": fig_w,
        "fig_h": fig_h,
        "export_dpi": export_dpi,
        "export_fmt": export_fmt,
        "transparent_bg": transparent_bg,
        "csv_br": csv_br,
    }


def main() -> None:
    addons_ativos, erros_addons = carregar_addons(ADDONS_DIR)
    cfg_plot = render_sidebar_global(addons_ativos, erros_addons)

    st.title("⚡ Central Analítica Modular — Filmes Finos & Preditor ZView")
    st.caption(
        "Arquitetura extensível de Add-ons para Keithley (I×V com Histerese e Sensoriamento I×t), "
        "Solartron (EIS: Bode, Nyquist e Circuito Equivalente ZView) e Espectroscopia UV-Vis."
    )

    if not addons_ativos:
        st.info(
            "Nenhum Add-on ativo encontrado na pasta `addons/`. "
            "Adicione seus arquivos `addon_*.py` na pasta `addons/` ou instale-os pela barra lateral esquerda."
        )
        if erros_addons:
            st.error("Foram detectados erros ao carregar arquivos presentes em `addons/`:")
            for err in erros_addons:
                with st.expander(f"Ver detalhes de erro em `{err['arquivo']}`", expanded=True):
                    st.code(err["erro"], language="python")
        return

    rotulos_abas = [f"{a['icon']} {a['title']}" for a in addons_ativos]
    abas_ui = st.tabs(rotulos_abas)

    for aba_container, info_addon in zip(abas_ui, addons_ativos):
        with aba_container:
            # Cada aba roda em seu próprio fragmento: interações internas não reexecutam as outras abas.
            _criar_fragmento_aba(info_addon)(cfg_plot)


if __name__ == "__main__":
    main()
