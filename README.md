# Suíte Analítica Modular (`lab-suite`)

Aplicação modular em Streamlit para análise, visualização e exportação de dados experimentais em Ciência de Materiais, com arquitetura desacoplada entre núcleo (`core/`) e extensões (`addons/`).

## Estrutura do Projeto

- `app.py`: aplicação principal em Streamlit, carregamento dinâmico de *add-ons* com cache por `mtime`, ordenação por `ORDER`, isolamento de abas via `@st.fragment`, barra lateral global de padronização gráfica e instalador de módulos `.py`.
- `core/`: motor compartilhado da aplicação (leitura/parsing, plotagem e utilitários comuns).
- `addons/`: módulos de análise e visualização carregados dinamicamente como abas.
- `addons/_template_addon.py`: contrato base com metadados (`TITLE`, `ICON`, `ORDER`) e a função obrigatória `render(cfg_plot)`.
- `requirements.txt`: dependências Python do ambiente.
- `Iniciar_Suite.bat` e `Parar_Suite.bat`: scripts auxiliares para iniciar e encerrar a suíte no Windows.

## Como Instalar e Rodar

1. Clona ou baixa o repositório:
   ```bash
   git clone [https://github.com/MKDV13/lab-suite.git](https://github.com/MKDV13/lab-suite.git)
   cd lab-suite