# Suíte Analítica Modular (`lab-suite`)

Aplicação modular em Streamlit para análise, visualização e exportação de dados experimentais em Ciência de Materiais, com arquitetura desacoplada entre núcleo (`core/`) e extensões (`addons/`).

## Estrutura do Projeto

- `app.py`: aplicação principal em Streamlit, carregamento dinâmico de *add-ons* com cache por `mtime`, ordenação por `ORDER`, isolamento de abas via `@st.fragment`, barra lateral global de padronização gráfica e instalador de módulos `.py`.
- `core/`: motor compartilhado da aplicação (leitura/parsing, plotagem e utilitários comuns).
- `addons/`: módulos de análise e visualização carregados dinamicamente como abas.
- `addons/_template_addon.py`: contrato base com metadados (`TITLE`, `ICON`, `ORDER`) e a função obrigatória `render(cfg_plot)`.
- `requirements.txt`: dependências Python do ambiente.
- `Instalar_Suite.bat`: script de instalação automática para Windows (verifica/oferece instalação do Python via `winget`, cria a `.venv` e instala o `requirements.txt`).
- `Iniciar_Suite.bat` e `Parar_Suite.bat`: scripts auxiliares para iniciar e encerrar a suíte no Windows.

## Como Instalar e Rodar

1. Clona ou baixa o repositório:
   ```bash
   git clone https://github.com/MKDV13/lab-suite.git
   cd lab-suite
   ```
2. Prepara o ambiente e instala as dependências:
   - **Windows (forma mais simples):** executa `Instalar_Suite.bat` (ele cuida da verificação do Python, cria a `.venv` e instala os pacotes de `requirements.txt`).
   - **Manual (Windows / Linux / macOS):** cria e ativa um ambiente virtual e instala as dependências:
     ```bash
     python -m venv .venv
     # Windows:
     .venv\Scripts\activate
     # Linux/macOS:
     source .venv/bin/activate

     pip install -r requirements.txt
     ```
3. Inicia a aplicação:
   - **Windows:** executa `Iniciar_Suite.bat` (ou corre `streamlit run app.py`).
   - **Outros sistemas:** executa `streamlit run app.py`.

## Como Criar ou Instalar um Add-on

1. Usa `addons/_template_addon.py` como base para o novo módulo.
2. Define `TITLE`, `ICON`, `ORDER` e implementa `render(cfg_plot)`.
3. Coloca o arquivo `.py` em `addons/` ou envia-o diretamente pela opção **Instalar / Gerenciar Add-ons** na barra lateral da aplicação.