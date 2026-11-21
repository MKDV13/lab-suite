# Suíte Analítica Modular (Streamlit)

Aplicação modular em Streamlit para análise e visualização de dados experimentais de laboratório (curvas I×V, corrente × tempo I×t, espectroscopia de impedância eletroquímica EIS, modelagem/exportação compatível com ZView e espectroscopia UV-Vis), com padronização visual estilo Origin e suporte a add-ons isolados por fragmento[cite: 2, 3, 5, 17, 23, 27].

## Estrutura do projeto

- `app.py`: núcleo da aplicação, carregamento dinâmico de add-ons com cache baseado em assinatura/`mtime`, validação AST e isolamento de abas via `@st.fragment`[cite: 2, 4, 17, 24].
- `core/`: utilitários centrais de leitura/inferência de arquivos científicos (`parsers.py`), formatação e plotagem (`plotting.py`), interface compartilhada (`ui.py`) e consolidação de resultados (`resultados.py`)[cite: 2, 7, 10, 12].
- `addons/`: módulos funcionais por técnica experimental (Keithley I×V e I×t, Solartron EIS, preditor/auxiliar ZView e Cary UV-Vis), além do `_template_addon.py` como modelo base para novos módulos[cite: 2, 6, 14, 23, 27].
- `Iniciar_Suite.bat` / `Parar_Suite.bat`: scripts de inicialização e encerramento no Windows (com detecção de ambiente virtual local e verificação de porta)[cite: 25].
- `requirements.txt`: dependências Python do projeto[cite: 5, 36].

## Como executar

1. Crie/ative o ambiente virtual (ex.: `.venv`) e instale as dependências[cite: 25, 31]:
   ```bash
   pip install -r requirements.txt