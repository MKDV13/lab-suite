@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

cd /d "%~dp0"
title Instalador - lab-suite

echo ========================================================
echo          INSTALADOR DO LAB-SUITE (WINDOWS)
echo ========================================================
echo.

set "PYTHON_CMD="

where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :tem_python
)

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :tem_python
)

echo [AVISO] Python nao encontrado neste computador.
echo.

where winget >nul 2>nul
if errorlevel 1 goto :sem_winget

set /p "INSTALAR_PY=Deseja tentar instalar o Python 3.12 agora via winget? (S/N): "
if /i not "!INSTALAR_PY!"=="S" goto :sem_winget

echo.
echo A instalar o Python 3.12...
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
echo.
echo [IMPORTANTE] Se o Python acabou de ser instalado, feche esta janela,
echo abra um novo terminal/CMD e execute Instalar_Suite.bat novamente
echo para o PATH do sistema atualizar.
pause
exit /b 0

:sem_winget
echo.
echo Instale o Python 3 marcando "Add python.exe to PATH" e execute este ficheiro de novo.
pause
exit /b 1

:tem_python
echo [OK] A usar interpretador: %PYTHON_CMD%
echo.

if exist ".venv\Scripts\python.exe" (
    echo [OK] Ambiente virtual .venv ja existe.
    goto :instalar_deps
)

echo A criar ambiente virtual em .venv...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo [ERRO] Falha ao criar o ambiente virtual .venv.
    pause
    exit /b 1
)

:instalar_deps
echo.
echo A atualizar o pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

if not exist "requirements.txt" (
    echo [AVISO] Ficheiro requirements.txt nao encontrado na pasta.
    goto :fim
)

echo.
echo A instalar dependencias de requirements.txt...
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Houve falha na instalacao de alguma dependencia.
    pause
    exit /b 1
)

:fim
echo.
echo ========================================================
echo       INSTALACAO CONCLUIDA COM SUCESSO!
echo ========================================================
echo.
set /p "ABRIR_AGORA=Deseja iniciar o lab-suite agora? (S/N): "
if /i not "%ABRIR_AGORA%"=="S" goto :encerrar

if exist "Iniciar_Suite.bat" (
    call "Iniciar_Suite.bat"
) else (
    ".venv\Scripts\streamlit.exe" run app.py
)

:encerrar
endlocal
exit /b 0