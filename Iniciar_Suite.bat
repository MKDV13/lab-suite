@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Inicializador - Suite Analitica Modular

cd /d "%~dp0"

rem ============================================================================
rem  Iniciar_Suite.bat
rem   - Detecta o Python: .venv, venv, Python global ou launcher "py".
rem   - Confere o requirements.txt e oferece instalar o que estiver faltando.
rem   - Inicia o Streamlit em segundo plano; TODA a saida vai para suite.log.
rem   - Abre a interface em janela de aplicativo Chrome ou Edge, ou no navegador.
rem  Para encerrar o servidor use o Parar_Suite.bat.
rem
rem  A logica detalhada fica no bloco PowerShell no final deste arquivo, depois
rem  do ultimo "exit /b". O cmd nunca le esse bloco, apenas o PowerShell.
rem ============================================================================

set "PORT=8501"
set "APP_URL=http://localhost:%PORT%"
set "LOG_FILE=%~dp0suite.log"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%~dp0app.py" goto :sem_app

rem ---- Localizar o Python ----------------------------------------------------
set "PY_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PY_EXE if exist "%~dp0venv\Scripts\python.exe" set "PY_EXE=%~dp0venv\Scripts\python.exe"
if not defined PY_EXE call :achar_python_global
if not defined PY_EXE call :achar_py_launcher
if not defined PY_EXE goto :sem_python

"%PY_EXE%" --version >nul 2>&1
if errorlevel 1 goto :python_quebrado

rem ---- Delegar ao bloco PowerShell: log, dependencias e inicio do servidor ---
set "SUITE_ROOT=%~dp0"
set "SUITE_PY=%PY_EXE%"
set "SUITE_PORT=%PORT%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$t = Get-Content -Raw -LiteralPath '%~f0'; $m = '#SUITEPS' + '_BEGIN'; Invoke-Expression $t.Substring($t.IndexOf($m))"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :falhou

rem ---- Abrir a interface -----------------------------------------------------
set "BROWSER_EXE="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "BROWSER_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER_EXE if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "BROWSER_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER_EXE if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "BROWSER_EXE=%LocalAppData%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER_EXE if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "BROWSER_EXE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER_EXE if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "BROWSER_EXE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER_EXE if exist "%LocalAppData%\Microsoft\Edge\Application\msedge.exe" set "BROWSER_EXE=%LocalAppData%\Microsoft\Edge\Application\msedge.exe"

if defined BROWSER_EXE goto :abrir_app
start "" "%APP_URL%"
exit /b 0

:abrir_app
start "" "%BROWSER_EXE%" --app=%APP_URL%
exit /b 0

rem ============================================================================
rem  Sub-rotinas de deteccao do Python
rem ============================================================================
:achar_python_global
python --version >nul 2>&1
if not errorlevel 1 set "PY_EXE=python"
exit /b 0

:achar_py_launcher
for /f "usebackq delims=" %%P in (`py -3 -c "import sys;print(sys.executable)" 2^>nul`) do set "PY_EXE=%%P"
exit /b 0

rem ============================================================================
rem  Tratamento de erros do launcher
rem ============================================================================
:sem_app
echo.
echo [ERRO] O arquivo "app.py" nao foi encontrado em:
echo        %~dp0
call :log_bat "ERRO: app.py nao encontrado na pasta do launcher"
echo.
pause
exit /b 1

:sem_python
echo.
echo [ERRO] Python nao encontrado.
echo        Instale o Python 3.9 ou superior e marque "Add python.exe to PATH",
echo        ou crie um ambiente virtual na pasta do projeto: python -m venv .venv
call :log_bat "ERRO: Python nao encontrado - nem .venv, nem venv, nem PATH, nem py launcher"
echo.
pause
exit /b 1

:python_quebrado
echo.
echo [ERRO] O Python selecionado nao executa:
echo        %PY_EXE%
echo        Se for o ambiente .venv ou venv, apague a pasta e recrie com: python -m venv .venv
call :log_bat "ERRO: o Python selecionado nao executa - %PY_EXE%"
echo.
pause
exit /b 1

:falhou
echo.
echo ============================================================================
echo  A Suite NAO foi iniciada. Detalhes registrados em:
echo    %LOG_FILE%
echo ============================================================================
echo.
pause
exit /b %RC%

:log_bat
>> "%LOG_FILE%" echo [%DATE% %TIME:~0,8%] %~1
exit /b 0

rem ----------------------------------------------------------------------------
rem  Nada abaixo desta linha e executado pelo cmd (todos os caminhos acima
rem  terminam em "exit /b").
rem ----------------------------------------------------------------------------
exit /b 0

#SUITEPS_BEGIN
# =============================================================================
# Bloco PowerShell do Iniciar_Suite.bat. Texto em ASCII de proposito.
# Variaveis recebidas do .bat: SUITE_ROOT, SUITE_PY, SUITE_PORT.
# Codigos de saida: 0 ok | 2 porta ocupada | 3 Streamlit falhou ao subir
#                   4 Streamlit nao respondeu a tempo | 5 dependencias pendentes
#                   99 erro inesperado do launcher
# =============================================================================
$ErrorActionPreference = 'Continue'

$root = $env:SUITE_ROOT
if (-not $root.EndsWith('\')) { $root = $root + '\' }
$py        = $env:SUITE_PY
$porta     = [int]$env:SUITE_PORT
$log       = $root + 'suite.log'
$logAntigo = $root + 'suite.old.log'
$pidArq    = $root + 'suite.pid'
$reqArq    = $root + 'requirements.txt'
$chkArq    = Join-Path $env:TEMP 'suite_checar_deps.py'
$limiteLog = 1MB
$esperaMax = 90      # tentativas de 500 ms (45 s) aguardando a porta abrir

# ---------------------------------------------------------------- utilitarios
function Write-SuiteLog([string]$texto) {
    $linha = '[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] ' + $texto
    try { Add-Content -LiteralPath $log -Value $linha -Encoding UTF8 -ErrorAction Stop } catch { }
}

function Test-Porta([int]$p) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', $p); return $true }
    catch { return $false }
    finally { $c.Close() }
}

function Get-ProcInfo([int]$procId) {
    return Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId = ' + $procId) -ErrorAction SilentlyContinue
}

# Processo da Suite = linha de comando com "streamlit" e "app.py"
function Test-ProcSuite($info) {
    if ($null -eq $info) { return $false }
    $cl = [string]$info.CommandLine
    return (($cl -match 'streamlit') -and ($cl -match 'app\.py'))
}

function Get-DonosDaPorta([int]$p) {
    $lista = @()
    $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $i = Get-ProcInfo ([int]$c.OwningProcess)
        if ($null -ne $i) { $lista += $i }
    }
    return $lista
}

function Get-FimDoLog([int]$n) {
    return @(Get-Content -LiteralPath $log -Tail $n -Encoding UTF8 -ErrorAction SilentlyContinue)
}

function Show-FimDoLog {
    Write-Host ''
    Write-Host '--- ultimas linhas de suite.log ---' -ForegroundColor DarkYellow
    foreach ($l in (Get-FimDoLog 18)) { Write-Host ('  ' + $l) }
    Write-Host '-----------------------------------' -ForegroundColor DarkYellow
}

function Show-DicaPorErro {
    $texto = (Get-FimDoLog 60) -join "`n"
    if ($texto -match 'ModuleNotFoundError|No module named') {
        Write-Host 'Dica: falta algum pacote. Rode: python -m pip install -r requirements.txt' -ForegroundColor Yellow
    }
    if ($texto -match 'already in use|Address already|10048') {
        Write-Host ('Dica: a porta ' + $porta + ' esta ocupada. Rode o Parar_Suite.bat ou mude PORT no Iniciar_Suite.bat.') -ForegroundColor Yellow
    }
}

# ------------------------------------------------- conferencia do requirements
$codigoChecagem = @'
import re
import sys
from importlib import metadata as md


def como_tupla(texto):
    nums = []
    for parte in re.split(r'[.+\-]', texto):
        m = re.match(r'\d+', parte)
        if not m:
            break
        nums.append(int(m.group()))
    return tuple(nums)


def satisfaz(instalada, op, exigida):
    a = como_tupla(instalada)
    b = como_tupla(exigida)
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    if op in ('>=', '~='):
        return a >= b
    if op == '==':
        return a == b
    return True


problemas = 0
with open(sys.argv[1], encoding='utf-8-sig') as arq:
    for bruta in arq:
        linha = bruta.split('#', 1)[0].strip()
        if not linha or linha.startswith('-'):
            continue
        m = re.match(r'([A-Za-z0-9][A-Za-z0-9._\-]*)(?:\[[^\]]*\])?\s*(?:(>=|==|~=)\s*([0-9][0-9A-Za-z.]*))?', linha)
        if not m:
            continue
        nome, op, exigida = m.groups()
        try:
            instalada = md.version(nome)
        except md.PackageNotFoundError:
            print('FALTA ' + nome + ' (nao instalado)')
            problemas += 1
            continue
        if op and not satisfaz(instalada, op, exigida):
            print('VERSAO ' + nome + ' instalada=' + instalada + ' exigida=' + op + exigida)
            problemas += 1
        else:
            print('OK ' + nome + ' ' + instalada)
sys.exit(1 if problemas else 0)
'@

function Invoke-Checagem {
    Set-Content -LiteralPath $chkArq -Value $codigoChecagem -Encoding ASCII
    $script:depLinhas = @(& $py $chkArq $reqArq 2>&1 | ForEach-Object { [string]$_ })
    $script:depRc = $LASTEXITCODE
}

function Get-LinhasOk      { return @($script:depLinhas | Where-Object { $_ -match '^OK ' } | ForEach-Object { $_.Substring(3) }) }
function Get-LinhasProblema { return @($script:depLinhas | Where-Object { $_.Trim() -and ($_ -notmatch '^OK ') }) }

function Install-Dependencias {
    $outTmp = Join-Path $env:TEMP 'suite_pip_out.txt'
    $errTmp = Join-Path $env:TEMP 'suite_pip_err.txt'
    $argsPip = @('-m', 'pip', 'install', '-r', ('"' + $reqArq + '"'), '--disable-pip-version-check')
    Write-SuiteLog 'Executando: pip install -r requirements.txt'
    $p = Start-Process -FilePath $py -ArgumentList $argsPip -WindowStyle Hidden -PassThru -RedirectStandardOutput $outTmp -RedirectStandardError $errTmp -ErrorAction Stop
    $null = $p.Handle
    Write-Host -NoNewline 'Instalando dependencias, pode levar alguns minutos '
    while (-not $p.HasExited) { Write-Host -NoNewline '.'; Start-Sleep -Seconds 2 }
    Write-Host ''
    $p.WaitForExit()
    $script:pipRc = $p.ExitCode
    Get-Content -LiteralPath $outTmp -Encoding UTF8 -ErrorAction SilentlyContinue | Add-Content -LiteralPath $log -Encoding UTF8
    Get-Content -LiteralPath $errTmp -Encoding UTF8 -ErrorAction SilentlyContinue | Add-Content -LiteralPath $log -Encoding UTF8
    Write-SuiteLog ('pip terminou com codigo ' + $script:pipRc)
}

# --------------------------------------------------------------------- fluxo
try {
    # 1) Rotacao: acima de 1 MB o log atual vira suite.old.log (um arquivo antigo)
    $rotacionou = $false
    if ((Test-Path -LiteralPath $log) -and ((Get-Item -LiteralPath $log).Length -gt $limiteLog)) {
        Move-Item -LiteralPath $log -Destination $logAntigo -Force
        $rotacionou = $true
    }

    # 2) Servidor ja ativo? Apenas reabre a interface.
    if (Test-Porta $porta) {
        $donos   = @(Get-DonosDaPorta $porta)
        $daSuite = @($donos | Where-Object { Test-ProcSuite $_ })
        if ($daSuite.Count -gt 0) {
            Write-SuiteLog ('Servidor ja ativo na porta ' + $porta + ' (PID ' + $daSuite[0].ProcessId + '). Interface reaberta.')
            Write-Host ('Servidor ja em execucao em http://localhost:' + $porta + '. Abrindo interface...')
            exit 0
        }
        Write-SuiteLog ('ERRO: porta ' + $porta + ' ocupada por processo que nao parece ser o Streamlit da Suite.')
        Write-Host ('[ERRO] A porta ' + $porta + ' ja esta em uso por um programa que nao parece ser o Streamlit da Suite.') -ForegroundColor Red
        Write-Host '       Feche esse programa ou altere o valor de PORT no inicio do Iniciar_Suite.bat.' -ForegroundColor Red
        exit 2
    }

    # 3) Cabecalho da sessao
    $versaoPy = ((& $py --version 2>&1) | Out-String).Trim()
    Add-Content -LiteralPath $log -Value @('', ('=' * 78)) -Encoding UTF8
    Write-SuiteLog 'NOVA SESSAO - Iniciar_Suite.bat'
    Write-SuiteLog ('Projeto : ' + $root)
    Write-SuiteLog ('Python  : ' + $py + '  [' + $versaoPy + ']')
    Write-SuiteLog ('Porta   : ' + $porta)
    if ($rotacionou) { Write-SuiteLog 'Log anterior passou de 1 MB e foi arquivado em suite.old.log' }

    # 4) Dependencias
    if (Test-Path -LiteralPath $reqArq) {
        Write-Host 'Verificando dependencias do requirements.txt...'
        Invoke-Checagem
        if ($script:depRc -ne 0) {
            $probs = Get-LinhasProblema
            Write-Host '[ATENCAO] Dependencias pendentes:' -ForegroundColor Yellow
            Write-SuiteLog 'DEPENDENCIAS PENDENTES:'
            foreach ($l in $probs) { Write-Host ('   - ' + $l) -ForegroundColor Yellow; Write-SuiteLog ('   - ' + $l) }
            $resp = Read-Host 'Instalar/atualizar agora com pip? (S/N)'
            if ($resp -notmatch '^[sSyY]') {
                Write-SuiteLog 'Usuario recusou a instalacao das dependencias. Inicializacao cancelada.'
                Write-Host 'Inicializacao cancelada. Instale com: python -m pip install -r requirements.txt' -ForegroundColor Red
                exit 5
            }
            Install-Dependencias
            if ($script:pipRc -ne 0) {
                Write-Host '[ERRO] O pip falhou ao instalar as dependencias.' -ForegroundColor Red
                Show-FimDoLog
                exit 5
            }
            Invoke-Checagem
            if ($script:depRc -ne 0) {
                Write-SuiteLog 'ERRO: dependencias ainda pendentes apos a instalacao.'
                Write-Host '[ERRO] Ainda ha dependencias pendentes apos a instalacao:' -ForegroundColor Red
                foreach ($l in (Get-LinhasProblema)) { Write-Host ('   - ' + $l) -ForegroundColor Red; Write-SuiteLog ('   - ' + $l) }
                exit 5
            }
        }
        Write-SuiteLog ('Dependencias OK: ' + ((Get-LinhasOk) -join '; '))
    }
    else {
        Write-SuiteLog 'AVISO: requirements.txt nao encontrado; checagem de dependencias ignorada.'
        Write-Host '[AVISO] requirements.txt nao encontrado; checagem de dependencias ignorada.' -ForegroundColor Yellow
    }

    # 5) Iniciar o Streamlit em segundo plano; saida e erros vao para suite.log
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $q = '"'
    $linhaCmd = '/c ' + $q + $q + $py + $q + ' -m streamlit run app.py --server.port=' + $porta + ' --server.headless=true --browser.gatherUsageStats=false >> ' + $q + $log + $q + ' 2>&1' + $q
    Write-SuiteLog 'Iniciando Streamlit em segundo plano...'
    Write-Host ('Iniciando servidor Streamlit na porta ' + $porta + '...')
    $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList $linhaCmd -WorkingDirectory $root -WindowStyle Hidden -PassThru -ErrorAction Stop

    # 6) Aguardar a porta abrir (ou o processo morrer)
    $pronto = $false
    for ($i = 0; $i -lt $esperaMax; $i++) {
        if ($proc.HasExited) { break }
        if (Test-Porta $porta) { $pronto = $true; break }
        Start-Sleep -Milliseconds 500
    }

    if ($pronto) {
        Set-Content -LiteralPath $pidArq -Value @([string]$proc.Id, [string]$porta) -Encoding ASCII
        Write-SuiteLog ('Servidor ativo em http://localhost:' + $porta + ' (PID do processo-pai ' + $proc.Id + ').')
        Write-Host 'Servidor ativo. Abrindo interface...' -ForegroundColor Green
        exit 0
    }

    if ($proc.HasExited) {
        Write-SuiteLog ('FALHA: o Streamlit encerrou durante a inicializacao (codigo ' + $proc.ExitCode + ').')
        Write-Host '[ERRO] O Streamlit encerrou durante a inicializacao.' -ForegroundColor Red
        Show-FimDoLog
        Show-DicaPorErro
        exit 3
    }

    # Processo vivo, mas a porta nao abriu no prazo: encerra a arvore para nao deixar orfao
    $idPai = $proc.Id
    & taskkill.exe /PID $idPai /T /F *> $null
    Write-SuiteLog ('FALHA: o Streamlit nao respondeu na porta ' + $porta + ' em ' + ($esperaMax / 2) + ' s. Processo encerrado.')
    Write-Host ('[ERRO] O Streamlit nao respondeu na porta ' + $porta + ' em ' + ($esperaMax / 2) + ' segundos.') -ForegroundColor Red
    Show-FimDoLog
    Show-DicaPorErro
    exit 4
}
catch {
    Write-SuiteLog ('ERRO INESPERADO no launcher: ' + $_.Exception.Message)
    Write-Host ('[ERRO] Falha inesperada no launcher: ' + $_.Exception.Message) -ForegroundColor Red
    exit 99
}
