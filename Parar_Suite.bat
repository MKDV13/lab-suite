@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Parar - Suite Analitica Modular

cd /d "%~dp0"

rem ============================================================================
rem  Parar_Suite.bat
rem  Encerra o servidor Streamlit iniciado pelo Iniciar_Suite.bat, junto com os
rem  processos filhos. Nao mata processos aleatorios. So sao encerrados:
rem   1. o PID registrado em suite.pid pelo Iniciar_Suite.bat, desde que a linha
rem      de comando desse processo ainda seja "streamlit ... app.py";
rem   2. se nao houver suite.pid valido, o processo que escuta na porta da Suite
rem      e cuja linha de comando e "streamlit ... app.py", mas somente depois
rem      de mostrar o PID e pedir confirmacao.
rem  Qualquer outro processo, mesmo na mesma porta, e deixado em paz.
rem
rem  A logica fica no bloco PowerShell no final deste arquivo, depois do ultimo
rem  "exit /b". O cmd nunca le esse bloco, apenas o PowerShell.
rem ============================================================================

set "SUITE_ROOT=%~dp0"
set "SUITE_PORT=8501"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$t = Get-Content -Raw -LiteralPath '%~f0'; $m = '#SUITEPS' + '_BEGIN'; Invoke-Expression $t.Substring($t.IndexOf($m))"
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%

rem ----------------------------------------------------------------------------
rem  Nada abaixo desta linha e executado pelo cmd.
rem ----------------------------------------------------------------------------
exit /b 0

#SUITEPS_BEGIN
# =============================================================================
# Bloco PowerShell do Parar_Suite.bat. Texto em ASCII de proposito.
# Variaveis recebidas do .bat: SUITE_ROOT, SUITE_PORT (padrao; suite.pid pode
# sobrescrever a porta com a que foi realmente usada pelo Iniciar_Suite.bat).
# Codigos de saida: 0 encerrado ou nada a encerrar | 1 falha ao encerrar
#                   2 usuario nao confirmou | 99 erro inesperado
# =============================================================================
$ErrorActionPreference = 'Continue'

$root = $env:SUITE_ROOT
if (-not $root.EndsWith('\')) { $root = $root + '\' }
$porta  = [int]$env:SUITE_PORT
$log    = $root + 'suite.log'
$pidArq = $root + 'suite.pid'

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

function Get-Resumo($info) {
    $cl = [string]$info.CommandLine
    if ($cl.Length -gt 120) { $cl = $cl.Substring(0, 120) + '...' }
    return ('PID ' + $info.ProcessId + ' [' + $info.Name + '] ' + $cl)
}

try {
    Write-Host '=== Parar Suite Analitica Modular ===' -ForegroundColor Cyan
    $alvos = @()

    # Criterio 1: PID registrado pelo Iniciar_Suite.bat (e a porta usada nele)
    if (Test-Path -LiteralPath $pidArq) {
        $linhas = @(Get-Content -LiteralPath $pidArq -ErrorAction SilentlyContinue)
        if (($linhas.Count -ge 2) -and ($linhas[1] -match '^\d+$')) { $porta = [int]$linhas[1] }
        if (($linhas.Count -ge 1) -and ($linhas[0] -match '^\d+$')) {
            $info = Get-ProcInfo ([int]$linhas[0])
            if (Test-ProcSuite $info) {
                $alvos += [int]$linhas[0]
                Write-Host ('Processo registrado pelo launcher: ' + (Get-Resumo $info))
            }
            else {
                Write-Host 'O PID registrado em suite.pid ja nao pertence a um Streamlit da Suite (ignorado).' -ForegroundColor Yellow
            }
        }
    }

    # Criterio 2 (reserva): quem escuta na porta E e "streamlit ... app.py", com confirmacao
    $donos = @(Get-DonosDaPorta $porta)
    $donosSuite = @($donos | Where-Object { Test-ProcSuite $_ })

    if ($alvos.Count -eq 0) {
        if ($donosSuite.Count -gt 0) {
            Write-Host ('Sem suite.pid valido, mas ha Streamlit escutando na porta ' + $porta + ':') -ForegroundColor Yellow
            foreach ($d in $donosSuite) { Write-Host ('   ' + (Get-Resumo $d)) }
            $resp = Read-Host 'Encerrar este(s) processo(s)? (S/N)'
            if ($resp -notmatch '^[sSyY]') {
                Write-Host 'Nada foi encerrado.'
                Write-SuiteLog 'Parar_Suite: usuario nao confirmou o encerramento por porta. Nada encerrado.'
                exit 2
            }
            foreach ($d in $donosSuite) { $alvos += [int]$d.ProcessId }
        }
        elseif ($donos.Count -gt 0) {
            Write-Host ('A porta ' + $porta + ' esta em uso por: ' + (Get-Resumo $donos[0])) -ForegroundColor Yellow
            Write-Host 'Esse processo nao parece ser o Streamlit da Suite. Nada foi encerrado.' -ForegroundColor Yellow
            Write-SuiteLog ('Parar_Suite: porta ' + $porta + ' ocupada por processo que nao e da Suite. Nada encerrado.')
            exit 0
        }
        else {
            Write-Host ('Nenhum servidor da Suite em execucao (porta ' + $porta + ' livre).') -ForegroundColor Green
            if (Test-Path -LiteralPath $pidArq) { Remove-Item -LiteralPath $pidArq -Force -ErrorAction SilentlyContinue }
            Write-SuiteLog 'Parar_Suite: nenhum servidor da Suite em execucao.'
            exit 0
        }
    }

    # Encerramento: arvore completa (cmd -> python -> filhos)
    $falhou = $false
    foreach ($id in ($alvos | Select-Object -Unique)) {
        & taskkill.exe /PID $id /T /F *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host ('   Encerrado: PID ' + $id + ' e processos filhos.') -ForegroundColor Green
            Write-SuiteLog ('Parar_Suite: encerrado PID ' + $id + ' (arvore completa).')
        }
        else {
            Write-Host ('   [ERRO] Nao foi possivel encerrar o PID ' + $id + ' (codigo ' + $LASTEXITCODE + ').') -ForegroundColor Red
            Write-SuiteLog ('Parar_Suite: FALHA ao encerrar PID ' + $id + ' (codigo ' + $LASTEXITCODE + ').')
            $falhou = $true
        }
    }

    # Confirmar que a porta foi liberada (ate 10 s)
    for ($i = 0; $i -lt 20; $i++) {
        if (-not (Test-Porta $porta)) { break }
        Start-Sleep -Milliseconds 500
    }
    if (Test-Path -LiteralPath $pidArq) { Remove-Item -LiteralPath $pidArq -Force -ErrorAction SilentlyContinue }

    if (Test-Porta $porta) {
        Write-Host ('[ATENCAO] A porta ' + $porta + ' continua ocupada por outro processo.') -ForegroundColor Yellow
        Write-SuiteLog ('Parar_Suite: porta ' + $porta + ' continua ocupada apos o encerramento.')
        exit 1
    }
    if ($falhou) { exit 1 }

    Write-Host ('Servidor encerrado. Porta ' + $porta + ' livre.') -ForegroundColor Green
    Write-Host 'Se a janela do aplicativo ainda estiver aberta, basta fecha-la.'
    exit 0
}
catch {
    Write-SuiteLog ('Parar_Suite: ERRO INESPERADO - ' + $_.Exception.Message)
    Write-Host ('[ERRO] Falha inesperada: ' + $_.Exception.Message) -ForegroundColor Red
    exit 99
}
