# TruthLens AI - All-in-One Startup Script (PowerShell)
# Usage:
#   .\run.ps1             - Starts both Backend (FastAPI) and Frontend (React/Vite)
#   .\run.ps1 -Backend    - Starts only Backend (http://localhost:8000)
#   .\run.ps1 -Frontend   - Starts only Frontend (http://localhost:5173)
#   .\run.ps1 -Streamlit  - Starts Streamlit Web App (http://localhost:8501)
#   .\run.ps1 -Test       - Runs complete pytest test suite

param(
    [switch]$Backend,
    [switch]$Frontend,
    [switch]$Streamlit,
    [switch]$Test
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Prefer repository virtual environment if available
$PythonExe = if (Test-Path "$RootDir\venv\Scripts\python.exe") { "$RootDir\venv\Scripts\python.exe" } else { "python" }
$StreamlitExe = if (Test-Path "$RootDir\venv\Scripts\streamlit.exe") { "$RootDir\venv\Scripts\streamlit.exe" } else { "streamlit" }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TruthLens AI - Multimodal Verification Platform" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

if ($Test) {
    Write-Host "[*] Executing test suite with $PythonExe..." -ForegroundColor Yellow
    Set-Location $RootDir
    & $PythonExe -m pytest -v
    exit $LASTEXITCODE
}

if ($Streamlit) {
    Write-Host "[*] Starting Streamlit Application on http://localhost:8501..." -ForegroundColor Green
    Set-Location $RootDir
    & $StreamlitExe run app.py
    exit 0
}

if ($Backend) {
    Write-Host "[*] Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
    Set-Location $RootDir
    & $PythonExe server.py
    exit 0
}

if ($Frontend) {
    Write-Host "[*] Starting React Frontend on http://localhost:5173..." -ForegroundColor Green
    Set-Location "$RootDir\frontend"
    npm run dev
    exit 0
}

# Default: Start both in separate PowerShell windows
Write-Host "[1/2] Launching FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RootDir'; Write-Host '--- TruthLens AI API Backend (FastAPI) ---' -ForegroundColor Cyan; & '$PythonExe' server.py"

Write-Host "[*] Waiting for backend to initialize models and start listening on http://localhost:8000..." -ForegroundColor Yellow
$backendReady = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
        if ($health.status -eq "healthy") {
            $backendReady = $true
            Write-Host "[OK] Backend is healthy and ready!" -ForegroundColor Green
            break
        }
    } catch {
        $elapsedSec = $i * 2
        Write-Host "    ... still initializing AI models (${elapsedSec}s)" -ForegroundColor DarkGray
    }
}

if (-not $backendReady) {
    Write-Host "[!] Warning: Backend is still loading, but launching frontend now..." -ForegroundColor Yellow
}

Write-Host "[2/2] Launching React Frontend on http://localhost:5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RootDir\frontend'; Write-Host '--- TruthLens AI Frontend (React 19 + Vite) ---' -ForegroundColor Cyan; npm run dev"

Write-Host ""
Write-Host "System launched successfully!" -ForegroundColor Green
Write-Host "  - React Dashboard: http://localhost:5173" -ForegroundColor White
Write-Host "  - API and Swagger Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
