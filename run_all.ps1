# TruthLens AI - Run All Services (FastAPI Server + Streamlit App)
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Activate Virtual Environment if available
if (Test-Path "$RootDir\venv\Scripts\Activate.ps1") {
    & "$RootDir\venv\Scripts\Activate.ps1"
} elseif (Test-Path "$RootDir\.venv\Scripts\Activate.ps1") {
    & "$RootDir\.venv\Scripts\Activate.ps1"
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Launching TruthLens AI Services" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

# 2. Launch FastAPI Backend
Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RootDir'; if (Test-Path '.\venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 } elseif (Test-Path '.\.venv\Scripts\Activate.ps1') { .\.venv\Scripts\Activate.ps1 }; python server.py"

# 3. Launch Streamlit Web UI
Write-Host "[2/2] Starting Streamlit App on http://localhost:8501..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$RootDir'; if (Test-Path '.\venv\Scripts\Activate.ps1') { .\venv\Scripts\Activate.ps1 } elseif (Test-Path '.\.venv\Scripts\Activate.ps1') { .\.venv\Scripts\Activate.ps1 }; streamlit run app.py"

Write-Host ""
Write-Host "Services started:" -ForegroundColor Cyan
Write-Host "  - Streamlit UI : http://localhost:8501" -ForegroundColor White
Write-Host "  - API Backend  : http://localhost:8000/docs" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
