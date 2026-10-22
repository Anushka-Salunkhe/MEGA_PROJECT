@echo off
title TruthLens AI Launcher
echo ==========================================================
echo   TruthLens AI - Multimodal Verification Platform
echo ==========================================================
echo Starting Backend (FastAPI) and Frontend (React Vite)...
echo.

cd /d "%~dp0"

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

start "TruthLens AI - Backend Server" cmd /k ""%PYTHON_EXE%" server.py"
timeout /t 3 /nobreak >nul

cd /d "%~dp0frontend"
start "TruthLens AI - React Frontend" cmd /k "npm run dev"

echo.
echo ==========================================================
echo System started in two separate console windows!
echo   Frontend UI:        http://localhost:5173
echo   Backend API / Docs: http://localhost:8000/docs
echo ==========================================================
pause
