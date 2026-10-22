@echo off
title TruthLens AI - Streamlit Dashboard
echo ==========================================================
echo   TruthLens AI - Streamlit Interactive Dashboard
echo ==========================================================
echo Starting Streamlit app on http://localhost:8501...
echo.

cd /d "%~dp0"

if exist "%~dp0venv\Scripts\streamlit.exe" (
    "%~dp0venv\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)
pause
