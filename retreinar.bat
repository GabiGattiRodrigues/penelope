@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Rode primeiro o rodar_penelope.bat
    pause
    exit /b 1
)
echo Gerando base sintetica...
".venv\Scripts\python.exe" src\gerar_base.py
echo Montando variaveis M0-M11...
".venv\Scripts\python.exe" src\variaveis.py
echo Treinando os 12 GAMs (leva uns 2 minutos)...
".venv\Scripts\python.exe" src\treinar.py
pause
