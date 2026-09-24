@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   Penelope - quem vai continuar fiel?
echo ==========================================

rem Isola o ambiente: ignora bibliotecas de outras instalacoes (Anaconda, PYTHONPATH, pasta do usuario)
set PYTHONPATH=
set PYTHONHOME=
set PYTHONNOUSERSITE=1
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Criando ambiente virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo Nao encontrei o Python. Instale em https://www.python.org/downloads/ e marque "Add to PATH".
        pause
        exit /b 1
    )
)

if not exist ".venv\instalado_v3.ok" (
    echo Instalando/reparando dependencias - pode levar alguns minutos...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip uninstall -y scipy scikit-learn
    "%PY%" -m pip install --no-cache-dir -r requirements.txt
    if errorlevel 1 (
        echo.
        echo A instalacao falhou. Tire um print desta tela e me mande.
        pause
        exit /b 1
    )
)

echo.
echo ---- diagnostico ----
"%PY%" -c "import sys; print('Python', sys.version.split()[0], '|', sys.executable)"
"%PY%" -c "import sys; sys.path.insert(0, 'src'); import scipy, sklearn, joblib, gam; joblib.load('artefatos/modelos_gam.joblib'); print('scipy', scipy.__version__, '| sklearn', sklearn.__version__, '| modelos OK')"
if errorlevel 1 (
    echo.
    echo O scipy ainda nao importa. Tire um print desta tela INTEIRA e me mande.
    pause
    exit /b 1
)
echo ok> ".venv\instalado_v3.ok"
echo ---------------------
echo.

if not exist "artefatos\modelos_gam.joblib" (
    echo Gerando base e treinando os modelos...
    "%PY%" src\gerar_base.py
    "%PY%" src\variaveis.py
    "%PY%" src\treinar.py
)

echo Abrindo o app no navegador...
"%PY%" -m streamlit run app.py
pause
