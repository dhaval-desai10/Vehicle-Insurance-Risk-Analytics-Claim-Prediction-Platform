@echo off
echo ============================================================
echo   InsureAI Platform - Full Stack Launcher
echo ============================================================

set PYTHON=C:\Users\DD\AppData\Local\Microsoft\WindowsApps\python.exe
set SCRIPTS=C:\Users\DD\AppData\Local\Python\pythoncore-3.14-64\Scripts
set PYTHONIOENCODING=utf-8

echo.
echo [1/4] Checking database...
if not exist "data\insurance.db" (
    echo  Running ETL pipeline...
    %PYTHON% src\etl_pipeline.py
) else (
    echo  Database found: data\insurance.db
)

echo.
echo [2/4] Checking ML models...
if not exist "src\models\saved\classifier.pkl" (
    echo  Training ML models ^(this takes ~2 mins^)...
    %PYTHON% src\models\train_models.py
) else (
    echo  Models found in src\models\saved\
)

echo.
echo [3/4] Starting FastAPI backend on port 8000...
start "InsureAI API" cmd /k "set PYTHONIOENCODING=utf-8 && %PYTHON% -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak > nul

echo.
echo [4/4] Starting Streamlit dashboard on port 8501...
start "InsureAI Dashboard" cmd /k "set PYTHONIOENCODING=utf-8 && %SCRIPTS%\streamlit.exe run app/streamlit_app.py --server.port 8501"

echo.
echo ============================================================
echo   PLATFORM IS RUNNING!
echo   Dashboard:  http://localhost:8501
echo   API Docs:   http://localhost:8000/docs
echo ============================================================
pause
