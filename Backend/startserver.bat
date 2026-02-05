@echo off
cd /d "%~dp0"
echo Starting Bank Churn API (FastAPI)...
echo Backend will run at http://127.0.0.1:8000
echo API docs at http://127.0.0.1:8000/docs
echo.
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt -q
uvicorn main:app --reload --host 127.0.0.1 --port 8000
pause
