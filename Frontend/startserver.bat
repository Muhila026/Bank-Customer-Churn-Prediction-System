@echo off
cd /d "%~dp0"
echo Starting Bank Churn Dashboard (React)...
echo Frontend will run at http://127.0.0.1:3000
echo Make sure Backend is running (run Backend\startserver.bat first) for predictions.
echo.
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)
call npm run dev
pause
