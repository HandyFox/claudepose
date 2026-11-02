@echo off
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py) || (set PY=python)
echo Installing / updating ScoreScribe...
%PY% -m pip install -e .
echo.
echo Done. Double-click "Start ScoreScribe.bat" to run it.
pause
