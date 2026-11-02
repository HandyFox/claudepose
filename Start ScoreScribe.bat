@echo off
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py) || (set PY=python)
echo Starting ScoreScribe - keep this window open while you use it.
%PY% -m scorescribe gui
pause
