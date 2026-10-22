@echo off
set PYTHONPATH=%~dp0src
echo Starting FedForge Live Server and Command Center Dashboard...
python run.py %*
pause
