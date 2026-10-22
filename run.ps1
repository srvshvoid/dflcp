$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$ScriptDir\src"
Write-Host "Starting FedForge Live Server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
python "$ScriptDir\run.py"
