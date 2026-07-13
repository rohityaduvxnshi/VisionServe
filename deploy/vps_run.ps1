# VisionServe service entry for the Windows VPS (native venv, no Docker).
# Lives at the bundle root on the VPS (folder containing server\ and model\).
# Registered via Task Scheduler — see CLAUDE.md Deployment record.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:OMP_NUM_THREADS = "2"     # tune to the VPS's physical cores
$env:APP_VERSION = "v1.0"
# frontend origins allowed to call the API — add the Vercel URL once it exists
$env:ALLOWED_ORIGINS = "https://dash-board.in,https://www.dash-board.in"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn server.app:app --host 127.0.0.1 --port 8002
