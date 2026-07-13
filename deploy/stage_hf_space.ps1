# Stages a minimal Hugging Face Space (Docker) repo into deploy/hf-space/.
# Push it with your HF token:
#   git -C deploy/hf-space init; git -C deploy/hf-space add -A; git -C deploy/hf-space commit -m "deploy"
#   git -C deploy/hf-space push --force https://<user>:<HF_TOKEN>@huggingface.co/spaces/<user>/visionserve main
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$stage = Join-Path $PSScriptRoot "hf-space"

Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$stage\server", "$stage\model" | Out-Null

Copy-Item "$root\server\app.py", "$root\server\inference.py", "$root\server\requirements.txt" "$stage\server\"
Copy-Item "$root\model\ppe_yolov8n_openvino_model" "$stage\model\" -Recurse
Copy-Item "$root\model\ppe_yolov8n_int8_openvino_model" "$stage\model\" -Recurse
Copy-Item "$root\server\Dockerfile" "$stage\Dockerfile"   # COPY paths are root-context, unchanged

# emoji built from codepoint so the script survives any .ps1 file encoding;
# BOM-less UTF-8 write so HF's YAML front-matter parser isn't tripped by a BOM
$emoji = [char]::ConvertFromUtf32(0x1F9BA)  # safety vest
$readme = @"
---
title: VisionServe
emoji: $emoji
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
---
# VisionServe backend
CPU-only PPE detection API (OpenVINO FP32/INT8). See the main repo for docs.
"@
[System.IO.File]::WriteAllText("$stage\README.md", $readme, [System.Text.UTF8Encoding]::new($false))

Write-Host "staged HF Space repo in $stage"
