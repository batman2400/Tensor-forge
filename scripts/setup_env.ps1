<#
.SYNOPSIS
  One-shot local environment setup for TensorForge (Windows / PowerShell).

.DESCRIPTION
  1. Installs uv (if missing) and Python 3.12.
  2. Creates the virtualenv OUTSIDE OneDrive (default: $HOME\.venvs\tensorforge).
  3. Installs CUDA 12.6 PyTorch, then the training + dev requirements.
  4. Enables the repo's git hooks (core.hooksPath = .githooks).
  5. Creates .env from .env.example if it does not exist (you fill in API_KEY).
  6. Verifies the GPU and the organizer assets.

  Safe to re-run.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#>
[CmdletBinding()]
param(
    [string]$VenvPath = (Join-Path $HOME ".venvs\tensorforge"),
    [switch]$SkipTorch,
    [switch]$UseLock
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$env:PYTHONUTF8 = "1"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "uv + Python 3.12"
python -m uv --version 2>$null
if ($LASTEXITCODE -ne 0) { python -m pip install --user uv }
python -m uv python install 3.12

Step "Virtualenv at $VenvPath (kept outside OneDrive on purpose)"
if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    python -m uv venv $VenvPath --python 3.12
}
$Py = Join-Path $VenvPath "Scripts\python.exe"

if (-not $SkipTorch) {
    Step "PyTorch (CUDA 12.6)"
    python -m uv pip install --python $Py -r requirements-torch-cu126.txt
}

Step "Training + dev requirements"
python -m uv pip install --python $Py -r requirements-train.txt -r requirements-dev.txt
if ($UseLock) {
    Write-Host "(-UseLock) Re-applying exact versions from requirements-train.lock.txt (torch excluded)"
    Get-Content requirements-train.lock.txt | Where-Object { $_ -match "==" -and $_ -notmatch "^torch==" } |
        Set-Content "$env:TEMP\tf_lock.txt"
    python -m uv pip install --python $Py -r "$env:TEMP\tf_lock.txt"
}

Step "Git hooks"
if (Test-Path ".git") {
    git config core.hooksPath .githooks
    Write-Host "core.hooksPath = $(git config core.hooksPath)"
} else {
    Write-Warning "Not a git repository yet. Run 'git init' then re-run this script."
}

Step ".env"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Warning ".env created from .env.example. Put your API_KEY in it (never commit it)."
} else {
    Write-Host ".env already exists"
}

Step "Verification"
& $Py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
if (Test-Path "data\train.csv") { & $Py scripts\verify_assets.py } else { & $Py scripts\verify_assets.py --no-data }

Write-Host "`nDone. Activate with:  $VenvPath\Scripts\Activate.ps1" -ForegroundColor Green
