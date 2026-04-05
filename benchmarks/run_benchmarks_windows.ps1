# PyIsolate Benchmark Runner for Windows (PowerShell Version)
# This script runs the same benchmarks as the .bat file but uses PowerShell
#
# HOW TO RUN THIS SCRIPT:
#
# Option 1: Use the launcher (EASIEST)
#   Double-click: run_benchmarks_powershell_launcher.bat
#
# Option 2: Enable PowerShell scripts permanently
#   1. Open PowerShell as regular user (not admin)
#   2. Run: Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser
#   3. Type Y and press Enter
#   4. Right-click this .ps1 file and select "Run with PowerShell"
#
# Option 3: Run with temporary bypass (one-time)
#   1. Open PowerShell in this directory
#   2. Run: powershell -ExecutionPolicy Bypass -File .\run_benchmarks_windows.ps1

$ErrorActionPreference = "Continue"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "PyIsolate Benchmark Runner for Windows (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Set up paths and filenames
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputFile = "benchmark_results_${env:COMPUTERNAME}_${Timestamp}.txt"
$VenvDir = ".benchmark_venv"

# Start output file
"[$(Get-Date)] Starting benchmark process..." | Out-File $OutputFile
"================================================================" | Add-Content $OutputFile
"SYSTEM INFORMATION" | Add-Content $OutputFile
"================================================================" | Add-Content $OutputFile
"System: Windows" | Add-Content $OutputFile
"Computer Name: $env:COMPUTERNAME" | Add-Content $OutputFile
"" | Add-Content $OutputFile

# Get system information
"Windows Version Details:" | Add-Content $OutputFile
(Get-WmiObject Win32_OperatingSystem).Caption | Add-Content $OutputFile
"" | Add-Content $OutputFile

# Step 1: Check for uv
Write-Host "Step 1: Checking for uv installation..."
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Host ""
    Write-Host "ERROR: uv is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install uv using one of these methods:"
    Write-Host ""
    Write-Host "Option 1: Using PowerShell (recommended):" -ForegroundColor Yellow
    Write-Host '  irm https://astral.sh/uv/install.ps1 | iex'
    Write-Host ""
    Write-Host "Option 2: Using pip:"
    Write-Host "  pip install uv"
    Write-Host ""
    Write-Host "After installation, please restart this script."
    Write-Host ""
    "[$(Get-Date)] ERROR: uv not found" | Add-Content $OutputFile
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "uv found: OK" -ForegroundColor Green
"[$(Get-Date)] uv found" | Add-Content $OutputFile

# Step 2: Create virtual environment
Write-Host ""
Write-Host "Step 2: Creating virtual environment..."
if (Test-Path $VenvDir) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
}

& uv venv $VenvDir 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    "[$(Get-Date)] ERROR: Failed to create venv" | Add-Content $OutputFile
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Virtual environment created: OK" -ForegroundColor Green
"[$(Get-Date)] Virtual environment created" | Add-Content $OutputFile

# Step 3: Activate virtual environment
Write-Host ""
Write-Host "Step 3: Activating virtual environment..."
& "$VenvDir\Scripts\Activate.ps1"
Write-Host "Virtual environment activated: OK" -ForegroundColor Green

# Step 4: Detect CUDA and install PyTorch
Write-Host ""
Write-Host "Step 4: Detecting GPU and installing PyTorch..."
Write-Host ""

$cudaAvailable = $false
$torchIndex = "https://download.pytorch.org/whl/cpu"

# Check for CUDA
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "NVIDIA GPU detected. Checking CUDA version..."
    $cudaInfo = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) {
        $cudaVersion = (& nvidia-smi | Select-String "CUDA Version" | ForEach-Object { $_ -match "CUDA Version:\s*(\d+\.\d+)" | Out-Null; $matches[1] })
        if ($cudaVersion) {
            Write-Host "Detected CUDA version: $cudaVersion" -ForegroundColor Green
            "[$(Get-Date)] CUDA detected: $cudaVersion" | Add-Content $OutputFile

            $cudaMajor = [int]($cudaVersion.Split('.')[0])
            if ($cudaMajor -ge 12) {
                $torchIndex = "https://download.pytorch.org/whl/cu121"
                Write-Host "Installing PyTorch with CUDA 12.1 support..."
            } elseif ($cudaMajor -eq 11) {
                $torchIndex = "https://download.pytorch.org/whl/cu118"
                Write-Host "Installing PyTorch with CUDA 11.8 support..."
            }
        }
    }
} else {
    Write-Host "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    "[$(Get-Date)] No CUDA detected, using CPU PyTorch" | Add-Content $OutputFile
}

Write-Host ""
Write-Host "Installing PyTorch from: $torchIndex"
# Suppress PowerShell's stderr warnings for this command
$ErrorActionPreference = "SilentlyContinue"
$output = & uv pip install torch torchvision torchaudio --index-url $torchIndex 2>&1
$ErrorActionPreference = "Continue"
$output | Out-String | Tee-Object -Append $OutputFile
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to install PyTorch." -ForegroundColor Red
    "[$(Get-Date)] ERROR: Failed to install PyTorch" | Add-Content $OutputFile
    Write-Host "Continuing without PyTorch - some benchmarks will be skipped"
} else {
    Write-Host "PyTorch installed successfully!" -ForegroundColor Green
    "[$(Get-Date)] PyTorch installed successfully" | Add-Content $OutputFile
}

# Step 5: Install remaining dependencies
Write-Host ""
Write-Host "Step 5: Installing remaining dependencies..."

$ErrorActionPreference = "SilentlyContinue"
$output = & uv pip install numpy psutil tabulate nvidia-ml-py3 pytest pytest-asyncio pyyaml 2>&1
$ErrorActionPreference = "Continue"
$output | Out-String | Tee-Object -Append $OutputFile

$ErrorActionPreference = "SilentlyContinue"
$output = & uv pip install -e . 2>&1
$ErrorActionPreference = "Continue"
$output | Out-String | Tee-Object -Append $OutputFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install pyisolate" -ForegroundColor Red
    "[$(Get-Date)] ERROR: Failed to install pyisolate" | Add-Content $OutputFile
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "pyisolate installed: OK" -ForegroundColor Green
"[$(Get-Date)] pyisolate installed" | Add-Content $OutputFile

# Step 6: Verify installation
Write-Host ""
Write-Host "Step 6: Verifying installation..."
"" | Add-Content $OutputFile
"Package Versions:" | Add-Content $OutputFile
& python --version 2>&1 | Add-Content $OutputFile
& python -c "import pyisolate; print(f'pyisolate: {pyisolate.__version__}')" 2>&1 | Add-Content $OutputFile
& python -c "import numpy; print(f'numpy: {numpy.__version__}')" 2>&1 | Add-Content $OutputFile
& python -c "import torch; print(f'torch: {torch.__version__}')" 2>&1 | Add-Content $OutputFile
& python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1 | Add-Content $OutputFile
& python -c "import psutil; print(f'psutil: {psutil.__version__}')" 2>&1 | Add-Content $OutputFile
"" | Add-Content $OutputFile

# Step 7: Run performance benchmarks
Write-Host ""
Write-Host "Step 7: Running performance benchmarks..."
"================================================================" | Add-Content $OutputFile
"PERFORMANCE BENCHMARKS" | Add-Content $OutputFile
"================================================================" | Add-Content $OutputFile
"" | Add-Content $OutputFile

Set-Location benchmarks
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: benchmarks directory not found" -ForegroundColor Red
    Write-Host "Make sure you're running this script from the pyisolate root directory"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Running benchmark.py (this may take several minutes)..."
Write-Host "Output is being saved to the results file..."

# Run benchmark - PowerShell handles subprocess differently
$env:PYTHONUNBUFFERED = "1"
$output = & python benchmark.py --quick 2>&1 | Out-String
$benchmarkResult = $LASTEXITCODE
$output | Tee-Object -Append "..\$OutputFile"

if ($benchmarkResult -ne 0) {
    Write-Host ""
    Write-Host "WARNING: Performance benchmark failed or was interrupted" -ForegroundColor Yellow
    "[$(Get-Date)] WARNING: Performance benchmark failed" | Add-Content "..\$OutputFile"
    "Error code: $benchmarkResult" | Add-Content "..\$OutputFile"
    "" | Add-Content "..\$OutputFile"
    Write-Host "Continuing with memory benchmarks..."
}

# Step 8: Run memory benchmarks
Write-Host ""
Write-Host "Step 8: Running memory benchmarks..."
"" | Add-Content "..\$OutputFile"
"================================================================" | Add-Content "..\$OutputFile"
"MEMORY BENCHMARKS" | Add-Content "..\$OutputFile"
"================================================================" | Add-Content "..\$OutputFile"
"" | Add-Content "..\$OutputFile"

Write-Host "Running memory_benchmark.py (this may take several minutes)..."
Write-Host "NOTE: This test intentionally pushes VRAM limits to find maximum capacity"

Write-Host "Starting memory benchmark at $(Get-Date -Format 'h:mm:ss tt')..."
Write-Host "NOTE: If nothing has changed after 90 minutes, press Ctrl+C" -ForegroundColor Yellow
Write-Host "The test intentionally pushes VRAM limits and may appear frozen when it hits limits."

# Run memory benchmark
$output = & python memory_benchmark.py --counts 1,2,5,10,25,50,100 2>&1 | Out-String
$memoryResult = $LASTEXITCODE
$output | Tee-Object -Append "..\$OutputFile"

if ($memoryResult -ne 0) {
    Write-Host ""
    Write-Host "WARNING: Memory benchmark failed or was interrupted" -ForegroundColor Yellow
    "[$(Get-Date)] WARNING: Memory benchmark failed" | Add-Content "..\$OutputFile"
    "Error code: $memoryResult" | Add-Content "..\$OutputFile"
}

Set-Location ..

# Step 9: Collect additional runtime information
Write-Host ""
Write-Host "Step 9: Collecting additional runtime information..."
"" | Add-Content $OutputFile
"================================================================" | Add-Content $OutputFile
"RUNTIME INFORMATION" | Add-Content $OutputFile
"================================================================" | Add-Content $OutputFile

# Final summary
"" | Add-Content $OutputFile
"================================================================" | Add-Content $OutputFile
"[$(Get-Date)] Benchmark collection completed" | Add-Content $OutputFile
"================================================================" | Add-Content $OutputFile

# Deactivate virtual environment
deactivate 2>$null

# Display completion message
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "BENCHMARK COLLECTION COMPLETED!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results have been saved to: $OutputFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "Please send the file '$OutputFile' back for analysis."
Write-Host ""
Write-Host "IMPORTANT NOTES:" -ForegroundColor Cyan
Write-Host "- The benchmarks intentionally push VRAM limits to find maximum capacity"
Write-Host "- CUDA out-of-memory errors are EXPECTED and part of the testing"
Write-Host "- If tests timeout, it may indicate Windows CUDA multiprocessing limitations"
Write-Host "- Partial results are still valuable and have been saved"
Write-Host ""
Write-Host "Thank you for running the benchmarks!"
Write-Host ""
Read-Host "Press Enter to exit"
