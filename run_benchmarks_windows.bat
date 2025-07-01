@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo PyIsolate Benchmark Runner for Windows
echo ================================================================
echo.
echo This script will:
echo   1. Check for uv installation
echo   2. Create a virtual environment
echo   3. Install PyTorch with appropriate CUDA support
echo   4. Install remaining dependencies
echo   5. Run performance and memory benchmarks
echo   6. Collect all results in a single file
echo.
echo ================================================================
echo.

REM Set up paths and filenames
set "SCRIPT_DIR=%~dp0"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_FILE=benchmark_results_%COMPUTERNAME%_%TIMESTAMP%.txt"
set "VENV_DIR=.benchmark_venv"
set "ERROR_LOG=benchmark_errors.log"
set "TEMP_OUTPUT=temp_output.txt"

REM Clean up any previous error log
if exist "%ERROR_LOG%" del "%ERROR_LOG%"
if exist "%TEMP_OUTPUT%" del "%TEMP_OUTPUT%"

echo [%date% %time%] Starting benchmark process... > "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo SYSTEM INFORMATION >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo System: Windows >> "%OUTPUT_FILE%"
echo Computer Name: %COMPUTERNAME% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Get detailed Windows version
echo Windows Version Details: >> "%OUTPUT_FILE%"
ver >> "%OUTPUT_FILE%" 2>&1
wmic os get Caption,Version,BuildNumber,OSArchitecture,ServicePackMajorVersion /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Get detailed CPU information
echo CPU Information: >> "%OUTPUT_FILE%"
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,Architecture /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
echo Legacy Processor Info: %PROCESSOR_IDENTIFIER% >> "%OUTPUT_FILE%"
echo Number of Processors: %NUMBER_OF_PROCESSORS% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Get detailed memory information
echo Memory Information: >> "%OUTPUT_FILE%"
wmic computersystem get TotalPhysicalMemory /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
for /f "tokens=2 delims==" %%a in ('wmic computersystem get TotalPhysicalMemory /format:list ^| findstr "="') do (
    set /a "RAM_GB=%%a/1024/1024/1024" 2>nul
    if not "!RAM_GB!"=="" echo Total RAM: !RAM_GB! GB >> "%OUTPUT_FILE%"
)
echo. >> "%OUTPUT_FILE%"

REM Get detailed video card information
echo Video Card Information: >> "%OUTPUT_FILE%"
wmic path win32_VideoController get Name,AdapterRAM,DriverVersion,DriverDate,VideoProcessor /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
for /f "tokens=2 delims==" %%a in ('wmic path win32_VideoController get AdapterRAM /format:list ^| findstr "AdapterRAM" ^| findstr -v "AdapterRAM=$"') do (
    set /a "VRAM_GB=%%a/1024/1024/1024" 2>nul
    if not "!VRAM_GB!"=="" echo Video RAM: !VRAM_GB! GB >> "%OUTPUT_FILE%"
)
echo. >> "%OUTPUT_FILE%"

REM Get motherboard and system information
echo System Hardware: >> "%OUTPUT_FILE%"
wmic baseboard get Manufacturer,Product,Version /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
wmic computersystem get Manufacturer,Model,SystemType /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

REM Step 1: Check for uv
echo Step 1: Checking for uv installation...
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: uv is not installed or not in PATH
    echo.
    echo Please install uv using one of these methods:
    echo.
    echo Option 1: Using PowerShell ^(recommended^):
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo Option 2: Using pip:
    echo   pip install uv
    echo.
    echo Option 3: Download from https://github.com/astral-sh/uv/releases
    echo.
    echo After installation, please restart this script.
    echo.
    echo [%date% %time%] ERROR: uv not found >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo uv found: OK
echo [%date% %time%] uv found >> "%OUTPUT_FILE%"

REM Step 2: Create virtual environment
echo.
echo Step 2: Creating virtual environment...
if exist "%VENV_DIR%" (
    echo Removing existing virtual environment...
    rmdir /s /q "%VENV_DIR%" 2>"%ERROR_LOG%"
    if !ERRORLEVEL! NEQ 0 (
        echo WARNING: Could not remove existing venv, continuing anyway...
        type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
    )
)

uv venv "%VENV_DIR%" 2>"%ERROR_LOG%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment
    echo Error details:
    type "%ERROR_LOG%"
    echo.
    echo [%date% %time%] ERROR: Failed to create venv >> "%OUTPUT_FILE%"
    type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo Virtual environment created: OK
echo [%date% %time%] Virtual environment created >> "%OUTPUT_FILE%"

REM Step 3: Activate virtual environment
echo.
echo Step 3: Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat" 2>"%ERROR_LOG%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    type "%ERROR_LOG%"
    echo [%date% %time%] ERROR: Failed to activate venv >> "%OUTPUT_FILE%"
    type "%ERROR_LOG%" >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo Virtual environment activated: OK

REM Step 4: Detect CUDA and install PyTorch appropriately
echo.
echo Step 4: Detecting GPU and installing PyTorch...
echo.

REM Check for CUDA availability
set "CUDA_AVAILABLE=0"
set "CUDA_VERSION="

where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected. Checking CUDA version...

    REM Get CUDA version from nvidia-smi
    for /f "tokens=*" %%i in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do (
        set "CUDA_LINE=%%i"
    )

    REM Extract CUDA version number
    if defined CUDA_LINE (
        for /f "tokens=4" %%a in ("!CUDA_LINE!") do (
            set "CUDA_VERSION=%%a"
            set "CUDA_AVAILABLE=1"
        )
    )

    if !CUDA_AVAILABLE! EQU 1 (
        echo Detected CUDA version: !CUDA_VERSION!
        echo [%date% %time%] CUDA detected: !CUDA_VERSION! >> "%OUTPUT_FILE%"

        REM Determine PyTorch CUDA version based on detected CUDA
        REM Extract major version
        for /f "tokens=1 delims=." %%a in ("!CUDA_VERSION!") do set "CUDA_MAJOR=%%a"

        REM Map CUDA version to PyTorch index
        if !CUDA_MAJOR! GEQ 12 (
            set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
            echo Installing PyTorch with CUDA 12.1 support...
        ) else if !CUDA_MAJOR! EQU 11 (
            set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
            echo Installing PyTorch with CUDA 11.8 support...
        ) else (
            set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
            echo CUDA version too old, installing CPU-only PyTorch...
        )
    ) else (
        echo Could not determine CUDA version, installing CPU-only PyTorch...
        set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    )
) else (
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch...
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    echo [%date% %time%] No CUDA detected, using CPU PyTorch >> "%OUTPUT_FILE%"
)

REM Install PyTorch with appropriate index
echo.
echo Installing PyTorch from: !TORCH_INDEX!
uv pip install torch torchvision torchaudio --index-url !TORCH_INDEX! > "%TEMP_OUTPUT%" 2>&1
set TORCH_INSTALL_RESULT=%ERRORLEVEL%
type "%TEMP_OUTPUT%"
type "%TEMP_OUTPUT%" >> "%OUTPUT_FILE%"

if %TORCH_INSTALL_RESULT% NEQ 0 (
    echo.
    echo ERROR: Failed to install PyTorch.
    echo [%date% %time%] ERROR: Failed to install PyTorch >> "%OUTPUT_FILE%"
    echo.
    echo Continuing without PyTorch - some benchmarks will be skipped
    echo.
) else (
    echo PyTorch installed successfully!
    echo [%date% %time%] PyTorch installed successfully >> "%OUTPUT_FILE%"
)

REM Step 5: Install remaining dependencies
echo.
echo Step 5: Installing remaining dependencies...

REM Install benchmark dependencies
uv pip install numpy psutil tabulate nvidia-ml-py3 pytest pytest-asyncio pyyaml > "%TEMP_OUTPUT%" 2>&1
type "%TEMP_OUTPUT%"
type "%TEMP_OUTPUT%" >> "%OUTPUT_FILE%"

REM Install pyisolate in editable mode
uv pip install -e . > "%TEMP_OUTPUT%" 2>&1
set PYISOLATE_RESULT=%ERRORLEVEL%
type "%TEMP_OUTPUT%"
type "%TEMP_OUTPUT%" >> "%OUTPUT_FILE%"

if %PYISOLATE_RESULT% NEQ 0 (
    echo ERROR: Failed to install pyisolate
    echo [%date% %time%] ERROR: Failed to install pyisolate >> "%OUTPUT_FILE%"
    pause
    exit /b 1
)
echo pyisolate installed: OK
echo [%date% %time%] pyisolate installed >> "%OUTPUT_FILE%"

REM Step 6: Verify installation
echo.
echo Step 6: Verifying installation...
echo. >> "%OUTPUT_FILE%"
echo Package Versions: >> "%OUTPUT_FILE%"
python --version >> "%OUTPUT_FILE%" 2>&1
python -c "import pyisolate; print(f'pyisolate: {pyisolate.__version__}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import numpy; print(f'numpy: {numpy.__version__}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import torch; print(f'torch: {torch.__version__}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" >> "%OUTPUT_FILE%" 2>&1
python -c "import psutil; print(f'psutil: {psutil.__version__}')" >> "%OUTPUT_FILE%" 2>&1
echo. >> "%OUTPUT_FILE%"

REM Step 7: Run performance benchmarks
echo.
echo Step 7: Running performance benchmarks...
echo ================================================================ >> "%OUTPUT_FILE%"
echo PERFORMANCE BENCHMARKS >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

cd benchmarks 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: benchmarks directory not found
    echo Make sure you're running this script from the pyisolate root directory
    pause
    exit /b 1
)

echo Running benchmark.py (this may take several minutes)...
echo Output is being saved to the results file...

REM Run benchmark with output capture
REM Note: Windows doesn't have a built-in timeout for console apps, but we can use start /wait
echo Starting performance benchmark...
start /B /WAIT python benchmark.py --quick > "%TEMP_OUTPUT%" 2>&1
set BENCHMARK_RESULT=!ERRORLEVEL!

REM Display and save the output
if exist "%TEMP_OUTPUT%" (
    type "%TEMP_OUTPUT%"
    type "%TEMP_OUTPUT%" >> "..\%OUTPUT_FILE%"
)

if %BENCHMARK_RESULT% EQU -1 (
    echo.
    echo WARNING: Performance benchmark timed out after 10 minutes
    echo This may indicate CUDA multiprocessing issues on Windows
    echo Partial results have been saved
    echo.
) else if %BENCHMARK_RESULT% NEQ 0 (
    echo.
    echo WARNING: Performance benchmark failed or was interrupted
    echo [%date% %time%] WARNING: Performance benchmark failed >> "..\%OUTPUT_FILE%"
    echo Error code: %BENCHMARK_RESULT% >> "..\%OUTPUT_FILE%"
    echo. >> "..\%OUTPUT_FILE%"
)
echo Continuing with memory benchmarks...

REM Step 8: Run memory benchmarks
echo.
echo Step 8: Running memory benchmarks...
echo. >> "..\%OUTPUT_FILE%"
echo ================================================================ >> "..\%OUTPUT_FILE%"
echo MEMORY BENCHMARKS >> "..\%OUTPUT_FILE%"
echo ================================================================ >> "..\%OUTPUT_FILE%"
echo. >> "..\%OUTPUT_FILE%"

echo Running memory_benchmark.py (this may take several minutes)...
echo Output is being saved to the results file...
echo NOTE: This test intentionally pushes VRAM limits to find maximum capacity

REM Run memory benchmark
REM Memory benchmarks take longer due to multiple extension creation

REM Calculate the time 15 minutes from now
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set /a "hours=%%a"
    set /a "minutes=%%b"
    set /a "seconds=%%c"
)

REM Add 15 minutes
set /a "minutes+=15"
if !minutes! GEQ 60 (
    set /a "hours+=1"
    set /a "minutes-=60"
)
if !hours! GEQ 24 (
    set /a "hours-=24"
)

REM Format the time nicely
if !hours! LSS 10 set "hours=0!hours!"
if !minutes! LSS 10 set "minutes=0!minutes!"

REM Determine AM/PM
set "ampm=AM"
set /a "display_hours=!hours!"
if !hours! GEQ 12 (
    set "ampm=PM"
    if !hours! GTR 12 (
        set /a "display_hours=!hours!-12"
    )
)
if !display_hours! EQU 0 set "display_hours=12"

echo Starting memory benchmark at %time%...
echo NOTE: Press Ctrl+C if nothing has changed in at least 15 minutes (at !display_hours!:!minutes! !ampm!)
echo The test intentionally pushes VRAM limits and may appear frozen when it hits limits.
start /B /WAIT python memory_benchmark.py --counts 1,2,5,10 --test-both-modes > "%TEMP_OUTPUT%" 2>&1
set MEMORY_RESULT=!ERRORLEVEL!

REM Display and save the output
if exist "%TEMP_OUTPUT%" (
    type "%TEMP_OUTPUT%"
    type "%TEMP_OUTPUT%" >> "..\%OUTPUT_FILE%"
)

if %MEMORY_RESULT% EQU -1 (
    echo.
    echo WARNING: Memory benchmark timed out after 20 minutes
    echo This may indicate CUDA multiprocessing issues on Windows
    echo Partial results have been saved
    echo.
    echo [%date% %time%] WARNING: Memory benchmark timed out >> "..\%OUTPUT_FILE%"
) else if %MEMORY_RESULT% NEQ 0 (
    echo.
    echo WARNING: Memory benchmark failed or was interrupted
    echo [%date% %time%] WARNING: Memory benchmark failed >> "..\%OUTPUT_FILE%"
    echo Error code: %MEMORY_RESULT% >> "..\%OUTPUT_FILE%"
)

cd ..

REM Step 9: Collect additional runtime information
echo.
echo Step 9: Collecting additional runtime information...
echo. >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo RUNTIME INFORMATION >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"

REM Get current memory usage
echo. >> "%OUTPUT_FILE%"
echo Current Memory Usage: >> "%OUTPUT_FILE%"
wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"

REM Try nvidia-smi if available for current GPU status
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo. >> "%OUTPUT_FILE%"
    echo Current NVIDIA GPU Status: >> "%OUTPUT_FILE%"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv >> "%OUTPUT_FILE%" 2>&1
    echo. >> "%OUTPUT_FILE%"
    echo Full nvidia-smi output: >> "%OUTPUT_FILE%"
    nvidia-smi >> "%OUTPUT_FILE%" 2>&1
)

REM Get disk space information
echo. >> "%OUTPUT_FILE%"
echo Disk Space Information: >> "%OUTPUT_FILE%"
wmic logicaldisk get size,freespace,caption /format:list 2>nul | findstr "=" >> "%OUTPUT_FILE%"

REM Final summary
echo. >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"
echo [%date% %time%] Benchmark collection completed >> "%OUTPUT_FILE%"
echo ================================================================ >> "%OUTPUT_FILE%"

REM Cleanup temporary files
if exist "%TEMP_OUTPUT%" del "%TEMP_OUTPUT%"
if exist "%ERROR_LOG%" del "%ERROR_LOG%"

REM Deactivate virtual environment
call deactivate 2>nul

REM Display completion message
echo.
echo ================================================================
echo BENCHMARK COLLECTION COMPLETED!
echo ================================================================
echo.
echo Results have been saved to: %OUTPUT_FILE%
echo.
echo Please send the file '%OUTPUT_FILE%' back for analysis.
echo.
echo IMPORTANT NOTES:
echo - The benchmarks intentionally push VRAM limits to find maximum capacity
echo - CUDA out-of-memory errors are EXPECTED and part of the testing
echo - If tests timeout, it may indicate Windows CUDA multiprocessing limitations
echo - Partial results are still valuable and have been saved
echo.
echo If you encountered any errors, please also include any error
echo messages shown above.
echo.
echo Thank you for running the benchmarks!
echo.
pause
