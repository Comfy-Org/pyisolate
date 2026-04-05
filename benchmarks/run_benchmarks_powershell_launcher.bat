@echo off
echo ================================================================
echo PowerShell Benchmark Launcher
echo ================================================================
echo.
echo This will run the benchmarks using PowerShell, which may work
echo better with CUDA multiprocessing on Windows.
echo.

REM Check if PowerShell scripts can run
powershell -Command "Get-ExecutionPolicy -Scope CurrentUser" > temp_policy.txt 2>&1
set /p CURRENT_POLICY=<temp_policy.txt
del temp_policy.txt

echo Current PowerShell execution policy: %CURRENT_POLICY%
echo.

if "%CURRENT_POLICY%"=="Restricted" (
    echo PowerShell scripts are currently BLOCKED on your system.
    echo.
    echo To enable the benchmark script, we need to allow PowerShell scripts.
    echo This is safe for scripts you trust.
    echo.
    set /p ENABLE_PS="Allow PowerShell scripts for current user? (Y/N): "
    if /i "!ENABLE_PS!"=="Y" (
        echo.
        echo Enabling PowerShell scripts for current user...
        powershell -Command "Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser -Force"
        echo Done! PowerShell scripts are now enabled for your user account.
        echo.
    ) else (
        echo.
        echo Cannot run PowerShell benchmarks without enabling scripts.
        echo You can still use run_benchmarks_windows.bat instead.
        echo.
        pause
        exit /b
    )
)

echo Running benchmarks via PowerShell...
echo.

REM Run the PowerShell script with bypass for this session only (extra safety)
powershell -ExecutionPolicy Bypass -File ".\run_benchmarks_windows.ps1"

pause
