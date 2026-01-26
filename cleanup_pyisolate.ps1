Write-Host "Cleaning up all extension venvs, .test_temps, and Python bytecode caches..." -ForegroundColor Cyan

# Remove extension venvs and test temp directories
$dirsToRemove = @(
    ".test_temps",
    ".benchmark_venv",
    "pyisolate\__pycache__",
    "pyisolate\_internal\__pycache__",
    "benchmarks\__pycache__",
    "example\__pycache__"
)

foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        Write-Host "Removing $dir ..."
        Remove-Item -Recurse -Force $dir
    }
}

# Remove all __pycache__ directories recursively
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Write-Host "Removing $($_.FullName) ..."
    Remove-Item -Recurse -Force $_.FullName
}

# Remove all .pyc files recursively
Get-ChildItem -Recurse -Include *.pyc | ForEach-Object {
    Write-Host "Removing $($_.FullName) ..."
    Remove-Item -Force $_.FullName
}

Write-Host "Cleanup complete!" -ForegroundColor Green