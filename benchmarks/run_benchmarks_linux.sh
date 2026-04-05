#!/bin/bash

set -euo pipefail

echo "================================================================"
echo "PyIsolate Benchmark Runner for Linux/macOS"
echo "================================================================"
echo ""
echo "This script will:"
echo "  1. Check for uv installation"
echo "  2. Create a virtual environment"
echo "  3. Install necessary dependencies"
echo "  4. Run performance and memory benchmarks"
echo "  5. Collect all results in a single file"
echo ""
echo "================================================================"
echo ""

# Set up paths and filenames
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HOSTNAME=$(hostname)
OUTPUT_FILE="benchmark_results_${HOSTNAME}_${TIMESTAMP}.txt"
VENV_DIR=".benchmark_venv"
ERROR_LOG="benchmark_errors.log"

# Cleanup function
cleanup() {
    if [ -f "$ERROR_LOG" ]; then
        rm -f "$ERROR_LOG"
    fi
}
trap cleanup EXIT

# Start output file with detailed system information
{
    echo "[$(date)] Starting benchmark process..."
    echo "================================================================"
    echo "SYSTEM INFORMATION"
    echo "================================================================"

    # Basic system info
    echo "System: $(uname -s)"
    echo "Hostname: $HOSTNAME"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo ""

    # Operating system details
    echo "Operating System Details:"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            echo "Distribution: $(grep '^NAME=' /etc/os-release | cut -d'"' -f2)"
            echo "Version: $(grep '^VERSION=' /etc/os-release | cut -d'"' -f2 2>/dev/null || echo 'Unknown')"
            echo "Version ID: $(grep '^VERSION_ID=' /etc/os-release | cut -d'"' -f2 2>/dev/null || echo 'Unknown')"
        elif [ -f /etc/lsb-release ]; then
            echo "Distribution: $(grep '^DISTRIB_DESCRIPTION=' /etc/lsb-release | cut -d'"' -f2)"
        else
            echo "Distribution: Unknown Linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Operating System: macOS"
        if command -v sw_vers &> /dev/null; then
            echo "Product Name: $(sw_vers -productName)"
            echo "Product Version: $(sw_vers -productVersion)"
            echo "Build Version: $(sw_vers -buildVersion)"
        fi
    fi
    echo ""

    # Detailed CPU information
    echo "CPU Information:"
    if [ -f /proc/cpuinfo ]; then
        echo "Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        echo "Physical Cores: $(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        echo "Logical Cores: $(nproc)"
        echo "CPU MHz: $(grep "cpu MHz" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        echo "Cache Size: $(grep "cache size" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        echo "CPU Flags: $(grep "flags" /proc/cpuinfo | head -1 | cut -d: -f2 | head -c 100 | xargs)..."
    elif command -v sysctl &> /dev/null; then
        # macOS
        echo "Model: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
        echo "Physical Cores: $(sysctl -n hw.physicalcpu 2>/dev/null || echo 'Unknown')"
        echo "Logical Cores: $(sysctl -n hw.logicalcpu 2>/dev/null || echo 'Unknown')"
        echo "CPU Frequency: $(sysctl -n hw.cpufrequency_max 2>/dev/null | awk '{print $1/1000000 " MHz"}' || echo 'Unknown')"
        echo "L1 Cache: $(sysctl -n hw.l1icachesize 2>/dev/null || echo 'Unknown') bytes (instruction), $(sysctl -n hw.l1dcachesize 2>/dev/null || echo 'Unknown') bytes (data)"
        echo "L2 Cache: $(sysctl -n hw.l2cachesize 2>/dev/null || echo 'Unknown') bytes"
        echo "L3 Cache: $(sysctl -n hw.l3cachesize 2>/dev/null || echo 'Unknown') bytes"
    fi
    echo ""

    # Detailed memory information
    echo "Memory Information:"
    if [ -f /proc/meminfo ]; then
        TOTAL_KB=$(grep "MemTotal" /proc/meminfo | awk '{print $2}')
        TOTAL_GB=$((TOTAL_KB / 1024 / 1024))
        echo "Total Memory: ${TOTAL_KB} KB (${TOTAL_GB} GB)"
        echo "Available Memory: $(grep "MemAvailable" /proc/meminfo | awk '{print $2}') KB"
        echo "Free Memory: $(grep "MemFree" /proc/meminfo | awk '{print $2}') KB"
        echo "Swap Total: $(grep "SwapTotal" /proc/meminfo | awk '{print $2}') KB"
        echo "Swap Free: $(grep "SwapFree" /proc/meminfo | awk '{print $2}') KB"
    elif command -v sysctl &> /dev/null; then
        # macOS
        TOTAL_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        TOTAL_GB=$((TOTAL_BYTES / 1024 / 1024 / 1024))
        echo "Total Memory: ${TOTAL_BYTES} bytes (${TOTAL_GB} GB)"
        if command -v vm_stat &> /dev/null; then
            vm_stat | grep -E "(Pages free|Pages active|Pages inactive|Pages speculative|Pages wired down)"
        fi
    fi
    echo ""

    # Video card information
    echo "Video Card Information:"
    if command -v lspci &> /dev/null; then
        # Linux with PCI
        lspci | grep -i vga || echo "No VGA controllers found via lspci"
        lspci | grep -i "3d controller" || true
        lspci | grep -i display || true
    elif command -v lshw &> /dev/null; then
        # Linux with lshw
        lshw -c display 2>/dev/null | grep -E "(product|vendor|description)" || echo "lshw display info not available"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v system_profiler &> /dev/null; then
            system_profiler SPDisplaysDataType 2>/dev/null | grep -E "(Chipset Model|Type|Bus|VRAM|Metal)" || echo "No display information available"
        fi
    else
        echo "Video card detection not available on this system"
    fi
    echo ""

    # Additional hardware information
    echo "Additional Hardware Information:"
    if [ -f /proc/version ]; then
        echo "Kernel Version: $(cat /proc/version)"
    fi

    if command -v dmidecode &> /dev/null && [ -r /dev/mem ]; then
        echo "System Manufacturer: $(dmidecode -s system-manufacturer 2>/dev/null || echo 'Unknown')"
        echo "System Product: $(dmidecode -s system-product-name 2>/dev/null || echo 'Unknown')"
        echo "BIOS Version: $(dmidecode -s bios-version 2>/dev/null || echo 'Unknown')"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "System Model: $(sysctl -n hw.model 2>/dev/null || echo 'Unknown')"
        echo "Machine Model: $(sysctl -n hw.machine 2>/dev/null || echo 'Unknown')"
    fi
    echo ""

} > "$OUTPUT_FILE"

# Step 1: Check for uv
echo "Step 1: Checking for uv installation..."
if ! command -v uv &> /dev/null; then
    echo ""
    echo "ERROR: uv is not installed or not in PATH"
    echo ""
    echo "Please install uv using one of these methods:"
    echo ""
    echo "Option 1: Using curl (recommended):"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Option 2: Using pip:"
    echo "  pip install uv"
    echo ""
    echo "Option 3: Using Homebrew (macOS):"
    echo "  brew install uv"
    echo ""
    echo "After installation, please restart this script."
    echo ""
    echo "[$(date)] ERROR: uv not found" >> "$OUTPUT_FILE"
    exit 1
fi
echo "uv found: OK"
echo "[$(date)] uv found: $(which uv)" >> "$OUTPUT_FILE"

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR" 2>"$ERROR_LOG" || {
        echo "WARNING: Could not remove existing venv, continuing anyway..."
        cat "$ERROR_LOG" >> "$OUTPUT_FILE" 2>/dev/null || true
    }
fi

uv venv "$VENV_DIR" 2>"$ERROR_LOG" || {
    echo "ERROR: Failed to create virtual environment"
    echo "Error details:"
    cat "$ERROR_LOG"
    echo ""
    echo "[$(date)] ERROR: Failed to create venv" >> "$OUTPUT_FILE"
    cat "$ERROR_LOG" >> "$OUTPUT_FILE"
    exit 1
}
echo "Virtual environment created: OK"
echo "[$(date)] Virtual environment created" >> "$OUTPUT_FILE"

# Step 3: Activate virtual environment
echo ""
echo "Step 3: Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate" 2>"$ERROR_LOG" || {
    echo "ERROR: Failed to activate virtual environment"
    cat "$ERROR_LOG"
    echo "[$(date)] ERROR: Failed to activate venv" >> "$OUTPUT_FILE"
    cat "$ERROR_LOG" >> "$OUTPUT_FILE"
    exit 1
}
echo "Virtual environment activated: OK"

# Step 4: Install pyisolate and dependencies
echo ""
echo "Step 4: Installing pyisolate and base dependencies..."
uv pip install -e ".[bench]" 2>"$ERROR_LOG" || {
    echo "ERROR: Failed to install pyisolate"
    cat "$ERROR_LOG"
    echo "[$(date)] ERROR: Failed to install pyisolate" >> "$OUTPUT_FILE"
    cat "$ERROR_LOG" >> "$OUTPUT_FILE"
    exit 1
}
echo "pyisolate installed: OK"
echo "[$(date)] pyisolate installed" >> "$OUTPUT_FILE"

# Step 5: Install PyTorch with correct CUDA version
echo ""
echo "Step 5: Installing PyTorch with appropriate CUDA support..."
echo "Running PyTorch installation helper..."
python install_torch_cuda.py 2>"$ERROR_LOG" || {
    echo "WARNING: PyTorch installation helper failed"
    echo "Attempting manual CPU-only installation..."

    # Detect OS for appropriate PyTorch installation
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        uv pip install torch torchvision torchaudio 2>"$ERROR_LOG" || {
            echo "ERROR: Failed to install PyTorch"
            cat "$ERROR_LOG"
            echo "[$(date)] ERROR: Failed to install PyTorch" >> "$OUTPUT_FILE"
            cat "$ERROR_LOG" >> "$OUTPUT_FILE"
            echo ""
            echo "Continuing without PyTorch - some benchmarks will be skipped"
        }
    else
        # Linux
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>"$ERROR_LOG" || {
            echo "ERROR: Failed to install PyTorch"
            cat "$ERROR_LOG"
            echo "[$(date)] ERROR: Failed to install PyTorch" >> "$OUTPUT_FILE"
            cat "$ERROR_LOG" >> "$OUTPUT_FILE"
            echo ""
            echo "Continuing without PyTorch - some benchmarks will be skipped"
        }
    fi
}
echo "[$(date)] PyTorch installation completed" >> "$OUTPUT_FILE"

# Verify Python and package versions
echo ""
echo "Step 6: Verifying installation..."
{
    echo ""
    echo "Package Versions:"
    python --version
    python -c "import pyisolate; print(f'pyisolate: {pyisolate.__version__}')" 2>&1 || echo "pyisolate: not available"
    python -c "import numpy; print(f'numpy: {numpy.__version__}')" 2>&1 || echo "numpy: not available"
    python -c "import torch; print(f'torch: {torch.__version__}')" 2>&1 || echo "torch: not available"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1 || echo "CUDA check: not available"
    python -c "import psutil; print(f'psutil: {psutil.__version__}')" 2>&1 || echo "psutil: not available"
    echo ""
} >> "$OUTPUT_FILE"

# Step 7: Run performance benchmarks
echo ""
echo "Step 7: Running performance benchmarks..."
{
    echo "================================================================"
    echo "PERFORMANCE BENCHMARKS"
    echo "================================================================"
    echo ""
} >> "$OUTPUT_FILE"

cd benchmarks 2>/dev/null || {
    echo "ERROR: benchmarks directory not found"
    echo "Make sure you're running this script from the pyisolate root directory"
    exit 1
}

echo "Running benchmark.py (this may take several minutes)..."
python benchmark.py --quick 2>&1 | tee -a "../$OUTPUT_FILE" || {
    echo "WARNING: Performance benchmark failed or was interrupted"
    echo "[$(date)] WARNING: Performance benchmark failed" >> "../$OUTPUT_FILE"
    echo "Exit code: $?" >> "../$OUTPUT_FILE"
    echo "" >> "../$OUTPUT_FILE"
    echo "Continuing with memory benchmarks..."
}

# Step 8: Run memory benchmarks
echo ""
echo "Step 8: Running memory benchmarks..."
{
    echo ""
    echo "================================================================"
    echo "MEMORY BENCHMARKS"
    echo "================================================================"
    echo ""
} >> "../$OUTPUT_FILE"

echo "Running memory_benchmark.py (this may take several minutes)..."
echo "Starting at: $(date)"
echo "NOTE: If nothing has changed after 90 minutes, press Ctrl+C"
echo "The test intentionally pushes VRAM limits and may appear frozen when it hits limits."
echo ""
python memory_benchmark.py --counts 1,2,5,10,25,50,100 2>&1 | tee -a "../$OUTPUT_FILE" || {
    echo "WARNING: Memory benchmark failed or was interrupted"
    echo "[$(date)] WARNING: Memory benchmark failed" >> "../$OUTPUT_FILE"
    echo "Exit code: $?" >> "../$OUTPUT_FILE"
}

cd ..

# Step 9: Collect additional runtime information
echo ""
echo "Step 9: Collecting additional runtime information..."
{
    echo ""
    echo "================================================================"
    echo "RUNTIME INFORMATION"
    echo "================================================================"

    # Current memory usage
    echo ""
    echo "Current Memory Usage:"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        free -h 2>/dev/null || echo "free command not available"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        vm_stat | grep -E "(Pages free|Pages active|Pages inactive|Pages speculative|Pages wired down)" || true
    fi

    # Current GPU status if available
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "Current NVIDIA GPU Status:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv 2>&1 || echo "nvidia-smi query failed"
        echo ""
        echo "Full nvidia-smi output:"
        nvidia-smi 2>&1 || echo "nvidia-smi failed"
    fi

    # Disk space information
    echo ""
    echo "Disk Space Information:"
    df -h 2>/dev/null | head -10 || echo "df command not available"

    # Process information
    echo ""
    echo "Top Memory Consuming Processes:"
    if command -v ps &> /dev/null; then
        ps aux --sort=-%mem 2>/dev/null | head -10 || ps aux 2>/dev/null | head -10 || echo "ps command not available"
    fi

    # Python environment details
    echo ""
    echo "Python Environment:"
    which python
    python -c "import sys; print(f'Python Path: {sys.executable}')"
    python -c "import site; print(f'Site Packages: {site.getsitepackages()}')" 2>/dev/null || true

    # Load averages (Linux/macOS)
    echo ""
    echo "System Load:"
    if [ -f /proc/loadavg ]; then
        echo "Load Average: $(cat /proc/loadavg)"
    elif command -v uptime &> /dev/null; then
        uptime
    fi

    echo ""
    echo "================================================================"
    echo "[$(date)] Benchmark collection completed"
    echo "================================================================"
} >> "$OUTPUT_FILE"

# Deactivate virtual environment
deactivate 2>/dev/null || true

# Display completion message
echo ""
echo "================================================================"
echo "BENCHMARK COLLECTION COMPLETED!"
echo "================================================================"
echo ""
echo "Results have been saved to: $OUTPUT_FILE"
echo ""
echo "Please send the file '$OUTPUT_FILE' back for analysis."
echo ""
echo "If you encountered any errors, please also include any error"
echo "messages shown above."
echo ""
echo "Thank you for running the benchmarks!"
echo ""
