#!/bin/bash
set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_SCRIPT="$SCRIPT_DIR/main.py"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

echo ""
echo "============================================================"
echo "  Graph Builder - CosmosDB Gremlin"
echo "============================================================"
echo ""

# Validate required files exist
if [ ! -f "$REQUIREMENTS" ]; then
    log_error "requirements.txt not found at: $REQUIREMENTS"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_error "main.py not found at: $PYTHON_SCRIPT"
    exit 1
fi

# Step 1: Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log_step "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    log_info "Virtual environment created at: $VENV_DIR"
else
    log_info "Virtual environment already exists"
fi

# Step 2: Activate virtual environment
log_step "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Step 3: Upgrade pip
log_step "Upgrading pip..."
pip install --quiet --upgrade pip

# Step 4: Install dependencies
log_step "Installing dependencies from requirements.txt..."
pip install --quiet -r "$REQUIREMENTS"
log_info "Dependencies installed successfully"

# Step 5: Run the Python script
echo ""
log_step "Running graph builder..."
echo ""

# Pass all command-line arguments to the Python script
python3 "$PYTHON_SCRIPT" "$@"
SCRIPT_EXIT_CODE=$?

# Step 6: Clean up
echo ""
deactivate

if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    log_info "Script completed successfully!"
else
    log_error "Script failed with exit code: $SCRIPT_EXIT_CODE"
fi

exit $SCRIPT_EXIT_CODE
