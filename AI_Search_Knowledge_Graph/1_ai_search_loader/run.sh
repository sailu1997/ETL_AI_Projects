#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_SCRIPT="$SCRIPT_DIR/main.py"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS" ]; then
    log_error "Requirements file not found: $REQUIREMENTS"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    log_step "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        log_error "Failed to create virtual environment"
        exit 1
    fi
    log_info "Virtual environment created successfully"
fi

# Activate virtual environment
log_step "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/upgrade dependencies
log_step "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REQUIREMENTS"
if [ $? -ne 0 ]; then
    log_error "Failed to install dependencies"
    deactivate
    exit 1
fi
log_info "Dependencies installed successfully"

# Run the Python script with all arguments passed to this script
log_step "Running ScholarBank to AI Search loader..."
python3 "$PYTHON_SCRIPT" "$@"
SCRIPT_EXIT_CODE=$?

# Deactivate virtual environment
deactivate

# Exit with the same code as the Python script
if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    log_info "Script completed successfully"
else
    log_error "Script failed with exit code: $SCRIPT_EXIT_CODE"
fi

exit $SCRIPT_EXIT_CODE
