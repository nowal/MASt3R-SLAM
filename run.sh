#!/bin/bash
# Run script for MASt3R-SLAM Real-Time Web App (using Gunicorn)

# Ensure we're in the correct directory where the script resides
cd "$(dirname "$0")" || exit 1
SCRIPT_DIR="$(pwd)"
echo "Running script from: $SCRIPT_DIR"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed or not in PATH. Please install Conda and try again."
    exit 1
fi

# Activate the existing mast3r-slam conda environment
echo "Activating mast3r-slam conda environment..."
# Source conda.sh if available, otherwise try to use eval
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "WARNING: conda.sh not found. Attempting 'eval \$(conda shell.bash hook)'"
    eval "$(conda shell.bash hook)"
fi
conda activate mast3r-slam

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate mast3r-slam conda environment."
    echo "Make sure you have created the environment as described in the README.md:"
    echo "  conda create -n mast3r-slam python=3.11" # Or your target Python version
    echo "  conda activate mast3r-slam"
    echo "  # Install PyTorch with matching CUDA version"
    echo "  # Install dependencies from thirdparty folders and requirements.txt"
    exit 1
fi
echo "Conda environment 'mast3r-slam' activated."

# Install/Upgrade additional packages needed for the web server
echo "Installing/upgrading additional packages for the web server..."
EXTRA_PACKAGES="gunicorn fastapi uvicorn[standard] websockets python-multipart scipy" # uvicorn[standard] for ws support
pip install --upgrade $EXTRA_PACKAGES

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install/upgrade additional packages."
    exit 1
fi
echo "Additional packages checked/installed successfully."

# Check if model checkpoint exists
MODEL_PATH="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "--------------------------------------------------------------------"
    echo "WARNING: MASt3R model checkpoint not found at:"
    echo "  $SCRIPT_DIR/$MODEL_PATH"
    echo "Please ensure the model checkpoint file exists in the correct location."
    echo "The SLAM process (slam_process_runner.py) will likely fail to load the model."
    echo "--------------------------------------------------------------------"
    # Consider exiting if this is critical for server startup, for now, it warns.
    # echo "ERROR: Model checkpoint missing. Exiting."
    # exit 1
fi

# --- SSL Certificate Generation ---
# Paths for SSL certificates
SSL_KEY_FILE="/etc/ssl/private/selfsigned_mast3r_slam.key"
SSL_CERT_FILE="/etc/ssl/certs/selfsigned_mast3r_slam.crt"

# Check if certificates exist, if not, generate them
if [ ! -f "$SSL_KEY_FILE" ] || [ ! -f "$SSL_CERT_FILE" ]; then
    echo "SSL certificates not found. Generating self-signed certificates..."
    echo "This will require sudo privileges."
    # Create directories if they don't exist
    sudo mkdir -p /etc/ssl/private
    sudo mkdir -p /etc/ssl/certs
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -keyout "$SSL_KEY_FILE" \
      -out "$SSL_CERT_FILE" \
      -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate self-signed SSL certificates."
        exit 1
    fi
    # Set permissions if needed, though gunicorn running as sudo will have access
    # sudo chmod 600 "$SSL_KEY_FILE"
    echo "Self-signed SSL certificates generated successfully."
else
    echo "Using existing SSL certificates:"
    echo "  Key file: $SSL_KEY_FILE"
    echo "  Cert file: $SSL_CERT_FILE"
fi


# --- Run the FastAPI app using Gunicorn ---
echo "Starting MASt3R-SLAM Real-Time Web App with Gunicorn on port 443 (HTTPS)..."

# Get the full path to the conda environment's Python
CONDA_PYTHON_PATH=$(which python)
if [ -z "$CONDA_PYTHON_PATH" ]; then
    echo "ERROR: Could not find Python interpreter in the activated conda environment."
    exit 1
fi
echo "Using Python interpreter: $CONDA_PYTHON_PATH"

# SET THE MULTIPROCESSING START METHOD VIA ENVIRONMENT VARIABLE
export PYTHON_MULTIPROCESSING_START_METHOD=spawn
echo "SCRIPT: PYTHON_MULTIPROCESSING_START_METHOD environment variable EXPORTED as 'spawn'"

# Gunicorn command
# --workers 1 is CRITICAL for this application.
# Use `new_fast:app` to point to your FastAPI application instance.
GUNICORN_CMD="sudo -E $CONDA_PYTHON_PATH -m gunicorn new_fast:app \
  --bind 0.0.0.0:443 \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --certfile=$SSL_CERT_FILE \
  --keyfile=$SSL_KEY_FILE \
  --access-logfile - \
  --error-logfile - \
  --log-level debug"
  # Add --daemon to run in background if desired, but console logs will be lost unless redirected.

echo "Executing Gunicorn command:"
echo "$GUNICORN_CMD"
echo "--------------------------------------------------------------------"
echo " Gunicorn will run in the foreground. Press Ctrl+C to stop it. "
echo " Ensure that no other service is using port 443.           "
echo " Logs will be printed to the console.                       "
echo "--------------------------------------------------------------------"

# Execute the command
eval "$GUNICORN_CMD"

# This line will be reached when Gunicorn stops
echo "Gunicorn process stopped."