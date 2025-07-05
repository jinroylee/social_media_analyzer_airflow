#!/bin/bash

# RunPod Setup Script for Airflow + MLflow ML Pipeline
# Run this script after cloning your repository to the RunPod instance

set -e  # Exit on any error

echo "🚀 Setting up ML Pipeline on RunPod..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update
apt-get install -y curl wget gnupg lsb-release ca-certificates

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Start Docker daemon manually (RunPod doesn't use systemd)
    echo "🔧 Starting Docker daemon..."
    dockerd > /dev/null 2>&1 &
    sleep 5
    
    # Add current user to docker group (if not root)
    if [ "$EUID" -ne 0 ]; then
        usermod -aG docker $USER
        echo "⚠️  Please logout and login again, or run 'newgrp docker' to use Docker without sudo"
    fi
else
    echo "✅ Docker already installed"
    # Check if Docker daemon is running
    if ! docker info > /dev/null 2>&1; then
        echo "🔧 Starting Docker daemon..."
        dockerd > /dev/null 2>&1 &
        sleep 5
    fi
fi

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
else
    echo "✅ Docker Compose already installed"
fi

# Install NVIDIA Container Toolkit for GPU support
echo "🎮 Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-container-runtime &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    # Use newer GPG key method for Ubuntu 22.04 compatibility
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    apt-get update
    apt-get install -y nvidia-container-toolkit nvidia-container-runtime
    
    # Configure Docker daemon for GPU support (RunPod specific)
    echo "🔧 Configuring Docker for GPU support..."
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
    
    # Restart Docker daemon
    pkill dockerd || true
    sleep 2
    dockerd > /dev/null 2>&1 &
    sleep 5
else
    echo "✅ NVIDIA Container Toolkit already installed"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models outputs mlruns plugins

# Set proper permissions
echo "🔐 Setting permissions..."
export AIRFLOW_UID=$(id -u)
echo "AIRFLOW_UID=$AIRFLOW_UID" >> .env
chown -R $AIRFLOW_UID:$AIRFLOW_UID logs data models outputs mlruns plugins

# Check GPU availability
echo "🎮 Checking GPU availability..."
nvidia-smi || echo "⚠️  GPU not detected or NVIDIA drivers not installed"

# Verify Docker can access GPU
echo "🧪 Testing Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi || echo "⚠️  Docker GPU access not working"

echo "✅ RunPod setup complete!"
echo ""
echo "🔥 Next steps:"
echo "1. Update your .env file with RunPod-specific settings"
echo "2. Run: docker-compose -f docker-compose.runpod.yml up -d"
echo "3. Access services:"
echo "   - Airflow: http://<runpod-ip>:8080"
echo "   - MLflow: http://<runpod-ip>:5001"
echo ""
echo "💡 Don't forget to open ports 8080 and 5001 in RunPod!"
echo "💡 Docker daemon is running in background. If you restart the container, run 'dockerd &' to start it again." 