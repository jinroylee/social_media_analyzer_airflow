#!/bin/bash

# Simple RunPod Setup Script for Airflow + MLflow ML Pipeline
# This assumes Docker is already available in your RunPod instance

set -e  # Exit on any error

echo "🚀 Simple RunPod Setup for ML Pipeline..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please use the full setup_runpod.sh script instead."
    exit 1
fi

# Start Docker daemon if not running
if ! docker info > /dev/null 2>&1; then
    echo "🔧 Starting Docker daemon..."
    dockerd > /dev/null 2>&1 &
    sleep 5
fi

# Install Docker Compose if not available
if ! command -v docker-compose &> /dev/null; then
    echo "🔧 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
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

# Test Docker GPU access
echo "🧪 Testing Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi || echo "⚠️  Docker GPU access not working - you may need to configure NVIDIA Container Toolkit"

echo "✅ Simple RunPod setup complete!"
echo ""
echo "🔥 Next steps:"
echo "1. Copy environment template: cp .env.runpod .env"
echo "2. Edit .env file with your AWS credentials and RunPod IP"
echo "3. Start services: docker-compose -f docker-compose.runpod.yml up -d"
echo "4. Access services:"
echo "   - Airflow: http://<runpod-ip>:8080"
echo "   - MLflow: http://<runpod-ip>:5001"
echo ""
echo "💡 Make sure ports 8080 and 5001 are exposed in RunPod!" 