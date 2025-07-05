# 🚀 RunPod Setup Guide for ML Pipeline

## Prerequisites
- RunPod account with GPU instance
- SSH access to your RunPod instance
- Git installed on RunPod instance

## ⚠️ Important Note for RunPod
RunPod containers don't use systemd, so we've created RunPod-specific setup scripts that work with the container environment.

## 🔥 Quick Setup Commands

### Option 1: Simple Setup (Recommended)
If Docker is already available in your RunPod instance:

```bash
# 1. Make simple setup script executable and run it
chmod +x setup_runpod_simple.sh
sudo ./setup_runpod_simple.sh

# 2. Copy and configure environment
cp .env.runpod .env
nano .env  # Update AWS credentials and RunPod IP

# 3. Start the services
docker-compose -f docker-compose.runpod.yml up -d
```

### Option 2: Full Setup
If you need to install Docker from scratch:

```bash
# 1. Make full setup script executable and run it
chmod +x setup_runpod.sh
sudo ./setup_runpod.sh

# 2. Copy and configure environment
cp .env.runpod .env
nano .env  # Update AWS credentials and RunPod IP

# 3. Start the services
docker-compose -f docker-compose.runpod.yml up -d
```

## 📋 Detailed Setup Steps

### Step 1: Initial System Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install git (if not already installed)
sudo apt install git -y

# Clone your repository
git clone <your-repo-url>
cd social_media_analyzer_airflow
```

### Step 2: Choose Your Setup Method

**For Simple Setup (Docker already available):**
```bash
chmod +x setup_runpod_simple.sh
sudo ./setup_runpod_simple.sh
```

**For Full Setup (Install Docker from scratch):**
```bash
chmod +x setup_runpod.sh
sudo ./setup_runpod.sh
```

### Step 3: Manual Docker Start (If Needed)
If Docker isn't running after setup:
```bash
# Start Docker daemon manually
sudo dockerd > /dev/null 2>&1 &
sleep 5

# Verify Docker is running
docker info
```

### Step 4: Configure Environment
```bash
# Copy RunPod environment template
cp .env.runpod .env

# Edit with your specific values
nano .env
```

**Important values to update:**
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `S3_BUCKET_NAME`: Your S3 bucket name
- `RUNPOD_PUBLIC_IP`: Your RunPod instance public IP

### Step 5: Configure RunPod Networking

**In RunPod Web Interface:**
1. Go to your pod details
2. Click "Edit" 
3. Add exposed ports:
   - `8080` (Airflow Web UI)
   - `5001` (MLflow Web UI)
4. Note your public IP address

### Step 6: Start Services
```bash
# Start all services
docker-compose -f docker-compose.runpod.yml up -d

# Check logs
docker-compose -f docker-compose.runpod.yml logs -f

# Check GPU access in containers
docker-compose -f docker-compose.runpod.yml exec airflow-scheduler nvidia-smi
```

## 🌐 Accessing Services

### Airflow Web UI
- URL: `http://<your-runpod-ip>:8080`
- Username: `airflow`
- Password: `airflow`

### MLflow Web UI
- URL: `http://<your-runpod-ip>:5001`
- No authentication required

## 🎮 GPU Configuration

### Verify GPU Access
```bash
# Check GPU in host
nvidia-smi

# Check GPU in Airflow scheduler (where ML tasks run)
docker-compose -f docker-compose.runpod.yml exec airflow-scheduler nvidia-smi

# Check CUDA availability in Python
docker-compose -f docker-compose.runpod.yml exec airflow-scheduler python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### GPU Memory Management
The configuration includes:
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` for better memory management
- `CUDA_VISIBLE_DEVICES=0` to use first GPU
- `NVIDIA_VISIBLE_DEVICES=all` for Docker access

## 🔧 Troubleshooting

### Common Issues

**1. "System has not been booted with systemd" Error**
This is normal in RunPod containers. Use our RunPod-specific setup scripts that don't rely on systemd.

**2. Docker Daemon Not Running**
```bash
# Start Docker daemon manually
sudo dockerd > /dev/null 2>&1 &
sleep 5

# Check if it's running
docker info
```

**3. Docker Permission Denied**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**4. GPU Not Accessible in Container**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# If that fails, you may need to configure NVIDIA Container Toolkit
```

**5. Port Access Issues**
- Ensure ports 8080 and 5001 are exposed in RunPod
- Check firewall settings: `sudo ufw status`

**6. Out of GPU Memory**
- Reduce batch size in `.env`: `BATCH_SIZE=8`
- Enable gradient checkpointing in training code
- Use mixed precision training

### Monitoring Commands
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor Docker containers
docker stats

# Check Airflow logs
docker-compose -f docker-compose.runpod.yml logs airflow-scheduler -f

# Check MLflow logs
docker-compose -f docker-compose.runpod.yml logs mlflow -f
```

## 🚦 Running Your ML Pipeline

### 1. Access Airflow UI
Navigate to `http://<your-runpod-ip>:8080`

### 2. Trigger DAG
- Find `ml_engagement_prediction_pipeline`
- Click the play button to trigger manually
- Monitor task progress in real-time

### 3. Monitor in MLflow
- Navigate to `http://<your-runpod-ip>:5001`
- Watch experiments and metrics in real-time
- View model artifacts and logs

### 4. Check GPU Utilization
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi
```

## 💡 Optimization Tips

### For Better Performance:
1. **Use larger batch sizes** if GPU memory allows
2. **Enable mixed precision** training (`fp16=True`)
3. **Use DataLoader with multiple workers**
4. **Pin memory** for faster data transfer

### For Cost Optimization:
1. **Stop pod when not training** to save costs
2. **Use spot instances** for non-critical workloads
3. **Monitor GPU utilization** to ensure efficient usage

## 🔄 Updating Your Code

When you update your code:
```bash
# Pull latest changes
git pull

# Rebuild and restart services
docker-compose -f docker-compose.runpod.yml down
docker-compose -f docker-compose.runpod.yml up -d --build
```

## 📊 Expected Behavior

With GPU acceleration, you should see:
- **Faster model training** (minutes instead of hours)
- **Higher GPU utilization** (70-90% during training)
- **MLflow tracking** with detailed metrics
- **Real-time progress** in Airflow UI

Your pipeline will now leverage the full power of RunPod's GPU infrastructure for efficient ML training and inference! 