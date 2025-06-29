# Local Airflow ML Pipeline Setup Guide

This guide will help you run your Airflow ML pipeline locally using Docker Compose instead of AWS.

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB of RAM available for Docker
- At least 10GB of free disk space

## Quick Start

### 1. Run the Setup Script
```bash
./setup_local_airflow.sh
```

This script will:
- Create necessary directories
- Set proper permissions
- Build the custom Docker image
- Initialize the Airflow database

### 2. Start Airflow Services
```bash
docker-compose -f docker-compose.airflow.yml up
```

Or run in detached mode:
```bash
docker-compose -f docker-compose.airflow.yml up -d
```

### 3. Access the Airflow UI
- Open your browser and go to: http://localhost:8080
- Login with: `airflow` / `airflow`

### 4. Run Your ML Pipeline
1. In the Airflow UI, find the `ml_engagement_prediction_pipeline` DAG
2. Toggle it ON (if it's paused)
3. Click "Trigger DAG" to run it manually

## Services and Ports

| Service | URL | Purpose |
|---------|-----|---------|
| Airflow Webserver | http://localhost:8080 | Main UI for managing pipelines |
| MLflow | http://localhost:5000 | Experiment tracking (optional) |
| PostgreSQL | localhost:5432 | Airflow metadata database |

## Local Development Features

### Mock Mode
The pipeline is configured to run in "mock mode" for local development:
- Uses local file system instead of S3
- Creates sample data and models for testing
- Shorter timeouts and fewer retries
- Disabled email notifications

### Directory Structure
```
├── data/           # Input data files
├── models/         # Trained models
├── outputs/        # Pipeline outputs and results
├── logs/           # Airflow logs
├── dags/           # Airflow DAGs
├── plugins/        # Airflow plugins
└── mlruns/         # MLflow experiment tracking
```

### Environment Variables
Key variables for local development (in `.env`):
- `ENVIRONMENT=local` - Enables local development mode
- `USE_S3_MODEL=false` - Disables S3 integration
- `DEBUG=true` - Enables debug logging

## Managing the Pipeline

### Start Services
```bash
docker-compose -f docker-compose.airflow.yml up -d
```

### Stop Services
```bash
docker-compose -f docker-compose.airflow.yml down
```

### View Logs
```bash
# All services
docker-compose -f docker-compose.airflow.yml logs

# Specific service
docker-compose -f docker-compose.airflow.yml logs airflow-scheduler
```

### Restart Services
```bash
docker-compose -f docker-compose.airflow.yml restart
```

### Rebuild After Changes
```bash
docker-compose -f docker-compose.airflow.yml down
docker-compose -f docker-compose.airflow.yml build
docker-compose -f docker-compose.airflow.yml up
```

## Pipeline Tasks

The ML pipeline includes these tasks:
1. **check_environment** - Validates local setup
2. **preprocess_data** - Data preprocessing (mock mode creates sample files)
3. **train_model** - Model training (mock mode creates dummy model)
4. **test_model** - Model testing (mock mode creates test results)
5. **notify_success** - Success notification
6. **cleanup_resources** - Resource cleanup

## Troubleshooting

### Common Issues

**Permission Errors:**
```bash
# Fix permissions
sudo chown -R $(id -u):$(id -g) ./logs ./data ./models ./outputs
```

**Port Already in Use:**
```bash
# Check what's using port 8080
lsof -i :8080
# Kill the process or change the port in docker-compose.yml
```

**Out of Memory:**
```bash
# Increase Docker memory limit to at least 4GB
# Docker Desktop: Settings > Resources > Memory
```

**Container Build Fails:**
```bash
# Clean up and rebuild
docker-compose -f docker-compose.airflow.yml down --volumes
docker system prune -f
docker-compose -f docker-compose.airflow.yml build --no-cache
```

### Logs and Debugging

**View Airflow Logs:**
- Airflow UI: Admin > Logs
- Command line: `docker-compose -f docker-compose.airflow.yml logs airflow-scheduler`

**View Task Logs:**
- Click on a task in the DAG graph
- Click "Logs" to see detailed execution logs

**Database Issues:**
```bash
# Reset the database
docker-compose -f docker-compose.airflow.yml down --volumes
docker-compose -f docker-compose.airflow.yml up airflow-init
```

## Customization

### Adding Real ML Code
1. Replace mock functions in `dags/ml_pipeline_dag.py`
2. Ensure your `modelfactory` modules are properly structured
3. Add any additional dependencies to `requirements.txt`
4. Rebuild the Docker image

### Changing Schedule
- Edit `schedule_interval` in the DAG definition
- Currently set to `None` (manual trigger only)
- Example: `'0 2 * * 1'` for weekly runs

### Adding More Tasks
1. Define new Python functions
2. Create PythonOperator tasks
3. Add to task dependencies

## Production Considerations

When moving to production:
1. Change `USE_S3_MODEL=true` in `.env`
2. Update AWS credentials
3. Set proper `schedule_interval`
4. Enable email notifications
5. Increase resource limits
6. Use proper secrets management

## Support

If you encounter issues:
1. Check the logs in Airflow UI
2. Verify Docker has enough resources
3. Ensure all required directories exist
4. Check file permissions 