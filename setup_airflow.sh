#!/bin/bash

# Setup script for Apache Airflow ML Pipeline
# This script sets up Airflow for your ML pipeline automation

set -e

echo "🚀 Setting up Apache Airflow for ML Pipeline..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p dags logs plugins mlruns

# Set proper permissions
echo "🔐 Setting up permissions..."
export AIRFLOW_UID=$(id -u)
echo "AIRFLOW_UID=$AIRFLOW_UID" > .env

# Add environment variables for AWS (you'll need to fill these in)
echo "AWS_ACCESS_KEY_ID=your-aws-access-key" >> .env
echo "AWS_SECRET_ACCESS_KEY=your-aws-secret-key" >> .env
echo "S3_BUCKET_NAME=your-s3-bucket-name" >> .env
echo "AWS_REGION=us-east-1" >> .env
echo "_AIRFLOW_WWW_USER_USERNAME=admin" >> .env
echo "_AIRFLOW_WWW_USER_PASSWORD=admin123" >> .env

echo "⚠️  IMPORTANT: Please update the .env file with your actual AWS credentials!"

# Move DAG file to dags directory
echo "📋 Moving DAG file..."
cp dags/ml_pipeline_dag.py dags/ 2>/dev/null || echo "DAG file already in place"

# Initialize Airflow
echo "🔄 Initializing Airflow database..."
docker-compose -f docker-compose.airflow.yml up airflow-init

# Install Python dependencies in Airflow container
echo "📦 Installing Python dependencies..."
docker-compose -f docker-compose.airflow.yml run --rm airflow-webserver pip install -r /opt/airflow/airflow_requirements.txt

echo "✅ Airflow setup completed!"
echo ""
echo "🎯 Next steps:"
echo "1. Update .env file with your AWS credentials"
echo "2. Run: docker-compose -f docker-compose.airflow.yml up -d"
echo "3. Access Airflow UI at: http://localhost:8080"
echo "4. Login with username: admin, password: admin123"
echo "5. Enable your DAG: ml_engagement_prediction_pipeline"
echo ""
echo "📚 For production deployment on AWS, consider using AWS MWAA (Managed Workflows for Apache Airflow)" 