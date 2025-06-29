#!/bin/bash

# Local Airflow Setup Script
echo "🚀 Setting up Airflow for local development..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Set proper permissions for Airflow
echo "📁 Setting up directories and permissions..."
mkdir -p ./dags ./logs ./plugins ./data ./models ./outputs ./mlruns
chmod -R 755 ./dags ./logs ./plugins ./data ./models ./outputs ./mlruns

# Set AIRFLOW_UID for proper permissions
export AIRFLOW_UID=$(id -u)
echo "Setting AIRFLOW_UID to $AIRFLOW_UID"

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.airflow.yml down --volumes --remove-orphans

echo "🐳 Building custom Airflow Docker image..."
if docker-compose -f docker-compose.airflow.yml build; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Docker build failed. Please check the error messages above."
    exit 1
fi

echo "🗄️ Initializing Airflow database..."
if docker-compose -f docker-compose.airflow.yml up airflow-init; then
    echo "✅ Airflow database initialized successfully!"
else
    echo "❌ Airflow initialization failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "✅ Setup complete! You can now start Airflow with:"
echo "   docker-compose -f docker-compose.airflow.yml up"
echo ""
echo "📊 Access points:"
echo "   Airflow UI: http://localhost:8080 (airflow/airflow)"
echo "   MLflow UI: http://localhost:5000"
echo ""
echo "🔧 To run the ML pipeline:"
echo "   1. Go to http://localhost:8080"
echo "   2. Find the 'ml_engagement_prediction_pipeline' DAG"
echo "   3. Turn it on and trigger it manually" 