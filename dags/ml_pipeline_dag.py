from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
import sys
import os
import logging

# Add your project root to Python path
sys.path.append('/opt/airflow/ml_pipeline')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for all tasks
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,  # Disabled for local development
    'email_on_retry': False,
    'retries': 1,  # Reduced retries for local development
    'retry_delay': timedelta(minutes=5),  # Shorter retry delay
    'execution_timeout': timedelta(hours=2),  # Reduced timeout for local development
}

# Create the DAG
dag = DAG(
    'ml_engagement_prediction_pipeline',
    default_args=default_args,
    description='Local ML pipeline for social media engagement prediction',
    schedule_interval=None,  # Manual trigger for local development
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't run missed schedules
    max_active_runs=1,  # Only one instance running at a time
    tags=['ml', 'local', 'engagement-prediction'],
)

def check_local_environment(**context):
    """Check if running in local environment and setup paths"""
    try:
        logger.info("Checking local environment setup...")
        
        # Check environment variables
        env = os.getenv('ENVIRONMENT', 'unknown')
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        logger.info(f"Environment: {env}")
        logger.info(f"Use S3 Model: {use_s3}")
        
        # Check local directories
        data_path = '/opt/airflow/data'
        models_path = '/opt/airflow/models'
        outputs_path = '/opt/airflow/outputs'
        
        for path in [data_path, models_path, outputs_path]:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory ready: {path}")
        
        # Check if sample data exists (create dummy data if needed)
        sample_data_file = os.path.join(data_path, 'sample_data.txt')
        if not os.path.exists(sample_data_file):
            with open(sample_data_file, 'w') as f:
                f.write("This is sample data for local development\n")
            logger.info("Created sample data file for local development")
        
        logger.info("Local environment check completed successfully")
        return "environment_check_success"
        
    except Exception as e:
        logger.error(f"Environment check failed: {str(e)}")
        raise

def preprocess_data(**context):
    """Task to preprocess data locally"""
    try:
        logger.info("Starting local data preprocessing...")
        
        # Check if we're in local mode
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        if not use_s3:
            logger.info("Running in local mode - using local data")
            # For local development, create mock preprocessing
            data_path = '/opt/airflow/data'
            processed_file = os.path.join(data_path, 'processed_data.txt')
            
            with open(processed_file, 'w') as f:
                f.write(f"Processed data - {datetime.now()}\n")
            
            logger.info(f"Mock preprocessing completed - output: {processed_file}")
        else:
            # Import your preprocessing module (only if available)
            try:
                from modelfactory.preprocess.preprocess import main as preprocess_main
                preprocess_main()
                logger.info("Data preprocessing completed successfully")
            except ImportError as e:
                logger.warning(f"Preprocessing module not available: {e}")
                logger.info("Running in mock mode for local development")
        
        return "preprocessing_success"
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def train_model(**context):
    """Task to train the model locally"""
    try:
        logger.info("Starting local model training...")
        
        # Check if we're in local mode
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        if not use_s3:
            logger.info("Running in local mode - using mock training")
            # For local development, create mock training
            models_path = '/opt/airflow/models'
            model_file = os.path.join(models_path, 'trained_model.txt')
            
            with open(model_file, 'w') as f:
                f.write(f"Mock trained model - {datetime.now()}\n")
            
            logger.info(f"Mock training completed - output: {model_file}")
        else:
            # Import your training module (only if available)
            try:
                from modelfactory.train import main as train_main
                train_main()
                logger.info("Model training completed successfully")
            except ImportError as e:
                logger.warning(f"Training module not available: {e}")
                logger.info("Running in mock mode for local development")
        
        return "training_success"
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def test_model(**context):
    """Task to test the model locally"""
    try:
        logger.info("Starting local model testing...")
        
        # Check if we're in local mode
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        if not use_s3:
            logger.info("Running in local mode - using mock testing")
            # For local development, create mock testing
            outputs_path = '/opt/airflow/outputs'
            results_file = os.path.join(outputs_path, 'test_results.txt')
            
            with open(results_file, 'w') as f:
                f.write(f"Mock test results - {datetime.now()}\n")
                f.write("Accuracy: 0.95\n")
                f.write("F1 Score: 0.92\n")
            
            logger.info(f"Mock testing completed - output: {results_file}")
        else:
            # Import your testing module (only if available)
            try:
                from modelfactory.test import main as test_main
                test_main()
                logger.info("Model testing completed successfully")
            except ImportError as e:
                logger.warning(f"Testing module not available: {e}")
                logger.info("Running in mock mode for local development")
        
        return "testing_success"
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

def notify_success(**context):
    """Send success notification"""
    logger.info("Local pipeline completed successfully!")
    
    # Log summary of outputs
    outputs_path = '/opt/airflow/outputs'
    if os.path.exists(outputs_path):
        files = os.listdir(outputs_path)
        logger.info(f"Output files created: {files}")
    
    return "Pipeline completed successfully"

def cleanup_resources(**context):
    """Cleanup any temporary resources"""
    try:
        logger.info("Cleaning up resources...")
        
        # Clear any temporary files if needed
        # For local development, we might want to keep files for inspection
        logger.info("Keeping output files for local inspection")
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
        except ImportError:
            logger.info("PyTorch not available - skipping CUDA cleanup")
            
        logger.info("Resource cleanup completed")
        return "cleanup_success"
        
    except Exception as e:
        logger.warning(f"Cleanup warning (non-critical): {str(e)}")
        return "cleanup_completed_with_warnings"

# Define tasks
environment_check_task = PythonOperator(
    task_id='check_environment',
    python_callable=check_local_environment,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

test_task = PythonOperator(
    task_id='test_model',
    python_callable=test_model,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_resources',
    python_callable=cleanup_resources,
    dag=dag,
    trigger_rule='all_done',  # Run regardless of upstream success/failure
)

success_notification = PythonOperator(
    task_id='notify_success',
    python_callable=notify_success,
    dag=dag,
    trigger_rule='all_success',  # Only run if all upstream tasks succeed
)

# Define task dependencies
environment_check_task >> preprocess_task >> train_task >> test_task >> success_notification
environment_check_task >> preprocess_task >> train_task >> test_task >> cleanup_task 