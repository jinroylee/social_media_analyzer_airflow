from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
import sys
import os
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv('/opt/airflow/.env')
    logging.info("Successfully loaded .env file")
except ImportError:
    logging.warning("python-dotenv not available, using system environment variables")
except Exception as e:
    logging.warning(f"Could not load .env file: {e}")

# Add your project root to Python path
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/modelfactory')

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
    description='ML pipeline for social media engagement prediction with S3 models',
    schedule_interval=None,  # Manual trigger for local development
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't run missed schedules
    max_active_runs=1,  # Only one instance running at a time
    tags=['ml', 'engagement-prediction', 's3'],
)

def check_local_environment(**context):
    """Check if running in local environment and setup paths"""
    try:
        logger.info("Checking environment setup...")
        
        # Check environment variables
        env = os.getenv('ENVIRONMENT', 'unknown')
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        logger.info(f"Environment: {env}")
        logger.info(f"Use S3 Model: {use_s3}")
        logger.info(f"Python path: {sys.path}")
        
        # Check local directories
        data_path = '/opt/airflow/data'
        models_path = '/opt/airflow/models'
        outputs_path = '/opt/airflow/outputs'
        
        for path in [data_path, models_path, outputs_path]:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory ready: {path}")
        
        # Check if modelfactory is available
        modelfactory_path = '/opt/airflow/modelfactory'
        if os.path.exists(modelfactory_path):
            logger.info(f"Modelfactory directory found: {modelfactory_path}")
            files = os.listdir(modelfactory_path)
            logger.info(f"Modelfactory contents: {files}")
        else:
            logger.warning(f"Modelfactory directory not found: {modelfactory_path}")
        
        # Test imports
        try:
            import modelfactory
            logger.info("Successfully imported modelfactory package")
        except ImportError as e:
            logger.warning(f"Cannot import modelfactory: {e}")
        
        # Check AWS credentials
        aws_key = os.getenv('AWS_ACCESS_KEY_ID', 'not_set')
        aws_region = os.getenv('AWS_REGION', 'not_set')
        s3_bucket = os.getenv('S3_BUCKET_NAME', 'not_set')
        logger.info(f"AWS Key: {aws_key[:10]}... (truncated)")
        logger.info(f"AWS Region: {aws_region}")
        logger.info(f"S3 Bucket: {s3_bucket}")
        
        logger.info("Environment check completed successfully")
        return "environment_check_success"
        
    except Exception as e:
        logger.error(f"Environment check failed: {str(e)}")
        raise

def preprocess_data(**context):
    """Task to preprocess data"""
    try:
        logger.info("Starting data preprocessing...")
        
        # Check if we're in S3 mode
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        if use_s3:
            logger.info("Running with S3 models - importing preprocessing module")
            try:
                # Import the preprocessing module
                from modelfactory.preprocess.preprocess import main as preprocess_main
                logger.info("Successfully imported preprocessing module")
                
                # Run the actual preprocessing
                logger.info("Starting preprocessing execution...")
                preprocess_main()
                logger.info("Data preprocessing completed successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import preprocessing module: {e}")
                logger.info("Available modules in modelfactory:")
                modelfactory_path = '/opt/airflow/modelfactory'
                if os.path.exists(modelfactory_path):
                    for root, dirs, files in os.walk(modelfactory_path):
                        for file in files:
                            if file.endswith('.py'):
                                logger.info(f"  {os.path.join(root, file)}")
                raise
            except Exception as e:
                logger.error(f"Preprocessing execution failed: {e}")
                raise
        else:
            logger.info("Running in local mode - using mock preprocessing")
            # Create mock preprocessing
            data_path = '/opt/airflow/data'
            processed_file = os.path.join(data_path, 'processed_data.txt')
            
            with open(processed_file, 'w') as f:
                f.write(f"Mock processed data - {datetime.now()}\n")
            
            logger.info(f"Mock preprocessing completed - output: {processed_file}")
        
        return "preprocessing_success"
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def train_model(**context):
    """Task to train the model"""
    try:
        logger.info("Starting model training...")
        
        # Check if we're in S3 mode
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        if use_s3:
            logger.info("Running with S3 models - importing training module")
            try:
                # Import the training module
                from modelfactory.train import main as train_main
                logger.info("Successfully imported training module")
                
                # Run the actual training
                logger.info("Starting training execution...")
                train_main()
                logger.info("Model training completed successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import training module: {e}")
                raise
            except Exception as e:
                logger.error(f"Training execution failed: {e}")
                raise
        else:
            logger.info("Running in local mode - using mock training")
            # Create mock training
            models_path = '/opt/airflow/models'
            model_file = os.path.join(models_path, 'trained_model.txt')
            
            with open(model_file, 'w') as f:
                f.write(f"Mock trained model - {datetime.now()}\n")
            
            logger.info(f"Mock training completed - output: {model_file}")
        
        return "training_success"
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def test_model(**context):
    """Task to test the model"""
    try:
        logger.info("Starting model testing...")
        
        # Check if we're in S3 mode
        use_s3 = os.getenv('USE_S3_MODEL', 'true').lower() == 'true'
        
        if use_s3:
            logger.info("Running with S3 models - importing testing module")
            try:
                # Import the testing module
                from modelfactory.test import main as test_main
                logger.info("Successfully imported testing module")
                
                # Run the actual testing
                logger.info("Starting testing execution...")
                test_main()
                logger.info("Model testing completed successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import testing module: {e}")
                raise
            except Exception as e:
                logger.error(f"Testing execution failed: {e}")
                raise
        else:
            logger.info("Running in local mode - using mock testing")
            # Create mock testing
            outputs_path = '/opt/airflow/outputs'
            results_file = os.path.join(outputs_path, 'test_results.txt')
            
            with open(results_file, 'w') as f:
                f.write(f"Mock test results - {datetime.now()}\n")
                f.write("Accuracy: 0.95\n")
                f.write("F1 Score: 0.92\n")
            
            logger.info(f"Mock testing completed - output: {results_file}")
        
        return "testing_success"
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

def notify_success(**context):
    """Send success notification"""
    logger.info("Pipeline completed successfully!")
    
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
        logger.info("Keeping output files for inspection")
        
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