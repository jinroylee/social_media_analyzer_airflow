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
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=6),  # Max 6 hours per task
}

# Create the DAG
dag = DAG(
    'ml_engagement_prediction_pipeline',
    default_args=default_args,
    description='Weekly ML pipeline for social media engagement prediction',
    schedule_interval='0 2 * * 1',  # Run every Monday at 2:00 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't run missed schedules
    max_active_runs=1,  # Only one instance running at a time
    tags=['ml', 'weekly', 'engagement-prediction'],
)

def preprocess_data(**context):
    """Task to preprocess data from S3"""
    try:
        logger.info("Starting data preprocessing...")
        
        # Import your preprocessing module
        from modelfactory.preprocess.preprocess import main as preprocess_main
        
        # Run preprocessing
        preprocess_main()
        
        logger.info("Data preprocessing completed successfully")
        return "preprocessing_success"
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def train_model(**context):
    """Task to train the model"""
    try:
        logger.info("Starting model training...")
        
        # Import your training module
        from modelfactory.train import main as train_main
        
        # Run training
        train_main()
        
        logger.info("Model training completed successfully")
        return "training_success"
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def test_model(**context):
    """Task to test the model"""
    try:
        logger.info("Starting model testing...")
        
        # Import your testing module
        from modelfactory.test import main as test_main
        
        # Run testing
        test_main()
        
        logger.info("Model testing completed successfully")
        return "testing_success"
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

def notify_success(**context):
    """Send success notification"""
    logger.info("Pipeline completed successfully!")
    return "Pipeline completed successfully"

def cleanup_resources(**context):
    """Cleanup any temporary resources"""
    try:
        logger.info("Cleaning up resources...")
        # Add any cleanup logic here if needed
        # For example, clearing temporary files, freeing GPU memory, etc.
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
            
        logger.info("Resource cleanup completed")
        return "cleanup_success"
        
    except Exception as e:
        logger.warning(f"Cleanup warning (non-critical): {str(e)}")
        return "cleanup_completed_with_warnings"

# Define tasks
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
    pool='ml_pool',  # We'll create this pool to limit concurrent ML tasks
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
    pool='ml_pool',
)

test_task = PythonOperator(
    task_id='test_model',
    python_callable=test_model,
    dag=dag,
    pool='ml_pool',
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

# Email notification for failures
failure_notification = EmailOperator(
    task_id='notify_failure',
    to=Variable.get("alert_email", default_var="admin@yourcompany.com"),
    subject='ML Pipeline Failed - {{ ds }}',
    html_content="""
    <h3>ML Pipeline Failure Alert</h3>
    <p>The weekly ML engagement prediction pipeline failed on {{ ds }}.</p>
    <p><strong>DAG:</strong> {{ dag.dag_id }}</p>
    <p><strong>Task:</strong> {{ task.task_id }}</p>
    <p><strong>Execution Date:</strong> {{ ds }}</p>
    <p><strong>Log Url:</strong> {{ task_instance.log_url }}</p>
    <p>Please check the logs for more details.</p>
    """,
    dag=dag,
    trigger_rule='one_failed',
)

# Define task dependencies
preprocess_task >> train_task >> test_task >> success_notification
preprocess_task >> train_task >> test_task >> cleanup_task
[preprocess_task, train_task, test_task] >> failure_notification 