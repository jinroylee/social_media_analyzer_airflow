import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import CLIPProcessor, CLIPTokenizer
from modelfactory.models.clip_regressor import CLIPEngagementRegressor
from modelfactory.utils.engagement_dataset import EngagementDataset
from modelfactory.utils.mlflow_utils import MLflowTracker, create_experiment_config
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import pickle
import math
from PIL import Image
import os
from datetime import datetime
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from io import BytesIO

# Load environment variables
load_dotenv()

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Training configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
LORA_LEARNING_RATE = 1e-3  # Higher learning rate for LoRA parameters
LORA_RANK = 8
USE_LORA = True

def load_pkl_from_s3(bucket_name, key):
    """Load pickle data from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        pkl_data = pickle.loads(response['Body'].read())
        return pkl_data
    except ClientError as e:
        print(f"Error loading pickle from S3 {key}: {e}")
        raise

def save_model_to_s3(model_state_dict, bucket_name, key):
    """Save PyTorch model state dict to S3"""
    try:
        # Serialize model to bytes
        model_buffer = BytesIO()
        torch.save(model_state_dict, model_buffer)
        model_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=model_buffer.getvalue()
        )
        print(f"Successfully saved model to s3://{bucket_name}/{key}")
        return True
    except Exception as e:
        print(f"Error saving model to S3 {key}: {e}")
        return False

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    criterion = nn.HuberLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(pixel_values, input_ids, attention_mask, sentiment)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    mae = mean_absolute_error(targets, predictions)
    correlation, _ = spearmanr(targets, predictions)
    r2 = r2_score(targets, predictions)
    avg_loss = total_loss / len(dataloader)
    
    return mae, correlation, r2, avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from S3 bucket: {S3_BUCKET_NAME}")
    
    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(
        experiment_name="social_media_engagement_prediction",
        tracking_uri=None,  # Uses local file store
        artifact_location=None
    )
    
    # Create run name with timestamp
    run_name = f"CLIP_LoRA_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start MLflow run
    with mlflow_tracker.start_run(run_name=run_name) as run:
        # Set tags for the run
        mlflow_tracker.set_tags({
            "model_type": "CLIP_LoRA",
            "task": "engagement_prediction",
            "framework": "pytorch",
            "device": str(device)
        })
        
        # Create experiment configuration
        config = create_experiment_config(
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lora_learning_rate=LORA_LEARNING_RATE,
            lora_rank=LORA_RANK,
            use_lora=USE_LORA
        )
        
        # Log hyperparameters
        mlflow_tracker.log_hyperparameters(config)
        
        # Load processors
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Load training data from S3
        data = load_pkl_from_s3(S3_BUCKET_NAME, "processed/train.pkl")
        
        print(f"Loaded {len(data)} training samples from S3")
        
        # Create dataset
        dataset = EngagementDataset(data, processor, tokenizer)
        
        # Split training data into train/validation (80%/20% of training data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"Train size: {train_size}, Validation size: {val_size}")
        
        # Log dataset information
        mlflow_tracker.log_dataset_info(train_size, val_size)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model with LoRA
        model = CLIPEngagementRegressor(use_lora=USE_LORA, lora_rank=LORA_RANK).to(device)
        
        # Log model architecture
        mlflow_tracker.log_model_architecture(model)
        
        # Setup optimizers with different learning rates for LoRA and non-LoRA parameters
        lora_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': LEARNING_RATE},
            {'params': lora_params, 'lr': LORA_LEARNING_RATE}
        ])
        
        criterion = nn.HuberLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_val_mae = float('inf')
        best_val_r2 = -float('inf')
        
        print("Starting training with LoRA...")
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0
            train_predictions = []
            train_targets = []
            
            for batch_idx, batch in enumerate(train_loader):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment = batch['sentiment'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(pixel_values, input_ids, attention_mask, sentiment)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.extend(outputs.squeeze().detach().cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Calculate training metrics
            train_mae = mean_absolute_error(train_targets, train_predictions)
            train_corr, _ = spearmanr(train_targets, train_predictions)
            train_r2 = r2_score(train_targets, train_predictions)
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_mae, val_corr, val_r2, val_loss = evaluate(model, val_loader, device)
            scheduler.step(val_loss)
            
            # Log metrics to MLflow
            epoch_metrics = {
                'train_loss': avg_train_loss,
                'train_mae': train_mae,
                'train_correlation': train_corr,
                'train_r2': train_r2,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_correlation': val_corr,
                'val_r2': val_r2,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'lora_learning_rate': optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None
            }
            
            mlflow_tracker.log_metrics(epoch_metrics, step=epoch)
            
            print(f'Epoch {epoch+1}/{EPOCHS}:')
            print(f'  Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, Corr: {train_corr:.4f}, R²: {train_r2:.4f}')
            print(f'  Val - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}, R²: {val_r2:.4f}')
            
            # Save best model based on validation MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_r2 = val_r2
                
                # Save model to S3
                model_key = "models/best_model_lora.pth"
                if save_model_to_s3(model.state_dict(), S3_BUCKET_NAME, model_key):
                    print(f'  New best model saved to S3! MAE: {val_mae:.4f}, R²: {val_r2:.4f}')
                
                # Log checkpoint to MLflow
                mlflow_tracker.log_model_checkpoint(model, f"best_model_epoch_{epoch+1}.pth")
                
                # Log best metrics
                mlflow_tracker.log_metrics({
                    'best_val_mae': best_val_mae,
                    'best_val_r2': best_val_r2,
                    'best_epoch': epoch + 1
                })
        
        # Log final model to MLflow
        print("Logging final model to MLflow...")
        
        # Create an example input for model signature
        sample_batch = next(iter(val_loader))
        example_input = {
            'pixel_values': sample_batch['pixel_values'][:1].to(device),
            'input_ids': sample_batch['input_ids'][:1].to(device),
            'attention_mask': sample_batch['attention_mask'][:1].to(device),
            'sentiment': sample_batch['sentiment'][:1].to(device)
        }
        
        # Log the final trained model
        mlflow_tracker.log_pytorch_model(
            model=model,
            input_example=example_input,
            model_path="final_model"
        )
        
        # Log training summary
        summary_text = f"""
        Training Summary:
        ================
        Best Validation MAE: {best_val_mae:.4f}
        Best Validation R²: {best_val_r2:.4f}
        Total Epochs: {EPOCHS}
        Training Samples: {train_size}
        Validation Samples: {val_size}
        Device: {device}
        Model: CLIP + LoRA (rank={LORA_RANK})
        Model saved to: s3://{S3_BUCKET_NAME}/models/best_model_lora.pth
        """
        
        mlflow_tracker.log_text_artifact(summary_text, "training_summary.txt")
        
        print(f'\nTraining completed!')
        print(f'Best validation MAE: {best_val_mae:.4f}')
        print(f'Best validation R²: {best_val_r2:.4f}')
        print(f'Model saved to S3: s3://{S3_BUCKET_NAME}/models/best_model_lora.pth')
        print(f'MLflow run ID: {run.info.run_id}')

if __name__ == "__main__":
    main()
