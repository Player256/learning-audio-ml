"""
Launch training jobs on SageMaker with different configurations
"""

import os
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime


def launch_training(
    run_name,
    model_name="Qwen/Qwen3-0.6B",
    hidden_size=896,
    num_layers=24,
    num_heads=14,
    num_kv_heads=2,
    intermediate_size=4864,
    batch_size=4,
    gradient_accumulation_steps=4,
    lr=1e-4,
    max_steps=50000,
    instance_type="ml.g5.xlarge",
    instance_count=1,
):
    """Launch a SageMaker training job."""

    # SageMaker session
    role = sagemaker.get_execution_role()
    session = sagemaker.Session()

    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"mimi-qwen-stt-{timestamp}" if run_name is None else run_name

    # Hyperparameters
    hyperparameters = {
        'model_name': model_name,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'intermediate_size': intermediate_size,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'lr': lr,
        'max_steps': max_steps,
        'warmup_steps': max_steps // 20,  # 5% warmup
        'run_name': run_name,
        's3_data_path': 's3://home-ml/stt_datasets',
        'log_interval': 10,
        'eval_interval': 500,
        'save_interval': 2000,
    }

    # Get W&B API key
    wandb_api_key = os.environ.get('WANDB_API_KEY', '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')
    if wandb_api_key:
        print("✅ W&B API key found")
    else:
        print("⚠️  W&B API key not found - logging will be skipped")

    # Estimator (use image_uri directly to bypass SDK version validation)
    estimator = PyTorch(
        entry_point='train_entry.py',
        source_dir='.',
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311',
        hyperparameters=hyperparameters,
        environment={
            'WANDB_API_KEY': wandb_api_key,
            'TRANSFORMERS_OFFLINE': '0',
            'HF_HUB_OFFLINE': '0',
        },
        output_path=f's3://home-ml/experiments/mimi-qwen-stt/{run_name}/',
        checkpoint_s3_uri=f's3://home-ml/checkpoints/mimi-qwen-stt/{run_name}/',
        max_run=4 * 24 * 60 * 60,  
        volume_size=250,  
        use_spot_instances=False,
    )

    print(f"\n{'='*60}")
    print(f"Launching job: {job_name}")
    print(f"Instance: {instance_type} x {instance_count}")
    print(f"Model: {model_name}")
    print(f"Hidden size: {hidden_size}, Layers: {num_layers}")
    print(f"Batch: {batch_size}, Grad accum: {gradient_accumulation_steps}")
    print(f"Effective batch: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {lr}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    estimator.fit(wait=False, job_name=job_name)

    print(f"✅ Job launched: {job_name}")
    print(f"📁 Output: s3://home-ml/experiments/mimi-qwen-stt/{run_name}/")

    return job_name


if __name__ == "__main__":
    print("Launching Mimi + Qwen3-0.6B STT training...\n")
    
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d-%H%M%S")
    
    job = launch_training(
        run_name=f"qwen3-mimi-stt-{ist_time}",
        model_name="Qwen/Qwen3-0.6B",
        hidden_size=1024,               
        num_layers=28,                 
        num_heads=16,                  
        num_kv_heads=8,                
        intermediate_size=3072,        
        batch_size=4,
        gradient_accumulation_steps=8,
        lr=3e-5,
        max_steps=50000,
        instance_type="ml.g5.xlarge",  
    )

    print(f"\n{'='*60}")
    print(f"Job launched: {job}")
    print(f"{'='*60}\n")

    print("Monitor with:")
    print(f"  python check_jobs.py --job-name={job}")
    print("\nStop with:")
    print(f"  python stop_jobs.py --job-name={job}")
