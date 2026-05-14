import sagemaker
from sagemaker.pytorch import PyTorch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(script_dir, "requirements.txt")

# Load environment variables from .env file if it exists
env_file = os.path.join(script_dir, ".env")
if os.path.exists(env_file):
    print(f"Loading environment variables from {env_file}")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes from value if present
                value = value.strip('"').strip("'")
                os.environ[key] = value
                print(f"  ✓ Loaded {key}")
else:
    print("⚠ No .env file found - will use system environment variables")

role = sagemaker.get_execution_role()

configs = [
    {
        "name": "mini-whisper-small",
        "encoder_d_model": 512,
        "encoder_num_layers": 8,
        "decoder_d_model": 512,
        "decoder_num_layers": 6,
        "batch_size": 16,
        "lr": 1e-4,
        "max_steps": 150000,  # ~3 epochs (820k samples / 16 batch = 51,250 steps/epoch)
        "warmup_steps": 2000,
        "eval_every": 2000,
    },
    {
        "name": "mini-whisper-medium",
        "encoder_d_model": 768,
        "encoder_num_layers": 12,
        "decoder_d_model": 768,
        "decoder_num_layers": 8,
        "batch_size": 8,
        "lr": 5e-5,
        "max_steps": 300000,  # ~3 epochs (820k samples / 8 batch = 102,500 steps/epoch)
        "warmup_steps": 4000,
        "eval_every": 4000,
    },
    # {
    #     "name": "mini-whisper-large",
    #     "encoder_d_model": 768,
    #     "encoder_num_layers": 12,
    #     "decoder_d_model": 768,
    #     "decoder_num_layers": 12,
    #     "batch_size": 4,
    #     "lr": 3e-5,
    #     "max_steps": 600000,  # ~3 epochs (820k samples / 4 batch = 205,000 steps/epoch)
    #     "warmup_steps": 8000,
    #     "eval_every": 8000,
    # }
]

launched_jobs = []

for cfg in configs:
    print(f"\nLaunching experiment: {cfg['name']}")
    print(f"Config: {cfg}")

    estimator = PyTorch(
        entry_point="mini-whisper-train.py",
        source_dir=script_dir,  
        role=role,
        instance_count=1,
        instance_type="ml.g5.xlarge",  
        framework_version="2.3.0",     
        py_version="py311",            
        hyperparameters=cfg,
        output_path=f"s3://home-ml/experiments/{cfg['name']}",
        dependencies=[requirements_path],  
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_TOKEN", os.environ.get("WANDB_API_KEY", "")),  # Try WANDB_TOKEN first, fallback to WANDB_API_KEY
            "WANDB_PROJECT": "mini-whisper",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),  # HuggingFace token for gated datasets
        },
        max_run=4 * 24 * 3600,  
        keep_alive_period_in_seconds=1800,  
        use_spot_instances=False,  
    )

    
    estimator.fit(wait=False)
    job_name = estimator.latest_training_job.name
    launched_jobs.append(job_name)
    print(f"✓ Job launched: {job_name}")

print("\n" + "="*80)
print("All jobs launched!")
print("="*80)
print("\nJob names:")
for job in launched_jobs:
    print(f"  - {job}")

