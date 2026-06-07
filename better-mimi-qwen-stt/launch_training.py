import os
import netrc
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime

session = sagemaker.Session()
role = sagemaker.get_execution_role()

job_name = f"qwen-mimi-stt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

estimator = PyTorch(
    entry_point="model.py",
    source_dir=".",
    role=role,
    instance_count=1,
    instance_type="ml.g5.2xlarge",
    volume_size=450,
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312",
    base_job_name="qwen-mimi-stt",
    environment={
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "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") or netrc.netrc().authenticators("api.wandb.ai")[2],
        "WANDB_PROJECT": "qwen-mimi-better",
    },
    max_run=172800,
    keep_alive_period_in_seconds=1800,
    disable_profiler=True,
)

estimator.fit(job_name=job_name, wait=False)
print(f"Training job launched: {job_name}")


# qwen-mimi-stt-20260606-073208

# aws sagemaker stop-training-job \
#     --training-job-name qwen-mimi-stt-20260606-073208