import sagemaker
from sagemaker.processing import FrameworkProcessor
from sagemaker.pytorch import PyTorchProcessor


session = sagemaker.Session()
role = sagemaker.get_execution_role()

processor = PyTorchProcessor(
    framework_version="2.1.0",
    py_version="py310",
    role=role,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    base_job_name="asr-dataset-upload",
)

processor.run(
    code="dataset_s3.py",
    source_dir=".",
    dependencies=["requirements.txt"],
    wait=False,
)

print(f"SageMaker processing job launched") 


# aws sagemaker describe-processing-job \
#     --processing-job-name asr-dataset-upload-2026-06-05-18-07-22-948
