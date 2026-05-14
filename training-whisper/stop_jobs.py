import argparse
import boto3

sm = boto3.client("sagemaker")
args = argparse.ArgumentParser()
args.add_argument("--job-name", type=str, required=True, help="Name of the SageMaker training job to stop")
args = args.parse_args()

response = sm.stop_training_job(
    TrainingJobName=args.job_name
)       

print(response)