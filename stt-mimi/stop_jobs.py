"""Stop SageMaker training jobs."""

import argparse
import boto3


def stop_job(job_name):
    """Stop a training job."""
    sagemaker = boto3.client('sagemaker')

    try:
        print(f"Stopping job: {job_name}...")
        sagemaker.stop_training_job(TrainingJobName=job_name)
        print(f"✅ Stop request sent for {job_name}")
        print(f"   Job will stop within a few minutes")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name', type=str, required=True, help='Job name to stop')
    args = parser.parse_args()

    stop_job(args.job_name)
