"""Check status of SageMaker training jobs."""

import argparse
import boto3
from datetime import datetime


def check_job(job_name=None):
    """Check status of training job(s)."""
    sagemaker = boto3.client('sagemaker')

    if job_name:
        # Check specific job
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            print_job_status(response)
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # List recent jobs
        response = sagemaker.list_training_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=10,
            NameContains='mimi-qwen-stt'
        )

        if not response['TrainingJobSummaries']:
            print("No mimi-qwen-stt jobs found")
            return

        print(f"\n{'='*80}")
        print(f"Recent Mimi+Qwen STT Jobs")
        print(f"{'='*80}")

        for job in response['TrainingJobSummaries']:
            name = job['TrainingJobName']
            status = job['TrainingJobStatus']
            created = job['CreationTime']

            status_emoji = {
                'InProgress': '🔄',
                'Completed': '✅',
                'Failed': '❌',
                'Stopping': '⏸️',
                'Stopped': '⏹️',
            }.get(status, '❓')

            print(f"\n{status_emoji} {name}")
            print(f"   Status: {status}")
            print(f"   Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")

            if status == 'InProgress':
                # Get more details
                detail = sagemaker.describe_training_job(TrainingJobName=name)
                if 'SecondaryStatusTransitions' in detail:
                    latest = detail['SecondaryStatusTransitions'][-1]
                    print(f"   Progress: {latest['Status']}")

        print(f"{'='*80}\n")


def print_job_status(job_info):
    """Print detailed job status."""
    name = job_info['TrainingJobName']
    status = job_info['TrainingJobStatus']
    created = job_info['CreationTime']

    print(f"\n{'='*80}")
    print(f"Job: {name}")
    print(f"{'='*80}")
    print(f"Status: {status}")
    print(f"Created: {created}")

    if 'TrainingStartTime' in job_info:
        print(f"Started: {job_info['TrainingStartTime']}")

    if 'TrainingEndTime' in job_info:
        print(f"Ended: {job_info['TrainingEndTime']}")
        duration = (job_info['TrainingEndTime'] - job_info['TrainingStartTime']).total_seconds() / 3600
        print(f"Duration: {duration:.2f} hours")

    if 'SecondaryStatus' in job_info:
        print(f"Secondary Status: {job_info['SecondaryStatus']}")

    if 'FailureReason' in job_info:
        print(f"❌ Failure Reason: {job_info['FailureReason']}")

    # Hyperparameters
    if 'HyperParameters' in job_info:
        print(f"\nHyperparameters:")
        for k, v in job_info['HyperParameters'].items():
            print(f"  {k}: {v}")

    # Resources
    if 'ResourceConfig' in job_info:
        rc = job_info['ResourceConfig']
        print(f"\nResources:")
        print(f"  Instance: {rc['InstanceType']} x {rc['InstanceCount']}")
        print(f"  Volume: {rc['VolumeSizeInGB']} GB")

    # Output
    if 'OutputDataConfig' in job_info:
        print(f"\nOutput: {job_info['OutputDataConfig']['S3OutputPath']}")

    if 'CheckpointConfig' in job_info:
        print(f"Checkpoints: {job_info['CheckpointConfig']['S3Uri']}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name', type=str, help='Specific job name to check')
    args = parser.parse_args()

    check_job(args.job_name)
