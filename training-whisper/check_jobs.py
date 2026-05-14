import argparse
import boto3

sm = boto3.client("sagemaker")
logs_client = boto3.client("logs")


# args = argparse.ArgumentParser()
# args.add_argument("--job-name", type=str, required=True, help="Name of the SageMaker training job to check")
# args = args.parse_args()

jobs = ["pytorch-training-2026-05-14-13-08-10-170", "pytorch-training-2026-05-14-13-08-11-004"]

for job_name in jobs:
    response = sm.describe_training_job(
        TrainingJobName=job_name
    )
    print("\n\n\n")
    print("="*80)
    print(response)
    print("="*80)

    print("="*80)
    log_group = "/aws/sagemaker/TrainingJobs"

    streams = logs_client.describe_log_streams(
        logGroupName=log_group,
        logStreamNamePrefix=job_name
    )

    for stream in streams["logStreams"]:
        log_stream = stream["logStreamName"]

        print(f"\n=== {log_stream} ===")

        events = logs_client.get_log_events(
            logGroupName=log_group,
            logStreamName=log_stream,
            startFromHead=True
        )

        for event in events["events"]:
            print(event["message"])
    print("="*80)
    
    print("\n\n\n")

"""
- pytorch-training-2026-05-14-08-57-39-718
  - pytorch-training-2026-05-14-08-57-40-564
  - pytorch-training-2026-05-14-08-57-41-193
  """