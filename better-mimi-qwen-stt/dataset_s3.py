from datasets import load_dataset, interleave_datasets, Features, Value, Audio
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import soundfile as sf
import boto3
import sys
import io


S3_BUCKET = "home-ml"
S3_PREFIX = "asr-dataset"
BATCH_SIZE = 100

TARGET_FEATURES = Features({
    "text": Value("string"),
    "audio": Audio(sampling_rate=16000),
})


def load_librispeech(splits):
    return [
        load_dataset(
            "openslr/librispeech_asr",
            "all",
            split=split,
            streaming=True,
        ).select_columns(["text", "audio"]).cast(TARGET_FEATURES)
        for split in splits
    ]


def load_voxpopuli(split):
    ds = load_dataset(
        "facebook/voxpopuli",
        "en",
        split=split,
        streaming=True,
    )
    return ds.map(
        lambda ex: {"text": ex["normalized_text"], "audio": ex["audio"]},
        remove_columns=ds.column_names,
    ).cast(TARGET_FEATURES)


def build_datasets():
    libri_train = load_librispeech(["train.clean.100", "train.clean.360", "train.other.500"])
    libri_val = load_librispeech(["validation.clean", "validation.other"])
    libri_test = load_librispeech(["test.clean", "test.other"])

    vox_train = load_voxpopuli("train")
    vox_val = load_voxpopuli("validation")
    vox_test = load_voxpopuli("test")

    train_combined = interleave_datasets(
        libri_train + [vox_train],
        probabilities=[0.25, 0.25, 0.25, 0.25],
        stopping_strategy="all_exhausted",
    )

    val_combined = interleave_datasets(
        libri_val + [vox_val],
        probabilities=[0.33, 0.33, 0.34],
        stopping_strategy="all_exhausted",
    )

    test_combined = interleave_datasets(
        libri_test + [vox_test],
        probabilities=[0.33, 0.33, 0.34],
        stopping_strategy="all_exhausted",
    )

    return train_combined, val_combined, test_combined


def upload_dataset_to_s3(dataset, split_name):
    s3 = boto3.client("s3")
    batch = []
    part_idx = 0

    for example in dataset:
        audio = example["audio"]
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio["array"], audio["sampling_rate"], format="WAV", subtype="PCM_16")
        batch.append({
            "text": example["text"],
            "audio_bytes": wav_buf.getvalue(),
        })

        if len(batch) >= BATCH_SIZE:
            table = pa.Table.from_pylist(batch)
            buf = io.BytesIO()
            pq.write_table(table, buf)
            buf.seek(0)
            key = f"{S3_PREFIX}/{split_name}/part_{part_idx:05d}.parquet"
            s3.upload_fileobj(buf, S3_BUCKET, key)
            print(f"Uploaded s3://{S3_BUCKET}/{key}")
            batch = []
            part_idx += 1

    if batch:
        table = pa.Table.from_pylist(batch)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        key = f"{S3_PREFIX}/{split_name}/part_{part_idx:05d}.parquet"
        s3.upload_fileobj(buf, S3_BUCKET, key)
        print(f"Uploaded s3://{S3_BUCKET}/{key}")


def run_upload():
    train, val, test = build_datasets()
    print("Uploading train split...")
    upload_dataset_to_s3(train, "train")
    print("Uploading val split...")
    upload_dataset_to_s3(val, "val")
    print("Uploading test split...")
    upload_dataset_to_s3(test, "test")
    print("Done.")


if __name__ == "__main__":
    run_upload()
