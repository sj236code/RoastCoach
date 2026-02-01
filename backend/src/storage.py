# backend/src/storage.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import boto3

S3_BUCKET = os.getenv("ROASTCOACH_S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"

_s3 = boto3.client("s3", region_name=AWS_REGION)


def _require_bucket():
    if not S3_BUCKET:
        raise RuntimeError("ROASTCOACH_S3_BUCKET env var is not set.")


def upload_file(local_path: Path, s3_key: str, content_type: Optional[str] = None) -> str:
    """
    Upload a local file to S3.
    Returns s3://bucket/key
    """
    _require_bucket()

    extra = {}
    if content_type:
        extra["ContentType"] = content_type

    if extra:
        _s3.upload_file(str(local_path), S3_BUCKET, s3_key, ExtraArgs=extra)
    else:
        _s3.upload_file(str(local_path), S3_BUCKET, s3_key)

    return f"s3://{S3_BUCKET}/{s3_key}"
