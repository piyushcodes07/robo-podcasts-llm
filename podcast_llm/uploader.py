import boto3
from botocore.client import Config
import os
from dotenv import load_dotenv

load_dotenv()

TEBI_ACCESS_KEY = os.getenv("TEBI_ACCESS_KEY")
TEBI_SECRET_KEY = os.getenv("TEBI_SECRET_KEY")
TEBI_ENDPOINT = "https://s3.tebi.io"
BUCKET_NAME = "music-r2"

s3 = boto3.client(
    "s3",
    aws_access_key_id=TEBI_ACCESS_KEY,
    aws_secret_access_key=TEBI_SECRET_KEY,
    endpoint_url=TEBI_ENDPOINT,
    config=Config(signature_version="s3v4"),
)


def upload_mp3(user_id: str, local_file_path: str):
    if not os.path.isfile(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")

    file_name = os.path.basename(local_file_path).replace(" ", "_")
    key = f"users/{user_id}/{file_name}"
    file_size = os.path.getsize(local_file_path)
    print("MAIN FILE   ", file_size)
    # Open file in binary mode
    with open(local_file_path, "rb") as f:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=f,  # File-like object
            ContentLength=file_size,  # Ensures HTTP header is set
            ContentType="audio/mpeg",
            ACL="public-read",
        )

    return f"https://s3.tebi.io/{BUCKET_NAME}/{key}"
