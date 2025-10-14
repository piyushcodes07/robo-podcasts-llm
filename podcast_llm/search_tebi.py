import boto3
import os
from dotenv import load_dotenv

load_dotenv()


def search_tebi_objects(bucket_name: str, prefix: str):
    """
    List all objects in a Tebi.io bucket that start with the given prefix.

    Args:
        bucket_name (str): Name of the Tebi bucket
        prefix (str): The key prefix to search for
        access_key (str): Your Tebi Access Key
        secret_key (str): Your Tebi Secret Key

    Returns:
        list[str]: List of matching object keys
    """
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.tebi.io",
        aws_access_key_id=os.getenv("TEBI_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("TEBI_SECRET_KEY"),
        region_name="global",
    )

    keys = []
    continuation_token = None

    while True:
        list_kwargs = {
            "Bucket": bucket_name,
            "Prefix": prefix,
        }
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**list_kwargs)

        for obj in response.get("Contents", []):
            keys.append(obj["Key"])

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    print(keys)
    return keys
