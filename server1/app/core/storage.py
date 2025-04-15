import boto3
from botocore.exceptions import ClientError
from loguru import logger
from app.core.config import settings

class S3Storage:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.AWS_S3_BUCKET

    def upload_file(self, file_data: bytes, key: str) -> str:
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=file_data)
            return f"https://{self.bucket_name}.s3.amazonaws.com/{key}"
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            raise

    def get_file(self, key: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Failed to get file from S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting file from S3: {e}")
            raise