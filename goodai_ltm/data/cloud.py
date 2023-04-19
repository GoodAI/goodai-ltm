import os
import pickle
from typing import Dict, Any
import boto3
from dotenv import load_dotenv

load_dotenv()


class CloudStorage:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        aws_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        if aws_key_id is None:
            raise SystemError('AWS_ACCESS_KEY_ID needs to be set')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if aws_secret_key is None:
            raise SystemError('AWS_SECRET_ACCESS_KEY needs to be set')
        self.client = boto3.client(
            's3',
            aws_access_key_id=aws_key_id,
            aws_secret_access_key=aws_secret_key,
        )

    def put_object(self, key: str, obj: Dict[str, Any]):
        obj_bytes = pickle.dumps(obj)
        self.client.put_object(Bucket=self.bucket_name, Body=obj_bytes, Key=key)

    def put_object_bytes(self, key: str, obj_bytes: bytes):
        self.client.put_object(Bucket=self.bucket_name, Body=obj_bytes, Key=key)

    @classmethod
    def get_instance(cls, bucket_name: str):
        instance_attr = f'__instance_{bucket_name}'
        try:
            return getattr(cls, instance_attr)
        except AttributeError:
            instance = cls(bucket_name)
            setattr(cls, instance_attr, instance)
            return instance
