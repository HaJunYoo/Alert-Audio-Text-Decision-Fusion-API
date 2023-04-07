from datetime import datetime, timedelta

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from airflow import Dataset

saved_data = Dataset('/Users/yoohajun/Desktop/grad_audio/output/spectrogram_fixed.zip')


def upload_to_s3(file_name: str, key: str, bucket_name: str) -> None:
    s3_hook = S3Hook(aws_conn_id='s3_conn')
    s3_hook.load_file(file_name, key, bucket_name=bucket_name, replace=True)


with DAG(
        dag_id='s3_dag',
        start_date=datetime(2023, 4, 5),
        # schedule_interval='@once',
        schedule=[saved_data],
        dagrun_timeout=timedelta(minutes=60),
        tags=['s3', 'upload'],
        catchup=False
) as dag:
    upload_to_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
        op_kwargs={
            'file_name': '/Users/yoohajun/Desktop/grad_audio/output/spectrogram_fixed.zip',
            'key': 'Audio-train-data/spectrogram_fixed.zip',
            'bucket_name': 'emerdy-bucket'
        }
    )
