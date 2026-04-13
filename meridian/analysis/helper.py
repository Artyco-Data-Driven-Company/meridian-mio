import functools
import os
from pathlib import Path
from google.cloud import storage, bigquery
import pandas as pd
import yaml


def singleton(cls):
  instances = {}

  @functools.wraps(cls)
  def get_instance(*args, **kwargs):
    if cls not in instances:
      instances[cls] = cls(*args, **kwargs)
    return instances[cls]

  return get_instance


class ClientConfig:
  """Class to manage client configuration loaded from a YAML file."""

  def __init__(self, config_path: str | None = None):
    self.config = {}
    if config_path:
      path = Path(config_path)
      if path.exists():
        with open(path, "r", encoding="utf-8") as f:
          self.config = yaml.safe_load(f) or {}

  def get(self, key_path, default=None):
    """
    Retrieve a value from the configuration using a dot-separated key path
    Returns the default value if the key is not found.
    """
    keys = key_path.split(".")
    value = self.config

    for k in keys:
      if isinstance(value, dict) and k in value:
        value = value[k]
      else:
        return default

    return value


@singleton
class GCPClient:
  """
  Helper class for interacting with Google Cloud Platform (GCP) services.

  This class centralizes the initialization of GCP service clients.
  Authentication must be configured beforehand, typically using the
  `GOOGLE_APPLICATION_CREDENTIALS` environment variable or
  workload identity in cloud environments.

  The configured credentials must have sufficient permissions
  for the operations being executed.
  """

  def __init__(self):
    """Initializes GCP service clients."""
    self.storage_client = storage.Client()
    self.bigquery_client = bigquery.Client()

  def upload_file_to_gcs(
      self, bucket_name: str, prefix: str, filepath: str, filename: str
  ) -> None:
    """
    Uploads a local file to a Google Cloud Storage bucket in a date-organized
    folder structure.

    Args:
        bucket_name (str): Target GCS bucket name.
        prefix (str): Prefix for the file path in GCS (e.g., "Reports").
        filepath (str): Local directory path where the file is located.
        filename (str): Name of the file to upload.
    """
    # Create fullpath
    full_path = os.path.join(filepath, filename)

    # Create the full GCS path with date-based organization
    bucket = self.storage_client.bucket(bucket_name)
    key = f"{prefix}/{filename}"
    blob = bucket.blob(key)

    # Upload the file to GCS
    blob.upload_from_filename(full_path)
    print(
        f"✅ File '{filename}' uploaded to 'gs://{bucket_name}/{key}' successfully."
    )

  def load_parquet_to_bq(self, parquet_path: str, table_id: str):
    """
    Loads a Parquet file into a BigQuery table.

    Args:
        parquet_path (str): Path to the Parquet file to load into BigQuery.
        table_id (str): BigQuery table ID in the format
                        'project.dataset.table'.
    """
    # DataFrame from the Parquet file
    df = pd.read_parquet(parquet_path)

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    # Load the DataFrame into BigQuery
    load_job = self.bigquery_client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )
    # Wait for the load job to complete
    load_job.result()

    print(f"✅ Loaded {len(df)} rows into {table_id}.")
