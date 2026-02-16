import os
from datetime import datetime, timezone
from google.cloud import storage, bigquery
import pandas as pd


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

  # -------------------------
  # Google Cloud Storage
  # -------------------------

  def upload_file_to_gcs(
      self, bucket_name: str, prefix: str, full_path: str
  ) -> None:
    """
    Uploads a local file to a Google Cloud Storage bucket in a date-organized
    folder structure.

    Args:
        bucket_name (str): Target GCS bucket name.
        prefix (str): Prefix for the file path in GCS (e.g., "Reports").
        full_path (str): Local file path to upload.
    """
    # Get the filename from the full path
    filename = os.path.basename(full_path)

    # Get the current date in YYYYMMDD format
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Create the full GCS path with date-based organization
    bucket = self.storage_client.bucket(bucket_name)
    key = f"{utc_now}/{prefix}/{filename}"
    blob = bucket.blob(key)

    # Upload the file to GCS
    blob.upload_from_filename(full_path)
    print(
        f"✅ File '{filename}' uploaded to 'gs://{bucket_name}/{key}' successfully."
    )

  def load_df_to_bq(self, df: pd.DataFrame, table_id: str):
    """
    Loads a pandas DataFrame into a BigQuery table.

    Args:
        df (pd.DataFrame): DataFrame to load into BigQuery.
        table_id (str): BigQuery table ID in the format
                        'project.dataset.table'.
    """
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
