import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.cloud import storage, bigquery
import pandas as pd
import yaml

from meridian.analysis import formatter


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


class CustomizeCharts:
  """Apply visual overrides to Altair charts based on a YAML configuration."""

  def __init__(self):
    self.global_config_charts: dict[str, str] = {}

  def register_chart_spec(
      self,
      chart_spec: formatter.ChartSpec,
      chart_overrides: dict[str, object] | None = None,
  ) -> formatter.ChartSpec:
    """Registers a ChartSpec with the provided overrides applied, and stores the
    resulting chart JSON in the global config charts dictionary.
    """
    if chart_overrides:
      chart_spec = self.apply(chart_spec, chart_overrides)
    self.global_config_charts[chart_spec.id] = chart_spec.chart_json
    return chart_spec

  def export_chart_json(self, filepath: str) -> None:
    """Exports chart JSON for all registered charts to a JSON file."""
    charts_data: dict[str, object] = {}

    # Iterate through the registered charts and add their JSON to the charts_data dict.
    for chart_id, chart_json in self.global_config_charts.items():
      chart_payload = json.loads(chart_json)
      charts_data[chart_id] = chart_payload

    # Write to the specified JSON file with indentation for readability.
    with open(filepath, "w") as f:
      json.dump(charts_data, f, indent=2)

  @staticmethod
  def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]):
    """Recursively merge *overrides* into *base* in place."""
    for key, value in overrides.items():
      if (
          key in base
          and isinstance(base[key], dict)
          and isinstance(value, dict)
      ):
        CustomizeCharts._deep_merge(base[key], value)
      elif (
          key in base
          and isinstance(base[key], list)
          and isinstance(value, list)
      ):
        CustomizeCharts._deep_merge_list(base[key], value)
      else:
        base[key] = value

  @staticmethod
  def _deep_merge_list(base: list[Any], overrides: list[Any]) -> None:
    """Recursively merge list items by index.

    - Dict items are merged recursively.
    - Scalar items replace the base item.
    - `null` (`None` in Python) keeps the base item unchanged.
    - Extra override items are appended.
    """
    for idx, override_item in enumerate(overrides):
      if idx >= len(base):
        base.append(override_item)
        continue

      if override_item is None:
        continue

      base_item = base[idx]
      if isinstance(base_item, dict) and isinstance(override_item, dict):
        CustomizeCharts._deep_merge(base_item, override_item)
      elif isinstance(base_item, list) and isinstance(override_item, list):
        CustomizeCharts._deep_merge_list(base_item, override_item)
      else:
        base[idx] = override_item

  def apply(
      self,
      chart_spec: formatter.ChartSpec,
      chart_overrides: dict[str, Any],
  ) -> formatter.ChartSpec:
    """
    Returns a new ChartSpec with the provided overrides applied to the original chart JSON.
    """
    # Get the original chart JSON as a dictionary
    chart_json_dict = json.loads(chart_spec.chart_json)

    # Recursively merge nested overrides without dropping sibling keys.
    self._deep_merge(chart_json_dict, chart_overrides)

    # Create a new ChartSpec with the updated config
    return formatter.ChartSpec(
        id=chart_spec.id,
        chart_json=json.dumps(chart_json_dict),
        description=chart_spec.description,
    )
