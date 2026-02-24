import yaml
from pathlib import Path


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
    (e.g., 'font_family.Arial').
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
