# Copyright 2025 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for formatting analysis outputs."""

from collections.abc import Sequence
from typing import Any
import dataclasses
import json
import math
import os

import altair as alt
import immutabledict
import jinja2
import pandas as pd
from meridian import constants as c


__all__ = [
    'CardSpec',
    'ChartSpec',
    'TableSpec',
    'StatsSpec',
    'create_template_env',
    'create_card_html',
]

Y_AXIS_TITLE_CONFIG = immutabledict.immutabledict(
    {
        'titleAngle': 0,
        'titleAlign': 'left',
        'titleY': -20,
    }
)

AXIS_CONFIG = immutabledict.immutabledict(
    {
        'ticks': False,
        'labelPadding': c.PADDING_10,
        'domainColor': c.GREY_300,
    }
)


_template_loader = jinja2.FileSystemLoader(
    os.path.abspath(os.path.dirname(__file__)) + '/templates'
)


@dataclasses.dataclass(frozen=True)
class CardSpec:
  id: str
  title: str


@dataclasses.dataclass(frozen=True)
class ChartSpec:
  id: str
  chart_json: str
  description: str | None = None

  def to_dataframe(self) -> pd.DataFrame:
    """Converts the ChartSpec's chart_json to a pandas DataFrame."""
    spec = json.loads(self.chart_json)
    datasets = spec.get('datasets')

    # Assuming there's only one dataset, extract the rows.
    _, rows = next(iter(datasets.items()))
    headers = rows[0].keys()
    headers_fix = [header.lower().replace(' ', '_') for header in headers]
    df = pd.DataFrame(rows, columns=headers_fix)
    df['id'] = self.id
    df.reset_index(drop=True, inplace=True)
    return df


@dataclasses.dataclass(frozen=True)
class TableSpec:
  id: str
  title: str
  column_headers: Sequence[str]
  row_values: Sequence[Sequence[str]]
  description: str | None = None

  def to_dataframe(self) -> pd.DataFrame:
    """Converts the TableSpec to a pandas DataFrame."""
    headers_fix = [
        header.lower().replace(' ', '_') for header in list(self.column_headers)
    ]
    df = pd.DataFrame(self.row_values, columns=headers_fix)
    df['id'] = self.id
    df.reset_index(drop=True, inplace=True)
    return df


@dataclasses.dataclass(frozen=True)
class StatsSpec:
  title: str
  stat: str
  delta: str | None = None


class CustomizeCharts:
  """Apply visual overrides to Altair charts based on a YAML configuration."""

  def __init__(self):
    self.global_config_charts: dict[str, str] = {}

  def register_chart_spec(
      self,
      chart_spec: ChartSpec,
      chart_overrides: dict[str, object] | None = None,
  ) -> ChartSpec:
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
    with open(filepath, 'w', encoding='utf-8') as f:
      json.dump(charts_data, f, indent=2)

  @staticmethod
  def _is_domain_range_pair(
      base: dict[str, Any], overrides: dict[str, Any]
  ) -> bool:
    """Returns True if *base* defines a `domain` and *overrides* defines a
    `domain`/`range` list pair (e.g. a Vega-Lite color scale).

    `base['range']` is not required: a chart may rely on Vega-Lite's default
    color scheme and only declare `domain`.
    """
    return (
        isinstance(base.get('domain'), list)
        and isinstance(overrides.get('domain'), list)
        and isinstance(overrides.get('range'), list)
    )

  @staticmethod
  def _merge_domain_range(
      base: dict[str, Any], overrides: dict[str, Any]
  ) -> None:
    """Merges a `domain`/`range` pair by matching domain labels.

    Only the color of domain values already present in `base['domain']` is
    replaced, using the matching entry from `overrides`. Domain values that
    only exist in the override (e.g. extra channels not used by this chart)
    are ignored so the legend never grows beyond the chart's own domain.
    """
    color_by_label = dict(zip(overrides['domain'], overrides['range']))
    base_range = base.get('range', [])
    base['range'] = [
        color_by_label.get(
            label, base_range[i] if i < len(base_range) else None
        )
        for i, label in enumerate(base['domain'])
    ]

  @staticmethod
  def _infer_color_domains(
      node: Any,
      datasets: dict[str, list[dict[str, Any]]],
      data_name: str | None,
  ) -> None:
    """Fills in `encoding.color.scale.domain` wherever it's missing.

    Vega-Lite infers a nominal color scale's domain from the unique values of
    its encoded field when `scale.domain` isn't set explicitly. That implicit
    domain needs to be made explicit before merging overrides, otherwise the
    override's full domain/range would be injected wholesale instead of being
    matched against the channels this specific chart actually plots.
    """
    if isinstance(node, dict):
      if isinstance(node.get('data'), dict) and 'name' in node['data']:
        data_name = node['data']['name']

      color = node.get('encoding', {}).get('color')
      if (
          isinstance(color, dict)
          and color.get('type') == 'nominal'
          and 'field' in color
          and not isinstance(color.get('scale', {}).get('domain'), list)
          and data_name is not None
      ):
        field = color['field']
        domain = []
        for row in datasets.get(data_name, []):
          value = row.get(field)
          if value is not None and value not in domain:
            domain.append(value)
        if domain:
          color.setdefault('scale', {})['domain'] = domain

      for value in node.values():
        CustomizeCharts._infer_color_domains(value, datasets, data_name)
    elif isinstance(node, list):
      for item in node:
        CustomizeCharts._infer_color_domains(item, datasets, data_name)

  @staticmethod
  def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]):
    """Recursively merge *overrides* into *base* in place."""
    if CustomizeCharts._is_domain_range_pair(base, overrides):
      CustomizeCharts._merge_domain_range(base, overrides)
      overrides = {
          k: v for k, v in overrides.items() if k not in ('domain', 'range')
      }

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
      chart_spec: ChartSpec,
      chart_overrides: dict[str, Any],
  ) -> ChartSpec:
    """
    Returns a new ChartSpec with the provided overrides applied to the original chart JSON.
    """
    # Get the original chart JSON as a dictionary
    chart_json_dict = json.loads(chart_spec.chart_json)

    # Make any implicit (data-derived) color domain explicit, so overrides
    # only replace colors for channels this chart actually plots.
    self._infer_color_domains(
        chart_json_dict, chart_json_dict.get('datasets', {}), None
    )

    # Recursively merge nested overrides without dropping sibling keys.
    self._deep_merge(chart_json_dict, chart_overrides)

    # Create a new ChartSpec with the updated config
    return ChartSpec(
        id=chart_spec.id,
        chart_json=json.dumps(chart_json_dict),
        description=chart_spec.description,
    )


def text_config(font_family: str):
  """Returns a dictionary with the text configuration for Vega-Lite charts."""
  return immutabledict.immutabledict(
      {
          'titleFont': font_family,
          'labelFont': font_family,
          'titleFontWeight': 'normal',
          'titleFontSize': c.AXIS_FONT_SIZE,
          'labelFontSize': c.AXIS_FONT_SIZE,
          'titleColor': c.GREY_700,
          'labelColor': c.GREY_700,
      }
  )


def custom_title_params(title: str, font_family: str) -> alt.TitleParams:
  """Formats the title to be at the top left of the plot."""
  return alt.TitleParams(
      text=title,
      anchor='start',
      fontSize=c.TITLE_FONT_SIZE,
      font=font_family,
      fontWeight='normal',
      offset=c.PADDING_10,
      color=c.GREY_800,
  )


def bar_chart_width(num_bars: int) -> int:
  """Returns the width for a bar chart based on the number of bars."""
  return (c.BAR_SIZE + c.PADDING_20) * num_bars


def format_percent(percent: float) -> str:
  """Formats a percentage value into a string format.

  Percentage values between 0 and 1 are formatted with 1 decimal place.
  Percentage values greater than 1 are formatted with 0 decimal places.

  Args:
    percent: The percentage value to format.

  Returns:
    A formatted string.
  """
  if percent >= 0.01:
    return '{:.0%}'.format(percent)
  else:
    return '{:.1g}%'.format(percent * 100)


def format_var_number(n: float, decimals: int) -> str:
  """
  Formats a number with thousands separators and given decimal places.
  Adds a '+' sign for positive numbers.
  """
  sign = ''
  if n > 0:
    sign = '+'

  fmt = f'{{:,.{decimals}f}}'
  fmt_replaces = (
      fmt.format(n).replace(',', 'X').replace('.', ',').replace('X', '.')
  )
  return sign + fmt_replaces


def format_var_pp(n: float, decimals: int) -> str:
  """
  Formats a number as a percentage point with given decimal places.
  Adds a '+' sign for positive numbers.
  """
  if n is None or not math.isfinite(n):
    return '-'

  sign = ''
  if n > 0:
    sign = '+'

  n_formatted = f'{n * 100:.{decimals}f} pp'
  return f'{sign}{n_formatted}'


def format_var_percent(n: float, decimals: int) -> str:
  """
  Formats a number as a percentage with given decimal places.
  Adds a '+' sign for positive numbers.
  """
  if n is None or not math.isfinite(n):
    return '-'

  sign = ''
  if n > 0:
    sign = '+'

  n_formatted = f'{n * 100:.{decimals}f}%'
  return f'{sign}{n_formatted}'


def format_percent_cm(n: float, decimals: int) -> str:
  """
  Formats a number as a percentage with given decimal places.
  """
  if n is None or not math.isfinite(n):
    return '-'

  n_formatted = f'{n * 100:.{decimals}f}%'
  return n_formatted


def format_number_cm(n: float, decimals: int) -> str:
  """
  Formats a number with thousands separators and given decimal places.
  """
  fmt = f'{{:,.{decimals}f}}'
  fmt_replaces = (
      fmt.format(n).replace(',', 'X').replace('.', ',').replace('X', '.')
  )
  return fmt_replaces


def compact_number(n: float, precision: int = 0, currency: str = '') -> str:
  """Formats a number into a compact notation to the specified precision.

  Ex. $15M

  Args:
    n: The number to format.
    precision: The number of decimals to use when rounding.
    currency: Optional string currency character. This is added at the beginning
      of the formatted string.

  Returns:
    A formatted string.
  """
  millnames = ['', 'k', 'M', 'B', 'T']
  millidx = max(
      0,
      min(
          len(millnames) - 1,
          int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
      ),
  )
  result = '{:.{precision}f}'.format(
      n / 10 ** (3 * millidx), precision=precision
  )
  suffixed = '{0}{dx}'.format(result, dx=millnames[millidx])

  # For negative numbers, add the currency after the negative sign.
  if n < 0:
    return suffixed[0] + currency + suffixed[1:]
  return currency + suffixed


def compact_number_expr(value: str = 'value', n_sig_digits: int = 3) -> str:
  """Returns the Vega expression to format the datum value with SI-prefixes.

  The scientific notation prefixes (k, M, G, T) are used for numeric values to
  make the numbers easier to read. Note, G for giga is replaced with B for
  billion.

  Args:
    value: Datum value to format.
    n_sig_digits: The number of significant digits for the formatted number.

  Returns: The Vega expression string to format the text into a compact form.
  """
  return f"replace(format(datum.{value}, '.{n_sig_digits}~s'), 'G', 'B')"


def format_number_text(percent_value: float, actual_value: float) -> str:
  """Formats the percent and actual value into a human-readable compact string.

  Ex. 40.2% (15M)

  Args:
    percent_value: Float between 0 and 1 representing a percentage value.
    actual_value: Float to display next to the percentage value.

  Returns:
    String with the percentage and the compact notation of the actual value.
  """
  return f'{round(percent_value * 100, 1)}% ({compact_number(actual_value)})'


def format_monetary_num(num: float, currency: str) -> str:
  """Formats a number into a readable monetary value (ex. $15M, $1.2B)."""
  precision = 1 if num != 0 and int(math.log10(abs(num))) % 3 == 0 else 0
  return compact_number(num, precision=precision, currency=currency)


def create_template_env() -> jinja2.Environment:
  """Creates a Jinja2 template environment."""
  return jinja2.Environment(
      loader=_template_loader,
      autoescape=jinja2.select_autoescape(),
  )


def create_card_html(
    template_env: jinja2.Environment,
    card_spec: CardSpec,
    insights: str,
    chart_specs: Sequence[ChartSpec | TableSpec] | None = None,
    stats_specs: Sequence[StatsSpec] | None = None,
) -> str:
  """Creates a card's HTML snippet that includes given card and chart specs."""
  insights_html = template_env.get_template('insights.html.jinja').render(
      text_html=insights
  )
  card_params = dataclasses.asdict(card_spec)
  card_params[c.CARD_CHARTS] = (
      _create_charts_htmls(template_env, chart_specs) if chart_specs else None
  )
  card_params[c.CARD_INSIGHTS] = insights_html
  card_params[c.CARD_STATS] = (
      _create_stats_htmls(template_env, stats_specs) if stats_specs else None
  )
  return template_env.get_template('card.html.jinja').render(card_params)


def _create_stats_htmls(
    template_env: jinja2.Environment, specs: Sequence[StatsSpec]
) -> Sequence[str]:
  """Creates a list of stats HTML snippets given a list of stats specs."""
  stats_htmls = []
  for spec in specs:
    stats_htmls.append(
        template_env.get_template('stats.html.jinja').render(
            dataclasses.asdict(spec)
        )
    )
  return stats_htmls


def _create_charts_htmls(
    template_env: jinja2.Environment,
    specs: Sequence[ChartSpec | TableSpec],
) -> Sequence[str]:
  """Creates a list of chart HTML snippets given a list of chart specs."""
  chart_template = template_env.get_template('chart.html.jinja')
  table_template = template_env.get_template('table.html.jinja')
  htmls = []
  for spec in specs:
    if isinstance(spec, ChartSpec):
      htmls.append(chart_template.render(dataclasses.asdict(spec)))
    else:
      htmls.append(table_template.render(dataclasses.asdict(spec)))
  return htmls
