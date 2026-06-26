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

"""Summarization module that creates a 2-page HTML report."""

from collections.abc import Sequence
from datetime import datetime, timezone
from functools import partial
import functools
import os

import jinja2
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import ClientConfig
from meridian.analysis import formatter
from meridian.analysis import summary_text
from meridian.analysis import visualizer
from meridian.data import time_coordinates as tc
from meridian.model import model
import pandas as pd


__all__ = [
    'Summarizer',
    'COMPARISON_METRICS_CARD_SPEC',
]

COMPARISON_METRICS_CARD_SPEC = formatter.CardSpec(
    id=summary_text.COMPARISON_METRICS_CARD_ID,
    title=summary_text.COMPARISON_METRICS_CARD_TITLE,
)


class Summarizer:
  """Generates HTML summary visualizations from the model fitting."""

  def __init__(
      self,
      meridian: model.Meridian,
      client_config: ClientConfig,
      use_kpi: bool = False,
  ):
    """Initialize the visualizer classes that are not time-dependent."""
    self._meridian = meridian
    self._use_kpi = analyzer.Analyzer(meridian)._use_kpi(use_kpi)
    self.customize_charts = formatter.CustomizeCharts()
    self.client_config = client_config

  @functools.cached_property
  def _model_fit(self):
    return visualizer.ModelFit(
        self._meridian, use_kpi=self._use_kpi, client_config=self.client_config
    )

  @functools.cached_property
  def _model_diagnostics(self):
    return visualizer.ModelDiagnostics(
        self._meridian, use_kpi=self._use_kpi, client_config=self.client_config
    )

  def output_comparison_metrics_summary(
      self,
      filename: str,
      filepath: str,
      start_date: tc.Date,
      end_date: tc.Date,
      start_date_cm: tc.Date,
      end_date_cm: tc.Date,
      bq_product_or_service: str = 'Unknown',
  ):
    """
    Generates and saves the HTML comparison metrics summary output.

    Args:
      filename: The filename for the generated HTML output.
      filepath: The path to the directory where the file will be saved.
      start_date: Optional start date selector, *inclusive*, in _yyyy-mm-dd_
        format.
      end_date: Optional end date selector, *inclusive* in _yyyy-mm-dd_ format.
      start_date_cm (str): Start date selector for comparison metrics,
        inclusive, in yyyy-mm-dd format.
      end_date_cm (str): End date selector for comparison metrics,
        inclusive, in yyyy-mm-dd format.
      bq_product_or_service: The product or service name to include in the
        BigQuery table.
    """
    self.utc_now = datetime.now(timezone.utc)
    self.filepath_cm = filepath
    self.prod_or_serv = bq_product_or_service

    # Generate the report content
    report = self._gen_comparison_metrics_summary(
        start_date, end_date, start_date_cm, end_date_cm
    )

    # Export chart JSON key comparison for debugging/inspection purposes.
    export_path = os.path.join(filepath, 'export_charts')
    os.makedirs(export_path, exist_ok=True)
    if self.client_config.get('export_charts_json', False):
      self.customize_charts.export_chart_json(
          os.path.join(
              export_path, 'json_charts_' + filename.replace('.html', '.json')
          )
      )

    # Create the output directory if it doesn't exist and save the report
    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, filename)
    with open(full_path, 'w') as f:
      f.write(report)
    print(f'✅ Report saved to {full_path}')

  def _gen_comparison_metrics_summary(
      self,
      start_date: tc.Date,
      end_date: tc.Date,
      start_date_cm: tc.Date,
      end_date_cm: tc.Date,
  ) -> str:
    """Generate HTML comparison metrics summary output (as sanitized content str)."""
    all_dates = self._meridian.input_data.time_coordinates.all_dates

    if start_date is None or end_date is None:
      raise ValueError(
          'Both start_date and end_date must be provided for comparison metrics summary.'
      )
    start_date = tc.normalize_date(start_date)
    end_date = tc.normalize_date(end_date)
    if start_date not in all_dates:
      raise ValueError(
          f'start_date ({start_date}) must be in the time coordinates!'
      )
    if end_date not in all_dates:
      raise ValueError(
          f'end_date ({end_date}) must be in the time coordinates!'
      )
    if start_date > end_date:
      raise ValueError(
          f'start_date ({start_date}) must be before end_date ({end_date})!'
      )

    selected_times = self._meridian.expand_selected_time_dims(
        start_date, end_date
    )

    if start_date_cm is None or end_date_cm is None:
      raise ValueError(
          'Both start_date_cm and end_date_cm must be provided for comparison metrics summary.'
      )
    start_date_cm = tc.normalize_date(start_date_cm)
    end_date_cm = tc.normalize_date(end_date_cm)
    if start_date_cm not in all_dates:
      raise ValueError(
          f'start_date_cm ({start_date_cm}) must be in the time coordinates!'
      )
    if end_date_cm not in all_dates:
      raise ValueError(
          f'end_date_cm ({end_date_cm}) must be in the time coordinates!'
      )
    if start_date_cm > end_date_cm:
      raise ValueError(
          f'start_date_cm ({start_date_cm}) must be before end_date_cm ({end_date_cm})!'
      )

    comparison_selected_times = self._meridian.expand_selected_time_dims(
        start_date_cm, end_date_cm
    )

    template_env = formatter.create_template_env()
    template_env.globals[c.START_DATE] = start_date.strftime(
        f'%b {start_date.day}, %Y'
    )

    interval_days = self._meridian.input_data.time_coordinates.interval_days
    end_date_adjusted = end_date + pd.Timedelta(days=interval_days)

    template_env.globals[c.END_DATE] = end_date_adjusted.strftime(
        f'%b {end_date_adjusted.day}, %Y'
    )
    template_env.globals['font_family'] = self.client_config.get(
        'html_reports.global.font_family', c.FONT_FAMILY
    )
    template_env.globals['font_link'] = self.client_config.get(
        'html_reports.global.font_link', c.FONT_LINK
    )

    html_template = template_env.get_template('summary.html.jinja')
    cards_htmls = self._create_cards_cm_htmls(
        template_env,
        selected_times=selected_times,
        cm_selected_times=comparison_selected_times,
    )

    return html_template.render(
        title=summary_text.MODEL_RESULTS_TITLE, cards=cards_htmls
    )

  def _create_cards_cm_htmls(
      self,
      template_env: jinja2.Environment,
      selected_times: Sequence[str] | None,
      cm_selected_times: Sequence[str] | None,
  ):
    """Creates the HTML snippets for cards in the comparison metrics summary page."""
    media_summary = visualizer.MediaSummary(
        self._meridian,
        selected_times=selected_times,
        use_kpi=self._use_kpi,
        client_config=self.client_config,
    )

    media_summary_cm = visualizer.MediaSummary(
        self._meridian,
        selected_times=cm_selected_times,
        use_kpi=self._use_kpi,
        client_config=self.client_config,
    )
    cards = [
        self._create_comparison_metrics_card_html(
            template_env,
            media_summary=media_summary,
            media_summary_cm=media_summary_cm,
            selected_times=selected_times,
            selected_times_cm=cm_selected_times,
        ),
    ]

    return cards

  def _create_comparison_metrics_card_html(
      self,
      template_env: jinja2.Environment,
      media_summary: visualizer.MediaSummary,
      media_summary_cm: visualizer.MediaSummary,
      selected_times: Sequence[str] | None,
      selected_times_cm: Sequence[str] | None,
  ) -> str:
    """Creates the HTML snippet for the Comparison Metrics card."""

    # Get summary metrics dataframes for both periods
    period_1_df = media_summary.get_summary_metrics_df()
    period_2_df = media_summary_cm.get_summary_metrics_df()

    # Get KPI sums for both periods
    kpi_period_1 = media_summary.get_kpi_sum()
    kpi_period_2 = media_summary_cm.get_kpi_sum()

    # Get labels for the two comparison periods
    period_1_label, period_2_label = self._get_labels_for_comparison_metrics(
        selected_times, selected_times_cm
    )

    kpi_resume_table = self._create_kpi_comparison_table_spec(
        kpi_period_1,
        kpi_period_2,
    )

    spend_resume_table = self._create_spend_comparison_table_spec(
        period_1_df,
        period_2_df,
    )

    contribution_resume_table = self._create_contribution_comparison_table_spec(
        period_1_df,
        period_2_df,
    )

    kpi_contribution_resume_table = (
        self._create_kpi_contribution_comparison_table_spec(
            period_1_df,
            period_2_df,
            kpi_period_1,
            kpi_period_2,
        )
    )

    roi_comparison_table = self._create_roi_comparison_table_spec(
        period_1_df,
        period_2_df,
        kpi_period_1,
        kpi_period_2,
    )

    spend_comparison_pie_chart = self._create_spend_comparison_pie_chart_spec(
        period_1_df,
        media_summary,
    )

    contribution_comparison_pie_chart = (
        self._create_contribution_comparison_pie_chart_spec(
            period_1_df,
            media_summary,
        )
    )

    insights = summary_text.COMPARISON_METRICS_INSIGHTS_FORMAT.format(
        period_1=period_1_label,
        period_2=period_2_label,
    )

    return formatter.create_card_html(
        template_env,
        COMPARISON_METRICS_CARD_SPEC,
        insights,
        [
            kpi_resume_table,
            spend_resume_table,
            contribution_resume_table,
            kpi_contribution_resume_table,
            roi_comparison_table,
            spend_comparison_pie_chart,
            contribution_comparison_pie_chart,
        ],
    )

  def _get_labels_for_comparison_metrics(
      self,
      selected_times: Sequence[str] | None,
      selected_times_cm: Sequence[str] | None,
  ) -> tuple[str, str]:
    """Generates labels for the two comparison periods based on selected times."""

    # Check that both selected_times and selected_times_cm are provided
    if selected_times is None or selected_times_cm is None:
      raise ValueError(
          'Both selected_times and selected_times_cm must be provided when '
          'comparison_metrics is True.'
      )

    interval_days = self._meridian.input_data.time_coordinates.interval_days

    start_date_normalized = tc.normalize_date(selected_times[0])
    end_date_normalized = tc.normalize_date(selected_times[-1])
    start_date_cm_normalized = tc.normalize_date(selected_times_cm[0])
    end_date_cm_normalized = tc.normalize_date(selected_times_cm[-1])

    start_date = start_date_normalized.strftime(
        f'%b {start_date_normalized.day}, %Y'
    )
    end_date_adjusted = end_date_normalized + pd.Timedelta(days=interval_days)
    end_date = end_date_adjusted.strftime(f'%b {end_date_adjusted.day}, %Y')
    period_1_label = f'{start_date} to {end_date}'

    start_date_cm = start_date_cm_normalized.strftime(
        f'%b {start_date_cm_normalized.day}, %Y'
    )
    end_date_cm_adjusted = end_date_cm_normalized + pd.Timedelta(
        days=interval_days
    )
    end_date_cm = end_date_cm_adjusted.strftime(
        f'%b {end_date_cm_adjusted.day}, %Y'
    )
    period_2_label = f'{start_date_cm} to {end_date_cm}'

    return period_1_label, period_2_label

  def _create_kpi_comparison_table_spec(
      self, kpi_period_1: float, kpi_period_2: float
  ) -> formatter.TableSpec:
    """Creates the KPI comparison table spec."""

    # Column headers for the comparison table
    column_names = [
        'Period 1',
        'Period 2',
        'Var %',
        'Var Abs',
    ]

    # Define KPI comparison values
    kpi_df = pd.DataFrame(
        {'period_1': [kpi_period_1], 'period_2': [kpi_period_2]}
    )
    kpi_df['var_pct'] = (kpi_df['period_1'] / kpi_df['period_2']) - 1
    kpi_df['var_abs'] = kpi_df['period_1'] - kpi_df['period_2']

    # Format KPI comparison values
    fmt_num = partial(formatter.format_number_cm, decimals=0)
    fmt_var_pct = partial(formatter.format_var_percent, decimals=2)
    fmt_var_num = partial(formatter.format_var_number, decimals=0)

    kpi_df['period_1'] = kpi_df['period_1'].apply(fmt_num)
    kpi_df['period_2'] = kpi_df['period_2'].apply(fmt_num)
    kpi_df['var_pct'] = kpi_df['var_pct'].apply(fmt_var_pct)
    kpi_df['var_abs'] = kpi_df['var_abs'].apply(fmt_var_num)

    # Create TableSpec for KPI comparison
    kpi_resume_table = formatter.TableSpec(
        id=summary_text.KPI_COMPARISON_ID,
        title=summary_text.KPI_COMPARISON_TITLE,
        description=summary_text.KPI_COMPARISON_DESCRIPTION,
        column_headers=column_names,
        row_values=kpi_df.values.tolist(),
    )

    # Save data as Parquet
    file_name = summary_text.KPI_COMPARISON_ID.replace(' ', '_') + '.parquet'
    df = kpi_resume_table.to_dataframe()
    self._comparison_df_to_parquet(df, file_name)

    return kpi_resume_table

  def _create_spend_comparison_table_spec(
      self,
      period_1_df: pd.DataFrame,
      period_2_df: pd.DataFrame,
  ) -> formatter.TableSpec:
    """Creates the Spend comparison table spec."""

    # Column headers for the comparison table
    column_names = [
        'Channel',
        'Period 1',
        'Period 2',
        'Var %',
        'Var Abs',
    ]

    # Define Spend comparison values
    spend_period_1 = period_1_df.set_index('channel')[c.SPEND]
    spend_period_2 = period_2_df.set_index('channel')[c.SPEND]
    spend_comparison_df = pd.DataFrame(
        {
            'period_1': spend_period_1,
            'period_2': spend_period_2,
        }
    ).fillna(0)
    spend_comparison_df['var_pct'] = (
        spend_comparison_df['period_1'] / spend_comparison_df['period_2']
    ) - 1
    spend_comparison_df['var_abs'] = (
        spend_comparison_df['period_1'] - spend_comparison_df['period_2']
    )
    spend_df = spend_comparison_df.reset_index()

    # Replace 'All Channels' with 'Total' in channel names
    spend_df['channel'] = spend_df['channel'].replace('All Channels', 'Total')

    # Create 'Digital' row
    digital_channels = self.client_config.get(
        'html_reports.comparison_metrics_summary.digital_channels', None
    )
    if digital_channels is not None:
      digital_sum = spend_df[spend_df['channel'].isin(digital_channels)].sum(
          numeric_only=True
      )
      digital_row = {
          'channel': 'Digital',
          'period_1': digital_sum['period_1'],
          'period_2': digital_sum['period_2'],
          'var_pct': digital_sum['period_1'] / digital_sum['period_2'] - 1,
          'var_abs': digital_sum['period_1'] - digital_sum['period_2'],
      }
      spend_df = pd.concat([spend_df, pd.DataFrame([digital_row])]).reset_index(
          drop=True
      )

    # Format Spend comparison values
    fmt_num = partial(formatter.format_number_cm, decimals=1)
    fmt_var_pct = partial(formatter.format_var_percent, decimals=2)
    fmt_var_num = partial(formatter.format_var_number, decimals=1)

    spend_df['period_1'] = spend_df['period_1'].apply(fmt_num)
    spend_df['period_2'] = spend_df['period_2'].apply(fmt_num)
    spend_df['var_pct'] = spend_df['var_pct'].apply(fmt_var_pct)
    spend_df['var_abs'] = spend_df['var_abs'].apply(fmt_var_num)

    # Create TableSpec for Spend comparison
    spend_resume_table = formatter.TableSpec(
        id=summary_text.SPEND_COMPARISON_ID,
        title=summary_text.SPEND_COMPARISON_TITLE,
        description=summary_text.SPEND_COMPARISON_DESCRIPTION,
        column_headers=column_names,
        row_values=spend_df.values.tolist(),
    )

    # Save data as Parquet
    file_name = summary_text.SPEND_COMPARISON_ID.replace(' ', '_') + '.parquet'
    df = spend_resume_table.to_dataframe()
    self._comparison_df_to_parquet(df, file_name)

    return spend_resume_table

  def _create_contribution_comparison_table_spec(
      self, period_1_df: pd.DataFrame, period_2_df: pd.DataFrame
  ) -> formatter.TableSpec:
    """Creates the Contribution comparison table spec."""

    # Column headers for the comparison table
    column_names = [
        'Channel',
        'Period 1',
        'Period 2',
        'Var pp',
    ]

    # Define Contribution comparison values
    contribution_period_1 = period_1_df.set_index('channel')[
        c.PCT_OF_CONTRIBUTION
    ].div(100)
    contribution_period_2 = period_2_df.set_index('channel')[
        c.PCT_OF_CONTRIBUTION
    ].div(100)
    contribution_comparison_df = pd.DataFrame(
        {
            'period_1': contribution_period_1,
            'period_2': contribution_period_2,
        }
    ).fillna(0)
    contribution_comparison_df['var_abs'] = (
        contribution_comparison_df['period_1']
        - contribution_comparison_df['period_2']
    )
    contribution_df = contribution_comparison_df.reset_index()

    # Replace 'All Channels' with 'MediaAtr' in channel names
    contribution_df['channel'] = contribution_df['channel'].replace(
        'All Channels', 'MediaAtr'
    )

    # Create 'Baseline' row
    baseline_pct_1 = (
        1
        - contribution_df[contribution_df['channel'] == 'MediaAtr'][
            'period_1'
        ].item()
    )
    baseline_pct_2 = (
        1
        - contribution_df[contribution_df['channel'] == 'MediaAtr'][
            'period_2'
        ].item()
    )
    baseline_row = {
        'channel': 'Baseline',
        'period_1': baseline_pct_1,
        'period_2': baseline_pct_2,
        'var_abs': baseline_pct_1 - baseline_pct_2,
    }
    contribution_df = pd.concat(
        [contribution_df, pd.DataFrame([baseline_row])]
    ).reset_index(drop=True)

    # Format Contribution comparison values
    fmt_pct = partial(formatter.format_percent_cm, decimals=1)
    fmt_var_pp = partial(formatter.format_var_pp, decimals=2)

    contribution_df['period_1'] = contribution_df['period_1'].apply(fmt_pct)
    contribution_df['period_2'] = contribution_df['period_2'].apply(fmt_pct)
    contribution_df['var_abs'] = contribution_df['var_abs'].apply(fmt_var_pp)

    # Create TableSpec for Contribution comparison
    contribution_resume_table = formatter.TableSpec(
        id=summary_text.CONTRIBUTION_COMPARISON_ID,
        title=summary_text.CONTRIBUTION_COMPARISON_TITLE,
        description=summary_text.CONTRIBUTION_COMPARISON_DESCRIPTION,
        column_headers=column_names,
        row_values=contribution_df.values.tolist(),
    )

    # Save data as Parquet
    file_name = (
        summary_text.CONTRIBUTION_COMPARISON_ID.replace(' ', '_') + '.parquet'
    )
    df = contribution_resume_table.to_dataframe()
    self._comparison_df_to_parquet(df, file_name)

    return contribution_resume_table

  def _create_kpi_contribution_comparison_table_spec(
      self,
      period_1_df: pd.DataFrame,
      period_2_df: pd.DataFrame,
      kpi_period_1: float,
      kpi_period_2: float,
  ) -> formatter.TableSpec:
    """Creates the KPI Contribution comparison table spec."""

    # Column headers for the comparison table
    column_names = [
        'Channel',
        'Period 1',
        'Period 2',
        'Var %',
        'Var Abs',
    ]

    # Define KPI Contribution comparison values
    # TODO: Debería tomarse de c.INCREMENTAL_OUTCOME y evitar el cálculo manual.
    kpi_contribution_period_1 = (
        period_1_df.set_index('channel')[c.PCT_OF_CONTRIBUTION].div(100)
        * kpi_period_1
    )
    kpi_contribution_period_2 = (
        period_2_df.set_index('channel')[c.PCT_OF_CONTRIBUTION].div(100)
        * kpi_period_2
    )
    kpi_contribution_comparison_df = pd.DataFrame(
        {
            'period_1': kpi_contribution_period_1,
            'period_2': kpi_contribution_period_2,
        }
    ).fillna(0)

    kpi_contribution_comparison_df['var_pct'] = (
        kpi_contribution_comparison_df['period_1']
        / kpi_contribution_comparison_df['period_2']
    ) - 1
    kpi_contribution_comparison_df['var_abs'] = (
        kpi_contribution_comparison_df['period_1']
        - kpi_contribution_comparison_df['period_2']
    )
    kpi_contribution_comparison_df = (
        kpi_contribution_comparison_df.reset_index()
    )

    # Replace 'All Channels' with 'MediaAtr' in channel names
    kpi_contribution_comparison_df['channel'] = kpi_contribution_comparison_df[
        'channel'
    ].replace('All Channels', 'MediaAtr')

    # Create "Baseline" row
    baseline_pct_1 = (
        kpi_period_1
        - kpi_contribution_comparison_df[
            kpi_contribution_comparison_df['channel'] == 'MediaAtr'
        ]['period_1'].item()
    )
    baseline_pct_2 = (
        kpi_period_2
        - kpi_contribution_comparison_df[
            kpi_contribution_comparison_df['channel'] == 'MediaAtr'
        ]['period_2'].item()
    )
    baseline_row = {
        'channel': 'Baseline',
        'period_1': baseline_pct_1,
        'period_2': baseline_pct_2,
        'var_pct': (baseline_pct_1 / baseline_pct_2) - 1,
        'var_abs': baseline_pct_1 - baseline_pct_2,
    }
    kpi_contribution_comparison_df = pd.concat(
        [kpi_contribution_comparison_df, pd.DataFrame([baseline_row])]
    ).reset_index(drop=True)

    # Format KPI Contribution comparison values
    fmt_num = partial(formatter.format_number_cm, decimals=0)
    fmt_var_pct = partial(formatter.format_var_percent, decimals=2)
    fmt_var_num = partial(formatter.format_var_number, decimals=0)

    kpi_contribution_comparison_df['period_1'] = kpi_contribution_comparison_df[
        'period_1'
    ].apply(fmt_num)
    kpi_contribution_comparison_df['period_2'] = kpi_contribution_comparison_df[
        'period_2'
    ].apply(fmt_num)
    kpi_contribution_comparison_df['var_pct'] = kpi_contribution_comparison_df[
        'var_pct'
    ].apply(fmt_var_pct)
    kpi_contribution_comparison_df['var_abs'] = kpi_contribution_comparison_df[
        'var_abs'
    ].apply(fmt_var_num)

    # Create TableSpec for KPI Contribution comparison
    kpi_contribution_resume_table = formatter.TableSpec(
        id=summary_text.KPI_CONTRIBUTION_COMPARISON_ID,
        title=summary_text.KPI_CONTRIBUTION_COMPARISON_TITLE,
        description=summary_text.KPI_CONTRIBUTION_COMPARISON_DESCRIPTION,
        column_headers=column_names,
        row_values=kpi_contribution_comparison_df.values.tolist(),
    )

    # Save data as Parquet
    file_name = (
        summary_text.KPI_CONTRIBUTION_COMPARISON_ID.replace(' ', '_')
        + '.parquet'
    )
    df = kpi_contribution_resume_table.to_dataframe()
    self._comparison_df_to_parquet(df, file_name)

    return kpi_contribution_resume_table

  def _create_roi_comparison_table_spec(
      self,
      period_1_df: pd.DataFrame,
      period_2_df: pd.DataFrame,
      kpi_period_1: float,
      kpi_period_2: float,
  ) -> formatter.TableSpec:
    """Creates the ROI comparison table spec."""

    # Column headers for the comparison table
    column_names = [
        'Period 1',
        'Period 2',
        'Var %',
        'Var Abs',
    ]

    # Define ROI comparison values
    # TODO: Debería tomarse de c.ROI y evitar el cálculo manual.
    kpi_contribution_period_1 = (
        period_1_df.set_index('channel')[c.PCT_OF_CONTRIBUTION].div(100)
        * kpi_period_1
    )
    kpi_contribution_period_2 = (
        period_2_df.set_index('channel')[c.PCT_OF_CONTRIBUTION].div(100)
        * kpi_period_2
    )
    kpi_contribution_comparison_df = pd.DataFrame(
        {
            'period_1': kpi_contribution_period_1,
            'period_2': kpi_contribution_period_2,
        }
    ).fillna(0)

    spend_period_1 = period_1_df.set_index('channel')[c.SPEND]
    spend_period_2 = period_2_df.set_index('channel')[c.SPEND]
    spend_comparison_df = pd.DataFrame(
        {
            'period_1': spend_period_1,
            'period_2': spend_period_2,
        }
    ).fillna(0)

    # Calculate ROI for both periods
    roi_comparison_df = kpi_contribution_comparison_df.merge(
        spend_comparison_df,
        left_index=True,
        right_index=True,
        suffixes=('_kpi_contribution', '_spend'),
    )
    roi_comparison_df['roi_period_1'] = (
        roi_comparison_df['period_1_kpi_contribution']
        / roi_comparison_df['period_1_spend']
    )
    roi_comparison_df['roi_period_2'] = (
        roi_comparison_df['period_2_kpi_contribution']
        / roi_comparison_df['period_2_spend']
    )

    # Select only ROI columns and 'All Channels' row
    roi_comparison_df = roi_comparison_df.loc[['All Channels']]
    roi_comparison_df = roi_comparison_df.reset_index()
    roi_comparison_df = roi_comparison_df[['roi_period_1', 'roi_period_2']]

    # Calculate variances
    roi_comparison_df['var_pct'] = (
        roi_comparison_df['roi_period_1'] / roi_comparison_df['roi_period_2']
    ) - 1
    roi_comparison_df['var_abs'] = (
        roi_comparison_df['roi_period_1'] - roi_comparison_df['roi_period_2']
    )

    # Format ROI comparison values
    fmt_num = partial(formatter.format_number_cm, decimals=1)
    fmt_var_pct = partial(formatter.format_var_percent, decimals=2)
    fmt_var_num = partial(formatter.format_var_number, decimals=1)

    roi_comparison_df['roi_period_1'] = roi_comparison_df['roi_period_1'].apply(
        fmt_num
    )
    roi_comparison_df['roi_period_2'] = roi_comparison_df['roi_period_2'].apply(
        fmt_num
    )
    roi_comparison_df['var_pct'] = roi_comparison_df['var_pct'].apply(
        fmt_var_pct
    )
    roi_comparison_df['var_abs'] = roi_comparison_df['var_abs'].apply(
        fmt_var_num
    )

    # Create TableSpec for ROI comparison
    roi_comparison_table = formatter.TableSpec(
        id=summary_text.ROI_COMPARISON_ID,
        title=summary_text.ROI_COMPARISON_TITLE,
        description=summary_text.ROI_COMPARISON_DESCRIPTION,
        column_headers=column_names,
        row_values=roi_comparison_df.values.tolist(),
    )

    # Save data as Parquet
    file_name = summary_text.ROI_COMPARISON_ID.replace(' ', '_') + '.parquet'
    df = roi_comparison_table.to_dataframe()
    self._comparison_df_to_parquet(df, file_name)

    return roi_comparison_table

  def _create_spend_comparison_pie_chart_spec(
      self,
      period_1_df: pd.DataFrame,
      media_summary: visualizer.MediaSummary,
  ) -> formatter.ChartSpec:
    """Creates the spend comparison pie chart spec."""

    # Define Spend pie chart values
    spend_period_1 = period_1_df.set_index('channel')[c.SPEND]
    spend_period_1 = spend_period_1.fillna(0)

    # Filter > 0 spend channels and exclude 'All Channels'
    spend_pie_chart = spend_period_1[spend_period_1 > 0]
    spend_pie_chart = spend_pie_chart[spend_pie_chart.index != 'All Channels']
    spend_df = spend_pie_chart.reset_index()

    # Calculate % of total spend
    total_spend = spend_df[c.SPEND].sum()
    spend_df['pct_of_total_spend'] = spend_df[c.SPEND] / total_spend

    # Create ChartSpec for Spend pie chart
    spend_pie_chart_spec = self.customize_charts.register_chart_spec(
        formatter.ChartSpec(
            id=summary_text.SPEND_COMPARISON_CHART_ID,
            description=summary_text.SPEND_COMPARISON_CHART_DESCRIPTION,
            chart_json=media_summary.plot_spend_comparison_pie_chart(
                spend_df
            ).to_json(),
        ),
        chart_overrides=self.client_config.get(
            'html_reports.comparison_metrics_summary.spend-comparison-pie-chart'
        ),
    )

    return spend_pie_chart_spec

  def _create_contribution_comparison_pie_chart_spec(
      self,
      period_1_df: pd.DataFrame,
      media_summary: visualizer.MediaSummary,
  ) -> formatter.ChartSpec:
    """Creates the contribution comparison pie chart spec."""

    # Define contribution pie chart values
    contribution_period_1 = period_1_df.set_index('channel')[
        c.PCT_OF_CONTRIBUTION
    ].div(100)
    contribution_period_1 = contribution_period_1.fillna(0)

    # Filter > 0 contribution channels and exclude 'All Channels'
    contribution_pie_chart = contribution_period_1[contribution_period_1 > 0]
    contribution_pie_chart = contribution_pie_chart[
        contribution_pie_chart.index != 'All Channels'
    ]
    contribution_df = contribution_pie_chart.reset_index()

    # Calculate % of total contribution
    total_contribution = contribution_df[c.PCT_OF_CONTRIBUTION].sum()
    contribution_df['pct_of_total_contribution'] = (
        contribution_df[c.PCT_OF_CONTRIBUTION] / total_contribution
    )

    # Create ChartSpec for Spend pie chart
    contribution_pie_chart_spec = self.customize_charts.register_chart_spec(
        formatter.ChartSpec(
            id=summary_text.CONTRIBUTION_COMPARISON_CHART_ID,
            description=summary_text.CONTRIBUTION_COMPARISON_CHART_DESCRIPTION,
            chart_json=media_summary.plot_contribution_comparison_pie_chart(
                contribution_df
            ).to_json(),
        ),
        chart_overrides=self.client_config.get(
            'html_reports.comparison_metrics_summary.contribution-comparison-pie-chart'
        ),
    )

    return contribution_pie_chart_spec

  def _comparison_df_to_parquet(self, df: pd.DataFrame, file_name: str) -> None:
    """Saves a DataFrame as a Parquet file in the comparison metrics directory."""
    full_path = os.path.join(self.filepath_cm, 'cm_tables', file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df['product_or_service'] = self.prod_or_serv
    df['updated_time'] = self.utc_now
    df.to_parquet(full_path, index=False, engine='pyarrow')
    print(f'✅ Data saved to {full_path}')
