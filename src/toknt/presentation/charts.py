"""plotext chart builders.

Charts are built into strings (``plotext.build()``) so the CLI decides how to
print them and tests can assert content without a TTY. This module knows
nothing about the database — it renders :class:`~toknt.analytics.aggregation.Series`
values and plain label/value pairs.

Missing data conventions:

* line charts keep gaps (``None`` renders as ``nan`` — plotext skips it),
* bar charts cannot show gaps, so missing buckets render as zero.

Chart axes never silently re-bucket or drop data; the caller controls the
series.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from datetime import datetime, tzinfo

import plotext as plt

from toknt.analytics.aggregation import Series, default_timezone

_MIN_WIDTH = 50
_MAX_WIDTH = 140
_LINE_HEIGHT = 15
_BAR_HEIGHT = 13


def _figure(width: int | None, height: int) -> None:
    size = shutil.get_terminal_size(fallback=(110, 32))
    if width is None:
        width = max(_MIN_WIDTH, min(size.columns - 2, _MAX_WIDTH))
    plt.clear_figure()
    plt.theme("clear")
    plt.plotsize(width, height)


def _shut() -> str:
    out = plt.build()
    plt.clear_figure()
    return out


def _bucket_text(moment: datetime, bucket: str) -> str:
    if bucket == "hour":
        return moment.strftime("%d %b %H:%M")
    if bucket == "week":
        return "wk " + moment.strftime("%d %b")
    return moment.strftime("%d %b")


def _tick_indexes(count: int, max_ticks: int = 7) -> list[int]:
    if count <= 0:
        return []
    if count <= max_ticks:
        return list(range(count))
    step = (count - 1) / (max_ticks - 1)
    return sorted({round(index * step) for index in range(max_ticks)})


_DAY_MS = 86_400_000


def _tick_axes(
    buckets: Sequence[int],
    bucket: str,
    tz: tzinfo,
) -> tuple[list[int], list[str]]:
    """Tick positions and their labels for a time-series chart.

    Labels stay compact (``%d %b``), but once the span exceeds a year every
    tick gains its two-digit year: bare month/day labels repeat across a
    multi-year axis and make the chronology ambiguous.
    """
    count = len(buckets)
    ticks = _tick_indexes(count)
    if not ticks:
        return ticks, []
    span_over_year = buckets[-1] - buckets[0] > 365 * _DAY_MS
    labels = []
    for index in ticks:
        moment = datetime.fromtimestamp(buckets[index] / 1000, tz=tz)
        text = _bucket_text(moment, bucket)
        if span_over_year:
            text += " '" + moment.strftime("%y")
        labels.append(text)
    return ticks, labels


def nice_scale(max_value: float) -> tuple[float, str]:
    """Divisor and suffix that keep chart axes human-readable."""
    if max_value >= 1_000_000:
        return 1_000_000.0, "M"
    if max_value >= 1_000:
        return 1_000.0, "K"
    return 1.0, ""


def _log_tick_label(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def _log_axis(raw_max: float, max_log: float) -> tuple[float, list[float], list[str]]:
    """Top, tick positions and labels for a log-scaled token axis.

    Plotext's own ``yscale('log')`` raises on zeros and does not accept
    ``None`` gaps. The values are transformed with ``log1p`` first (the same
    transform the TUI uses), so a zero bucket sits on the baseline instead
    of breaking the line, and the tick labels stay in token units
    (1K, 1M, …) rather than log exponents.
    """
    if raw_max <= 1:
        exponents = [0]
    else:
        highest = int(math.floor(math.log10(raw_max)))
        step = max(1, math.ceil((highest + 1) / 5))
        exponents = list(range(0, highest + 1, step))
    positions = [math.log1p(10.0**exponent) for exponent in exponents]
    labels = [_log_tick_label(10.0**exponent) for exponent in exponents]
    top = max(max_log * 1.05, 1.0)
    return top, positions, labels


def _finite(value: float | None) -> float | None:
    """The value when it is a real number, else ``None`` (a gap)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def line_chart(
    series: Series,
    *,
    title: str,
    ylabel: str,
    tz: tzinfo | None = None,
    percent: bool = False,
    log: bool = False,
    width: int | None = None,
    height: int = _LINE_HEIGHT,
) -> str | None:
    """Multi-series line chart; ``None`` values render as gaps.

    ``log`` plots token counts on a log axis so a 5M-token series remains
    legible next to a 250M one. Rates and percentages stay linear: the
    ``percent`` flag wins over ``log``.
    """
    if series.is_empty:
        return None
    tz = tz or default_timezone()
    x = list(range(len(series.buckets)))
    # plotext 5.3.2 crashes on series without a single value (legend path);
    # skip those instead of plotting an invisible all-gap line.
    plotted = {
        key: [_finite(value) for value in values]
        for key, values in series.series.items()
    }
    plotted = {
        key: values
        for key, values in plotted.items()
        if any(value is not None for value in values)
    }
    if not plotted:
        return None
    use_log = log and not percent
    raw_max = max(
        (value for values in plotted.values() for value in values if value is not None),
        default=0,
    )
    if use_log:
        # log1p keeps a zero bucket on the baseline (the same transform the
        # TUI charts use) instead of breaking the line or inventing a floor
        plotted = {
            key: [
                None if value is None else math.log1p(max(value, 0.0))
                for value in values
            ]
            for key, values in plotted.items()
        }
        plotted = {
            key: values
            for key, values in plotted.items()
            if any(value is not None for value in values)
        }
        if not plotted:
            return None
        max_log = max(
            (
                value
                for values in plotted.values()
                for value in values
                if value is not None
            ),
            default=0,
        )
        top, y_positions, y_labels = _log_axis(raw_max, max_log)
        divisor, suffix = 1.0, ""
    else:
        max_value = raw_max
        divisor, suffix = (1.0, "") if percent else nice_scale(max_value)
        top = y_positions = y_labels = None

    _figure(width, height)
    for key, values in plotted.items():
        y = [None if value is None else value / divisor for value in values]
        # Labels render inline in plotext 5.x; keep them short enough to
        # stay out of the plot area.
        plt.plot(x, y, label=str(key)[:24])
    ticks, tick_labels = _tick_axes(series.buckets, series.bucket, tz)
    plt.xticks(ticks, tick_labels)
    plt.title(title)
    if use_log:
        plt.ylim(0, (top or 1.0) * 1.05)
        plt.yticks(y_positions or [], y_labels or [])
        plt.ylabel(f"{ylabel} (log)")
    else:
        plt.ylabel(f"{ylabel} ({suffix})" if suffix else ylabel)
    return _shut()


def bar_chart(
    series: Series,
    *,
    title: str,
    ylabel: str,
    stacked: bool = False,
    tz: tzinfo | None = None,
    percent: bool = False,
    width: int | None = None,
    height: int = _BAR_HEIGHT,
) -> str | None:
    """Grouped or stacked bar chart; missing buckets render as zero."""
    if series.is_empty:
        return None
    tz = tz or default_timezone()
    max_value = max(
        (value for values in series.series.values() for value in values if value is not None),
        default=0,
    )
    divisor, suffix = (1.0, "") if percent else nice_scale(max_value)

    _figure(width, height)
    data = [
        [
            0 if _finite(value) is None else _finite(value) / divisor  # type: ignore[operator]
            for value in values
        ]
        for values in series.series.values()
    ]
    names = [str(key)[:24] for key in series.series]
    if stacked:
        plt.stacked_bar(data, labels=names)
    else:
        # grouped bars: plotext's multiple_bar() takes one list of series
        # and a label per series (plain bar() draws a single series)
        plt.multiple_bar(data, labels=names)
    ticks, tick_labels = _tick_axes(series.buckets, series.bucket, tz)
    plt.xticks(ticks, tick_labels)
    plt.title(title)
    plt.ylabel(f"{ylabel} ({suffix})" if suffix else ylabel)
    return _shut()


def category_bar(
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    title: str,
    unit: str = "",
    width: int | None = None,
    height: int | None = None,
) -> str | None:
    """Horizontal bar chart for ranked categories (largest at the top).

    plotext renders the first category at the bottom, so pairs are sorted
    ascending by value to put the largest at the top.
    """
    pairs = [
        (str(label), float(value))
        for label, value in zip(labels, values, strict=True)
        if value is not None
    ]
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[1])
    # every category needs a bar row and a label row, or plotext squeezes
    # the smallest ones out of the plot
    chart_height = height or max(8, min(16, len(pairs) * 2 + 2))
    divisor, suffix = nice_scale(max(pair[1] for pair in pairs))
    _figure(width, chart_height)
    # a horizontal bar takes the heights alone; the categories are the
    # y-axis ticks (passing labels and values as two arguments would draw
    # a single bar)
    plt.bar([pair[1] / divisor for pair in pairs], orientation="horizontal")
    plt.yticks(list(range(1, len(pairs) + 1)), [pair[0] for pair in pairs])
    plt.title(title)
    if unit:
        plt.xlabel(f"{unit} ({suffix})" if suffix else unit)
    return _shut()
