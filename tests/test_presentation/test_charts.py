"""Time-series chart tick labels.

Tick positions must be ascending and match their labels; labels stay compact
(``%d %b``) except on spans longer than a year, where each tick carries its
two-digit year because bare month/day labels repeat across the axis.
"""

from __future__ import annotations

from datetime import UTC, datetime

from toknt.presentation.charts import _tick_axes, _tick_indexes

_DAY_MS = 86_400_000
_EPOCH = 1_700_000_000_000  # 2023-11-14 UTC


def _daily_buckets(days: int) -> list[int]:
    return [_EPOCH + index * _DAY_MS for index in range(days)]


class TestTickAxes:
    def test_positions_and_labels_align(self) -> None:
        buckets = _daily_buckets(60)
        ticks, labels = _tick_axes(buckets, "day", UTC)
        assert ticks == _tick_indexes(len(buckets))
        assert len(labels) == len(ticks)

    def test_short_span_labels_stay_compact(self) -> None:
        _ticks, labels = _tick_axes(_daily_buckets(30), "day", UTC)
        assert all("'" not in label for label in labels)

    def test_multi_year_labels_carry_the_year(self) -> None:
        buckets = _daily_buckets(800)
        ticks, labels = _tick_axes(buckets, "day", UTC)
        expected = [
            datetime.fromtimestamp(buckets[index] / 1000, tz=UTC).strftime(
                "%d %b '%y"
            )
            for index in ticks
        ]
        assert labels == expected

    def test_hour_labels_include_time_of_day(self) -> None:
        buckets = [_EPOCH + index * 3_600_000 for index in range(10)]
        _ticks, labels = _tick_axes(buckets, "hour", UTC)
        expected = datetime.fromtimestamp(
            _EPOCH / 1000, tz=UTC
        ).strftime("%d %b %H:%M")
        assert labels[0] == expected


class TestChartRendering:
    """Every chart kind must render — plotext's API differs per plot type."""

    @staticmethod
    def _series():
        from toknt.analytics.aggregation import Series

        buckets = tuple(_daily_buckets(5))
        return Series(
            buckets,
            {
                "alpha": (10, 20, None, 40, 50),
                "beta": (5, 5, 5, 5, 5),
            },
            "model",
            "day",
        )

    def test_line_chart_renders(self) -> None:
        from toknt.presentation.charts import line_chart

        out = line_chart(self._series(), title="t", ylabel="tokens", tz=UTC)
        assert out and "t" in out

    def test_token_line_chart_can_use_a_log_axis(self) -> None:
        from toknt.analytics.aggregation import Series
        from toknt.presentation.charts import line_chart

        series = Series(
            tuple(_daily_buckets(5)),
            {"alpha": (5_000_000, 250_000_000, None, 50_000, 1_000_000)},
            "model",
            "day",
        )
        out = line_chart(series, title="t", ylabel="tokens", tz=UTC, log=True)
        assert out and "(log)" in out
        # labels stay in token units, not log exponents
        assert "1M" in out and ("100M" in out or "1B" in out)

    def test_log_chart_keeps_zero_buckets_on_the_baseline(self) -> None:
        from toknt.analytics.aggregation import Series
        from toknt.presentation.charts import line_chart

        series = Series(
            tuple(_daily_buckets(4)),
            {"alpha": (0, 5, 0, 10)},
            "model",
            "day",
        )
        out = line_chart(series, title="t", ylabel="tokens", tz=UTC, log=True)
        assert out and "(log)" in out
        # a zero is a real value on the baseline, not a missing bucket:
        # the series stays in the legend
        assert "alpha" in out

    def test_non_finite_values_do_not_crash_the_chart(self) -> None:
        from toknt.analytics.aggregation import Series
        from toknt.presentation.charts import line_chart

        series = Series(
            tuple(_daily_buckets(4)),
            {"alpha": (float("nan"), float("inf"), 1.0, 10.0)},
            "model",
            "day",
        )
        out = line_chart(series, title="t", ylabel="tokens", tz=UTC, log=True)
        assert out is not None

    def test_grouped_bar_chart_renders(self) -> None:
        from toknt.presentation.charts import bar_chart

        # plotext's bar() takes one series per call and a single legend
        # label; passing several series at once raises a TypeError
        out = bar_chart(self._series(), title="t", ylabel="tokens", tz=UTC)
        assert out and "t" in out
        assert "alpha" in out and "beta" in out

    def test_stacked_bar_chart_renders(self) -> None:
        from toknt.presentation.charts import bar_chart

        out = bar_chart(self._series(), title="t", ylabel="tokens", tz=UTC, stacked=True)
        assert out and "alpha" in out and "beta" in out

    def test_category_bar_chart_renders(self) -> None:
        from toknt.presentation.charts import category_bar

        out = category_bar(["one", "two"], [3, 5], title="t", unit="tokens")
        assert out and "one" in out
