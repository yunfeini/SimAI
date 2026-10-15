"""Shared presentation helpers; raw scheduler statistics are never modified."""

import math


DEFAULT_CURVE_BINS = 60
DEFAULT_MAX_LATENCY_JOBS = 15


def format_number(value):
    if not math.isfinite(value):
        return "N/A"
    if value and abs(value) < 0.01:
        return "{:.2g}".format(value)
    if abs(value) >= 10000:
        return "{:.3g}".format(value)
    return "{:.2f}".format(value).rstrip("0").rstrip(".")


def label_bars(axis, bars, rotation=0, counts=False):
    """Label bars and histogram bins, including zero-valued comparison bars."""
    for bar in bars:
        value = bar.get_height()
        if not math.isfinite(value) or (counts and value == 0):
            continue
        axis.annotate(str(int(value)) if counts else format_number(value),
                      (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 4), textcoords="offset points",
                      ha="center", va="bottom", rotation=rotation,
                      fontsize=8, clip_on=False)
    axis.margins(y=0.18)


def time_bin_curve(times, values, bins=DEFAULT_CURVE_BINS):
    """Average a post-event step signal over equal-duration time bins.

    Keep the last state at duplicate timestamps. Integrate duration, not
    sample count, so bursts of scheduler events do not bias the trend.
    Return bin centers, means, minima and maxima; bins=0 returns raw states.
    """
    if bins < 0:
        raise ValueError("curve bins cannot be negative")
    if len(times) != len(values):
        raise ValueError("curve timestamps and values must have equal lengths")
    states = []
    for time, value in sorted(zip(times, values), key=lambda item: item[0]):
        if not math.isfinite(time) or not math.isfinite(value):
            raise ValueError("curve values must be finite")
        if states and time == states[-1][0]:
            states[-1] = (time, value)
        else:
            states.append((time, value))
    if bins == 0 or len(states) <= bins or len(states) < 2:
        x = [item[0] for item in states]
        y = [item[1] for item in states]
        return x, y, list(y), list(y)

    start, end = states[0][0], states[-1][0]
    width = (end - start) / bins
    x, mean, low, high = [], [], [], []
    index = 0
    for bucket in range(bins):
        left = start + width * bucket
        right = end if bucket == bins - 1 else start + width * (bucket + 1)
        position = left
        area = 0.0
        minimum, maximum = math.inf, -math.inf
        while position < right:
            while index + 1 < len(states) and states[index + 1][0] <= position:
                index += 1
            stop = min(right, states[index + 1][0])
            value = states[index][1]
            area += value * (stop - position)
            minimum, maximum = min(minimum, value), max(maximum, value)
            position = stop
        x.append((left + right) / 2)
        mean.append(area / (right - left))
        low.append(minimum)
        high.append(maximum)
    return x, mean, low, high


def plot_curve(axis, times, values, label, color=None, bins=DEFAULT_CURVE_BINS):
    x, mean, low, high = time_bin_curve(times, values, bins)
    if not x:
        return
    # No interpolation/splines: averaged values stay inside the observed range.
    if low != high:
        start, end = min(times), max(times)
        x = [start] + x + [end]
        mean = mean[:1] + mean + mean[-1:]
        low = low[:1] + low + low[-1:]
        high = high[:1] + high + high[-1:]
        line, = axis.plot(x, mean, label=label, color=color, linewidth=2)
        axis.fill_between(x, low, high, color=line.get_color(), alpha=0.09,
                          linewidth=0)
    else:
        axis.step(x, mean, where="post", label=label, color=color, linewidth=1.8)
    axis.margins(x=0)


def curve_note(bins):
    if bins == 0:
        return "Curves: raw event states."
    return ("Dense curves: {} time bins, duration-weighted mean (sparse: raw); "
            "shading: min-max. Statistics: raw data.").format(bins)
