#!/usr/bin/env python
# pyright: basic

from fractions import Fraction
from src.config import INVERSE_CLOCK_GRANULARITY


def float_to_clock(value: float) -> Fraction:
    return Fraction(round(value * INVERSE_CLOCK_GRANULARITY), INVERSE_CLOCK_GRANULARITY)


def str_to_clock(s: str) -> Fraction:
    return float_to_clock(float(s))


def clock_to_percent_str(clock: Fraction) -> str:
    return f"{float(100 * clock)}%"
