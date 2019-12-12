import logging
import re
from datetime import datetime

import pytest

from openeo.util import first_not_none, get_temporal_extent, TimingLogger


@pytest.mark.parametrize(['input', 'expected'], [
    ((123,), 123),
    ((1, 2, 3), 1),
    ((0, 2, 3), 0),
    ((0.0, 2, 3), 0.0),
    ((None, 2, 3), 2),
    ((None, 'foo', 3), 'foo'),
    ((None, None, 3), 3),
    ((None, [], [1, 2]), []),
])
def test_first_not_none(input, expected):
    assert expected == first_not_none(*input)


def test_first_not_none_failures():
    with pytest.raises(ValueError):
        first_not_none()
    with pytest.raises(ValueError):
        first_not_none(None)
    with pytest.raises(ValueError):
        first_not_none(None, None)


def test_get_temporal_extent():
    assert get_temporal_extent("2019-03-15") == ("2019-03-15", None)
    assert get_temporal_extent("2019-03-15", "2019-10-11") == ("2019-03-15", "2019-10-11")
    assert get_temporal_extent(["2019-03-15", "2019-10-11"]) == ("2019-03-15", "2019-10-11")
    assert get_temporal_extent(("2019-03-15", "2019-10-11")) == ("2019-03-15", "2019-10-11")
    assert get_temporal_extent(extent=["2019-03-15", "2019-10-11"]) == ("2019-03-15", "2019-10-11")
    assert get_temporal_extent(extent=("2019-03-15", "2019-10-11")) == ("2019-03-15", "2019-10-11")
    assert get_temporal_extent(extent=(None, "2019-10-11")) == (None, "2019-10-11")
    assert get_temporal_extent(extent=("2019-03-15", None)) == ("2019-03-15", None)
    assert get_temporal_extent(start_date="2019-03-15", end_date="2019-10-11") == ("2019-03-15", "2019-10-11")
    assert get_temporal_extent(start_date="2019-03-15") == ("2019-03-15", None)
    assert get_temporal_extent(end_date="2019-10-11") == (None, "2019-10-11")


def test_timing_logger_basic(caplog):
    caplog.set_level(logging.INFO)
    with TimingLogger("Testing"):
        logging.info("Hello world")

    pattern = re.compile(r'''
        .*Testing.+start.+\d{4}-\d{2}-\d{2}\ \d{2}:\d{2}:\d{2}.*\n
        .*Hello\ world.*\n
        .*Testing.*end.+\d{4}-\d{2}-\d{2}\ \d{2}:\d{2}:\d{2}.*elapsed.+[0-9.]+\n
        .*
        ''', re.VERBOSE | re.DOTALL)
    assert pattern.match(caplog.text)


def test_timing_logger_custom():
    logs = []

    def logger(msg):
        logs.append(msg)

    timing_logger = TimingLogger("Testing", logger=logger)

    # Trick to have a "time" function that returns different times in subsequent calls
    times = [datetime(2019, 12, 12, 10, 10, 10, 10000), datetime(2019, 12, 12, 11, 12, 13, 14141)]
    timing_logger._now = iter(times).__next__

    with timing_logger:
        logger("Hello world")

    assert logs == [
        "Testing: start 2019-12-12 10:10:10.010000",
        "Hello world",
        "Testing: end 2019-12-12 11:12:13.014141, elapsed 1:02:03.004141"
    ]


def test_timing_logger_fail():
    logs = []

    def logger(msg):
        logs.append(msg)

    timing_logger = TimingLogger("Testing", logger=logger)

    # Trick to have a "time" function that returns different times in subsequent calls
    times = [datetime(2019, 12, 12, 10, 10, 10, 10000), datetime(2019, 12, 12, 11, 12, 13, 14141)]
    timing_logger._now = iter(times).__next__

    try:
        with timing_logger:
            raise ValueError("Hello world")
    except ValueError:
        pass

    assert logs == [
        "Testing: start 2019-12-12 10:10:10.010000",
        "Testing: fail 2019-12-12 11:12:13.014141, elapsed 1:02:03.004141"
    ]
