import pytest

from src.monitor import monitor_performance


def test_monitor_performance_normal():
    result = monitor_performance(predicted_ac_power=100.0, actual_ac_power=95.0, threshold=0.15)
    assert result.status == "Normal"
    assert result.percentage_difference == -5.0
    assert result.absolute_error == 5.0


def test_monitor_performance_anomaly():
    result = monitor_performance(predicted_ac_power=100.0, actual_ac_power=70.0, threshold=0.15)
    assert result.status == "Potential Performance Anomaly"
    assert result.percentage_difference == -30.0
    assert result.absolute_error == 30.0


def test_monitor_performance_zero_predicted():
    result = monitor_performance(predicted_ac_power=0.0, actual_ac_power=10.0)
    assert result.status == "Normal"
    assert result.predicted_ac_power == 0.0
    assert result.recommendation.startswith("No generation expected")
