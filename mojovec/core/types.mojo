"""
Defines fundamental types and compile-time constants for metrics and quantizers.
"""

comptime MetricType = Int
comptime METRIC_L2 = 0
comptime METRIC_INNER_PRODUCT = 1

comptime QuantizerType = Int
comptime QT_8bit = 0
comptime QT_fp16 = 1
