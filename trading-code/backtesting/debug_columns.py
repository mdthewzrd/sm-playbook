#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))

import pandas as pd
from dual_deviation_cloud import DualDeviationCloud

# Create sample data
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

data.index = pd.date_range('2025-01-01', periods=5, freq='H')

# Initialize indicator
indicator = DualDeviationCloud({
    'ema_fast_length': 9,
    'ema_slow_length': 20,
    'positive_dev_1': 1.0,
    'positive_dev_2': 0.5,
    'negative_dev_1': 2.0,
    'negative_dev_2': 2.4
})

# Calculate indicators
result = indicator.calculate(data)

print("Available columns:")
for col in result.columns:
    print(f"  {col}")

print("\nSample values:")
print(result.head())