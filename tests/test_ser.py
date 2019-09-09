import pytest
import doctest
import re
import numpy as np
import babypandas.bpd as bpd
import pandas as pd

#########
# Utils #
#########

# Sorted Series
ser1 = bpd.Series(data=[1, 2, 3, 4, 5])
pser1 = pd.Series(data=[1, 2, 3, 4, 5])

# Numbered Series
ser2 = bpd.Series(data=['a', 'b', 'c', 'd'])
pser2 = pd.Series(data=['a', 'b', 'c', 'd'])

# Unsorted Series
ser3 = bpd.Series(data=[7, 1, 3, 4, 2])
pser3 = pd.Series(data=[7, 1, 3, 4, 2])

def assert_df_equal(df, pdf, method=None, **kwargs):
	if method:
		df = getattr(df, method)(**kwargs)
		pdf = getattr(pdf, method)(**kwargs)

	assert (np.all(df.columns == pdf.columns)), 'Columns do not match'
	assert (np.all(df.index == pdf.index)), 'Indices do not match'
	assert (np.all(df.values == pdf.values)), 'Values do not match'

def assert_series_equal(ser, pser, method=None, **kwargs):
	if method:
		ser = getattr(ser, method)(**kwargs)
		pser = getattr(pser, method)(**kwargs)

	assert (np.all(ser.index == pser.index)), 'Indices do not match'
	assert (np.all(ser.values == pser.values)), 'Values do not match'

#########
# Tests #
#########

def test_basic():
	assert_series_equal(ser1, pser1)
	assert_series_equal(ser2, pser2)
	assert_series_equal(ser3, pser3)

def test_take():
	assert_series_equal(ser1, pser1, 'take', indices=[0, 2])
	assert_series_equal(ser3, pser3, 'take', indices=[0, 2])

def test_sample():
	assert_series_equal(ser1, pser1, 'sample', n=3, random_state=0)
	assert_series_equal(ser1, pser1, 'sample', n=8, replace=True, random_state=0)
	assert_series_equal(ser2, pser2, 'sample', random_state=0)

def test_apply():
	f = lambda x: x + 2
	assert_series_equal(ser1, pser1, 'apply', func=f)

def test_sort_values():
	assert_series_equal(ser3, pser3, 'sort_values')
	assert_series_equal(ser3, pser3, 'sort_values', ascending=False)

def test_describe():
	assert_series_equal(ser3, pser3, 'describe')

def test_reset_index():
	ser = ser3.sort_values()
	pser = pser3.sort_values()
	assert_df_equal(ser, pser, 'reset_index')
	assert_series_equal(ser, pser, 'reset_index', drop=True)

def test_to_numpy():
	assert isinstance(ser1.to_numpy(), np.ndarray)
