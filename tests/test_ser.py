import pytest
import doctest
import re
import numpy as np
from numpy.testing import assert_array_equal
import babypandas.bpd as bpd
import pandas as pd

#########
# Utils #
#########

def ser1():
	'''Ordered numbers'''
	return bpd.Series(data=[1, 2, 3, 4, 5])

def ser2():
	'''Letters'''
	return bpd.Series(data=['a', 'b', 'c', 'd'])

def ser3():
	'''Unordered numbers'''
	return bpd.Series(data=[7, 1, 3, 4, 2])

def assert_equal(string1, string2):
    string1, string2 = str(string1), str(string2)
    whitespace = re.compile(r'\s')
    purify = lambda s: whitespace.sub('', s)
    assert purify(string1) == purify(string2), "\n%s\n!=\n%s" % (string1, string2)

############
# Doctests #
############


# def test_doctests():
#     results = doctest.testmod(bpd, optionflags=doctest.NORMALIZE_WHITESPACE)
#     assert results.failed == 0

#########
# Tests #
#########

def test_basic():
	assert_array_equal(ser1().values, np.array([1, 2, 3, 4, 5]))
	assert_array_equal(ser2().values, np.array(['a', 'b', 'c', 'd']))
	assert_array_equal(ser3().values, np.array([7, 1, 3, 4, 2]))

def test_take():
	ser = ser1()
	assert_array_equal(ser.take([0, 2]).values, np.array([1, 3]))
	ser = ser3()
	assert_array_equal(ser.take([0, 2]).values, np.array([7, 3]))

def test_sample():
	ser = ser1()
	assert_array_equal(ser.sample(3, random_state=0).values, np.array([3, 1, 2]))
	assert_array_equal(ser.sample(8, replace=True, random_state=0).values, np.array([5, 1, 4, 4, 4, 2, 4, 3]))
	ser = ser2()
	assert_array_equal(ser.sample(random_state=0).values, np.array(['c']))

def test_apply():
	ser = ser1()
	f = lambda x: x + 2
	assert_array_equal(ser.apply(f).values, np.array([3, 4, 5, 6, 7]))

def test_sort_values():
	ser = ser3()
	assert_array_equal(ser.sort_values().values, np.array([1, 2, 3, 4, 7]))
	assert_array_equal(ser.sort_values().index, np.array([1, 4, 2, 3, 0]))
	assert_array_equal(ser.sort_values(ascending=False).values, np.array([7, 4, 3, 2, 1]))

def test_describe():
	ser = ser3()
	arr = pd.Series([7, 1, 3, 4, 2])
	assert_array_equal(ser.describe().values, np.array([5, arr.mean(), arr.std(), 1, 2, 3, 4, 7]))

def test_reset_index():
	ser = ser3().sort_values()
	assert_equal(ser.reset_index(),
	'''
	   index  0
	0      1  1
	1      4  2
	2      2  3
	3      3  4
	4      0  7
	''')
	assert_array_equal(ser.reset_index(drop=True).values, ser.values)
	assert_array_equal(ser.reset_index(drop=True).index, np.array([0, 1, 2, 3, 4]))

def test_to_numpy():
	ser = ser1()
	assert isinstance(ser.to_numpy(), np.ndarray)
