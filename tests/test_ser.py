import pytest
import doctest
import re
import numpy as np
from numpy.testing import assert_array_equal
import babypandas.bpd as bpd

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
	assert_array_equal(ser1(), np.array([1, 2, 3, 4, 5]))
	assert_array_equal(ser2(), np.array(['a', 'b', 'c', 'd']))
	assert_array_equal(ser3(), np.array([7, 1, 3, 4, 2]))

def test_take():
	ser = ser1()
	assert_array_equal(ser.take([0, 2]), np.array([1, 3]))
	ser = ser3()
	assert_array_equal(ser.take([0, 2]), np.array([7, 3]))

# def test_sample():
# 	# TODO