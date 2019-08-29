import pytest
import doctest
import re
import numpy as np
from numpy.testing import assert_array_equal
import babypandas.bpd as bpd

def test_df_length():
	'''For first test'''
	df = bpd.DataFrame().assign(name=['Al', 'Bob', 'Jill', 'Sally'],
								age=['20', '25', '22', '23'])
	assert(df.shape[0] == 4), 'Length incorrect'

#########
# Utils #
#########

def df1():
	'''Simple 3x3 table'''
	return bpd.DataFrame().assign(letter=['a', 'b', 'c'],
		 						  count=[9, 3, 3],
		 						  points=[1, 2, 2])

def df2():
	'''Table with only numbers'''
	return bpd.DataFrame().assign(col1=[5, 2, 7, 5],
								  col2=[2, 7, 1, 8],
								  col3=[6, 6, 1, 3],
								  col4=[5, 5, 5, 9])

def df3():
	'''Animals and types'''
	return bpd.DataFrame().assign(name=['dog', 'cat', 'pidgeon', 'chicken', 'snake'],
								  type=['mammal', 'mammal', 'bird', 'bird', 'reptile'])

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
	df = df1()
	assert_equal(df, 
	'''
		  letter  count  points
	0      a      9       1
	1      b      3       2
	2      c      3       2
	''')

def test_column():
	df = df1()
	assert_array_equal(df.get('letter'), ['a', 'b', 'c'])
	assert_array_equal(df.get('count'), [9, 3, 3])

def test_row():
	df = df2()
	assert_array_equal(df.iloc[0], [5, 2, 6, 5])

def test_take(): 
	df = df1()
	assert_equal(df.take([0, 2]), 
	'''
		  letter  count  points
	0      a      9       1
	2      c      3       2
	''')

def test_drop():
	df = df1()
	assert_equal(df.drop(columns=['count']), 
	'''
		  letter  points
	0      a       1
	1      b       2
	2      c       2
	''')
	assert_equal(df.drop(columns='count'), 
	'''
		  letter  points
	0      a       1
	1      b       2
	2      c       2
	''')

def test_sample():
	df = df1()
	assert_equal(df.sample(1, random_state=0), 
	'''
		letter  count  points
	2      c      3       2
	''')
	assert_equal(df.sample(2, random_state=50),
	'''
		letter  count  points
	1      b      3       2
	2      c      3       2
	''')

def test_apply():
	df = df2()
	f = lambda x: x + 2
	assert_equal(df.apply(f),
	'''
	   col1  col2  col3  col4
	0     7     4     8     7
	1     4     9     8     7
	2     9     3     3     7
	3     7    10     5    11
	''')
	