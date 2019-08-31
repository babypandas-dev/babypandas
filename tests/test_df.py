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

def df4():
	'''Types'''
	return bpd.DataFrame().assign(type=['mammal', 'bird', 'reptile'],
								  short=['m',  'b', 'r'])

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
	assert_equal(df1(), 
	'''
		  letter  count  points
	0      a      9       1
	1      b      3       2
	2      c      3       2
	''')
	assert_equal(df2(),
	'''
	   col1  col2  col3  col4
	0     5     2     6     5
	1     2     7     6     5
	2     7     1     1     5
	3     5     8     3     9
	''')
	assert_equal(df3(),
	'''
	      name     type
	0      dog   mammal
	1      cat   mammal
	2  pidgeon     bird
	3  chicken     bird
	4    snake  reptile
	''')
	assert_equal(df4(),
	'''
	      type short
	0   mammal     m
	1     bird     b
	2  reptile     r
	''')

def test_iloc():
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

def test_get():
	df = df1()
	assert_array_equal(df.get('letter'), ['a', 'b', 'c'])
	assert_array_equal(df.get('count'), [9, 3, 3])

def test_assign():
	df = df2()
	assert_equal(df.assign(col5=[1, 1, 1, 1]),
	'''
	   col1  col2  col3  col4  col5
	0     5     2     6     5     1
	1     2     7     6     5     1
	2     7     1     1     5     1
	3     5     8     3     9     1
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

def test_sort_values():
	df = df1()
	assert_equal(df.sort_values(by='count'),
	'''
	  letter  count  points
	1      b      3       2
	2      c      3       2
	0      a      9       1
	''')
	df = df2()
	assert_equal(df.sort_values(by='col2', ascending=False),
	'''
	   col1  col2  col3  col4
	3     5     8     3     9
	1     2     7     6     5
	0     5     2     6     5
	2     7     1     1     5
	''')

def test_describe():
	df = df2()
	assert_equal(df.describe(),
	'''
	           col1      col2     col3  col4
	count  4.000000  4.000000  4.00000   4.0
	mean   4.750000  4.500000  4.00000   6.0
	std    2.061553  3.511885  2.44949   2.0
	min    2.000000  1.000000  1.00000   5.0
	25%    4.250000  1.750000  2.50000   5.0
	50%    5.000000  4.500000  4.50000   5.0
	75%    5.500000  7.250000  6.00000   6.0
	max    7.000000  8.000000  6.00000   9.0
	''')

def test_groupby():
	df = df3()
	gb = df.groupby('type')
	assert isinstance(gb, bpd.DataFrameGroupBy)
	assert_equal(gb.count(),
	'''
	         name
	type
	bird        2
	mammal      2
	reptile     1
	''')

def test_reset_index():
	df = df2().sort_values(by='col2')
	assert_equal(df.reset_index(),
	'''
	   index  col1  col2  col3  col4
	0      2     7     1     1     5
	1      0     5     2     6     5
	2      1     2     7     6     5
	3      3     5     8     3     9
	''')
	assert_equal(df.reset_index(drop=True),
	'''
		   col1  col2  col3  col4
	0     7     1     1     5
	1     5     2     6     5
	2     2     7     6     5
	3     5     8     3     9
	''')

def test_set_index():
	df = df2()
	assert_equal(df.set_index('col1'),
	'''
	      col2  col3  col4
	col1
	5        2     6     5
	2        7     6     5
	7        1     1     5
	5        8     3     9
	''')
	assert_equal(df.set_index('col1', drop=False),
	'''
	      col1  col2  col3  col4
	col1
	5        5     2     6     5
	2        2     7     6     5
	7        7     1     1     5
	5        5     8     3     9
	''')
