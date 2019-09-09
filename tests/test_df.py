import pytest
import doctest
import re
import numpy as np
import babypandas.bpd as bpd
import pandas as pd

def test_df_length():
	'''For first test'''
	df = bpd.DataFrame().assign(name=['Al', 'Bob', 'Jill', 'Sally'],
								age=['20', '25', '22', '23'])
	assert(df.shape[0] == 4), 'Length incorrect'

#########
# Utils #
#########

df1 = bpd.DataFrame().assign(letter=['a', 'b', 'c'],
 						     count=[9, 3, 3],
 						     points=[1, 2, 2])
pdf1 = pd.DataFrame().assign(letter=['a', 'b', 'c'],
  						     count=[9, 3, 3],
 						     points=[1, 2, 2])

df2 = bpd.DataFrame().assign(col1=[5, 2, 7, 5],
						     col2=[2, 7, 1, 8],
			       	 		 col3=[6, 6, 1, 3],
							 col4=[5, 5, 5, 9])
pdf2 = pd.DataFrame().assign(col1=[5, 2, 7, 5],
						     col2=[2, 7, 1, 8],
			       	 		 col3=[6, 6, 1, 3],
							 col4=[5, 5, 5, 9])

df3 = bpd.DataFrame().assign(name=['dog', 'cat', 'pidgeon', 'chicken', 'snake'],
							 kind=['mammal', 'mammal', 'bird', 'bird', 'reptile'])
pdf3 = pd.DataFrame().assign(name=['dog', 'cat', 'pidgeon', 'chicken', 'snake'],
							 kind=['mammal', 'mammal', 'bird', 'bird', 'reptile'])

df4 = bpd.DataFrame().assign(kind=['mammal', 'bird', 'reptile'],
							 short=['m',  'b', 'r'])
pdf4 = pd.DataFrame().assign(kind=['mammal', 'bird', 'reptile'],
							 short=['m',  'b', 'r'])

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
	assert_df_equal(df1, pdf1)
	assert_df_equal(df2, pdf2)
	assert_df_equal(df3, pdf3)
	assert_df_equal(df4, pdf4)

def test_iloc():
	assert_series_equal(df2.iloc[0], pdf2.iloc[0])

def test_take():
	assert_df_equal(df1, pdf1, 'take', indices=[0, 2])

def test_drop():
	assert_df_equal(df1, pdf1, 'drop', columns=['count'])

def test_sample():
	assert_df_equal(df1, pdf1, 'sample', n=1, random_state=0)
	assert_df_equal(df1, pdf1, 'sample', n=2, random_state=50)

def test_get():
	assert_series_equal(df1, pdf1, 'get', key='letter')
	assert_df_equal(df1, pdf1, 'get', key=['letter', 'points'])

def test_assign():
	assert_df_equal(df2, pdf2, 'assign', col5=[1, 1, 1, 1])

def test_apply():
	f = lambda x: x + 2
	assert_df_equal(df2, pdf2, 'apply', func=f)

def test_sort_values():
	assert_df_equal(df1, pdf1, by='count')
	assert_df_equal(df2, pdf2, by='col2', ascending=False)

def test_describe():
	assert_df_equal(df2, pdf2, 'describe')

def test_groupby():
	gb = df3.groupby('kind')
	pgb = pdf3.groupby('kind')
	assert isinstance(gb, bpd.DataFrameGroupBy)
	assert_df_equal(gb.sum(), pgb.sum())

def test_reset_index():
	df = df2.sort_values(by='col2')
	pdf = pdf2.sort_values(by='col2')
	assert_df_equal(df, pdf, 'reset_index')
	assert_df_equal(df, pdf, 'reset_index', drop=True)

def test_set_index():
	assert_df_equal(df2, pdf2, 'set_index', keys='col1')
	assert_df_equal(df2, pdf2, 'set_index', keys='col1', drop=False)

def test_merge():
	assert_df_equal(df3.merge(df4), pdf3.merge(pdf4))

# # def test_append():
# # 	...

def test_to_numpy():
	assert isinstance(df1.to_numpy(), np.ndarray)