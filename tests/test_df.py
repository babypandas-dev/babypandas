import pytest
import doctest
import re
import numpy as np
import babypandas.bpd as bpd
import pandas as pd

#########
# Utils #
#########

df1_input = {'data': {'letter': ['a', 'b', 'c'], 'count': [9,3,3]}, 'index': [0,1,2]}
df2_input = {'data': {'col1': [5, 2, 7, 5], 'col2': [2, 7, 1, 8], 'col3': [6, 6, 1, 3], 'col4': [5, 5, 5, 9]}}
df3_input = {'data': {'name': ['dog', 'cat', 'pidgeon', 'chicken', 'snake'], 'kind': ['mammal', 'mammal', 'bird', 'bird', 'reptile']}}
df4_input = {'data': {'kind': ['mammal', 'bird', 'reptile'], 'short': ['m',  'b', 'r']}}

@pytest.fixture(scope='function')
def dfs():
	df1_input = {'data': {'letter': ['a', 'b', 'c'], 
						  'count': [9,3,3]}, 
						  'index': [0,1,2]}
	df2_input = {'data': {'col1': [5, 2, 7, 5], 
						  'col2': [2, 7, 1, 8], 
						  'col3': [6, 6, 1, 3], 
						  'col4': [5, 5, 5, 9]}}
	df3_input = {'data': {'name': ['dog', 'cat', 'pidgeon', 'chicken', 'snake'], 
						  'kind': ['mammal', 'mammal', 'bird', 'bird', 'reptile']}}
	df4_input = {'data': {'kind': ['mammal', 'bird', 'reptile'], 
						  'short': ['m',  'b', 'r']}}

	dct = {}
	dct['df1'] = (bpd.DataFrame(**df1_input), pd.DataFrame(**df1_input))
	dct['df2'] = (bpd.DataFrame(**df2_input), pd.DataFrame(**df2_input))
	dct['df3'] = (bpd.DataFrame(**df3_input), pd.DataFrame(**df3_input))
	dct['df4'] = (bpd.DataFrame(**df4_input), pd.DataFrame(**df4_input))
	return dct

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

def test_basic(dfs):
	for df, pdf in dfs.values():
		assert_df_equal(df, pdf)

def test_iloc(dfs):
	for df, pdf in dfs.values():
		assert_series_equal(df.iloc[0], pdf.iloc[0])

def test_take(dfs):
	for df, pdf in dfs.values():
		indices = np.random.choice(len(pdf), 2, replace=False)
		assert_df_equal(df, pdf, 'take', indices=indices)

	# Exceptions
	df1 = dfs['df1'][0]
	assert pytest.raises(TypeError, df1.take, indices=1)
	assert pytest.raises(ValueError, df1.take, indices=['foo'])
	assert pytest.raises(IndexError, df1.take, indices=np.arange(4))

def test_drop(dfs):
	for df, pdf in dfs.values():
		col = pdf.columns.to_series().sample()
		assert_df_equal(df, pdf, 'drop', columns=col)

	# Exceptions
	df1 = dfs['df1'][0]
	assert pytest.raises(TypeError, df1.drop, columns=0)
	assert pytest.raises(KeyError, df1.drop, columns=['count', 'foo'])

def test_sample(dfs):
	for df, pdf in dfs.values():
		assert_df_equal(df, pdf, 'sample', n=1, random_state=0)
		assert_df_equal(df, pdf, 'sample', n=2, random_state=50)

	# Exceptions
	df1 = dfs['df1'][0]
	assert pytest.raises(TypeError, df1.sample, n='foo')
	assert pytest.raises(TypeError, df1.sample, replace='foo')
	assert pytest.raises(TypeError, df1.sample, random_state='foo')
	assert pytest.raises(ValueError, df1.sample, n=8)

def test_get(dfs):
	for df, pdf in dfs.values():
		key = pdf.columns.to_series().sample(2).values
		assert_series_equal(df, pdf, 'get', key=key[0])
		assert_df_equal(df, pdf, 'get', key=key)

	# Exceptions
	df1 = dfs['df1'][0]
	assert pytest.raises(TypeError, df1.get, key=1)
	assert pytest.raises(KeyError, df1.get, key='foo')

def test_assign(dfs):
	df2, pdf2 = dfs['df2']
	assert_df_equal(df2, pdf2, 'assign', col5=[1, 1, 1, 1])

	# Exceptions
	assert pytest.raises(ValueError, df2.assign, col5=[1, 1, 1, 1], col6=[2, 2])
	assert pytest.raises(ValueError, df2.assign, col5=[1, 1])

def test_apply(dfs):
	df2, pdf2 = dfs['df2']
	f = lambda x: x + 2
	assert_df_equal(df2, pdf2, 'apply', func=f)

	# Exceptions
	assert pytest.raises(TypeError, df2.apply, func=2)
	assert pytest.raises(ValueError, df2.apply, func=f, axis=3)

def test_sort_values(dfs):
	for df, pdf in dfs.values():
		by = pdf.columns.to_series().sample().iloc[0]
		assert_df_equal(df, pdf, by=by)
		assert_df_equal(df, pdf, by=by, ascending=False)

	# Exceptions
	df1 = dfs['df1'][0]
	assert pytest.raises(TypeError, df1.sort_values, by=0)
	assert pytest.raises(KeyError, df1.sort_values, by='foo')
	assert pytest.raises(TypeError, df1.sort_values, by='count', ascending='foo')

def test_describe(dfs):
	for df, pdf in dfs.values():
		assert_df_equal(df, pdf, 'describe')

def test_groupby(dfs):
	df3, pdf3 = dfs['df3']
	gb = df3.groupby('kind')
	pgb = pdf3.groupby('kind')
	assert isinstance(gb, bpd.DataFrameGroupBy)
	assert_df_equal(gb.sum(), pgb.sum())

	# Exceptions
	assert pytest.raises(TypeError, df3.groupby, by=0)
	assert pytest.raises(KeyError, df3.groupby, by='foo')

def test_reset_index(dfs):
	for df, pdf in dfs.values():
		by = pdf.columns.to_series().sample().iloc[0]
		df = df.sort_values(by=by)
		pdf = pdf.sort_values(by=by)
		assert_df_equal(df, pdf, 'reset_index')
		assert_df_equal(df, pdf, 'reset_index', drop=True)

	# Exceptions
	df2 = dfs['df2'][0]
	assert pytest.raises(TypeError, df2.reset_index, drop='foo')

def test_set_index(dfs):
	for df, pdf in dfs.values():
		keys = pdf.columns.to_series().sample().iloc[0]
		assert_df_equal(df, pdf, 'set_index', keys=keys)
		assert_df_equal(df, pdf, 'set_index', keys=keys, drop=False)

	# Exceptions
	df2 = dfs['df2'][0]
	assert pytest.raises(TypeError, df2.set_index, keys=0)
	assert pytest.raises(KeyError, df2.set_index, keys='foo')
	assert pytest.raises(TypeError, df2.set_index, keys='col1', drop='foo')

def test_merge(dfs):
	df3, pdf3 = dfs['df3']
	df4, pdf4 = dfs['df4']
	assert_df_equal(df3.merge(df4), pdf3.merge(pdf4))

	# Exceptions
	assert pytest.raises(TypeError, df3.merge, right=np.array([1, 2, 3]))
	assert pytest.raises(ValueError, df3.merge, right=df4, how='on')
	assert pytest.raises(KeyError, df3.merge, right=df4, on='foo')
	assert pytest.raises(KeyError, df3.merge, right=df4, left_on='kind')
	assert pytest.raises(KeyError, df3.merge, right=df4, left_on='foo', right_on='kind')
	assert pytest.raises(KeyError, df3.merge, right=df4, left_on='kind', right_on='foo')

# # def test_append():
# # 	...

def test_to_numpy(dfs):
	df1 = dfs['df1'][0]
	assert isinstance(df1.to_numpy(), np.ndarray)