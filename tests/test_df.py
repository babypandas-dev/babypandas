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

@pytest.fixture(scope='function')
def dfs():
    inputs = []
    # df1 input: Strings and numbers DataFrame
    inputs.append({'data': {'letter': ['a', 'b', 'c'],
                            'count': [9,3,3],
                            'idx': [0,1,2]}})
    # df2 input: All numbers DataFrame
    inputs.append({'data': {'col1': [5, 2, 7, 5],
                            'col2': [2, 7, 1, 8],
                            'col3': [6, 6, 1, 3],
                            'col4': [5, 5, 5, 9]}})
    # df3 input: DataFrame with groups
    inputs.append({'data': {'name': ['dog', 'cat', 'pidgeon', 'chicken', 'snake'],
                            'kind': ['mammal', 'mammal', 'bird', 'bird', 'reptile']}})
    # df4 input: DataFrame for merge
    inputs.append({'data': {'kind': ['mammal', 'bird', 'reptile'],
                            'short': ['m',  'b', 'r']}})
    # df5 input: DataFrame for append
    inputs.append({'data': {'letter': ['d' ,'e'],
                            'count': [4, 1],
                            'idx': [3, 4]}})
    # df6 input: DataFrame for merge
    inputs.append({'data': {'kind': ['mammal', 'bird', 'reptile'],
                            'len': [6,  4, 7]}})

    dct = {}
    for key in range(len(inputs)):
        dct['df{}'.format(key + 1)] = (bpd.DataFrame(**inputs[key]), pd.DataFrame(**inputs[key]))
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

def test_merge_on_index(dfs):
    df3, pdf3 = dfs['df3']
    df4, pdf4 = dfs['df4']
    df4 = df4.set_index('kind')
    pdf4 = pdf4.set_index('kind')
    assert_df_equal(df3.merge(df4, left_on='kind', right_index=True), pdf3.merge(pdf4, left_on='kind', right_index=True))

def test_merge_on_both_index(dfs):
    df4, pdf4 = dfs['df4']
    df6, pdf6 = dfs['df6']
    df4 = df4.set_index('kind')
    pdf4 = pdf4.set_index('kind')
    df6 = df6.set_index('kind')
    pdf6 = pdf6.set_index('kind')
    assert_df_equal(df4.merge(df6, left_index=True, right_index=True), pdf4.merge(pdf6, left_index=True, right_index=True))

def test_append(dfs):
    df1, pdf1 = dfs['df1']
    df5, pdf5 = dfs['df5']
    assert_df_equal(df1.append(df5), pdf1.append(pdf5))
    assert_df_equal(df1.append(df5, ignore_index=True), pdf1.append(pdf5, ignore_index=True))

    # Exceptions
    assert pytest.raises(TypeError, df1.append, right=np.array([1, 2, 3]))
    assert pytest.raises(TypeError, df1.append, right=df5, ignore_index='foo')

def test_to_numpy(dfs):
    for df, pdf in dfs.values():
        assert isinstance(df.to_numpy(), np.ndarray)
        assert_array_equal(df.to_numpy(), pdf.to_numpy())


def test_indexing(dfs):
    # Check that boolean indexing works as expected.
    bp_df, df = dfs['df2']
    n, p = bp_df.shape
    col1_is_5 = bp_df.get('col1') == 5
    col4_is_5 = bp_df.get('col4') == 5
    # Simple indexing cases, Series, and array.
    for indexer in (col1_is_5, col4_is_5):
        for this_ind in (indexer, np.array(indexer)):
            indexed = bp_df[this_ind]
            assert indexed.shape[0] == np.count_nonzero(this_ind)
            assert list(indexed.index) == list(np.arange(n)[this_ind])
    # Sort Series index, and get the same output (because it depends on the
    # index).
    sorted_indexer = col1_is_5.sort_values()
    with pytest.warns(UserWarning):  # Reindex generates warning.
        indexed = bp_df[sorted_indexer]
    assert indexed.shape[0] == 2
    assert list(indexed.index) == [0, 3]
    # Any other type of indexing generates an error
    for indexer in ('col2', ['col1', 'col2'], 2, slice(1, 3),
                   (col1_is_5, 'col1')):
        with pytest.raises(IndexError):
            bp_df[indexer]

def test_transpose_produces_bpd_frame():
    """Check that .T produces a babypandas frame, not a pd.DataFrame."""
    df = bpd.DataFrame()
    df = df.assign(**{'foo': [1, 2, 3], 'bar': [4, 5, 6]})

    assert isinstance(df, bpd.DataFrame)
    assert isinstance(df.T, bpd.DataFrame)
