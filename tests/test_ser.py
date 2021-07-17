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
def sers():
    inputs = []
    # ser1 input: Sorted Series
    inputs.append({'data': [1, 2, 3, 4, 5]})
    # ser2 input: String Series
    inputs.append({'data': ['a', 'b', 'c', 'd']})
    # ser3 input: Unsorted Series
    inputs.append({'data': [7, 1, 3, 4, 2]})
    # ser4 input: Duplicates
    inputs.append({'data': [1, 1, 2, 2, 2, 3]})

    dct = {}
    for key in range(len(inputs)):
        dct['ser{}'.format(key + 1)] = (bpd.Series(**inputs[key]), pd.Series(**inputs[key]))
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

def test_basic(sers):
    for ser, pser in sers.values():
        assert_series_equal(ser, pser)

def test_take(sers):
    for ser, pser in sers.values():
        indices = np.random.choice(len(pser), 2, replace=False)
        assert_series_equal(ser, pser, 'take', indices=indices)

    # Exceptions
    ser1 = sers['ser1'][0]
    assert pytest.raises(TypeError, ser1.take, indices=0)
    assert pytest.raises(ValueError, ser1.take, indices=['foo'])
    assert pytest.raises(IndexError, ser1.take, indices=np.arange(6))

def test_sample(sers):
    for ser, pser in sers.values():
        assert_series_equal(ser, pser, 'sample', n=3, random_state=0)
        assert_series_equal(ser, pser, 'sample', n=8, replace=True, random_state=0)

    # Exceptions
    ser1 = sers['ser1'][0]
    assert pytest.raises(TypeError, ser1.sample, n='foo')
    assert pytest.raises(TypeError, ser1.sample, replace='foo')
    assert pytest.raises(TypeError, ser1.sample, random_state='foo')
    assert pytest.raises(ValueError, ser1.sample, n=8)

def test_get(sers):
    
    ser1 = bpd.Series(data=[1, 2, 3, 4])
    assert ser1.get(key=2) == 3

def test_apply(sers):
    ser1, pser1 = sers['ser1']
    f = lambda x: x + 2
    assert_series_equal(ser1, pser1, 'apply', func=f)

    # Exceptions
    assert pytest.raises(TypeError, ser1.apply, func='foo')

def test_sort_values(sers):
    for ser, pser in sers.values():
        assert_series_equal(ser, pser, 'sort_values')
        assert_series_equal(ser, pser, 'sort_values', ascending=False)

    # Exceptions
    ser3 = sers['ser3'][0]
    assert pytest.raises(TypeError, ser3.sort_values, ascending='foo')

def test_describe(sers):
    for ser, pser in sers.values():
        assert_series_equal(ser, pser, 'describe')

def test_unique(sers):
    for ser, pser in sers.values():
        assert_array_equal(ser, pser)

def test_reset_index(sers):
    for ser, pser in sers.values():
        ser = ser.sort_values()
        pser = pser.sort_values()
        assert_df_equal(ser, pser, 'reset_index')
        assert_series_equal(ser, pser, 'reset_index', drop=True)

    # Exceptions
    ser3 = sers['ser3'][0]
    assert pytest.raises(TypeError, ser3.reset_index, drop='foo')

def test_to_numpy(sers):
    for ser, pser in sers.values():
        assert isinstance(ser.to_numpy(), np.ndarray)
        assert_array_equal(ser.to_numpy(), pser.to_numpy())


def test_where():

    # replace with an array
    s = bpd.Series(data=[1, 5, 3, 5, 6])
    t = bpd.Series(data=[0, 5, 2, 5, 4])
    cond = s == 5
    other = np.array([0, 1, 2, 3, 4])
    assert_series_equal(s, t, method='where', cond=cond, other=other)

    # replace with a broadcasted float
    s = bpd.Series(data=[1, 5, 3, 5, 6])
    t = bpd.Series(data=[10, 5, 10, 5, 6])
    cond = s >= 5
    other = 10
    assert_series_equal(s, t, method='where', cond=cond, other=other)

    # throw an error if other is not supplied
    s = bpd.Series(data=[1, 5, 3, 5, 6])
    cond = s == 5
    assert pytest.raises(TypeError, s.where, cond=cond)
    

def test_indexing():
    # Check that boolean indexing works as expected.
    s = bpd.Series(data=[1, 5, 3, 5, 6])
    n = len(s)
    s_is_5 = s == 5
    # Simple indexing cases, Series, and array.
    for this_ind in (s_is_5, np.array(s_is_5)):
        indexed = s[this_ind]
        assert len(indexed) == np.count_nonzero(this_ind)
        assert list(indexed.index) == list(np.arange(n)[this_ind])
    # Sort Series index, and get the same output (because it depends on the
    # index).
    sorted_indexer = s_is_5.sort_values()
    indexed = s[sorted_indexer]
    assert len(indexed) == 2
    assert list(indexed.index) == [1, 3]
    # Any other type of indexing generates an error
    for indexer in (2, slice(1, 3)):
        with pytest.raises(IndexError):
            s[indexer]


def test_str():
    # check that we can use Series.str methods

    # given
    s = bpd.Series(data=['this', 'is', 'a', 'test'])

    # when
    result = s.str.upper()

    # then
    assert isinstance(result, bpd.Series)
    assert result.iloc[0] == 'THIS'
    assert result.iloc[1] == 'IS'
    assert result.iloc[2] == 'A'
    assert result.iloc[3] == 'TEST'


def test_str_dir():
    # check that Series.str is discoverable with dir()

    # given
    s = bpd.Series(data=['this', 'is', 'a', 'test'])

    # then
    assert 'isupper' in dir(s.str)
    assert 'wrap' in dir(s.str)
    assert '__lt__' not in dir(s.str)


def test_bitwise_and():
    # check that bitwise and between two Series works

    # given
    s = bpd.Series(data=[1,2,3,4])

    # when
    result = ((s >= 2) & (s <= 3))

    # then
    assert not result.iloc[0]
    assert result.iloc[1]
    assert result.iloc[2]
    assert not result.iloc[3]


def test_bitwise_or():
    # check that bitwise or between two Series works

    # given
    s = bpd.Series(data=[1,2,3,4])

    # when
    result = ((s > 2) | (s < 2))

    # then
    assert result.iloc[0]
    assert not result.iloc[1]
    assert result.iloc[2]
    assert result.iloc[3]


def test_bitwise_xor():
    # check that bitwise or between two Series works

    # given
    s = bpd.Series(data=[1,2,3,4])

    # when
    result = ((s >= 2) ^ (s <= 2))

    # then
    assert result.iloc[0]
    assert not result.iloc[1]
    assert result.iloc[2]
    assert result.iloc[3]


def test_unary_not():
    # check that ~x works on boolean series

    # given
    s = bpd.Series(data=[True, True, False])

    # when
    result = ~s

    # then
    assert not result.iloc[0]
    assert not result.iloc[1]
    assert result.iloc[2]


def test_reverse_add():
    # given
    s = bpd.Series(data=[1,2,3])

    # when
    t = 1 + s

    # then
    assert t.iloc[0] == 2
    assert t.iloc[1] == 3
    assert t.iloc[2] == 4


def test_reverse_subtract():
    # given
    s = bpd.Series(data=[1,2,3])

    # when
    t = 1 - s

    # then
    assert t.iloc[0] == 0
    assert t.iloc[1] == -1
    assert t.iloc[2] == -2


def test_negation():
    # given
    s = bpd.Series(data=[1,2,3])

    # when
    t = -s

    # then
    assert t.iloc[0] == -1
    assert t.iloc[1] == -2
    assert t.iloc[2] == -3
