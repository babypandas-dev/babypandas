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
