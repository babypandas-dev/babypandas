import pandas as pd

pd.set_option("display.max_rows", 10)

class DataFrame(object):
    '''
    Custom DataFrame Class; Pandas DataFrames with methods removed.

    :Example:
    >>> df = DataFrame.from_records([[1,2,3],[4,5,6]], columns=['a', 'b', 'c'])
    >>> df.shape
    (2, 3)
    >>> df.assign(d=[1,2]).shape
    (2, 4)
    >>> df.loc[1, 'b']
    5
    '''

    def __init__(self, **kwargs):
        
        # hidden pandas dataframe object
        self._pd = pd.DataFrame(**kwargs)
        
        # lift loc/iloc back to custom DataFrame objects
        self.loc = DataFrameIndexer(self._pd.loc)
        self.iloc = DataFrameIndexer(self._pd.iloc)

        # Properties
        self.shape = _lift_to_pd(self._pd.shape)
        self.columns = _lift_to_pd(self._pd.columns)
        self.index = _lift_to_pd(self._pd.index)
        self.values = _lift_to_pd(self._pd.values)
        self.T = _lift_to_pd(self._pd.T)


    # Formatting
    def __repr__(self):
        return self._pd.__repr__()

    def __str__(self):
        return self._pd.__str__()

    # return the underlying DataFrame
    def to_df(self):
        '''return the full pandas dataframe'''
        return self._pd
    
    # Creation
    @classmethod
    def from_dict(cls, data):
        return cls(data=data)
        
    @classmethod
    def from_records(cls, data, columns):
        
        return cls(data=data, columns=columns)

    # Dunder Attributes
    def __repr_html__():
        # TODO
        return

    # Selection
    def take(self, indices, axis=0, is_copy=True):
        '''
        Return the elements in the given positional indices along an axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.take(**inp))

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        '''
        Drop specified labels from rows or columns.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.drop(**inp))

    def sample(sself, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None):
        '''
        Return a random sample of items from an axis of object.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.sample(**inp))

    def get(self, key, default=None):
        '''
        Get item from object for given key (ex: DataFrame column).
        '''
        return _lift_to_pd(self._pd.get(key, default))

    # Creation
    def assign(self, **kwargs):
        '''
        Assign new columns to a DataFrame.
        '''
        return _lift_to_pd(self._pd.assign(**kwargs))

    # Transformation
    def apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        '''
        Apply a function along an axis of the DataFrame.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.apply(**inp))

    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        '''
        Sort by the values along either axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.sort_values(**inp))

    def describe(self):
        '''
        Generate descriptive statistics that summarize the central 
        tendency, dispersion and shape of a dataset’s distribution.
        '''
        return _lift_to_pd(self._pd.describe())

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, observed=False, **kwargs):
        '''
        Group DataFrame or Series using a mapper or by a Series of columns.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.groupby(**inp))

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        '''
        Reset the index of the DataFrame, and use the default one 
        instead. If the DataFrame has a MultiIndex, this method can 
        remove one or more levels.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.reset_index(**inp))

    # Combining
    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        '''
        Merge DataFrame or named Series objects with a database-style join.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.merge(**inp))

    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        '''
        Append rows of other to the end of caller, returning a new object.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.append(**inp))

    # Plotting
    def plot(self, *args, **kwargs):
        '''
        Plot the data in the DataFrame.
        '''
        return _lift_to_pd(self.pd_plot(*args, **kwargs))

    # IO
    def to_csv(self, path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.'):
        '''
        Write object to a comma-separated values (csv) file.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.to_csv(**inp))

    def to_numpy(self, dtype=None, copy=False):
        '''
        Convert the DataFrame to a NumPy array.
        '''
        return _lift_to_pd(self._pd.to_numpy(dtype, copy))

    # Calculations
    def count(self, axis=0, level=None, numeric_only=False):
        '''
        Count cells for each column or row.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.count(**inp))

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the mean of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.mean(**inp))

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the median of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.median(**inp))

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the minimum of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.min(**inp))

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the maximum of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.max(**inp))

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        '''
        Return the sum of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.sum(**inp))

    def abs(self):
        '''
        Return a Series/DataFrame with absolute numeric value of each element.
        '''
        return _lift_to_pd(self._pd.abs())

class Series(object):
    '''
    Custom Series class; Pandas Series with methods removed.
    '''

    def __init__(self, **kwargs):
        
        # hidden pandas dataframe object
        self._pd = pd.Series(**kwargs)
        
        # lift loc/iloc back to custom DataFrame objects
        self.loc = DataFrameIndexer(self._pd.loc)
        self.iloc = DataFrameIndexer(self._pd.iloc)

        # List of Pandas DataFrame methods to be made "public"
        _props = ['shape']
        _selection = ['take', 'sample']
        _transformation = ['apply', 'sort_values', 'describe', 'reset_index'] # added reset_index
        _plotting = ['plot']
        _io = ['to_csv', 'to_numpy']
        
        _attrs = (
            _props + _selection +
            _transformation + _plotting + _io + _calcs)

        for meth in _attrs:
            setattr(self, meth, _lift_to_pd(getattr(self._pd, meth)))
            # self.__dict__[meth] = _lift_to_pd(getattr(self._pd, meth))

    # Formatting
    def __repr__(self):
        return self._pd.__repr__()

    def __str__(self):
        return self._pd.__str__()

    # Calculations
    def count(self, level=None):
        '''
        Return number of observations in the Series
        '''
        return _lift_to_pd(self._pd.count(level))

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the mean of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.mean(**inp))

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the median of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.median(**inp))

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the minimum of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.min(**inp))

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        '''
        Return the maximum of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.max(**inp))

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs):
        '''
        Return the sum of the values for the requested axis.
        '''
        inp = locals()
        del inp['self']
        return _lift_to_pd(self._pd.sum(**inp))

    def abs(self):
        '''
        Return a Series with absolute numeric value of each element.
        '''
        return _lift_to_pd(self._pd.abs())

    # Arithmetic
    def __add__(self, other):
        f = _lift_to_pd(self._pd.__add__)
        return f(other)

    def __mul__(self, other):
        f = _lift_to_pd(self._pd.__mul__)
        return f(other)

    def __rmul__(self, other):
        f = _lift_to_pd(self._pd.__rmul__)
        return f(other)

    def __pow__(self, other):
        f = _lift_to_pd(self._pd.__pow__)
        return f(other)

    def __sub__(self, other):
        f = _lift_to_pd(self._pd.__sub__)
        return f(other)

    def __truediv__(self, other):
        f = _lift_to_pd(self._pd.__truediv__)
        return f(other)

    # comparison
    def __eq__(self, other):
        f = _lift_to_pd(self._pd.__eq__)
        return f(other)

    def __ne__(self, other):
        f = _lift_to_pd(self._pd.__ne__)
        return f(other)

    def __gt__(self, other):
        f = _lift_to_pd(self._pd.__gt__)
        return f(other)

    def __lt__(self, other):
        f = _lift_to_pd(self._pd.__lt__)
        return f(other)

    def __ge__(self, other):
        f = _lift_to_pd(self._pd.__ge__)
        return f(other)

    def __le__(self, other):
        f = _lift_to_pd(self._pd.__le__)
        return f(other)

    # othe dunder methods
    def __len__(self):
        return self._pd.__len__()

    # array interface (for applying numpy functions)
    def __array__(self, *vargs, **kwargs):
        return self._pd.__array__(*vargs, **kwargs)

    # return the underlying Series
    def to_ser(self):
        '''return the full pandas series'''
        return self._pd


class DataFrameGroupBy(object):
    '''
    '''

    def __init__(self, groupby):
        
        # hidden pandas dataframe object
        self._pd = groupby
        
        # List of Pandas methods to be made "public".
        _attrs = ['count', 'mean', 'median', 'min', 'max', 'sum', 'size'] 

        for meth in _attrs:
            setattr(self, meth, _lift_to_pd(getattr(self._pd, meth)))

    # return the underlying groupby object
    def to_gb(self):
        '''return the full pandas dataframe'''
        return self._pd

    def aggregate(self, func):
        if not callable(func):
            raise Exception('Provide a function to aggregate')

        return self._pd.aggregate(func)
    

class DataFrameIndexer(object):
    '''
    Class for lifting results of loc/iloc back to the
    custom DataFrame class.
    '''
    def __init__(self, indexer):        
        self.idx = indexer
        
    def __getitem__(self, item):

        # convert to pandas if item is baby-pandas object
        try:
            item = item._pd
        except AttributeError:
            pass

        # TODO: restrict what item can be? (e.g. boolean array)
        data = self.idx[item]

        if isinstance(data, pd.DataFrame):
            return DataFrame(data=data)
        elif isinstance(data, pd.Series):
            return Series(data=data)
        else:
            return data


def _lift_to_pd(func):
    '''checks output-type of function and if output is a
    Pandas object, lifts the output to a babypandas class'''

    if not callable(func):
        return func

    types = (DataFrame, DataFrameGroupBy, Series)

    def closure(*vargs, **kwargs):
        vargs = [x._pd if isinstance(x, types) else x for x in vargs]
        kwargs = {k: x._pd if isinstance(x, types) else x 
                  for (k, x) in kwargs.items()}

        a = func(*vargs, **kwargs)
        if isinstance(a, pd.DataFrame):
            return DataFrame(data=a)
        elif isinstance(a, pd.Series):
            return Series(data=a)
        elif isinstance(a, pd.core.groupby.generic.DataFrameGroupBy):
            return DataFrameGroupBy(a)
        else:
            return a

    closure.__doc__ = func.__doc__

    return closure


def read_csv(filepath, **kwargs):
    '''read_csv'''
    df = pd.read_csv(filepath, **kwargs)
    return DataFrame(data=df)
