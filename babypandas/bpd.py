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
    def _repr_html_(self):
        f = _lift_to_pd(self._pd._repr_html_)
        return f()

    # Selection
    def take(self, indices):
        '''
        Return the elements in the given positional indices along an axis.

        :param indices: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.take)
        return f(indices=indices)

    def drop(self, columns=None):
        '''
        Drop specified labels from rows or columns.

        :param columns: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.drop)
        return f(columns=columns)

    def sample(self, n=None, replace=False):
        '''
        Return a random sample of items from an axis of object.

        :param n: TODO
        :param replace: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.sample)
        return f(n=n, replace=replace)

    def get(self, key):
        '''
        Get item from object for given key (ex: DataFrame column).

        :param key: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.get)
        return f(key=key)
    # Creation
    def assign(self, **kwargs):
        '''
        Assign new columns to a DataFrame.

        :param kwargs: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.assign)
        return f(**kwargs)

    # Transformation
    def apply(self, func, axis=0):
        '''
        Apply a function along an axis of the DataFrame.

        :param func: TODO
        :param axis: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.apply)
        return f(func=func, axis=axis)

    def sort_values(self, by, ascending=True):
        '''
        Sort by the values along either axis.

        :param by: TODO
        :param ascending: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.sort_values)
        return f(by=by, ascending=ascending)

    def describe(self):
        '''
        Generate descriptive statistics that summarize the central 
        tendency, dispersion and shape of a dataset’s distribution.
        '''
        f = _lift_to_pd(self._pd.describe)
        return f()

    def groupby(self, by=None):
        '''
        Group DataFrame or Series using a mapper or by a Series of columns.

        :param by: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.groupby)
        return f(by=by)

    def reset_index(self, drop=False):
        '''
        Reset the index of the DataFrame, and use the default one 
        instead. If the DataFrame has a MultiIndex, this method can 
        remove one or more levels.

        :param drop: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.reset_index)
        return f(drop=drop)

    def set_index(self, keys, drop=True):
        '''
        Set the DataFrame index using existing columns.

        :param keys: TODO
        :param drop: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.set_index)
        return f(keys=keys, drop=drop)

    # Combining
    def merge(self, right, how='inner', on=None, left_on=None, right_on=None):
        '''
        Merge DataFrame or named Series objects with a database-style join.

        :param right: TODO
        :param how: TODO
        :param on: TODO
        :param left_on: TODO
        :param right_on: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.merge)
        return f(right=right, how=how, on=on, left_on=left_on, right_on=right_on)

    def append(self, other):
        '''
        Append rows of other to the end of caller, returning a new object.

        :param other: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.append)
        return f(other=other)

    # Plotting
    def plot(self, *args, **kwargs):
        '''
        Plot the data in the DataFrame.
        '''
        f = _lift_to_pd(self._pd.plot)
        return f(*args, **kwargs)

    # IO
    def to_csv(self, path_or_buf=None):
        '''
        Write object to a comma-separated values (csv) file.

        :param path_or_buf: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.to_csv)
        return f(path_or_buf=path_or_buf)

    def to_numpy(self):
        '''
        Convert the DataFrame to a NumPy array.
        '''
        f = _lift_to_pd(self._pd.to_numpy)
        return f()


class Series(object):
    '''
    Custom Series class; Pandas Series with methods removed.
    '''

    def __init__(self, **kwargs):
        
        # hidden pandas dataeriesframe object
        self._pd = pd.Series(**kwargs)
        
        # lift loc/iloc back to custom Series objects
        self.loc = DataFrameIndexer(self._pd.loc)
        self.iloc = DataFrameIndexer(self._pd.iloc)

        self.shape = _lift_to_pd(self._pd.shape)

    # Formatting
    def __repr__(self):
        return self._pd.__repr__()

    def __str__(self):
        return self._pd.__str__()

    # Selection
    def take(self, indices):
        '''
        Return the elements in the given positional indices along an axis.

        :param indices: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.take)
        return f(indices)

    def sample(self, n=None, replace=False):
        '''
        Return a random sample of items from an axis of object.
        
        :param n: TODO
        :param replace: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.sample)
        return f(n=n, replace=replace)

    # Transformation
    def apply(self, func):
        '''
        Invoke function on values of Series.

        :param func: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.apply)
        return f(func=func)

    def sort_values(self, ascending=True):
        '''
        Sort by the values

        :param ascending: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.sort_values)
        return f(ascending=ascending)

    def describe(self):
        '''
        Generate descriptive statistics that summarize the central tendency, 
        dispersion and shape of a dataset’s distribution.
        '''
        f = _lift_to_pd(self._pd.describe)
        return f()

    def reset_index(self, drop=False):
        '''
        Generate a new DataFrame or Series with the index reset.

        :param drop: TODO
        :return: TODO
        '''
        f = _lift_to_pd(self._pd.reset_index)
        return f(drop=drop)

    # Plotting
    def plot(self, *args, **kwargs):
        '''
        Plot the data in the DataFrame.
        '''
        f = _lift_to_pd(self._pd.plot)
        return f(*args, **kwargs)

    # IO
    def to_csv(self, *args, **kwargs):
        '''
        Write object to a comma-separated values (csv) file.
        '''
        f = _lift_to_pd(self._pd.to_csv)
        return f(*args, **kwargs)

    def to_numpy(self):
        '''
        A NumPy ndarray representing the values in this Series or Index.
        '''
        f = _lift_to_pd(self._pd.to_numpy)
        return f()

    # Calculations
    def count(self):
        '''
        Return number of observations in the Series
        '''
        f = _lift_to_pd(self._pd.count)
        return f()

    def mean(self):
        '''
        Return the mean of the values for the requested axis.
        '''
        f = _lift_to_pd(self._pd.mean)
        return f()

    def median(self):
        '''
        Return the median of the values for the requested axis.
        '''
        f = _lift_to_pd(self._pd.median)
        return f()

    def min(self):
        '''
        Return the minimum of the values for the requested axis.
        '''
        f = _lift_to_pd(self._pd.min)
        return f()

    def max(self):
        '''
        Return the maximum of the values for the requested axis.
        '''
        f = _lift_to_pd(self._pd.max)
        return f()

    def sum(self):
        '''
        Return the sum of the values for the requested axis.
        '''
        f = _lift_to_pd(self._pd.sum)
        return f()

    def abs(self):
        '''
        Return a Series with absolute numeric value of each element.
        '''
        f = _lift_to_pd(self._pd.abs)
        return f()

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

    def __mod__(self, other):
        f = _lift_to_pd(self._pd.__mod__)
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

    # return the underlying groupby object
    def to_gb(self):
        '''return the full pandas dataframe'''
        return self._pd

    def aggregate(self, func):
        if not callable(func):
            raise Exception('Provide a function to aggregate')

        return self._pd.aggregate(func)

    # Calculations
    def count(self):
        '''
        Compute count of group.
        '''
        f = _lift_to_pd(self._pd.count)
        return f()

    def mean(self):
        '''
        Compute mean of group.
        '''
        f = _lift_to_pd(self._pd.mean)
        return f()

    def median(self):
        '''
        Compute median of group.
        '''
        f = _lift_to_pd(self._pd.median)
        return f()

    def min(self):
        '''
        Compute min of group.
        '''
        f = _lift_to_pd(self._pd.min)
        return f()

    def max(self):
        '''
        Compute max of group.
        '''
        f = _lift_to_pd(self._pd.max)
        return f()

    def sum(self):
        '''
        Compute sum of group.
        '''
        f = _lift_to_pd(self._pd.sum)
        return f()

    def size(self):
        '''
        Compute group sizes.
        '''
        f = _lift_to_pd(self._pd.size)
        return f()
    

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
