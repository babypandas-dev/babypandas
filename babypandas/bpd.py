import pandas as pd
import numpy as np
from collections.abc import Iterable

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
        '''
        Create an empty DataFrame.
        '''
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
        '''Return the full pandas DataFrame.'''
        return self._pd
    
    # Creation
    @classmethod
    def from_dict(cls, data):
        '''
        Create a DataFrame from a dictionary.
        '''
        return cls(data=data)
        
    @classmethod
    def from_records(cls, data, columns):
        '''
        Create a DataFrame from a sequence of records.
        '''
        return cls(data=data, columns=columns)

    # Dunder Attributes
    def _repr_html_(self):
        f = _lift_to_pd(self._pd._repr_html_)
        return f()

    # Selection
    def take(self, indices):
        '''
        Return the elements in the given positional indices along an axis.

        :param indices: An array of ints indicating which positions to take.
        :type indices: list of ints
        :return: DataFrame with the given positional indices.
        :rtype: DataFrame
        :raises IndexError: if any `indices` are out of bounds with respect to DataFrame length.

        :example:
        >>> df = bpd.DataFrame().assign(name=['falcon', 'parrot', 'lion'],
        ...                             kind=['bird', 'bird', 'mammal'])
        >>> df
             name    kind
        0  falcon    bird
        1  parrot    bird
        2    lion  mammal
        >>> df.take([0, 2])
             name    kind
        0  falcon    bird
        2    lion  mammal
        '''
        if not isinstance(indices, Iterable):
            raise TypeError('Argument `indices` must be a list-like object')
        if not all(isinstance(x, (int, np.integer)) for x in indices):
            raise ValueError('Argument `indices` must only contain integers')
        if not all(x < self._pd.shape[0] for x in indices):
            raise IndexError('Indices are out-of-bounds')

        f = _lift_to_pd(self._pd.take)
        return f(indices=indices)

    def drop(self, columns=None):
        '''
        Drop specified labels from rows or columns.

        :param columns: Column labels to drop.
        :type columns: str label or list of str labels
        :return: DataFrame with the dropped columns.
        :rtype: DataFrame
        :raises KeyError: if `columns` not found in columns

        :example:
        >>> df = bpd.DataFrame().assign(A=[0, 4, 8],
        ...                             B=[1, 5, 9],
        ...                             C=[2, 6, 10],
        ...                             D=[3, 7, 11])
        >>> df
           A  B   C   D
        0  0  1   2   3
        1  4  5   6   7
        2  8  9  10  11
        >>> df.drop(columns=['B', 'C'])
           A   D
        0  0   3
        1  4   7
        2  8  11
        '''
        if not isinstance(columns, Iterable):
            raise TypeError('Argument `columns` must be a string label or list of string labels')
        mask = [columns not in self.columns] if isinstance(columns, str) else [x not in self.columns for x in columns]
        if any(mask):
            c = [columns] if isinstance(columns, str) else columns
            raise KeyError('{} not found in columns'.format(np.array(c)[mask]))

        f = _lift_to_pd(self._pd.drop)
        return f(columns=columns)

    def sample(self, n=None, replace=False, random_state=None):
        '''
        Return a random sample of items from an axis of object.

        :param n: Number of items from axis to return.
        :param replace: Sample with or without replacement.
        :param random_state: Seed for the random number generator
        :type n: int, optional
        :type replace: bool, default False
        :type random_state: int, optional
        :return: DataFrame with `n` randomly sampled rows.
        :rtype: DataFrame
        :raises ValueError: if a sample larger than the length of the DataFrame is taken without replacement.

        :example:
        >>> df = bpd.DataFrame().assign(letter=['a', 'b', 'c'],
        ...                             count=[9, 3, 3],
        ...                             points=[1, 2, 2])
        >>> df.sample(1, random_state=0)
            letter  count  points
        2      c      3       2
        '''
        if not isinstance(n, int) and n != None:
            raise TypeError('Argument `n` not an integer')
        if not isinstance(replace, bool):
            raise TypeError('Argument `replace` not a boolean')
        if not isinstance(random_state, int) and random_state != None:
            raise TypeError('Argument `random_state` must be an integer or None')
        if n != None and n > self._pd.shape[0] and replace == False:
            raise ValueError('Cannot take a larger sample than length of DataFrame when `replace=False`')

        f = _lift_to_pd(self._pd.sample)
        return f(n=n, replace=replace, random_state=random_state)

    def get(self, key):
        '''
        Get item from object for given key (ex: DataFrame column).

        :param key: Column label or list of column labels
        :type key: str label or list of str labels 
        :return: Series with the corresponding label or DataFrame with the corresponding labels
        :rtype: Series or DataFrame
        :raises KeyError: if `key` not found in columns

        :example:
        >>> df = bpd.DataFrame().assign(letter=['a', 'b', 'c'],
        ...                             count=[9, 3, 3],
        ...                             points=[1, 2, 2])
        >>> df.get('letter')
        0    a
        1    b
        2    c
        Name: letter, dtype: object
        >>> df.get(['count', 'points'])
           count  points
        0      9       1
        1      3       2
        2      3       2
        '''
        if not isinstance(key, str) and not isinstance(key, Iterable):
            raise TypeError('Argument `key` must be a string label or list of string labels')
        mask = [key not in self.columns] if isinstance(key, str) else [x not in self.columns for x in key]
        if any(mask):
            k = [key] if isinstance(key, str) else key
            raise KeyError('{} not found in columns'.format(np.array(k)[mask]))

        f = _lift_to_pd(self._pd.get)
        return f(key=key)

    # Creation
    def assign(self, **kwargs):
        '''
        Assign new columns to a DataFrame.

        :param kwargs: Keyword column names with a list of values.
        :return: DataFrame with the additional column(s).
        :rtype: DataFrame
        :raises ValueError: if columns have different lengths or if new columns have different lengths than the existing DataFrame

        :example:
        >>> df = bpd.DataFrame().assign(flower=['sunflower', 'rose'])
        >>> df.assign(color=['yellow', 'red'])
              flower   color
        0  sunflower  yellow
        1       rose     red        
        '''
        if len(set(map(len, kwargs.values()))) not in (0, 1):
            raise ValueError('Not all columns have the same length')
        if self._pd.shape[1] != 0:
            if len(list(kwargs.values())[0]) != self._pd.shape[0]:
                raise ValueError('New column does not have the same length as existing DataFrame')

        f = _lift_to_pd(self._pd.assign)
        return f(**kwargs)

    # Transformation
    def apply(self, func, axis=0):
        '''
        Apply a function along an axis of the DataFrame.

        :param func: Function to apply to each column or row.
        :param axis: Axis along which the function is applied:
            
            - 0 or 'index': apply function to each column.
            - 1 or 'columns': apply function to each row.

        :type func: function
        :type axis: {0 or ‘index’, 1 or ‘columns’}, default 0
        :return: Result of applying func along the given axis of the DataFrame.
        :rtype: Series or DataFrame

        :example:
        >>> def add_two(row):
        ...     return row + 2
        >>> df = bpd.DataFrame(A=[1, 1],
        ...                    B=[2, 2])
        >>> df.apply(add_two)
           A  B
        0  3  4
        1  3  4
        '''
        if not callable(func):
            raise TypeError('Argument `func` must be a function')
        if axis not in [0, 1]:
            raise ValueError('Argument `axis` must be either 0 or 1')

        f = _lift_to_pd(self._pd.apply)
        return f(func=func, axis=axis)

    def sort_values(self, by, ascending=True):
        '''
        Sort by the values along either axis.

        :param by: String label or list of string labels to sort by.
        :param ascending: Sort ascending vs. descending.
        :type by: str or list of str
        :type param: bool, default True
        :return: DataFrame with sorted values.
        :rtype: DataFrame
        :raises KeyError: if `by` not found in columns

        :example:
        >>> df = bpd.DataFrame().assign(name=['Sally', 'George', 'Bill', 'Ann'],
        ...                             age=[21, 25, 18, 28],
        ...                             height_cm=[161, 168, 171, 149])
        >>> df.sort_values(by='age')
             name  age  height_cm
        2    Bill   18        171
        0   Sally   21        161
        1  George   25        168
        3     Ann   28        149
        >>> df.sort_values(by='height_cm', ascending=False)
             name  age  height_cm
        2    Bill   18        171
        1  George   25        168
        0   Sally   21        161
        3     Ann   28        149
        '''
        if not isinstance(by, Iterable):
            raise TypeError('Argument `by` must be a string label or list of string labels')
        mask = [by not in self.columns] if isinstance(by, str) else [x not in self.columns for x in by]
        if any(mask):
            b = [by] if isinstance(by, str) else by
            raise KeyError('{} not found in columns'.format(np.array(b)[mask]))
        if not isinstance(ascending, bool):
            raise TypeError('Argument `ascending` must be a boolean')

        f = _lift_to_pd(self._pd.sort_values)
        return f(by=by, ascending=ascending)

    def describe(self):
        '''
        Generate descriptive statistics that summarize the central 
        tendency, dispersion and shape of a dataset’s distribution.

        :return: Summary statistics of the DataFrame provided.
        :rtype: DataFrame

        :example:
        >>> df = bpd.DataFrame().assign(A=[0, 10, 20],
        ...                             B=[1, 2, 3])
        >>> df.describe()
                  A    B
        count   3.0  3.0
        mean   10.0  2.0
        std    10.0  1.0
        min     0.0  1.0
        25%     5.0  1.5
        50%    10.0  2.0
        75%    15.0  2.5
        max    20.0  3.0
        '''
        f = _lift_to_pd(self._pd.describe)
        return f()

    def groupby(self, by=None):
        '''
        Group DataFrame or Series using a mapper or by a Series of columns.

        :param by: Used to determine the groups for the groupby.
        :type by: label, or list of labels
        :return: Groupby object that contains information about the groups.
        :rtype: DataFrameGroupBy
        :raises KeyError: if `by` not found in columns

        :example:
        >>> df =bpd.DataFrame(animal=['Falcon', 'Falcon', 'Parrot', 'Parrot'],
        ...                   max_speed=[380, 370, 24, 26])
        >>> df.groupby('animal').mean()
                max_speed
        animal
        Falcon      375.0
        Parrot       25.0
        '''
        if not isinstance(by, Iterable):
            raise TypeError('Argument `by` must be a string label or list of string labels')
        mask = [by not in self.columns] if isinstance(by, str) else [x not in self.columns for x in by]
        if any(mask):
            b = [by] if isinstance(by, str) else by
            raise KeyError('{} not found in columns'.format(np.array(b)[mask]))

        f = _lift_to_pd(self._pd.groupby)
        return f(by=by)

    def reset_index(self, drop=False):
        '''
        Reset the index of the DataFrame, and use the default one 
        instead. If the DataFrame has a MultiIndex, this method can 
        remove one or more levels.

        :param drop: Does not insert index as a column.
        :type drop: bool, default False
        :return: DataFrame with the new index.
        :rtype: DataFrame

        :example:
        >>> df = bpd.DataFrame().assign(name=['Sally', 'George', 'Bill', 'Ann'],
        ...                             age=[21, 25, 18, 28],
        ...                             height_cm=[161, 168, 171, 149])
        >>> sorted = df.sort_values(by='age')
        >>> sorted
             name  age  height_cm
        2    Bill   18        171
        0   Sally   21        161
        1  George   25        168
        3     Ann   28        149
        >>> sorted.reset_index(drop=True)
             name  age  height_cm
        0    Bill   18        171
        1   Sally   21        161
        2  George   25        168
        3     Ann   28        149

        '''
        if not isinstance(drop, bool):
            raise TypeError('Argument `drop` must be a boolean')

        f = _lift_to_pd(self._pd.reset_index)
        return f(drop=drop)

    def set_index(self, keys, drop=True):
        '''
        Set the DataFrame index using existing columns.

        :param keys: Key(s) to set index on.
        :param drop: Delete column(s) to be used as the new index.
        :type keys: str label or list of str labels
        :type drop: bool, default True
        :return: DataFrame with changed row labels.
        :rtype: DataFrame
        :raises KeyError: if `keys` not found in columns

        :example:
        >>> df = bpd.DataFrame().assign(name=['Sally', 'George', 'Bill', 'Ann'],
        ...                             age=[21, 25, 18, 28],
        ...                             height_cm=[161, 168, 171, 149])
        >>> df.set_index('name')
                age  height_cm
        name
        Sally    21        161
        George   25        168
        Bill     18        171
        Ann      28        149
        '''
        if not isinstance(keys, Iterable):
            raise TypeError('Argument `keys` must be a string label or list of string labels')
        mask = [keys not in self.columns] if isinstance(keys, str) else [x not in self.columns for x in keys]
        if any(mask):
            k = [keys] if isinstance(keys, str) else keys
            raise KeyError('{} not found in columns'.format(np.array(k)[mask]))
        if not isinstance(drop, bool):
            raise TypeError('Argument `drop` must be a boolean')

        f = _lift_to_pd(self._pd.set_index)
        return f(keys=keys, drop=drop)

    # Combining
    def merge(self, right, how='inner', on=None, left_on=None, right_on=None):
        '''
        Merge DataFrame or named Series objects with a database-style join.

        :param right: Object to merge with
        :param how: Type of merge to be performed.
            
            - left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
            - right: use only keys from right frame, similar to a SQL right outer join; preserve key order.
            - outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
            - inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
        
        :param on: Column or index level names to join on. These must be found in both DataFrames.
        :param left_on: Column or index level names to join on in the left DataFrame.
        :param right_on: Column or index level names to join on in the right DataFrame.
        :type right: DataFrame or named Series
        :type how: {'left', 'right', 'outer', 'inner'}, default 'inner'
        :type on: label or list of labels
        :type left_on: label or list of labels
        :type right_on: label or list of labels
        :return: A DataFrame of the two merged objects.
        :rtype: DataFrame
        :raises KeyError: if any input labels are not found in the corresponding DataFrame's columns

        :example:
        >>> df1 = bpd.DataFrame().assign(pet=['dog', 'cat', 'lizard', 'turtle'],
        ...                              kind=['mammal', 'mammal', 'reptile', 'reptile'])
        >>> df2 = bpd.DataFrame().assign(kind=['mammal', 'reptile', 'amphibian'],
        ...                              abr=['m', 'r', 'a'])
        >>> df1.merge(df2, on='kind')
              pet     kind abr
        0     dog   mammal   m
        1     cat   mammal   m
        2  lizard  reptile   r
        3  turtle  reptile   r
        '''
        if not isinstance(right, DataFrame):
            raise TypeError('Argument `right` must by a DataFrame')
        if how not in ['left', 'right', 'outer', 'inner']:
            raise ValueError('Argument `how` must be either \'left\', \'right\', \'outer\', or \'inner\'')
        if (on not in self._pd.columns or on not in right.columns) and on != None:
            raise KeyError('Label \'{}\' not found in both DataFrames'.format(on))
        if (left_on == None and right_on != None) or (left_on != None and right_on == None):
            raise KeyError('Both `left_on` and `right_on` must be column labels')
        if left_on != None and right_on != None:
            if left_on not in self._pd.columns:
                raise KeyError('Label \'{}\' not found in left DataFrame'.format(left_on))
            if right_on not in right.columns:
                raise KeyError('Label \'{}\' not found in right DataFrame'.format(right_on))

        f = _lift_to_pd(self._pd.merge)
        return f(right=right, how=how, on=on, left_on=left_on, right_on=right_on)

    def append(self, other, ignore_index=False):
        '''
        Append rows of other to the end of caller, returning a new object.

        :param other: The data to append.
        :type other: DataFrame or Series/dict-like object, or list of these
        :return: DataFrame with appended rows.
        :rtype: DataFrame

        :example:
        '''
        if not isinstance(other, DataFrame):
            raise TypeError('Argument `other` must by a DataFrame')
        if not isinstance(ignore_index, bool):
            raise TypeError('Argument `ignore_index` must be a boolean')

        f = _lift_to_pd(self._pd.append)
        return f(other=other, ignore_index=ignore_index)

    # Plotting
    def plot(self, *args, **kwargs):
        '''
        Plot the data in the DataFrame.
        '''
        f = _lift_to_pd(self._pd.plot)
        return f(*args, **kwargs)

    # IO
    def to_csv(self, path_or_buf=None, index=True):
        '''
        Write object to a comma-separated values (csv) file.

        :param path_or_buf: File path or object, if None is provided the result is returned as a string.
        :param index: Write row names (index).
        :type path_or_buf: str or file handle, default None
        :type index: bool, default True
        :return: If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        :rtype: None or str
        '''
        if not isinstance(index, bool):
            raise TypeError('Argument `index` must be a boolean')

        f = _lift_to_pd(self._pd.to_csv)
        return f(path_or_buf=path_or_buf, index=index)

    def to_numpy(self):
        '''
        Convert the DataFrame to a NumPy array.

        :return: DataFrame as a NumPy array.
        :rtype: NumPy array
        '''
        f = _lift_to_pd(self._pd.to_numpy)
        return f()


class Series(object):
    '''
    Custom Series class; Pandas Series with methods removed.
    '''

    def __init__(self, **kwargs):
        '''
        Create an empty Series.
        '''
        # hidden pandas dataeriesframe object
        self._pd = pd.Series(**kwargs)
        
        # lift loc/iloc back to custom Series objects
        self.loc = DataFrameIndexer(self._pd.loc)
        self.iloc = DataFrameIndexer(self._pd.iloc)

        self.shape = _lift_to_pd(self._pd.shape)
        self.index = _lift_to_pd(self._pd.index)
        self.values = _lift_to_pd(self._pd.values)

    # Formatting
    def __repr__(self):
        return self._pd.__repr__()

    def __str__(self):
        return self._pd.__str__()

    # Selection
    def take(self, indices):
        '''
        Return the elements in the given positional indices along an axis.

        :param indices: An array of ints indicating which positions to take.
        :type indices: list of ints
        :return: Series with the given positional indices.
        :raises IndexError: if any `indices` are out of bounds with respect to DataFrame length.

        :example:
        >>> s = bpd.Series(data=[1, 2, 3], index=['A', 'B', 'C'])
        >>> s.take([0, 3])
        A    1
        C    3
        dtype: int64
        >>> s.take(np.arange(2))
        A    1
        B    2
        dtype: int64
        '''
        if not isinstance(indices, Iterable):
            raise TypeError('Argument `indices` must be a list-like object')
        if not all(isinstance(x, (int, np.integer)) for x in indices):
            raise ValueError('Argument `indices` must only contain integers')
        if not all(x < self._pd.shape[0] for x in indices):
            raise IndexError('Indices are out-of-bounds')

        f = _lift_to_pd(self._pd.take)
        return f(indices)

    def sample(self, n=None, replace=False, random_state=None):
        '''
        Return a random sample of items from an axis of object.
        
        :param n: Number of items from axis to return.
        :param replace: Sample with or without replacement.
        :param random_state: Seed for the random number generator
        :type n: int, optional
        :type replace: bool, default False
        :type random_state: int, optional
        :return: Series with `n` randomly sampled items.
        :rtype: Series
        :raises ValueError: if a sample larger than the length of the DataFrame is taken without replacement.

        :example:
        >>> s = bpd.Series(data=[1, 2, 3, 4, 5])
        >>> s.sample(3, random_state=0)
        2    3
        0    1
        1    2
        dtype: int64
        >>> s.sample(7, replace=True, random_state=10)
        1    2
        4    5
        0    1
        1    2
        3    4
        4    5
        1    2
        dtype: int64
        '''
        if not isinstance(n, int) and n != None:
            raise TypeError('Argument `n` not an integer')
        if not isinstance(replace, bool):
            raise TypeError('Argument `replace` not a boolean')
        if not isinstance(random_state, int) and random_state != None:
            raise TypeError('Argument `random_state` must be an integer or None')
        if n != None and n > self._pd.shape[0] and replace == False:
            raise ValueError('Cannot take a larger sample than length of DataFrame when `replace=False`')

        f = _lift_to_pd(self._pd.sample)
        return f(n=n, replace=replace, random_state=random_state)

    # Transformation
    def apply(self, func):
        '''
        Invoke function on values of Series.

        :param func: Function to apply.
        :type func: function
        :return: Result of applying func to the Series.
        :rtype: Series

        :example:
        >>> def cut_off_5(val):
        ...     if val > 5:
        ...         return 5
        ...     else:
        ...         return val
        >>> s = bpd.Series(data=[1, 3, 5, 7, 9]
        >>> s.apply(cut_off_5)
        0    1
        1    3
        2    5
        3    5
        4    5
        dtype: int64
        '''
        if not callable(func):
            raise TypeError('Argument `func` must be a function')

        f = _lift_to_pd(self._pd.apply)
        return f(func=func)

    def sort_values(self, ascending=True):
        '''
        Sort by the values

        :param ascending: Sort ascending vs. descending.
        :type ascending: bool, default True
        :return: Series with sorted values.
        :rtype: Series

        :example:
        >>> s = bpd.Series(data=[6, 4, 3, 9, 5])
        >>> s.sort_values()
        2    3
        1    4
        4    5
        0    6
        3    9
        dtype: int64
        >>> s.sort_values(ascending=False)
        3    9
        0    6
        4    5
        1    4
        2    3
        dtype: int64
        '''
        if not isinstance(ascending, bool):
            raise TypeError('Argument `ascending` must be a boolean')

        f = _lift_to_pd(self._pd.sort_values)
        return f(ascending=ascending)

    def describe(self):
        '''
        Generate descriptive statistics that summarize the central tendency, 
        dispersion and shape of a dataset’s distribution.

        :return: Summary statistics of the Series provided.
        :rtype: Series

        :example:
        >>> s = bpd.Series(data=[6, 7, 7, 5, 9, 5, 1])
        >>> s.describe()
        count    7.000000
        mean     5.714286
        std      2.497618
        min      1.000000
        25%      5.000000
        50%      6.000000
        75%      7.000000
        max      9.000000
        dtype: float64
        '''
        f = _lift_to_pd(self._pd.describe)
        return f()

    def reset_index(self, drop=False):
        '''
        Generate a new DataFrame or Series with the index reset.

        :param drop: Does not insert index as a column.
        :type drop: bool, default False
        :return: When drop is False (the default), a DataFrame is returned. The newly created columns will come first in the DataFrame, followed by the original Series values. When drop is True, a Series is returned.
        :rtype: Series or DataFrame

        :example:
        >>> s = bpd.Series([6, 4, 3, 9, 5])
        >>> sorted = s.sort_values()
        >>> sorted.reset_index()
           index  0
        0      2  3
        1      1  4
        2      4  5
        3      0  6
        4      3  9
        >>> sorted.reset_index(drop=True)
        0    3
        1    4
        2    5
        3    6
        4    9
        dtype: int64
        '''
        if not isinstance(drop, bool):
            raise TypeError('Argument `drop` must be a boolean')

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
    def to_csv(self, path_or_buf=None, index=True):
        '''
        Write object to a comma-separated values (csv) file.
        :param path_or_buf: File path or object, if None is provided the result is returned as a string.
        :param index: Write row names (index).
        :type path_or_buf: str or file handle, default None
        :type index: bool, default True
        :return: If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        :rtype: None or str
        '''
        if not isinstance(index, bool):
            raise TypeError('Argument `index` must be a boolean')

        f = _lift_to_pd(self._pd.to_csv)
        return f(path_or_buf=path_or_buf, index=index)

    def to_numpy(self):
        '''
        A NumPy ndarray representing the values in this Series or Index.

        :return: Series as a NumPy array.
        :rtype: NumPy array
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
