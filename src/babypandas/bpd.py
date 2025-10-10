from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.core import common as com
from pandas.core import indexing

from babypandas.utils import decorate_all_methods, suppress_warnings

pd.set_option("display.max_rows", 10)


@decorate_all_methods(suppress_warnings, exclude=["__getitem__"])
class DataFrame(object):
    """
    Custom DataFrame Class; Pandas DataFrames with methods removed.

    Examples
    --------
    >>> df = DataFrame.from_records([[1,2,3],[4,5,6]], columns=['a', 'b', 'c'])
    >>> df.shape
    (2, 3)
    >>> df.assign(d=[1,2]).shape
    (2, 4)
    >>> df.loc[1, 'b']
    5
    """

    def __init__(self, **kwargs):
        """
        Create an empty DataFrame.
        """
        # hidden pandas dataframe object
        self._pd = pd.DataFrame(**kwargs)

        # lift loc/iloc back to custom DataFrame objects
        self.loc = DataFrameIndexer(self._pd.loc)
        self.iloc = DataFrameIndexer(self._pd.iloc)

    @property
    def T(self):
        return self.__class__(data=self._pd.T)

    @property
    def index(self):
        return self._pd.index

    @property
    def columns(self):
        return self._pd.columns

    @property
    def values(self):
        return self._pd.values

    @property
    def shape(self):
        return self._pd.shape

    # Formatting
    def __repr__(self):
        return self._pd.__repr__()

    def __str__(self):
        return self._pd.__str__()

    # return the underlying DataFrame
    def to_df(self):
        """Return the full pandas DataFrame."""
        return self._pd

    # Creation
    @classmethod
    def from_dict(cls, data):
        """
        Construct DataFrame from dict of array-like or dicts.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.

        Returns
        -------
        DataFrame
        """
        return cls(data=data)

    @classmethod
    def from_records(cls, data, *, columns=None):
        """
        Convert structured or record ndarray to DataFrame.

        Parameters
        ----------
        data : ndarray (structured dtype), list of tuples, dict, or DataFrame
        columns : sequence, default None, keyword-only
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns)

        Returns
        -------
        DataFrame
        """
        return cls(data=data, columns=columns)

    # Dunder Attributes
    def _repr_html_(self):
        f = _lift_to_pd(self._pd._repr_html_)
        return f()

    def __getitem__(self, key):
        if getattr(key, "to_ser", None):  # Convert to pd.Series
            key = key.to_ser()
        if not com.is_bool_indexer(key):
            raise IndexError(
                "BabyPandas only accepts Boolean objects "
                "when indexing against the data frame; "
                "please use .get to get columns, and "
                ".loc or .iloc for more complex cases."
            )
        f = _lift_to_pd(self._pd._getitem_bool_array)
        return f(key)

    # Selection
    def take(self, indices):
        """
        Return the rows in the given *positional* indices.

        This means that we are not indexing according to actual values in the
        index attribute of the object. We are indexing according to the actual
        position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.

        Returns
        -------
        taken : DataFrame
            An DataFrame containing the elements taken from the object.

        Raises
        ------
        IndexError
            If any `indices` are out of bounds with respect to DataFrame
            length.

        Examples
        --------
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
        """
        if not isinstance(indices, Iterable):
            raise TypeError("Argument `indices` must be a list-like object")
        if not all(isinstance(x, (int, np.integer)) for x in indices):
            raise ValueError("Argument `indices` must only contain integers")
        if not all(x < self._pd.shape[0] for x in indices):
            raise IndexError("Indices are out-of-bounds")

        f = _lift_to_pd(self._pd.take)
        return f(indices=indices)

    def drop(self, *, columns=None):
        """
        Remove columns by specifying column names.

        Parameters
        ----------
        columns : single label or list-like
            Column names to drop.

        Returns
        -------
        df : DataFrame
            DataFrame with the dropped columns.

        Raises
        ------
        KeyError
            If none of the column labels are found.

        Examples
        --------
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
        """
        if not isinstance(columns, Iterable):
            raise TypeError(
                "Argument `columns` must be a string label or list of string labels"
            )
        mask = (
            [columns not in self.columns]
            if isinstance(columns, str)
            else [x not in self.columns for x in columns]
        )
        if any(mask):
            c = [columns] if isinstance(columns, str) else columns
            raise KeyError("{} not found in columns".format(np.array(c)[mask]))

        f = _lift_to_pd(self._pd.drop)
        return f(columns=columns)

    def sample(self, n=None, *, replace=False, random_state=None):
        """
        Return a random sample of rows from a data frame.

        You can use `random_state` for reproducibility.

        Parameters
        ----------
        n : None or int, optional
            Number of rows to return.  None corresponds to 1.
        replace : {False, True}, optional, keyword only.
            Sample with or without replacement.
        random_state : int or numpy.random.RandomState, optional, keyword only
            Seed for the random number generator (if int), or numpy RandomState
            object.

        Returns
        -------
        s_df : DataFrame
            A new DataFrame containing `n` items randomly sampled from the
            caller object.

        Raises
        ------
        ValueError
            If a sample larger than the length of the DataFrame is taken
            without replacement.

        Examples
        --------
        >>> df = bpd.DataFrame().assign(letter=['a', 'b', 'c'],
        ...                             count=[9, 3, 3],
        ...                             points=[1, 2, 2])
        >>> df.sample(1, random_state=0)
            letter  count  points
        2      c      3       2
        """
        if not isinstance(n, int) and n != None:
            raise TypeError("Argument `n` not an integer")
        if not isinstance(replace, bool):
            raise TypeError("Argument `replace` not a boolean")
        if not isinstance(random_state, int) and random_state != None:
            raise TypeError(
                "Argument `random_state` must be an integer or None"
            )
        if n != None and n > self._pd.shape[0] and replace == False:
            raise ValueError(
                "Cannot take a larger sample than length of DataFrame when `replace=False`"
            )

        f = _lift_to_pd(self._pd.sample)
        return f(n=n, replace=replace, random_state=random_state)

    def get(self, key):
        """Return column or columns from data frame.

        Parameters
        ----------
        key : str or iterable of strings
            Column label or iterable of column labels

        Returns
        -------
        series_or_df : Series or DataFrame
            Series with the corresponding label or DataFrame with the
            corresponding column labels.

        Raises
        ------
        KeyError
            If any column named in `key` not found in columns.

        Examples
        --------
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
        """
        if not isinstance(key, str) and not isinstance(key, Iterable):
            raise TypeError(
                "Argument `key` must be a string label or list of string labels"
            )
        mask = (
            [key not in self.columns]
            if isinstance(key, str)
            else [x not in self.columns for x in key]
        )
        if any(mask):
            k = [key] if isinstance(key, str) else key
            raise KeyError("{} not found in columns".format(np.array(k)[mask]))

        f = _lift_to_pd(self._pd.get)
        return f(key=key)

    # Creation
    def assign(self, **kwargs):
        """
        Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : dict of {str: callable or Series}
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though pandas doesn't check it).
            If the values are not callable, (e.g. a Series, scalar, or array),
            they are simply assigned.

        Returns
        -------
        df_with_cols : DataFrame
            A new DataFrame with the new columns in addition to all the
            existing columns.

        Raises
        ------
        ValueError
            If columns have different lengths or if new columns have different lengths than the existing DataFrame

        Examples
        --------
        >>> df = bpd.DataFrame().assign(flower=['sunflower', 'rose'])
        >>> df.assign(color=['yellow', 'red'])
              flower   color
        0  sunflower  yellow
        1       rose     red
        """
        if len(set(map(len, kwargs.values()))) not in (0, 1):
            raise ValueError("Not all columns have the same length")
        if self._pd.shape[1] != 0:
            if len(list(kwargs.values())[0]) != self._pd.shape[0]:
                raise ValueError(
                    "New column does not have the same length as existing DataFrame"
                )

        f = _lift_to_pd(self._pd.assign)
        return f(**kwargs)

    # Transformation
    def apply(self, func, axis=0):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is either
        the DataFrame's index (``axis=0``) or the DataFrame's columns
        (``axis=1``). The final return type is inferred from the return type of
        the applied function.

        Parameters
        ----------
        func : function
            Function to apply to each column or row.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the function is applied:

            * 0 or 'index': apply function to each column.
            * 1 or 'columns': apply function to each row.

        Returns
        -------
        applied : Series or DataFrame
            Result of applying ``func`` along the given axis of the DataFrame.

        Examples
        --------
        >>> def add_two(row):
        ...     return row + 2
        >>> df = bpd.DataFrame(A=[1, 1],
        ...                    B=[2, 2])
        >>> df.apply(add_two)
           A  B
        0  3  4
        1  3  4
        """
        if not callable(func):
            raise TypeError("Argument `func` must be a function")
        if axis not in [0, 1, "index", "columns"]:
            raise ValueError(
                'Argument `axis` must be one of 0, 1 "index" or "columns"'
            )

        f = _lift_to_pd(self._pd.apply)
        return f(func=func, axis=axis)

    def sort_values(self, by, *, ascending=True):
        """
        Sort by the values in column(s) named in `by`.

        Parameters
        ----------
        by : str or list of str
            Name or list of column names to sort by.
        ascending : {True, False} or list of bool, keyword only
            Sort ascending vs. descending. Specify list for multiple sort
            orders.  If this is a list of bools, must match the length of the
            `by`.  Default is True.

        Returns
        -------
        sorted_obj : DataFrame

        Raises
        ------
        KeyError
            If `by` not found in columns.

        Examples
        --------
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
        """
        if not isinstance(by, Iterable):
            raise TypeError(
                "Argument `by` must be a string label or list of string labels"
            )
        mask = (
            [by not in self.columns]
            if isinstance(by, str)
            else [x not in self.columns for x in by]
        )
        if any(mask):
            b = [by] if isinstance(by, str) else by
            raise KeyError("{} not found in columns".format(np.array(b)[mask]))
        if not isinstance(ascending, bool):
            raise TypeError("Argument `ascending` must be a boolean")

        f = _lift_to_pd(self._pd.sort_values)
        return f(by=by, ascending=ascending)

    def describe(self):
        """
        Generate descriptive statistics.

        Statistics summarize the central tendency, dispersion and shape of a
        dataset's distribution, excluding ``NaN`` values.

        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types.

        Parameters
        ----------
        None

        Returns
        -------
        descr : DataFrame
            Summary statistics of the Dataframe provided.

        Examples
        --------
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
        """
        f = _lift_to_pd(self._pd.describe)
        return f()

    def groupby(self, by):
        """
        Group DataFrame by values in columns specified in `by`.

        A groupby operation involves some combination of splitting the object,
        applying a function, and combining the results. this can be used to
        group large amounts of data and compute operations on these groups.

        Parameters
        ----------
        by : label, or list of labels
            Used to determine the groups for the groupby. Should be a label or
            list of labels that will group by the named columns in ``self``.
            Notice that a tuple is interpreted a (single) key.

        Returns
        -------
        df_gb : DataFrameGroupBy
            groupby object that contains information about the groups.

        Raises
        -------
        KeyError
            If `by` not found in columns

        Examples
        --------
        >>> df = bpd.DataFrame(animal=['Falcon', 'Falcon', 'Parrot', 'Parrot'],
        ...                   max_speed=[380, 370, 24, 26])
        >>> df.groupby('animal').mean()
                max_speed
        animal
        Falcon      375.0
        Parrot       25.0
        """
        if not isinstance(by, Iterable):
            raise TypeError(
                "Argument `by` must be a string label or list of string labels"
            )
        mask = (
            [by not in self.columns]
            if isinstance(by, str)
            else [x not in self.columns for x in by]
        )
        if any(mask):
            b = [by] if isinstance(by, str) else by
            raise KeyError("{} not found in columns".format(np.array(b)[mask]))

        f = _lift_to_pd(self._pd.groupby)
        return f(by=by)

    def reset_index(self, *, drop=False):
        """
        Reset the index.

        Reset the index of the DataFrame, and use the default one instead.

        Parameters
        ----------
        drop : bool, default False, keyword only
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.

        Returns
        -------
        DataFrame
            DataFrame with the new index.

        Reset the index of the DataFrame, and use the default one
        instead. If the DataFrame has a MultiIndex, this method can
        remove one or more levels.

        Examples
        --------
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

        """
        if not isinstance(drop, bool):
            raise TypeError("Argument `drop` must be a boolean")

        f = _lift_to_pd(self._pd.reset_index)
        return f(drop=drop)

    def set_index(self, keys, drop=True):
        """
        Set the DataFrame index using existing columns.

        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index replaces the
        existing index.

        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single array of
            the same length as the calling DataFrame, or a list containing an
            arbitrary combination of column keys and arrays. Here, "array"
            encompasses :class:`Series`, :class:`Index` and ``np.ndarray``.
        drop : bool, default True
            Delete columns to be used as the new index.

        Returns
        -------
        DataFrame
            Data frame with changed row labels.

        Raises
        ------
        KeyError
            If `keys` not found in columns.

        Examples
        --------
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
        """
        if not isinstance(keys, Iterable):
            raise TypeError(
                "Argument `keys` must be a string label or list of string labels"
            )
        mask = (
            [keys not in self.columns]
            if isinstance(keys, str)
            else [x not in self.columns for x in keys]
        )
        if any(mask):
            k = [keys] if isinstance(keys, str) else keys
            raise KeyError("{} not found in columns".format(np.array(k)[mask]))
        if not isinstance(drop, bool):
            raise TypeError("Argument `drop` must be a boolean")

        f = _lift_to_pd(self._pd.set_index)
        return f(keys=keys, drop=drop)

    # Combining
    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
    ):
        r"""
        Merge DataFrame or named Series objects with a database-style join.

        The join is done on columns or indexes. If joining columns on columns,
        the DataFrame indexes *will be ignored*. Otherwise if joining indexes
        on indexes or indexes on a column or columns, the index will be passed
        on.

        Parameters
        ----------
        right : DataFrame or named Series
            Object to merge with.
        how : {'left', 'right', 'outer', 'inner'}, default 'inner'
            Type of merge to be performed.

            \* left: use only keys from left frame, similar to a SQL left outer
              join; preserve key order.
            \* right: use only keys from right frame, similar to a SQL right
              outer join; preserve key order.
            \* outer: use union of keys from both frames, similar to a SQL full
              outer join; sort keys lexicographically.
            \* inner: use intersection of keys from both frames, similar to a
              SQL inner join; preserve the order of the left keys.
        on : label or list
            Column or index level names to join on. These must be found in both
            DataFrames. If `on` is None and not merging on indexes then this
            defaults to the intersection of the columns in both DataFrames.
        left_on : label or list, or array-like
            Column or index level names to join on in the left DataFrame. Can
            also be an array or list of arrays of the length of the left
            DataFrame.  These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame. Can
            also be an array or list of arrays of the length of the right
         left_index : boolean, default False
            Use the index from the left DataFrame as the join key(s). If it is
            a MultiIndex, the number of keys in the other DataFrame (either the
            index or a number of columns) must match the number of levels
        right_index : boolean, default False
            Use the index from the right DataFrame as the join key. Same
            caveats as left_index   DataFrame.  These arrays are treated as if
            they are columns.

        Returns
        -------
        DataFrame
            A DataFrame of the two merged objects.

        Raises
        ------
        KeyError
            If any input labels are not found in the corresponding DataFrame's
            columns.

        Examples
        --------
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
        """
        using_index = left_index or right_index
        if not isinstance(right, DataFrame):
            raise TypeError("Argument `right` must by a DataFrame")
        if how not in ["left", "right", "outer", "inner"]:
            raise ValueError(
                "Argument `how` must be either 'left', 'right', 'outer', or 'inner'"
            )
        if (
            on not in self._pd.columns or on not in right.columns
        ) and on != None:
            raise KeyError(
                "Label '{}' not found in both DataFrames".format(on)
            )
        if not using_index and (
            (left_on == None and right_on != None)
            or (left_on != None and right_on == None)
        ):
            raise KeyError(
                "Both `left_on` and `right_on` must be column labels"
            )
        if left_on != None and right_on != None:
            if left_on not in self._pd.columns:
                raise KeyError(
                    "Label '{}' not found in left DataFrame".format(left_on)
                )
            if right_on not in right.columns:
                raise KeyError(
                    "Label '{}' not found in right DataFrame".format(right_on)
                )

        f = _lift_to_pd(self._pd.merge)
        return f(
            right=right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
        )

    # Plotting
    def plot(self, *args, **kwargs):
        """
        DataFrame plotting accessor and method

        Examples
        --------
        >>> df.plot.line()
        >>> df.plot.scatter('x', 'y')
        >>> df.plot.hexbin()
        """
        f = _lift_to_pd(self._pd.plot)
        return f(*args, **kwargs)

    # IO
    def to_csv(self, path_or_buf=None, *, index=True):
        """
        Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.
        index : bool, default True
            Write row names (index).

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting csv format as a
            string. Otherwise returns None.
        """
        if not isinstance(index, bool):
            raise TypeError("Argument `index` must be a boolean")

        f = _lift_to_pd(self._pd.to_csv)
        return f(path_or_buf=path_or_buf, index=index)

    def to_numpy(self):
        """
        Convert the DataFrame to a NumPy array.

        By default, the dtype of the returned array will be the common NumPy
        dtype of all types in the DataFrame. For example, if the dtypes are
        ``float16`` and ``float32``, the results dtype will be ``float32``.
        This may require copying data and coercing values, which may be
        expensive.

        Parameters
        ----------
        None

        Returns
        -------
        df_arr : numpy.ndarray
            DataFrame as a NumPy array.
        """
        f = _lift_to_pd(self._pd.to_numpy)
        return f()


@decorate_all_methods(suppress_warnings)
class SeriesStringMethods(object):
    """
    String methods on Series objects. Will return bpd.Series
    """

    def __init__(self, methods):
        self._methods = methods

    def __getattr__(self, name):
        return _lift_to_pd(getattr(self._methods, name))

    def __dir__(self):
        return [x for x in dir(self._methods) if not x.startswith("_")]


@decorate_all_methods(suppress_warnings)
class Series(object):
    """
    Custom Series class; Pandas Series with methods removed.
    """

    def __init__(self, **kwargs):
        """
        Create an empty Series.
        """
        # hidden pandas dataeriesframe object
        self._pd = pd.Series(**kwargs)

        # lift loc/iloc back to custom Series objects
        self.loc = DataFrameIndexer(self._pd.loc)
        self.iloc = DataFrameIndexer(self._pd.iloc)

        self.shape = _lift_to_pd(self._pd.shape)
        self.index = _lift_to_pd(self._pd.index)
        self.values = _lift_to_pd(self._pd.values)

    @property
    def str(self):
        """
        String methods on Series.
        """
        # accessing the `.str` attribute of a pd.Series will raise an
        # AttributeError if the series does not consist of string values. We
        # use a property here to replicate this behavior.
        return SeriesStringMethods(self._pd.str)

    # Formatting
    def __repr__(self):
        return self._pd.__repr__()

    def __str__(self):
        return self._pd.__str__()

    def __getitem__(self, key):
        if getattr(key, "to_ser", None):  # Convert to pd.Series
            key = key.to_ser()
        if not com.is_bool_indexer(key):
            raise IndexError(
                "BabyPandas only accepts Boolean objects "
                "when indexing against the Series; please use "
                ".loc or .iloc for more complex cases."
            )
        key = indexing.check_bool_indexer(self.index, key)
        f = _lift_to_pd(self._pd._get_with)
        return f(key)

    # Selection
    def take(self, indices):
        """
        Return the elements in the given *positional* indices.

        This means that we are not indexing according to actual values in the
        index attribute of the object. We are indexing according to the actual
        position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.

        Returns
        -------
        taken : Series
            A Series containing the elements taken from the object.

        Raises
        ------
        IndexError
            If any `indices` are out of bounds with respect to Series
            length.

        Examples
        --------
        >>> s = bpd.Series(data=[1, 2, 3], index=['A', 'B', 'C'])
        >>> s.take([0, 3])
        A    1
        C    3
        dtype: int64
        >>> s.take(np.arange(2))
        A    1
        B    2
        dtype: int64
        """
        if not isinstance(indices, Iterable):
            raise TypeError("Argument `indices` must be a list-like object")
        if not all(isinstance(x, (int, np.integer)) for x in indices):
            raise ValueError("Argument `indices` must only contain integers")
        if not all(x < self._pd.shape[0] for x in indices):
            raise IndexError("Indices are out-of-bounds")

        f = _lift_to_pd(self._pd.take)
        return f(indices)

    def sample(self, n=None, replace=False, random_state=None):
        """
        Return a random sample of elements from a Series.

        You can use `random_state` for reproducibility.

        Parameters
        ----------
        n : None or int, optional
            Number of elements to return.  None corresponds to 1.
        replace : {False, True}, optional, keyword only.
            Sample with or without replacement.
        random_state : int or numpy.random.RandomState, optional, keyword only
            Seed for the random number generator (if int), or numpy RandomState
            object.

        Returns
        -------
        s_series : Series
            A new Series containing `n` items randomly sampled from the caller
            object.

        Raises
        ------
        ValueError
            If a sample larger than the length of the Series is taken
            without replacement.

        Examples
        --------
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
        """
        if not isinstance(n, int) and n != None:
            raise TypeError("Argument `n` not an integer")
        if not isinstance(replace, bool):
            raise TypeError("Argument `replace` not a boolean")
        if not isinstance(random_state, int) and random_state != None:
            raise TypeError(
                "Argument `random_state` must be an integer or None"
            )
        if n != None and n > self._pd.shape[0] and replace == False:
            raise ValueError(
                "Cannot take a larger sample than length of DataFrame when `replace=False`"
            )

        f = _lift_to_pd(self._pd.sample)
        return f(n=n, replace=replace, random_state=random_state)

    def get(self, key, default=None):
        """
        Get item from object for given key (ex: Series entry).
        Returns default value if not found.
        Parameters
        ----------
        key : object
        Returns
        -------
        value : same type as items contained in object
        """

        f = _lift_to_pd(self._pd.get)
        return f(key, default=default)

    # Transformation
    def apply(self, func):
        """
        Invoke function on values of Series.

        Can be ufunc (a NumPy function that applies to the entire Series)
        or a Python function that only works on single values.

        Parameters
        ----------
        func : function
            Python function or NumPy ufunc to apply.

        Returns
        -------
        a_obj : Series or DataFrame
            If func returns a Series object the result will be a DataFrame.

        Examples
        --------
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
        """
        if not callable(func):
            raise TypeError("Argument `func` must be a function")

        f = _lift_to_pd(self._pd.apply)
        return f(func=func)

    def sort_values(self, *, ascending=True):
        """
        Sort by the values.

        Sort a Series in ascending or descending order.

        Parameters
        ----------
        ascending : bool, default True, keyword only
            If True, sort values in ascending order, otherwise descending.

        Returns
        -------
        s_series : Series
            Series ordered by values.

        Example
        -------
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
        """
        if not isinstance(ascending, bool):
            raise TypeError("Argument `ascending` must be a boolean")

        f = _lift_to_pd(self._pd.sort_values)
        return f(ascending=ascending)

    def unique(self):
        """
        Return unique values of Series object.

        Parameters
        ----------
        None

        Returns
        -------
        values : ndarray
            A NumPy array containing the unique values, in order of appearance.

        Examples
        --------
        >>> s = bpd.Series(data=[6, 7, 7, 5, 9, 5, 1])
        >>> s.unique()
        array([6, 7, 5, 9, 1])
        """
        f = _lift_to_pd(self._pd.unique)
        return f()

    def describe(self):
        """
        Generate descriptive statistics.

        Statistics summarize the central tendency, dispersion and shape of a
        Series' distribution, excluding ``NaN`` values.

        Analyzes both numeric and object series.

        Parameters
        ----------
        None

        Returns
        -------
        descr : Series
            Summary statistics of the Series provided.

        Examples
        --------
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
        """
        f = _lift_to_pd(self._pd.describe)
        return f()

    def reset_index(self, *, drop=False):
        """
        Reset the index.

        This is useful when the index is meaningless and needs to be reset to
        the default before another operation.

        Parameters
        ----------
        drop : bool, default False, keyword only
            When True, do not try to insert index into dataframe columns. This
            resets the index to the default integer index.  If False, then turn
            input Series into DataFrame, adding original index as column.

        Returns
        -------
        Series or DataFrame
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.

        Examples
        --------
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
        """
        if not isinstance(drop, bool):
            raise TypeError("Argument `drop` must be a boolean")

        f = _lift_to_pd(self._pd.reset_index)
        return f(drop=drop)

    def where(self, cond, other):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : boolean Series, array-like, or callable
            Where cond is True, keep the original value. Where False, replace
            with corresponding value from other. If cond is callable, it is
            computed on the Series and should return boolean Series or array.
        other : scalar, Series/DataFrame, or callable
            Entries where cond is False are replaced with corresponding value
            from other. If other is callable, it is computed on the Series
            and should return scalar or Series.

        Returns
        -------
        s_series : Series
            A new Series with the values replaced when the condition is False.

        Notes
        -----
        The `where` method is an application of the if-then idiom. For each
        element in the calling Series, if ``cond`` is ``True`` the
        element is used; otherwise the corresponding element from the Series
        ``other`` is used.
        The signature for :func:`Series.where` differs from
        :func:`numpy.where`. Roughly ``ser1.where(m, ser2)`` is equivalent to
        ``np.where(m, ser1, ser2)``.
        Examples
        --------
        >>> s = pd.Series(range(5))
        >>> s.where(s > 1, 10)
        0    10
        1    10
        2    2
        3    3
        4    4
        dtype: int64
        """

        f = _lift_to_pd(self._pd.where)
        return f(cond, other)

    # Plotting
    def plot(self, *args, **kwargs):
        """
        Series plotting accessor and method.

        Examples
        --------
        >>> s.plot.line()
        >>> s.plot.bar()
        >>> s.plot.hist()
        """
        f = _lift_to_pd(self._pd.plot)
        return f(*args, **kwargs)

    # IO
    def to_csv(self, path_or_buf=None, index=True):
        """
        Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.
        index : bool, default True
            Write row names (index).

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting csv format as a
            string. Otherwise returns None.
        """
        if not isinstance(index, bool):
            raise TypeError("Argument `index` must be a boolean")

        f = _lift_to_pd(self._pd.to_csv)
        return f(path_or_buf=path_or_buf, index=index)

    def to_numpy(self):
        """
        A NumPy ndarray representing the values in this Series or Index.

        Parameters
        ----------
        None

        Returns
        -------
        arr : numpy.ndarray
        """
        f = _lift_to_pd(self._pd.to_numpy)
        return f()

    # Calculations
    def count(self):
        """
        Return number of non-NA/null observations in the Series.
        """
        f = _lift_to_pd(self._pd.count)
        return f()

    def mean(self):
        """
        Return the mean of the values for the requested axis.
        """
        f = _lift_to_pd(self._pd.mean)
        return f()

    def median(self):
        """
        Return the median of the values for the requested axis.
        """
        f = _lift_to_pd(self._pd.median)
        return f()

    def min(self):
        """
        Return the minimum of the values in the Series.
        """
        f = _lift_to_pd(self._pd.min)
        return f()

    def max(self):
        """
        Return the maximum of the values in the Series.
        """
        f = _lift_to_pd(self._pd.max)
        return f()

    def sum(self):
        """
        Return the sum of the values in the Series.
        """
        f = _lift_to_pd(self._pd.sum)
        return f()

    def abs(self):
        """
        Return a Series with absolute numeric value of each element.
        """
        f = _lift_to_pd(self._pd.abs)
        return f()

    # Arithmetic
    def __add__(self, other):
        f = _lift_to_pd(self._pd.__add__)
        return f(other)

    def __radd__(self, other):
        f = _lift_to_pd(self._pd.__radd__)
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

    def __rsub__(self, other):
        f = _lift_to_pd(self._pd.__rsub__)
        return f(other)

    def __neg__(self):
        f = _lift_to_pd(self._pd.__neg__)
        return f()

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

    # bitwise operators

    def __and__(self, other):
        f = _lift_to_pd(self._pd.__and__)
        return f(other)

    def __or__(self, other):
        f = _lift_to_pd(self._pd.__or__)
        return f(other)

    def __xor__(self, other):
        f = _lift_to_pd(self._pd.__xor__)
        return f(other)

    # othe dunder methods
    def __len__(self):
        return self._pd.__len__()

    def __invert__(self):
        """unary inversion, ~ operator"""
        f = _lift_to_pd(self._pd.__invert__)
        return f()

    # array interface (for applying numpy functions)
    def __array__(self, *vargs, **kwargs):
        return self._pd.__array__(*vargs, **kwargs)

    # return the underlying Series
    def to_ser(self):
        """Return the underlying Pandas series"""
        return self._pd


@decorate_all_methods(suppress_warnings)
class DataFrameGroupBy(object):
    """ """

    def __init__(self, groupby):
        # hidden pandas dataframe object
        self._pd = groupby

    # return the underlying groupby object
    def to_gb(self):
        """return the underlying pandas groupby object"""
        return self._pd

    def aggregate(self, func):
        if not callable(func):
            raise Exception("Provide a function to aggregate")

        return self._pd.aggregate(func)

    # Calculations
    def count(self):
        """
        Compute count of group.
        """
        f = _lift_to_pd(self._pd.count)
        return f()

    def mean(self):
        """
        Compute mean of group.
        """
        f = _lift_to_pd(self._pd.mean)
        return f()

    def median(self):
        """
        Compute median of group.
        """
        f = _lift_to_pd(self._pd.median)
        return f()

    def min(self):
        """
        Compute min of group.
        """
        f = _lift_to_pd(self._pd.min)
        return f()

    def max(self):
        """
        Compute max of group.
        """
        f = _lift_to_pd(self._pd.max)
        return f()

    def sum(self):
        """
        Compute sum of group.
        """
        f = _lift_to_pd(self._pd.sum)
        return f()

    def size(self):
        """
        Compute group sizes.
        """
        f = _lift_to_pd(self._pd.size)
        return f()


@decorate_all_methods(suppress_warnings)
class DataFrameIndexer(object):
    """
    Class lifts results of loc/iloc back to the custom DataFrame class.
    """

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


@suppress_warnings
def _lift_to_pd(func):
    """Checks output-type of function and if output is a
    Pandas object, lifts the output to a babypandas class"""

    if not callable(func):
        return func

    types = (DataFrame, DataFrameGroupBy, Series)

    def closure(*vargs, **kwargs):
        vargs = [x._pd if isinstance(x, types) else x for x in vargs]
        kwargs = {
            k: x._pd if isinstance(x, types) else x
            for (k, x) in kwargs.items()
        }

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


@suppress_warnings
def read_csv(filepath, **kwargs):
    """read_csv"""
    df = pd.read_csv(filepath, **kwargs)
    return DataFrame(data=df)


read_csv.__doc__ = pd.read_csv.__doc__
