import pandas as pd


class DataFrameIndexer(object):
    '''
    Class for lifting results of loc/iloc back to the
    custom DataFrame class.
    '''
    def __init__(self, indexer):        
        self.idx = indexer
        
    def __getitem__(self, item):
        
        data = self.idx[item]
        if isinstance(data, pd.DataFrame):
            return DataFrame(data=self.idx[item])
        else:
            return data
    

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
        self._df = pd.DataFrame(**kwargs)
        
        # lift loc/iloc back to custom DataFrame objects
        self.loc = DataFrameIndexer(self._df.loc)
        self.iloc = DataFrameIndexer(self._df.iloc)

        # List of Pandas DataFrame methods to be made "public".
        _dunder_attrs = ['__repr__', '_repr_html_']
        _props = ['shape', 'columns', 'index', 'values', 'T']
        _selection = ['take', 'drop', 'sample', 'get']
        _creation = ['assign']
        _transformation = ['apply', 'sort_values', 'describe']
        _combining = ['merge', 'append']
        _plotting = ['plot']
        _io = ['to_csv', 'to_numpy']
        
        # TODO: pivot, group,...
        # TODO: figure out series (especially for .loc selection / comparison operators)
        _attrs = (
            _dunder_attrs + _props + _selection + 
            _creation + _transformation + _combining +
            _plotting + _io)

        for meth in _attrs:
            setattr(self, meth, getattr(self._df, meth))
            setattr(self, meth, self._lift2DF(getattr(self._df, meth)))
        
    def to_df(self):
        '''return the full pandas dataframe'''
        return self._df
    
    # Creation    
    @classmethod
    def from_dict(cls, data):
        return cls(data=data)
        
    @classmethod
    def from_records(cls, data, columns):
        
        return cls(data=data, columns=columns)
        
    def _lift2DF(self, func):
        '''checks output-type of function and if output is a
        Pandas object, lifts the output to the DataFrame class'''
        
        if not callable(func):
            return func

        def closure(*vargs, **kwargs):
            vargs = [x._df if isinstance(x, DataFrame) else x for x in vargs]
            kwargs = {k: x._df if isinstance(x, DataFrame) else x for (k, x) in kwargs.items()}

            a = func(*vargs, **kwargs)
            if isinstance(a, pd.DataFrame):
                return DataFrame(data=a)
            else:
                return a

        closure.__doc__ = func.__doc__

        return closure

