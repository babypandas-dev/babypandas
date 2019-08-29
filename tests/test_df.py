import pytest

import babypandas.bpd as bpd

def test_df_length():
	df = bpd.DataFrame().assign(name=['Al', 'Bob', 'Jill', 'Sally'],
								age=['20', '25', '22', '23'])
	assert(df.shape[0] == 4), 'Length incorrect'

def table1():
	return bpd.DataFrame().assign(letter=['a', 'b', 'c'],
		 						  count=[9, 3, 3],
		 						  points=[1, 2, 2])