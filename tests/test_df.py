import pytest

import os

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

import src.babypandas.bpd

def test_df_length():
	df = bpd.DataFrame().assign(name=['Al', 'Bob', 'Jill', 'Sally'],
								age=['20', '25', '22', '23'])
	assert(df.shape[0] == 4), 'Length incorrect'