import sys
from glob import glob
from setuptools import setup, find_packages


with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]


setup(
    name = 'babypandas',
    packages = find_packages('src'),
    package_dir = {'': 'src'},
    version = '0.1.0',
    install_requires = install_requires,
    description = 'A restricted Pandas API',
    author = 'Aaron Fraenkel',
    author_email = '',
    url = 'https://github.com/afraenkel/babypandas',
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')]
)
