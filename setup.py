import sys
from setuptools import setup


with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]


setup(
    name = 'babypandas',
    packages = ['babypandas'],
    version = '0.1.0',
    install_requires = install_requires,
    description = 'A restricted Pandas API',
    author = 'Aaron Fraenkel, Darren Liu',
    author_email = '',
    url = 'https://github.com/afraenkel/babypandas'
)
