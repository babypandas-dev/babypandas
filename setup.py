import sys
from setuptools import setup


with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]

with open('README.md') as fh:
    long_description = fh.read()

setup(
    name = 'babypandas',
    packages = ['babypandas'],
    version = '0.1.7',
    setup_requires=['pytest-runner'],
    install_requires = install_requires,
    tests_require = ['pytest'],
    description = 'A restricted Pandas API',
    long_description = long_description,
    long_description_content_type="text/markdown",
    author = 'Aaron Fraenkel, Darren Liu',
    author_email = 'afraenkel@ucsd.edu',
    url = 'https://github.com/afraenkel/babypandas'
)
