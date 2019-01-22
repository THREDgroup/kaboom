from setuptools import setup, find_packages

setup(
    name='kaboom',
    version='0.0.1',
    author='Sam Lapp',
    author_email='sam.m.lapp@gmail.com',
    packages=find_packages(),#exclude=('heuristic_bursts', 'tests', 'wec'))
    include_package_data = True
)
