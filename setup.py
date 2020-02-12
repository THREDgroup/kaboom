from setuptools import setup, find_packages

setup(
    name='kaboom',
    version='0.0.1',
    author='Sam Lapp',
    author_email='sam.m.lapp@gmail.com',
    packages=find_packages(),#exclude=('heuristic_bursts', 'tests', 'wec'))
    include_package_data = True,
    install_requires=["pandas>=0.23.0","matplotlib>=2.2.2","numpy>=1.14.3","setuptools>=40.6.3","scipy>=1.1.0"],
)
