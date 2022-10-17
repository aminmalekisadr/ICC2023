""" Module setup """
from setuptools import setup, find_packages

setup(
    name='src',
    version='0.1',
    description='GA based anomaly detection',
    url='https://github.com/aminmalekisadr/Genetic-Algorithm-Guided-Satellite-Anomaly-Detection',
    author='Mohammad Amin Maleki Sadr',
    author_email='aminmalekisadr@gmail.com',
    packages=find_packages(),
    include_package_data=True
)