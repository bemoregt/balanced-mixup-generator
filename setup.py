#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Balanced Mixup Generator',
    version='0.0.1',
    description='(Class) Balanced Mixup Generator',
    #long_description=readme,
    author='daisukelab',
    author_email='foo@bar.com',
    install_requires=['numpy'],
    url='https://github.com/daisukelab/balanced-mixup-generator',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)