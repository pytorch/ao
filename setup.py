from setuptools import setup, find_packages

setup(
    name='ao',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    description='Package for applying ao techniques to GPU models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pytorch-labs/ao',
)
