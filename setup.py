from setuptools import setup, find_packages

setup(
    name='gpbacay_arcane',
    version='0.1',
    author='Gianne P. Bacay',
    author_email='giannebacay2004@gmail.com',
    description='A Python library for custom neuromorphic neural network mechanisms built on top of TensorFlow and Keras',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gpbacay/gpbacay_arcane',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'keras',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
