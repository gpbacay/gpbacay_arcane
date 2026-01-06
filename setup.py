"""
A.R.C.A.N.E. - Neuromimetic Language Foundation Model

Augmented Reconstruction of Consciousness through Artificial Neural Evolution

A Python library for neuromimetic neural network mechanisms.
"""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gpbacay-arcane',
    version='3.0.1',
    author='Gianne P. Bacay',
    author_email='giannebacay2004@gmail.com',
    description='A neuromimetic semantic foundation model library with biologically-inspired neural mechanisms including spiking neural networks, Hebbian learning, and homeostatic plasticity',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gpbacay/gpbacay_arcane',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'tensorflow>=2.12.0',
        'keras>=2.12.0',
        'matplotlib>=3.5.0',
    ],
    extras_require={
        'ollama': [
            'ollama>=0.1.0',
            'sentence-transformers>=2.2.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'gpbacay-arcane-about = gpbacay_arcane.cli_commands:about',
            'gpbacay-arcane-list-models = gpbacay_arcane.cli_commands:list_models',
            'gpbacay-arcane-list-layers = gpbacay_arcane.cli_commands:list_layers',
            'gpbacay-arcane-version = gpbacay_arcane.cli_commands:version',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords=[
        'neuromimetic',
        'neural-network',
        'language-model',
        'spiking-neural-network',
        'hebbian-learning',
        'reservoir-computing',
        'tensorflow',
        'deep-learning',
        'computational-neuroscience',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/gpbacay/gpbacay_arcane/issues',
        'Source': 'https://github.com/gpbacay/gpbacay_arcane',
        'Documentation': 'https://github.com/gpbacay/gpbacay_arcane#readme',
    },
)
