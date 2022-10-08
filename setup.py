from setuptools import setup
from pathlib import Path


with open('VERSION') as f:
    version = f.read()

setup(
    name="bn_testing",
    version=version,
    packages=["bn_testing"],
    python_requires='>=3.8.0',

    # dependencies
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'networkx>=2.5',
        'pymc>4.1.2',
    ],
    tests_require=[
        "pytest",
    ],

    # metadata for upload to PyPI
    author="Tobias Windisch",
    author_email="tobias.windisch@posteo.de",
    description="A test bench to benchmark learn algorithms for graphical models",
    license="GNU GPL3",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    keywords="graphical models",
    url="https://github.com/windisch/bn_testing",
)
