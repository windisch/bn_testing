from setuptools import setup
from pathlib import Path

setup(
    name="bn_testing",
    version="0.2.0",
    packages=["bn_testing"],

    # dependencies
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'networkx>=2.5',
        'tqdm>4.6.0',
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
    url="https://github.com/windisch/bn-test-bench",
)
