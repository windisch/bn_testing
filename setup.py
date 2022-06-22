from setuptools import setup

setup(
    name="bn_testing",
    version="0.0.1",
    packages=["bn_testing"],

    # dependencies
    install_requires=[
        'numpy>=1.17.2',
        'sklearn',
        'pandas',
    ],
    tests_require=[
        "pytest",
    ],

    # metadata for upload to PyPI
    author="Tobias Windisch",
    author_email="tobias.windisch@posteo.de",
    description="A test bench to benchmark learn algorithms for graphical models",
    license="GNU GPL3",
    keywords="graphical models",
    url="https://github.com/windisch/bn-test-bench",
)
