from setuptools import setup, find_packages

setup(
    name="gwsiren",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "emcee",
        "astropy",
    ],
    python_requires=">=3.8",
) 