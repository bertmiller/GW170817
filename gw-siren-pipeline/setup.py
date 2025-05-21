from setuptools import setup, find_packages

setup(
    name="gwsiren",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.0",
        "pandas==2.1.1",
        "emcee==3.1.4",
        "astropy==5.3.3",
        "jax[cpu]==0.4.*",
    ],
    python_requires=">=3.8",
) 