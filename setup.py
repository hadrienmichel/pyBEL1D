import setuptools
try:
    import pygimli as pg
except ImportError:
    raise ImportError('You need to install pyGIMLi through conda')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyBEL1D",
    version="0.1",
    description="A Python implementation of the BEL1D codes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MICHEL Hadrien",
    author_email="hadrien.michel@uliege.be",
    url="https://github.com/hadrienmichel/pyBEL1D",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering"
    ],
    python_requiers="==3.7"
)