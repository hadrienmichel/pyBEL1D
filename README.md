# pyBEL1D
pyBEL1D is a python implementation of the BEL1D matlab codes ([BEL1D](https://github.com/hadrienmichel/BEL1D)). It is a work under devellopment and not in any form a finished product. 

# Installation:
The following instructions are working on Windows 10. 

Build a new conda environment with python 3.7.
```
conda create -n bel1d
conda activate bel1d
```
Install the different libraries in the new environment:
```
conda install python=3.7.7 # For the python version
conda install numpy
conda install scipy
conda install scikit-learn
conda install matplotlib
# For the sNMR application
conda config --add channels gimli --add channels conda-forge
conda install pygimli
# For DC application:
conda install libpython
conda install -c msys2 m2w64-toolchain
pip install git+https://github.com/miili/pysurf96
```

For MACOS installation, replace the last 3 lines with (not tested):
```
conda install -c anaconda gfortran_osx-64
pip install git+https://github.com/miili/pysurf96
```

On Linux machines, run (not tested):
```
conda install -c anaconda gfortran_linux-64
pip install git+https://github.com/miili/pysurf96
```

Then run the code in this environment (bel1d for the example above).

# Utilization
All the functions must be in the pyBEL1D folder to run (or you need to import the library, not yet implemented) and respect the folder architecture that is in the repository.

- The file exampleSNMR.py provides a commented example on how to run the codes for SNMR data.
- The file exampleDC.py provides an example on how to use BEL1D with a dispersion curve originating from real data.

# Acknowledgement
The forward model for sNMR is provided by [pygimli](https://www.pygimli.org).

The forward model for Surface Waves dispersion curves is a Python inteface of the Computer programs in seismology (R. Hermans) provided by miili on [github](https://github.com/miili/pysurf96).