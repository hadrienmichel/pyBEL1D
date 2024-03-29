# TODO/DONE:
#   - (Done on 24/04/2020) Add conditions (function for checking that samples are within a given space)
#   - (Done on 13/04/2020) Add Noise propagation (work in progress 29/04/20 - OK for SNMR 30/04/20 - DC OK) -> Noise impact is always very low???
#   - (Done on 11/05/2020) Add DC example (uses pysurf96 for forward modelling: https://github.com/miili/pysurf96 - compiled with msys2 for python)
#   - (Done on 18/05/2020) Add postprocessing (partially done - need for True model visualization on top and colorscale of graphs)
#   - (Done on 12/05/2020) Speed up kernel density estimator (vecotization?) - result: speed x4
#   - (Done on 13/05/2020) Add support for iterations
#   - Parallelization of the computations:
#       - (Done) KDE (one core/thread per dimension) -> Most probable gain
#       - (Done on 14/07/2020) Forward modelling (all cores/threads operating) -> Needed for more complex forward models
#       - (Not possible to parallelize (same seed for different workers))Sampling and checking conditions (all cores/thread operating) -> Can be usefull - not priority
#   - (Done) Add iteration convergence critereon!
#   - Lower the memory needs (how? not urgent)
#   - (Done 19-10-21) Comment the codes!
#   - (Done) Check KDE behaviour whit outliers (too long computations and useless?)

# Importing custom libraries
from .utilities import Tools
from .utilities.Tools import round_to_n
from .utilities.KernelDensity import KDE
#Importing common libraries
import numpy as np                          # For common matrix operations
import math as mt                           # Common mathematical functions
import matplotlib                           # For graphical outputs
from matplotlib import pyplot               # For matlab-like graphs
import sklearn                              # For PCA and CCA decompositions
import sklearn.decomposition                # For PCA decompositions
import sklearn.cross_decomposition          # For CCA decompositions
from scipy import stats                     # For the statistical distributions
from pathos import multiprocessing as mp    # For parallelization (No issues with pickeling)
from pathos import pools as pp              # For parallelization
from functools import partial               # For building parallelizable functions
import time                                 # For CPU time measurements
from numpy import random                    # For random sampling
from typing import Callable                 # For typing of functions in calls
from copy import deepcopy                   # For copying dataframes without links

# Forward models:
from pygimli.physics.sNMR import MRS, MRS1dBlockQTModelling # sNMR (from pyGIMLI: https://www.pygimli.org/)
from pysurf96 import surf96                                 # Dispersion Curves (from Github: https://github.com/hadrienmichel/pysurf96)
from scipy.integrate import solve_ivp                       # Pendulum

'''
In order for parallelization to work efficiently for different type of forward models,
some functions are requiered:

    - ForwardParallelFun: Enables an output to the function even if the function fails
    - ForwardSNMR: Function that defines the forward model directly instead of directly
                   calling a class method (not pickable).

In order for other forward model to run properly, please, use a similar method! The 
forward model MUST be a callable function DIRECTLY, not a class method.
'''
# Parralelization functions:
def ForwardParallelFun(Model, function, nbVal):
    '''This function enables the use of any function to be parralelized.

    WARNING: The function MUST be pickable by dill.

    Inputs: - Model (np.ndarray): the model for which the forward must be run
            - function (lambda function): the function that, when given a model,
                                          returns the corresponding dataset
            - nbVal (int): the number of values that the forward function 
                           is supposed to output. Used only in case of error.
    Output: The computed forward model or a None array of the same size.
    '''
    try:
        ForwardComputed = function(Model)
    except:
        ForwardComputed = [None]*nbVal
    return ForwardComputed

def ForwardSNMR(Model, nlay=None ,K=None ,zvec=None ,t=None):
    '''Function that extracts the forward model from the pygimli class object for SNMR.

    Inputs: - Model (np.ndarray): the model for which the forward must be run
            - nlay (int): the number of layers in the inputed model
            - K (np.ndarray): the kernel matrix (computed from MRSMatlab)
            - zvec (np.ndarray): the vertical discretization of the kernel matrix
            - t (np.ndarray): the timings for the data measurements
    
    Output: The forward model for the inputted Model

    FURTHER INFORMATIONS:

    For enabling parrallel runs and pickeling, the forward model MUST be declared at the 
    top of the file. 
    In the case of pysurf96, the function (surf96) is directly imported and usable as is. 
    In the case of SNMR, the function (method) is part of a class object and therefore not
    pickable as is. We create a pickable instance of the forward by creating a function 
    that calls the class object and its method directly. 
    
    This function is pickable (using dill).
    '''
    return MRS1dBlockQTModelling(nlay, K, zvec, t).response(Model)

### Forward modelling for pendulum:
def forwardPendulum(param, times):
    
    H = 10 # Heigth of the pivot point
    l = param[0]
    h = param[1]
    M = param[2] # Not used in the forward --> unsensitive parameter

    
    tmp = H - h # Removing the component above ground
    theta_0 = mt.acos(tmp/l)
    omega_0 = mt.sqrt(9.81/l)

    # Defining the system of equations:
    f = lambda t, in2: np.asarray([in2[1], omega_0**2 * np.sin(in2[0])]).transpose()

    x0 = np.asarray([theta_0, 0]).transpose()
    tInit = 0

    # Solve the equation system:
    y = solve_ivp(f, (tInit, np.max(times)), x0, t_eval=times)
    yReturn = np.ones_like(times)*H - np.cos(y.y[1,:])*l
    return yReturn

def forwardDCimage(freqsImage, velImage, model, nLayer):
    imageDC = np.zeros(len(freqsImage), len(velImage))
    periods = np.divide(1, freqsImage)
    dispCurve = np.zeros_like(periods)
    nbPeriods = np.ceil(periods/5)
    fails = 0
    for i in range(periods/5):
        periodCurr = periods[i*5:min([(i+1)*5, len(periods)])]
        try:
            dispCurve[i*5:(i+1)*5] = surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=model[2*nLayer-1:3*nLayer-1],vs=model[nLayer-1:2*nLayer-1],rho=model[3*nLayer-1:4*nLayer-1],periods=periodCurr,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        except:
            fails += 1
    if fails == nbPeriods:
        raise Exception('Unable to compute dispersion curve for current model')
    for i, vCompute in enumerate(dispCurve):
        idx = np.argmin(np.abs(velImage-vCompute))
        imageDC[i,idx] = 1
    imageFlatten = np.flatten(imageDC)
    return imageFlatten

class MODELSET:
    '''MODELSET is an object class that can be initialized using:
        - the dedicated class methods (DC and SNMR) - see dedicated help
        - the __init__ method
    
    To initialize with the init method, the different arguments are:
        - prior (list of scipy stats objects): a list describing the statistical
                                                distributions for the prior model space
        - cond (callable lambda function): a function that returns True or False if the
                                            model given in argument respects (True) the 
                                            conditions or not (False)
        - method (string): name of the method (e.g. "sNMR")
        - forwardFun (dictionary): a dictionary with two entries
                - "Fun" (callable lambda function): the forward model function for a given 
                                                    model
                - "Axis" (np.array): the X axis along which the computation is done
        - paramNames (dictionary): a dictionary with multiple entries
                - "NamesFU" (list): Full names of all the parameters with units
                - "NamesSU" (list): Short names of all the parameters with units
                - "NamesS" (list): Short names of all the parameters without units
                - "NamesGlobal" (list): Full names of the global parameters (not layered)
                - "NamesGlobalS" (list): Short names of the global parameters (not layered)
                - "DataUnits" (string): Units for the dataset,
                - "DataName" (string): Name of the Y-axis of the dataset (result from the 
                                       forward model)
                - "DataAxis" (string): Name of the X-axis of the dataset
        - nbLayer (int): the number of layers for the model (None if not layered)
        - logTransform ([bool, bool]): Applying a log transform to the models parameters 
                                       (first value) and/or the datasets (second value).
                                       The first boolean can also be a list of booleans 
                                       with the length of the prior which will mean that
                                       the log transform can be applied parameter by parameter.
    '''

    def __init__(self, prior=None, cond=None, method=None, forwardFun=None, paramNames=None, nbLayer=None, logTransform=[False, False], thicknessesFixed=None):
        if (prior is None) or (method is None) or (forwardFun is None) or (paramNames is None):
            self.prior = []
            self.method = []
            self.forwardFun = []
            self.paramNames = []
            self.nbLayer = nbLayer # If None -> Model with parameters and no layers (not geophy?)
            self.cond = cond
            self.logTransform = logTransform
            self.thicknessesFixed = None
        else:
            self.prior = prior
            self.method = method
            self.forwardFun = forwardFun
            self.cond = cond
            self.paramNames = paramNames
            self.nbLayer = nbLayer
            self.logTransform = logTransform
            self.thicknessesFixed = thicknessesFixed
    
    @classmethod
    def pendulum(cls, prior=None, timings=None):
        """Synthetic case of a pendulum.
        
        The class method takes as arguments:
            - prior (ndarray): a 1D numpy array containing the prior model space.
                                [l_min, l_max, h_min, h_max, M_min, M_max]
            - Timings (ndarray): a 1D array containing the times at which the
                                 position is sampled.
        By default, all inputs are None and the prior is the exemple case.

        Units of the prior are:
            - Length (l) in m
            - Height (h) in m
            - Mass (M) in kg 
        """
        # Defining the prior and the names of the variables
        H = 10 # Height of the axis 
        if prior is None:
            prior = np.asarray([1, 9, 1, 9, 0, 50])
        if timings is None:
            timings = np.arange(10, 121)
        nbLayer = None
        nbParam = 3
        ListPrior = [None] * nbParam
        NamesFullUnits = [None] * nbParam
        NamesShort = [None] * nbParam
        NamesShortUnits = [None] * nbParam
        Mins = np.zeros((nbParam,))
        Maxs = np.zeros((nbParam,))
        Units = [" [m]", " [m]", " [kg]"]
        NFull = ["Lenght ","Initial Heigth ","Mass "]
        NShort = ["l", "h", "M"]
        for i in range(nbParam):
            ListPrior[i] = stats.uniform(loc=prior[i*2],scale=prior[i*2+1]-prior[i*2])
            Mins[i] = prior[i*2]
            Maxs[i] = prior[i*2+1]
            NamesFullUnits[i] = NFull[i] + Units[i]
            NamesShortUnits[i] = NShort[i] + Units[i]
            NamesShort[i] = NShort[i]
        method = "Pendulum"
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["Depth [m]", "W [/]", "T_2^* [sec]"],"DataUnits":"[V]","DataName":"Amplitude [V]","DataAxis":"Time/pulses [/]"}# The representation is automated -> no time displayed since pulses are agregated
        
        forward = {"Fun":lambda model: forwardPendulum(model,timings),"Axis":timings}
        cond = lambda model: np.logical_and(np.greater_equal(model[0]+model[1],H),(np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs)))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nbLayer, logTransform=[False, False])

    @classmethod
    def SNMR(cls,prior=None,Kernel=None,Timing=None):
        """SNMR is a class method that generates a MODELSET class object for sNMR.

        The class method takes as arguments:
            - prior (ndarray): a 2D numpy array containing the prior model space 
                               decsription. The array is structured as follow:
                               [[e_1_min, e_1_max, W_1_min, W_1_max, T_2,1_min, T_2,1_max],
                               [e_2_min, ...                               ..., T_2,1_max],
                               [:        ...                               ...          :],
                               [e_nLay-1_min, ...                     ..., T_2,nLay-1_max],
                               [0, 0, W_nLay_min, ...                   ..., T_2,nLay_max]]

                               It has 6 columns and nLay lines, nLay beiing the number of 
                               layers in the model.
            
            - Kernel (str): a string containing the path to the matlab generated '*.mrsk'
                            kernel file.
            
            - Timing (array): a numpy array containing the timings for the dataset simulation.

            By default, all inputs are None and this generates the example sNMR case.

            Units for the prior are:
                - Thickness (e) in m
                - Water content (w) in m^3/m^3
                - Decay time (T_2^*) in sec

        """
        if prior is None:
            prior = np.array([[2.5, 7.5, 0.035, 0.10, 0.005, 0.350], [0, 0, 0.10, 0.30, 0.005, 0.350]])
        if Kernel is None:
            Kernel = "Data/sNMR/KernelTest.mrsk"
        if Timing is None:
            Timing = np.arange(0.005,0.5,0.001)
        nLayer, nParam = prior.shape
        nParam /= 2
        nParam = int(nParam)
        # prior = np.multiply(prior,np.matlib.repmat(np.array([1, 1, 1/100, 1/100, 1/1000, 1/1000]),nLayer,1))
        ListPrior = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesFullUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShort = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        Mins = np.zeros(((nLayer*nParam)-1,))
        Maxs = np.zeros(((nLayer*nParam)-1,))
        Units = [" [m]", " [/]", " [s]"]
        NFull = ["Thickness ","Water Content ","Relaxation Time "]
        NShort = ["th_{", "W_{", "T_{2,"]
        ident = 0
        for j in range(nParam):
            for i in range(nLayer):
                if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                    Mins[ident] = prior[i,j*2]
                    Maxs[ident] = prior[i,j*2+1]
                    NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                    NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                    NamesShort[ident] = NShort[j] + str(i+1) + "}"
                    ident += 1
        method = "sNMR"
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["Depth [m]", "W [/]", "T_2^* [sec]"],"DataUnits":"[V]","DataName":"Amplitude [V]","DataAxis":"Time/pulses [/]"}# The representation is automated -> no time displayed since pulses are agregated
        KFile = MRS()
        KFile.loadKernel(Kernel)
        forwardFun = lambda model: ForwardSNMR(model, nLayer, KFile.K, KFile.z, Timing)
        forward = {"Fun":forwardFun,"Axis":Timing}
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer, logTransform=[False, False])

    @classmethod
    def DC(cls,prior=None,Frequency=None):
        """DC is a class method that generates a MODELSET class object for DC.

        The class method takes as arguments:
            - prior (ndarray): a 2D numpy array containing the prior model space 
                               decsription. The array is structured as follow:
                               [[e_1_min, e_1_max, Vs_1_min, Vs_1_max, Vp_1_min, Vp_1_max, rho_1_min, rho_1_max],
                               [e_2_min, ...                                                     ..., rho_2_max],
                               [:        ...                                                     ...          :],
                               [e_nLay-1_min, ...                                           ..., rho_nLay-1_max],
                               [0, 0, Vs_nLay_min, ...                                        ..., rho_nLay_max]]

                               It has 8 columns and nLay lines, nLay beiing the number of 
                               layers in the model.
            
            - Frequency (array): a numpy array containing the frequencies for the dataset simulation.

            By default, all inputs are None and this generates the example sNMR case.

            Units for the prior are:
                - Thickness (e) in km
                - S-wave velocity (Vs) in km/sec
                - P-wave velocity (Vp) in km/sec
                - Density (rho) in T/m^3
        """
        # from pysurf96 import surf96
        if prior is None:
            prior = np.array([[0.0025, 0.0075, 0.002, 0.1, 0.05, 0.5, 1.0, 3.0], [0, 0, 0.1, 0.5, 0.3, 0.8, 1.0, 3.0]])
        if Frequency is None:
            Frequency = np.linspace(1,50,50)
        nLayer, nParam = prior.shape
        nParam /= 2
        nParam = int(nParam)
        # prior = np.multiply(prior,np.matlib.repmat(np.array([1/1000, 1/1000, 1, 1, 1, 1, 1, 1]),nLayer,1))
        ListPrior = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesFullUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShort = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        Mins = np.zeros(((nLayer*nParam)-1,))
        Maxs = np.zeros(((nLayer*nParam)-1,))
        Units = ["\\ [km]", "\\ [km/s]", "\\ [km/s]", "\\ [T/m^3]"]
        NFull = ["Thickness\\ ","s-Wave\\ velocity\\ ","p-Wave\\ velocity\\ ", "Density\\ "]
        NShort = ["th_{", "Vs_{", "Vp_{", "\\rho_{"]
        ident = 0
        for j in range(nParam):
            for i in range(nLayer):
                if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                    Mins[ident] = prior[i,j*2]
                    Maxs[ident] = prior[i,j*2+1]
                    NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                    NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                    NamesShort[ident] = NShort[j] + str(i+1) + "}"
                    ident += 1
        method = "DC"
        Periods = np.divide(1,Frequency)
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["Depth\\ [km]", "Vs\\ [km/s]", "Vp\\ [km/s]", "\\rho\\ [T/m^3]"],"DataUnits":"[km/s]","DataName":"Phase\\ velocity\\ [km/s]","DataAxis":"Periods\\ [s]"}
        forwardFun = lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=model[2*nLayer-1:3*nLayer-1],vs=model[nLayer-1:2*nLayer-1],rho=model[3*nLayer-1:4*nLayer-1],periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        forward = {"Fun":forwardFun,"Axis":Periods}
        def PoissonRatio(model):
            vp=model[2*nLayer-1:3*nLayer-1]
            vs=model[nLayer-1:2*nLayer-1]
            ratio = 1/2 * (np.power(vp,2) - 2*np.power(vs,2))/(np.power(vp,2)-np.power(vs,2))
            return ratio
        RatioMin = [0.2]*nLayer
        RatioMax = [0.45]*nLayer
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all() and (np.logical_and(np.greater(PoissonRatio(model),RatioMin),np.less(PoissonRatio(model),RatioMax))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer, logTransform=[False, False])

    @classmethod
    def DCVs(cls,prior=None,Frequency=None, VpFixed=None, RhoFixed=None):
        """DCVs is a class method that generates a MODELSET class object for DC. Contrary
        to the simple DC MODELSET class method, here, only Vs and depth are taken into account.

        The class method takes as arguments:
            - prior (ndarray): a 2D numpy array containing the prior model space 
                               decsription. The array is structured as follow:
                               [[e_1_min, e_1_max, Vs_1_min, Vs_1_max],
                               [e_2_min, ...            ..., Vs_2_max],
                               [:        ...           ...          :],
                               [e_nLay-1_min, ...  ..., Vs_nLay-1_max],
                               [0, 0, Vs_nLay_min, ....., Vs_nLay_max]]

                               It has 4 columns and nLay lines, nLay beiing the number of 
                               layers in the model.
            
            - Frequency (array): a numpy array containing the frequencies for the dataset simulation.

            By default, all inputs are None and this generates the example sNMR case.

            Units for the prior are:
                - Thickness (e) in km
                - S-wave velocity (Vs) in km/sec
        """
        # from pysurf96 import surf96
        if prior is None:
            prior = np.array([[0.0025, 0.0075, 0.002, 0.1], [0, 0, 0.1, 0.5]])
        if Frequency is None:
            Frequency = np.linspace(1,50,50)
        nLayer, nParam = prior.shape
        nParam /= 2
        nParam = int(nParam)
        # prior = np.multiply(prior,np.matlib.repmat(np.array([1/1000, 1/1000, 1, 1, 1, 1, 1, 1]),nLayer,1))
        ListPrior = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesFullUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShort = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        Mins = np.zeros(((nLayer*nParam)-1,))
        Maxs = np.zeros(((nLayer*nParam)-1,))
        Units = ["\\ [km]", "\\ [km/s]"]
        NFull = ["Thickness\\ ","s-Wave\\ velocity\\ "]
        NShort = ["th_{", "Vs_{"]
        ident = 0
        for j in range(nParam):
            for i in range(nLayer):
                if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                    Mins[ident] = prior[i,j*2]
                    Maxs[ident] = prior[i,j*2+1]
                    NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                    NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                    NamesShort[ident] = NShort[j] + str(i+1) + "}"
                    ident += 1
        method = "DC"
        Periods = np.divide(1,Frequency)
        PoissonRatio = 0.25
        RhoTypical = 2.0
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["Depth\\ [km]", "Vs\\ [km/s]"],"DataUnits":"[km/s]","DataName":"Phase\\ velocity\\ [km/s]","DataAxis":"Periods\\ [s]"}
        if (VpFixed is None) and (RhoFixed is None):
            forwardFun = lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(model[nLayer-1:2*nLayer-1],2)),vs=model[nLayer-1:2*nLayer-1],rho=np.ones((nLayer,))*RhoTypical,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        elif (VpFixed is not None) and (RhoFixed is not None):
            forwardFun = lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=VpFixed,vs=model[nLayer-1:2*nLayer-1],rho=RhoFixed,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        elif VpFixed is not None:
            forwardFun = lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=VpFixed,vs=model[nLayer-1:2*nLayer-1],rho=np.ones((nLayer,))*RhoTypical,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        elif RhoFixed is not None:
            forwardFun = lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(model[nLayer-1:2*nLayer-1],2)),vs=model[nLayer-1:2*nLayer-1],rho=RhoFixed,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        
        forward = {"Fun":forwardFun,"Axis":Periods}
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer, logTransform=[False, False])

    @classmethod
    def sNMR_logLayers(cls,prior=None, Kernel=None, Timing=None, logUniform=True, maxDepth:float=150, minThick=0.05, maxThick=5, nbLayers:int=25):
        """SNMR is a class method that generates a MODELSET class object for sNMR.

        The class method takes as arguments:
            - prior (ndarray): a 2D numpy array containing the prior model space 
                               decsription. The array is structured as follow:
                               [[e_1_min, e_1_max, W_1_min, W_1_max, T_2,1_min, T_2,1_max],
                               [e_2_min, ...                               ..., T_2,1_max],
                               [:        ...                               ...          :],
                               [e_nLay-1_min, ...                     ..., T_2,nLay-1_max],
                               [0, 0, W_nLay_min, ...                   ..., T_2,nLay_max]]

                               It has 6 columns and nLay lines, nLay beiing the number of 
                               layers in the model.
            
            - Kernel (str): a string containing the path to the matlab generated '*.mrsk'
                            kernel file.
            
            - Timing (array): a numpy array containing the timings for the dataset simulation.

            By default, all inputs are None and this generates the example sNMR case.

            Units for the prior are:
                - Thickness (e) in m
                - Water content (w) in m^3/m^3
                - Decay time (T_2^*) in sec

        """
        if prior is None:
            prior = np.repeat(np.atleast_2d([0.025, 0.30, 0.005, 0.400]), nbLayers, axis=0)
        if Kernel is None:
            Kernel = "Data/sNMR/KernelTest.mrsk"
        if Timing is None:
            Timing = np.arange(0.005,0.5,0.001)
        nParam = 2
        # prior = np.multiply(prior,np.matlib.repmat(np.array([1, 1, 1/100, 1/100, 1/1000, 1/1000]),nLayer,1))
        ListPrior = [None] * ((nbLayers*nParam))# Half space at bottom
        NamesFullUnits = [None] * ((nbLayers*nParam))# Half space at bottom
        NamesShort = [None] * ((nbLayers*nParam))# Half space at bottom
        NamesShortUnits = [None] * ((nbLayers*nParam))# Half space at bottom
        Mins = np.zeros(((nbLayers*nParam),))
        Maxs = np.zeros(((nbLayers*nParam),))
        Units = [" [/]", " [s]"]
        NFull = ["Water Content ","Relaxation Time "]
        NShort = ["W_{", "T_{2,"]
        ## Linear-Log mixing for depth discretization (inspiered from MRSMatlab):
        # notOK = True
        # linLayers = 0
        # while notOK:
        #     depths = np.logspace(np.log10(minThick), np.log10(maxDepth-(linLayers*maxThick)), nbLayers-(linLayers))
        #     thickness = np.append([minThick], np.diff(depths))
        #     if thickness[-1] > maxThick:
        #         linLayers += 1
        #     else: 
        #         notOK = False
        # if linLayers > 0:
        #     depths = np.append(depths, np.cumsum(np.ones((linLayers,))*maxThick)+depths[-1])
        #     thickness = np.append([minThick], np.diff(depths))
        thickness = np.ones((nbLayers,))*maxThick
        # else:
        #     raise Exception('Impossible to generate a thickness model with those parameters.')
        ident = 0
        for j in range(nParam):
            for i in range(nbLayers):
                if logUniform:
                    ListPrior[ident] = stats.loguniform(a=prior[i,j*2],b=prior[i,j*2+1]-prior[i,j*2])
                else:
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                Mins[ident] = prior[i,j*2]
                Maxs[ident] = prior[i,j*2+1]
                NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                NamesShort[ident] = NShort[j] + str(i+1) + "}"
                ident += 1
        method = "sNMR"
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["Depth [m]", "W [/]", "T_2^* [sec]"],"DataUnits":"[V]","DataName":"Amplitude [V]","DataAxis":"Time/pulses [/]"}# The representation is automated -> no time displayed since pulses are agregated
        KFile = MRS()
        KFile.loadKernel(Kernel)
        forwardFun = lambda model: ForwardSNMR(np.append(thickness[:-1], model), nbLayers, KFile.K, KFile.z, Timing)
        forward = {"Fun":forwardFun,"Axis":Timing}
        cond = lambda model: (np.logical_and(np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs)),1)).all() #np.logical_and(np.max(np.abs(np.diff(model[:nbLayers]))) < 0.05, np.max(np.abs(np.diff(model[nbLayers:]))) < 0.05))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nbLayers, logTransform=[False, False], thicknessesFixed=thickness)

    @classmethod
    def DCVs_logLayers(cls,prior=None,Frequency=None, logUniform=True, maxDepth:float=0.150, minThick=5e-4, maxThick=0.005, nbLayers:int=50):
        """DCVs_logLayers is a class method that generates a MODELSET class object for DC. Contrary
        to the simple DC MODELSET class method, here, only Vs and depth are taken into account.

        The class method takes as arguments:
            - prior (ndarray): a 2D numpy array containing the prior model space 
                               decsription. The array is structured as follow:
                               [[e_1_min, e_1_max, Vs_1_min, Vs_1_max],
                               [e_2_min, ...            ..., Vs_2_max],
                               [:        ...           ...          :],
                               [e_nLay-1_min, ...  ..., Vs_nLay-1_max],
                               [0, 0, Vs_nLay_min, ....., Vs_nLay_max]]

                               It has 4 columns and nLay lines, nLay beiing the number of 
                               layers in the model.
            
            - Frequency (array): a numpy array containing the frequencies for the dataset simulation.

            By default, all inputs are None and this generates the example sNMR case.

            Units for the prior are:
                - Thickness (e) in km
                - S-wave velocity (Vs) in km/sec
        """
        # from pysurf96 import surf96
        if prior is None:
            prior = np.repeat(np.atleast_2d([0.1, 3.0]), nbLayers, axis=0)
        if Frequency is None:
            Frequency = np.linspace(1,50,50)
        nParam = 1
        ## Linear-Log mixing for depth discretization (inspiered from MRSMatlab):
        notOK = True
        linLayers = 0
        while notOK:
            depths = np.logspace(np.log10(minThick), np.log10(maxDepth-(linLayers*maxThick)), nbLayers-(linLayers+1))
            thickness = np.append([minThick], np.diff(depths))
            if thickness[-1] > maxThick:
                linLayers += 1
            else: 
                notOK = False
        if linLayers > 0:
            depths = np.append(depths, np.cumsum(np.ones((linLayers,))*maxThick)+depths[-1])
            thickness = np.append([minThick], np.diff(depths))
        else:
            raise Exception('Impossible to generate a thickness model with those parameters.')
        # prior = np.multiply(prior,np.matlib.repmat(np.array([1/1000, 1/1000, 1, 1, 1, 1, 1, 1]),nLayer,1))
        ListPrior = [None] * nbLayers# Half space at bottom
        NamesFullUnits = [None] * nbLayers# Half space at bottom
        NamesShort = [None] * nbLayers# Half space at bottom
        NamesShortUnits = [None] * nbLayers# Half space at bottom
        Mins = np.zeros((nbLayers,))
        Maxs = np.zeros((nbLayers,))
        Units = ["\\ [km/s]"]
        NFull = ["s-Wave\\ velocity\\ "]
        NShort = ["Vs_{"]
        ident = 0
        for j in range(nParam):
            for i in range(nbLayers):
                if logUniform:
                    ListPrior[ident] = stats.loguniform(a=prior[i,j*2],b=prior[i,j*2+1]-prior[i,j*2])
                else:
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                Mins[ident] = prior[i,j*2]
                Maxs[ident] = prior[i,j*2+1]
                NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                NamesShort[ident] = NShort[j] + str(i+1) + "}"
                ident += 1
        method = "DC"
        Periods = np.divide(1,Frequency)
        PoissonRatio = 0.25
        RhoTypical = 2.0
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["Depth\\ [km]", "Vs\\ [km/s]"],"DataUnits":"[km/s]","DataName":"Phase\\ velocity\\ [km/s]","DataAxis":"Periods\\ [s]"}
        forwardFun = lambda model: surf96(thickness=np.append(thickness, [0]),vp=np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(model,2)),vs=model,rho=np.ones((nbLayers,))*RhoTypical,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        forward = {"Fun":forwardFun,"Axis":Periods}
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nbLayers, logTransform=[False, False], thicknessesFixed=thickness)

class PREBEL:
    """Object that is used to store the PREBEL elements:
    
    For a given model set (see MODELSET class), the PREBEL class
    enables all the computations that takes place previous to any 
    field data knowledge. It takes as argument:
        - MODPARAM (MODELSET class object): a previously defined
                                            MODLESET class object
        - nbModels (int): the number of models to sample from the
                          prior (dafault = 1000)

    The class object has different attributes:
        - PRIOR (list): a list containing the different distributions
                        for the prior. The statistical distributions 
                        must be scipy.stats objects in order to be
                        sampled.
        - CONDITIONS (callable): a lambda functions that will return
                                 either *True* if the selected model 
                                 is within the prior conditions of 
                                 *False* otherwise.
        - nbModels (int): the number of models to sample from the
                          prior.
        - MODPARAM (MODELSET): the full MODELSET object (see class 
                               description).
        - MODELS (np.ndarray): the sampled models with dimensions
                               (nbModels * nbParam)
        - FORWARD (np.ndarray): the corresponding datasets with 
                                dimensions (nbModels * len(data))
        - PCA (dict): a dictionnary containing the PCA reduction with 
                      their mathematical descriptions:
            - 'Data': a sklearn.decompostion.PCA object with the PCA
                      decomposition for the data dimensions.
            - 'Models': a sklearn.decompostion.PCA object with the PCA
                        decomposition for the models dimensions. If no
                        PCA reduction is applied to the model space, 
                        the value is *None*
        - CCA (object): a sklearn.cross_decomposition.CCA object 
                        containing the CCA decomposition.
        - KDE (object): a KDE object containing the Kernel Density 
                        Estimation for the CCA space (see 
                        utilities.KernelDensity for more details).

    """
    def __init__(self,MODPARAM:MODELSET,nbModels:int=1000):
        self.PRIOR = MODPARAM.prior
        self.CONDITIONS = MODPARAM.cond
        self.nbModels = nbModels
        self.MODPARAM = MODPARAM
        self.MODELS = []
        self.FORWARD = []
        self.PCA = dict()
        self.CCA = []
        self.KDE = []

    def run(self, Parallelization:list=[False, None], RemoveOutlier:bool=False, reduceModels:bool=False, verbose:bool=False, PriorSampled=None):
        """The RUN method runs all the computations for the preparation of BEL1D

        It is an instance method that does not need any arguments.
        Howerev, optional arguments are:
            - Parallelization (list): instructions for parallelization
                - [False, ?]: no parallel runs
                - [True, None]: parallel runs without pool provided
                - [True, pool]: parallel runs with pool (defined bypathos.pools) 
                                provided
                The default is no parallel runs.
            - RemoveOutlier (bool): simplifie the KDE computation by removing models
                                    that are way outside the space (default=False).
            - reduceModels (bool): apply PCA reduction to the models (True) or not (False).
                                   Default value is *False*
            - verbose (bool): receive feedback from the code while it is running (True)
                              or not (False). The default is *False*.
        """
        # 1) Sampling (if not done already):
        if verbose:
            print('Sampling the prior . . .')
        if PriorSampled is None:
            if self.nbModels is None: # Normally, we should never enter this
                self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS)
                self.nbModels = 1000
            else:
                self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS,self.nbModels)
        else:
            self.MODELS = PriorSampled
            self.nbModels = PriorSampled.shape[0]
        
        # 2) Running the forward model
        if verbose:
            print('Running the forward modelling . . .')
        # For DC, sometimes, the code will return an error --> need to remove the model from the prior
        # Initialization of the FORWARD attribute. Need to compute 
        indexCurr = 0
        while True:
            try:
                tmp = self.MODPARAM.forwardFun["Fun"](self.MODELS[indexCurr,:])
                break
            except:
                indexCurr += 1
                if indexCurr > self.nbModels:
                    raise Exception('The forward modelling failed!')
        self.FORWARD = np.zeros((self.nbModels,len(tmp)))
        timeBegin = time.time()
        if Parallelization[0]:
            # We create a partial function that has a fixed fowrard function. The remaining arguments are :
            #   - Model: a numpy array containing the model to compute
            # It returns the Forward Computed, either a list of None or a list of values corresponding to the forward
            functionParallel = partial(ForwardParallelFun, function=self.MODPARAM.forwardFun["Fun"], nbVal=len(tmp))
            inputs = [self.MODELS[i,:] for i in range(self.nbModels)]
            if Parallelization[1] is not None:
                pool = Parallelization[1]
                terminatePool = False
            else:
                pool = pp.ProcessPool(mp.cpu_count()) # Create the pool for paralelization
                Parallelization[1] = pool
                terminatePool = True
            outputs = pool.map(functionParallel,inputs)
            self.FORWARD = np.vstack(outputs) #ForwardParallel
            notComputed = [i for i in range(self.nbModels) if self.FORWARD[i,0] is None]
            self.MODELS = np.array(np.delete(self.MODELS,notComputed,0),dtype=np.float64)
            self.FORWARD = np.array(np.delete(self.FORWARD,notComputed,0),dtype=np.float64)
            newModelsNb = np.size(self.MODELS,axis=0) # Get the number of models remaining
            timeEnd = time.time()
            if verbose:
                print('The Parallelized Forward Modelling took {} seconds.'.format(timeEnd-timeBegin))
        else:
            notComputed = []
            for i in range(self.nbModels):
                # print(i)
                try:
                    self.FORWARD[i,:] = self.MODPARAM.forwardFun["Fun"](self.MODELS[i,:])
                except:
                    self.FORWARD[i,:] = [None]*len(tmp)
                    notComputed.append(i)
            # Getting the uncomputed models and removing them:
            self.MODELS = np.delete(self.MODELS,notComputed,0)
            self.FORWARD = np.delete(self.FORWARD,notComputed,0)
            newModelsNb = np.size(self.MODELS,axis=0) # Get the number of models remaining
            timeEnd = time.time()
            if verbose:
                print('The Unparallelized Forward Modelling took {} seconds.'.format(timeEnd-timeBegin))
        if self.MODPARAM.method == "DC":
            # In the case of surface waves, the forward model sometimes provide datasets that have a sharp
            # transition that is not possible in practice. We therefore need to remove those models. They 
            # are luckily easy to identify. Their maximum variabilty is way larger than the other models.
            VariabilityMax = np.max(np.abs(self.FORWARD[:,1:]-self.FORWARD[:,:-1]),axis=1)
            from scipy.special import erfcinv # https://github.com/PyCQA/pylint/issues/3744 pylint: disable=no-name-in-module
            c = -1/(mt.sqrt(2)*erfcinv(3/2))
            VariabilityMaxAuthorized = np.median(VariabilityMax) + 3 * c*np.median(np.abs(VariabilityMax-np.median(VariabilityMax)))
            isOutlier = np.greater(np.abs(VariabilityMax),VariabilityMaxAuthorized)
            self.MODELS = np.delete(self.MODELS,np.where(isOutlier),0)
            self.FORWARD = np.delete(self.FORWARD,np.where(isOutlier),0)
            newModelsNb = np.size(self.FORWARD,axis=0) # Get the number of models remaining
            pass
        if verbose:
            print('{} models remaining after forward modelling!'.format(newModelsNb))
        self.nbModels = newModelsNb
        # 3) PCA on data (and optionally model):
        if verbose:
            print('Reducing the dimensionality . . .')
        varRepresented = 0.90
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=varRepresented) # Keeping 90% of the variance
            if type(self.MODPARAM.logTransform[0]) is list:
                if len(self.MODPARAM.logTransform[0]) != len(self.PRIOR):
                    raise Exception('The length of the log-transform is not the same as the legth of the prior.')
                ModelsTransform = deepcopy(self.MODELS)
                for count, logT in enumerate(self.MODPARAM.logTransform[0]):
                    if logT:
                        ModelsTransform[:,count] = np.log(ModelsTransform[:,count])
                m_h = pca_model.fit_transform(ModelsTransform)
            elif (type(self.MODPARAM.logTransform[0]) is bool) and self.MODPARAM.logTransform[0]:
                m_h = pca_model.fit_transform(np.log(self.MODELS))
            else:
                m_h = pca_model.fit_transform(self.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            if self.MODPARAM.logTransform[1]:
                d_h = pca_data.fit_transform(np.log(self.FORWARD))
            else:
                d_h = pca_data.fit_transform(self.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions ({}) than the models ({})!'.format(n_CompPCA_Data, n_CompPCA_Mod))
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                if self.MODPARAM.logTransform[1]:
                    d_h = pca_data.fit_transform(np.log(self.FORWARD))
                else:
                    d_h = pca_data.fit_transform(self.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            self.PCA = {'Data':pca_data,'Model':pca_model}
        else:
            if type(self.MODPARAM.logTransform[0]) is list:
                if len(self.MODPARAM.logTransform[0]) != len(self.PRIOR):
                    raise Exception('The length of the log-transform is not the same as the legth of the prior.')
                ModelsTransform = deepcopy(self.MODELS)
                for count, logT in enumerate(self.MODPARAM.logTransform[0]):
                    if logT:
                        ModelsTransform[:,count] = np.log(ModelsTransform[:,count])
                m_h = ModelsTransform
            elif (type(self.MODPARAM.logTransform[0]) is bool) and self.MODPARAM.logTransform[0]:
                m_h = np.log(self.MODELS)
            else:
                m_h = self.MODELS
            n_CompPCA_Mod = m_h.shape[1]
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            if self.MODPARAM.logTransform[1]:
                d_h = pca_data.fit_transform(np.log(self.FORWARD))
            else:
                d_h = pca_data.fit_transform(self.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions ({}) than the models ({})!'.format(n_CompPCA_Data, n_CompPCA_Mod))
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                if self.MODPARAM.logTransform[1]:
                    d_h = pca_data.fit_transform(np.log(self.FORWARD))
                else:
                    d_h = pca_data.fit_transform(self.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            self.PCA = {'Data':pca_data,'Model':None}
        # 4) CCA:
        if verbose:
            print('Building the CCA space . . .')
        cca_transform = sklearn.cross_decomposition.CCA(n_components=n_CompPCA_Mod)
        d_c,m_c = cca_transform.fit_transform(d_h,m_h)
        self.CCA = cca_transform
        # 5) KDE:
        if verbose:
            print('Running Kernel Density Estimation . . .')
        self.KDE = KDE(d_c,m_c)
        self.KDE.KernelDensity(RemoveOutlier=RemoveOutlier,Parallelization=Parallelization, verbose=verbose)
        if Parallelization[0] and terminatePool:
            pool.terminate()
    
    @classmethod
    def POSTBEL2PREBEL(cls, PREBEL, POSTBEL, Dataset=None, NoiseModel=None, nbMaxPrior:int=1000000, Parallelization:list=[False,None], reduceModels:bool=False, verbose:bool=False):
        ''' POSTBEL2PREBEL is a class method that converts a POSTBEL object to a PREBEL one.

        It takes as arguments:
            - PREBEL (PREBEL): The previous PREBEL object
            - POSTBEL (POSTBEL): the current POSTBEL object
        And optional arguments are:
            - Dataset (np.array): the field dataset
            - NoiseModel (list): the list defining the noise model (see dedicated functions)
            - nbMaxPrior (int=1000000): the maximum number of models that can be selected
                                        in the built prior
            - Parallelization (list): parallelization instructions
                    - [False, _]: no parallel runs (default)
                    - [True, None]: parallel runs without pool provided
                    - [True, pool]: parallel runs with pool (defined by pathos.pools) 
                                    provided 
            - reduceModels (bool): apply PCA reduction to the models (True) or not (False).
                                   Default value is *False*
            - verbose (bool): output progresses messages (True) or not (False - default)

        '''
        # 1) Initialize the Prebel class object
        if verbose:
            print('Initializing the PREBEL object . . .')
        Modelset = POSTBEL.MODPARAM # A MODELSET class object
        PrebelNew = cls(Modelset)
        # 2) Running the forward model
        if not(len(POSTBEL.SAMPLESDATA) != 0):
            if verbose: 
                print('Running the forward modelling . . .')
            # We are using the built-in method of POSTBEL to run the forward model
            POSTBEL.DataPost(Parallelization=Parallelization)
        if verbose:
            print('Building the informed prior . . .')
        PrebelNew.MODELS = np.append(PREBEL.MODELS,POSTBEL.SAMPLES,axis=0)
        PrebelNew.FORWARD = np.append(PREBEL.FORWARD,POSTBEL.SAMPLESDATA,axis=0)
        PrebelNew.nbModels = np.size(PrebelNew.MODELS,axis=0) # Get the number of sampled models
        # Downsampling in case of too many models:
        if PrebelNew.nbModels > nbMaxPrior:
            idxKeep = np.random.choice(PrebelNew.nbModels, nbMaxPrior, replace=False)
            PrebelNew.MODELS = PrebelNew.MODELS[idxKeep,:]
            PrebelNew.FORWARD = PrebelNew.FORWARD[idxKeep,:]
            PrebelNew.nbModels = len(idxKeep)
        # 3) PCA on data (and optionally model):
        varRepresented = 0.90
        if verbose:
            print('Reducing the dimensionality . . .')
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=varRepresented) # Keeping 90% of the variance
            if type(PrebelNew.MODPARAM.logTransform[0]) is list:
                if len(PrebelNew.MODPARAM.logTransform[0]) != len(PrebelNew.PRIOR):
                    raise Exception('The length of the log-transform is not the same as the legth of the prior.')
                ModelsTransform = deepcopy(PrebelNew.MODELS)
                for count, logT in enumerate(PrebelNew.MODPARAM.logTransform[0]):
                    if logT:
                        ModelsTransform[:,count] = np.log(ModelsTransform[:,count])
                m_h = pca_model.fit_transform(ModelsTransform)
            elif (type(PrebelNew.MODPARAM.logTransform[0]) is bool) and PrebelNew.MODPARAM.logTransform[0]:
                m_h = pca_model.fit_transform(np.log(PrebelNew.MODELS))
            else:
                m_h = pca_model.fit_transform(PrebelNew.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            if PrebelNew.MODPARAM.logTransform[1]:
                d_h = pca_data.fit_transform(np.log(PrebelNew.FORWARD))
            else:
                d_h = pca_data.fit_transform(PrebelNew.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions than the models!')
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                if PrebelNew.MODPARAM.logTransform[1]:
                    d_h = pca_data.fit_transform(np.log(PrebelNew.FORWARD))
                else:
                    d_h = pca_data.fit_transform(PrebelNew.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            PrebelNew.PCA = {'Data':pca_data,'Model':pca_model}
        else:
            if type(PrebelNew.MODPARAM.logTransform[0]) is list:
                if len(PrebelNew.MODPARAM.logTransform[0]) != len(PrebelNew.PRIOR):
                    raise Exception('The length of the log-transform is not the same as the legth of the prior.')
                ModelsTransform = deepcopy(PrebelNew.MODELS)
                for count, logT in enumerate(PrebelNew.MODPARAM.logTransform[0]):
                    if logT:
                        ModelsTransform[:,count] = np.log(ModelsTransform[:,count])
                m_h = ModelsTransform
            elif (type(PrebelNew.MODPARAM.logTransform[0]) is bool) and PrebelNew.MODPARAM.logTransform[0]:
                m_h = np.log(PrebelNew.MODELS)
            else:
                m_h = PrebelNew.MODELS
            n_CompPCA_Mod = m_h.shape[1]
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            if PrebelNew.MODPARAM.logTransform[1]:
                d_h = pca_data.fit_transform(np.log(PrebelNew.FORWARD))
            else:
                d_h = pca_data.fit_transform(PrebelNew.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions than the models!')
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                if PrebelNew.MODPARAM.logTransform[1]:
                    d_h = pca_data.fit_transform(np.log(PrebelNew.FORWARD))
                else:
                    d_h = pca_data.fit_transform(PrebelNew.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            PrebelNew.PCA = {'Data':pca_data,'Model':None}
        # 4) CCA:
        if verbose:
            print('Building the CCA space . . .')
        cca_transform = sklearn.cross_decomposition.CCA(n_components=n_CompPCA_Mod)
        d_c,m_c = cca_transform.fit_transform(d_h,m_h)
        PrebelNew.CCA = cca_transform
        # 5-pre) If dataset already exists:
        if Dataset is not None:
            Dataset = np.reshape(Dataset,(1,-1))# Convert for reverse transform
            if PrebelNew.MODPARAM.logTransform[1]:
                Dataset = np.log(Dataset)
            d_obs_h = PrebelNew.PCA['Data'].transform(Dataset)
            d_obs_c = PrebelNew.CCA.transform(d_obs_h)
            if NoiseModel is not None:
                Noise = np.sqrt(Tools.PropagateNoise(PrebelNew, NoiseModel, DatasetIn=Dataset))
            else:
                Noise = None
        # 5) KDE:
        if verbose:
            print('Running Kernel Density Estimation . . .')
        PrebelNew.KDE = KDE(d_c,m_c)
        if Dataset is None:
            PrebelNew.KDE.KernelDensity(Parallelization=Parallelization, verbose=verbose)
        else:
            PrebelNew.KDE.KernelDensity(XTrue=np.squeeze(d_obs_c), NoiseError=Noise, Parallelization=Parallelization, verbose=verbose)
        if verbose:
            print('PREBEL object build!')
        return PrebelNew
    
    def runMCMC(self, Dataset=None, NoiseModel=None, nbSamples:int=100000, nbChains:int=10, noData:bool=False, verbose:bool=False):
        ''' RUNMCMC is a class method that runs a simple metropolis McMC algorithm
        on the prior model space (PREBEL). 

        It takes as arguments:
            - Dataset (np.array): the field dataset
            - NoiseModel (np.array): the list defining the noise model
            - nbSamples (int): the number of models to sample per chains (larger for larger 
                               priors). The default value is 100000
            - nbChains (int): the number of chains to run. The larger, the better to avoid 
                              remaining in a local optimum. The default value is 10.
            - noData (bool): Return data (False - default) or not (True).
            - verbose (bool): output progresses messages (True) or not (False - default)

        It returns 2 arrays containing the samples models and the associated datasets.
        '''
        if Dataset is None:
            raise Exception('No Dataset given to compute likelihood')
        if len(Dataset) != self.FORWARD.shape[1]:
            raise Exception('Dataset given not compatible with forward model')
        if NoiseModel is None:
            raise Exception('No noise model provided. Impossible to compute the likelihood')
        if len(NoiseModel) != len(Dataset):
            raise Exception('NoiseModel should have the same size as the dataset')
        timeIn = time.time() # For the timer
        nbParam = len(self.MODPARAM.prior)
        accepted = np.zeros((nbChains, nbSamples, nbParam))
        if noData:
            acceptedData = None 
        else:
            acceptedData = np.zeros((nbChains, nbSamples, len(Dataset)))
        for j in range(nbChains):
            if verbose:
                print('Running chain {} out of {}. . .'.format(j, nbChains))
            rejectedNb = 0
            i = 0
            LikelihoodLast = 1e-50 # First sample most likely accepted
            Covariance = 0.01*np.cov(self.MODELS.T) # Compute the initial covariance from the prior distribution
            passed = False
            passedVerbose = False
            while i < nbSamples:
                if i == 0:
                    ## Sampling a random model from the prior distribution
                    sampleCurr = Tools.Sampling(self.PRIOR,self.CONDITIONS,nbModels=1)
                else:
                    ## Random change to the sampled model (according to the covariance):
                    sampleCurr = sampleLast + np.random.multivariate_normal(np.zeros((nbParam,)),Covariance)
                ## Computing the likelihood from a data misfit:
                if self.MODPARAM.cond(sampleCurr):
                    try:
                        SynData = self.MODPARAM.forwardFun['Fun'](sampleCurr[0])
                        DataDiff = Dataset - SynData
                        FieldError = NoiseModel
                        A = np.divide(1,np.sqrt(2*np.pi*np.power(FieldError,2)))
                        B = np.exp(-1/2 * np.power(np.divide(DataDiff,FieldError),2))
                        Likelihood = np.prod(np.multiply(A, B))
                        if Likelihood > 1e307:
                            Likelihood = 1e307
                    except:
                        rejectedNb += 1
                        continue
                else:
                    rejectedNb += 1
                    continue
                ## Sampling (or not) the model:
                ratio = Likelihood/LikelihoodLast
                if ratio > np.random.uniform(0,1):
                    sampleLast = sampleCurr
                    accepted[j,i,:] = sampleCurr[0]
                    if not(noData):
                        acceptedData[j,i,:] = SynData
                    i += 1
                    passed = False
                    passedVerbose = False
                else:
                    rejectedNb += 1
                if np.mod(i,int(nbSamples/100)) == 0 and not(passedVerbose):
                    if verbose:
                        print('{} models sampled (out of {}) in chain {}.'.format(i, nbSamples, j))
                    passedVerbose = True
                    # LikelihoodLast = 1e-50
                if  (i < nbSamples/2) and (np.mod(i,50) == 0) and not(passed):
                    AcceptanceRatio = i/(rejectedNb+i)
                    if AcceptanceRatio < 0.80: # 0.75
                        if verbose:
                            print('Acceptance ratio too low, reducing covariance.')
                        Covariance *= 0.8 # We are reducing the covariance to increase the acceptance rate
                    elif AcceptanceRatio > 0.90: # 0.85
                        if verbose:
                            print('Acceptance ratio too high, increasing covariance.')
                        Covariance *= 1.2 # We are increasing the covariance to decrease the acceptance rate
                    passed = True
                LikelihoodLast = Likelihood
        if verbose:
            print(f'MCMC on PREBEL executed in {time.time()-timeIn} seconds.')
        return np.asarray(accepted), np.asarray(acceptedData)
    
    def ShowPreModels(self,TrueModel=None):
        '''SHOWPREMODELS is a function that displays the models sampled from the prior model space.

        The optional argument TrueModel (np.array) is an array containing the benchmark model.
        '''
        nbParam = self.MODELS.shape[1]
        nbLayer = self.MODPARAM.nbLayer
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        sortIndex = np.arange(self.nbModels)
        if nbLayer is not None:# If the model can be displayed as layers
            nbParamUnique = int(np.ceil(nbParam/nbLayer))-1 # Number of parameters minus the thickness
            fig = pyplot.figure(figsize=[4*nbParamUnique,10])
            Param = list()
            Param.append(np.cumsum(self.MODELS[:,0:nbLayer-1],axis=1))
            for i in range(nbParamUnique):
                Param.append(self.MODELS[:,(i+1)*nbLayer-1:(i+2)*nbLayer-1])
            if TrueModel is not None:
                TrueMod = list()
                TrueMod.append(np.cumsum(TrueModel[0:nbLayer-1]))
                for i in range(nbParamUnique):
                    TrueMod.append(TrueModel[(i+1)*nbLayer-1:(i+2)*nbLayer-1])
                
            maxDepth = np.max(Param[0][:,-1])*1.25
            if nbParamUnique > 1:
                axes = fig.subplots(1,nbParamUnique) # One graph per parameter
                for j in range(nbParamUnique):
                    for i in sortIndex:
                        axes[j].step(np.append(Param[j+1][i,:], Param[j+1][i,-1]),np.append(np.append(0, Param[0][i,:]), maxDepth),where='pre',color='gray')
                    if TrueModel is not None:
                        axes[j].step(np.append(TrueMod[j+1][:], TrueMod[j+1][-1]),np.append(np.append(0, TrueMod[0][:]), maxDepth),where='pre',color='k')
                    axes[j].invert_yaxis()
                    axes[j].set_ylim(bottom=maxDepth,top=0.0)
                    axes[j].set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][j+1]),fontsize=14)
                    axes[j].set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][0]),fontsize=14)
            else:
                j = 0
                axes = fig.subplots(1,nbParamUnique) # One graph per parameter
                for i in sortIndex:
                    axes.step(np.append(Param[j+1][i,:], Param[j+1][i,-1]),np.append(np.append(0, Param[0][i,:]), maxDepth),where='pre',color='gray')
                if TrueModel is not None:
                    axes.step(np.append(TrueMod[j+1][:], TrueMod[j+1][-1]),np.append(np.append(0, TrueMod[0][:]), maxDepth),where='pre',color='k')
                axes.invert_yaxis()
                axes.set_ylim(bottom=maxDepth,top=0.0)
                axes.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][j+1]),fontsize=14)
                axes.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][0]),fontsize=14)
        if nbParamUnique > 1:
            for ax in axes.flat:
                ax.label_outer()

        # fig.suptitle("Prior model visualization",fontsize=16)
        pyplot.show(block=False)

    def ShowPriorDataset(self):
        '''SHOWPRIORDATASET is a function that displays the ensemble of datasets modelled from
        sampled prior models.
        '''
        sortIndex = np.arange(self.nbModels)
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        for j in sortIndex:
            ax.plot(self.MODPARAM.forwardFun["Axis"],np.squeeze(self.FORWARD[j,:]),color='gray')
            ax.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["DataAxis"]),fontsize=14)
            ax.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["DataName"]),fontsize=14)
        pyplot.show(block=False)
        
class POSTBEL:
    """Object that is used to store the POSTBEL elements:
    
    For a given PREBEL set (see PREBEL class), the POSTBEL class
    enables all the computations that takes place aftre the  
    field data acquisition. It takes as argument:
        - PREBEL (PREBEL class object): the PREBEL object from 
                                        PREBEL class

    The class object has multiple attributes. Some of those are 
        - nbModels (int): Number of models in the prior (from the 
                          PREBEL object)
        - nbSamples (int): Number of models in the posterior (setup
                           in the *run* method). Default value of 
                           1000.
        - FORWARD (np.ndarray): Forward model for the prior models.
                                Originating from the PREBEL object.
        - MODELS (np.ndarray): Models sampled in the prior. Originating
                               from the PREBEL object.
        - KDE (object): Kernel Density Estimation descriptor.
                            Originating from the PREBEL object.
        - PCA (dict): a dictionnary containing the PCA reduction with 
                      their mathematical descriptions:
            - 'Data': a sklearn.decompostion.PCA object with the PCA
                      decomposition for the data dimensions.
            - 'Models': a sklearn.decompostion.PCA object with the PCA
                        decomposition for the models dimensions. If no
                        PCA reduction is applied to the model space, 
                        the value is *None*
        - CCA (object): a sklearn.cross_decomposition.CCA object 
                        containing the CCA decomposition.
        - MODPARAM (MODELSET): the full MODELSET object (see class 
                               description).
        - DATA (dict): dictionary containing the field dataset.
            - 'True': The dataset in the original dimension.
            - 'PCA': The dataset in the PCA-reduced space.
            - 'CCA': The dataset in the CCA-projected space.
        - SAMPLES (np.ndarray): The models sampled from the posterior
                                model space.
        - SAMPLESDATA (np.ndarray): The datasets corresponding to the 
                                    sampled models.
    """
    def __init__(self,PREBEL:PREBEL):
        self.nbModels = PREBEL.nbModels
        self.nbSamples = 1000 # Default value for the parameter
        self.FORWARD = PREBEL.FORWARD # Forward from the prior
        self.MODELS = PREBEL.MODELS
        self.KDE = PREBEL.KDE
        self.PCA = PREBEL.PCA
        self.CCA = PREBEL.CCA
        self.MODPARAM = PREBEL.MODPARAM
        self.DATA = dict()
        self.SAMPLES = []
        self.SAMPLESDATA = []

    def run(self, Dataset, nbSamples:int=1000, Graphs:bool=False, NoiseModel:list=None, verbose:bool=False):
        '''RUN is a method that runs POSTBEL operations for a given dataset.

        It takes as argument:
            - Dataset (np.array): the field dataset
        
        Optional arguments are:
            - nbSamples (int): the number of posterior models to sample
                               (defalut=1000)
            - Graphs (bool): show KDE graphs (True) or not (False)
                             (default=False)
            - NoiseModel (list): the list defining the noise model 
                                 (see dedicated functions)
                                 (default=None)
            - verbose (bool): output progresses messages (True) or
                              not (False - default)
        '''
        self.nbSamples = nbSamples
        # Transform dataset to CCA space:
        if verbose:
            print('Projecting the dataset into the CCA space . . .')
        Dataset = np.reshape(Dataset,(1,-1))# Convert for reverse transform
        d_obs_h = self.PCA['Data'].transform(Dataset)
        d_obs_c = self.CCA.transform(d_obs_h)
        self.DATA = {'True':Dataset,'PCA':d_obs_h,'CCA':d_obs_c}
        # Propagate Noise:
        if NoiseModel is not None:
            if verbose:
                print("Propagating the noise model . . .")
            Noise = np.sqrt(Tools.PropagateNoise(self,NoiseModel))
        else:
            Noise = None
        # Obtain corresponding distribution (KDE)
        if verbose:
            print('Obtaining the distribution in the CCA space . . .')
        if (self.KDE.Dist[0] is None):
            self.KDE.GetDist(Xvals=d_obs_c,Noise=Noise)
        if Graphs:
            self.KDE.ShowKDE(Xvals=d_obs_c)
        # Sample models:
        if verbose:
            print('Sampling models and back-transformation . . .')
        if self.MODPARAM.cond is None:
            samples_CCA = self.KDE.SampleKDE(nbSample=nbSamples)
            # Back transform models to original space:
            samples_PCA = np.matmul(samples_CCA,self.CCA.y_loadings_.T)
            samples_PCA *= self.CCA.y_std_
            samples_PCA += self.CCA.y_mean_
            # samples_PCA = self.CCA.inverse_transform(samples_CCA)
            if self.PCA['Model'] is None:
                samples_Init = samples_PCA 
            else:
                samples_Init = self.PCA['Model'].inverse_transform(samples_PCA)
            if type(self.MODPARAM.logTransform[0]) is list:
                if len(self.MODPARAM.logTransform[0]) != samples_Init.shape[1]:
                    raise Exception('The length of the log-transform is not the same as the length of the prior.')
                for count, logT in enumerate(self.MODPARAM.logTransform[0]):
                    if logT:
                        samples_Init[:,count] = np.exp(samples_Init[:,count])
            elif (type(self.MODPARAM.logTransform[0]) is bool) and self.MODPARAM.logTransform[0]:
                samples_Init = np.exp(samples_Init)
            self.SAMPLES = samples_Init
        else: # They are conditions to respect!
            nbParam = len(self.MODPARAM.prior)
            Samples = np.zeros((nbSamples,nbParam))
            achieved = False
            modelsOK = 0
            nbTestsMax = nbSamples*10 # At max, we could be at 0.1 sample every loop.
            while not(achieved):
                samples_CCA = self.KDE.SampleKDE(nbSample=(nbSamples-modelsOK))
                # Back transform models to original space:
                samples_PCA = np.matmul(samples_CCA,self.CCA.y_loadings_.T)
                samples_PCA *= self.CCA.y_std_
                samples_PCA += self.CCA.y_mean_
                # samples_PCA = self.CCA.inverse_transform(samples_CCA)
                if self.PCA['Model'] is None:
                    Samples[modelsOK:,:] = samples_PCA 
                else:
                    Samples[modelsOK:,:] = self.PCA['Model'].inverse_transform(samples_PCA)
                if type(self.MODPARAM.logTransform[0]) is list:
                    if len(self.MODPARAM.logTransform[0]) != Samples.shape[1]:
                        raise Exception('The length of the log-transform is not the same as the length of the prior.')
                    for count, logT in enumerate(self.MODPARAM.logTransform[0]):
                        if logT:
                            Samples[modelsOK:,count] = np.exp(Samples[modelsOK:,count])
                elif (type(self.MODPARAM.logTransform[0]) is bool) and self.MODPARAM.logTransform[0]:
                    Samples[modelsOK:,:] = np.exp(Samples[modelsOK:,:]) 
                keep = np.ones((nbSamples,))
                for i in range(nbSamples-modelsOK):
                    keep[modelsOK+i] = self.MODPARAM.cond(Samples[modelsOK+i,:])
                indexKeep = np.where(keep)
                modelsOK = np.shape(indexKeep)[1]
                tmp = np.zeros((nbSamples,nbParam))
                tmp[range(modelsOK),:] = np.squeeze(Samples[indexKeep,:])
                Samples = tmp
                if modelsOK == nbSamples:
                    achieved = True
                nbTestsMax -= 1
                if nbTestsMax < 0:
                    raise Exception('Impossible to sample models in the current posterior under reasonable timings!')
            if verbose:
                print('{} models sampled from the posterior model space!'.format(nbSamples))
            self.SAMPLES = Samples
            self.SAMPLESDATA = []
    
    def runMCMC(self, NoiseModel=None, nbSamples:int=20000, nbChains=10, verbose:bool=False):
        ''' RUNMCMC is a class method that runs a simple metropolis McMC algorithm
        on the last posterior model space (POSTBEL). 

        It takes as arguments:
            - NoiseModel (np.array): the list defining the noise model
            - nbSamples (int): the number of models to sample per chains (larger for larger 
                               priors). The default value is 20000
            - nbChains (int): the number of chains to run. The larger, the better to avoid 
                              remaining in a local optimum. The default value is 10.
            - verbose (bool): output progresses messages (True) or not (False - default)

        It returns 2 arrays containing the samples models and the associated datasets.
        '''
        if NoiseModel is None:
            raise Exception('No noise model provided. Impossible to compute the likelihood')
        if len(NoiseModel) != len(self.DATA['True'][0,:]):
            raise Exception('NoiseModel should have the same size as the dataset')
        timeIn = time.time() # For the timer
        nbParam = len(self.MODPARAM.prior)
        accepted = np.zeros((nbChains, nbSamples, nbParam))
        acceptedData = np.zeros((nbChains, nbSamples, len(self.DATA['True'][0,:])))
        for j in range(nbChains):
            if verbose:
                print('Running chain {} out of {}. . .'.format(j, nbChains))
            rejectedNb = 0
            i = 0
            LikelihoodLast = 1e-50 # First sample most likely accepted
            Covariance = 0.01*np.cov(self.SAMPLES.T) # Compute the initial covariance from the posterior distribution
            passed = False
            passedVerbose = False
            while i < nbSamples:
                if i == 0:
                    ## Sampling a random model from the posterior distribution
                    samples_CCA = self.KDE.SampleKDE(nbSample=1)
                    # Back transform models to original space:
                    samples_PCA = np.matmul(samples_CCA,self.CCA.y_loadings_.T)
                    samples_PCA *= self.CCA.y_std_
                    samples_PCA += self.CCA.y_mean_
                    if self.PCA['Model'] is None:
                        sampleCurr = samples_PCA 
                    else:
                        sampleCurr = self.PCA['Model'].inverse_transform(samples_PCA)
                else:
                    ## Random change to the sampled model:
                    sampleCurr = sampleLast + np.random.multivariate_normal(np.zeros((len(self.MODPARAM.prior),)),Covariance)# np.random.uniform(0,0.01)*np.random.randn()*sampleAdd
                ## Computing the likelihood from a data misfit:
                if self.MODPARAM.cond(sampleCurr[0,:]):
                    try:
                        SynData = self.MODPARAM.forwardFun['Fun'](sampleCurr[0,:])
                        DataDiff = self.DATA['True'] - SynData
                        FieldError = NoiseModel
                        A = np.divide(1,np.sqrt(2*np.pi*np.power(FieldError,2)))
                        B = np.exp(-1/2 * np.power(np.divide(DataDiff,FieldError),2))
                        Likelihood = np.prod(np.multiply(A, B))
                        if Likelihood > 1e305:
                            Likelihood = 1e305
                    except:
                        rejectedNb += 1
                        continue
                else:
                    rejectedNb += 1
                    continue
                ## Sampling (or not) the model:
                ratio = Likelihood/LikelihoodLast
                if ratio > np.random.uniform(0,1):
                    sampleLast = sampleCurr
                    accepted[j,i,:] = sampleCurr[0,:]
                    acceptedData[j,i,:] = SynData
                    i += 1
                    passed = False
                    passedVerbose = False
                else:
                    rejectedNb += 1
                if np.mod(i,int(nbSamples/100)) == 0 and not(passedVerbose):
                    if verbose:
                        print('{} models sampled (out of {}) in chain {}.'.format(i, nbSamples, j))
                    passedVerbose = True
                if (i < nbSamples/2) and (np.mod(i,50) == 0) and not(passed):
                    AcceptanceRatio = i/(rejectedNb+i)
                    if AcceptanceRatio < 0.80: # 0.75
                        if verbose:
                            print('Acceptance ratio too low, reducing covariance.')
                        Covariance *= 0.8 # We are reducing the covariance to increase the acceptance rate
                    elif AcceptanceRatio > 0.90: # 0.85
                        if verbose:
                            print('Acceptance ratio too high, increasing covariance.')
                        Covariance *= 1.2 # We are increasing the covariance to decrease the acceptance rate
                    passed = True
                LikelihoodLast = Likelihood
        if verbose:
            print(f'MCMC on POSTBEL executed in {time.time()-timeIn} seconds.')
        return np.asarray(accepted), np.asarray(acceptedData)

    def DataPost(self, Parallelization=[False,None], OtherModels=None, verbose:bool=False):
        '''DATAPOST is a function that computes the forward model for all the 
        models sampled from the posterior.

        The optional arguments are:
            - Parallelization (list): parallelization instructions
                    - [False, _]: no parallel runs (default)
                    - [True, None]: parallel runs without pool provided
                    - [True, pool]: parallel runs with pool (defined by pathos.pools) 
                                    provided 
            - OtherModels (np.ndarray): a numpy array containing models that have the
                                        same formatting as the one originating from the 
                                        POSTBEL method. The methdo returns the forward 
                                        models for those instead of the POSTBEL sampled.
            - verbose (bool): output progresses messages (True) or not (False - default)
        
        The function returns the models and the corresponding computed dataset. 

        WARNING: Some models might be removed from the forward modelling. They 
        correspond to models that are not computable using the selected forward
        modelling function.
        '''
        tInit = time.time() # For the timer
        if OtherModels is not None:
            SAMPLES = OtherModels
            SAMPLESDATA = []
        else:
            SAMPLES = self.SAMPLES
            SAMPLESDATA = self.SAMPLESDATA
        nbSamples = np.size(SAMPLES,axis=0)
        if len(SAMPLESDATA)!=0:# The dataset is already simulated
            if verbose:
                print('Forward modelling already conducted!')
            return SAMPLESDATA
        if verbose:
            print('Computing the forward model . . .')
        indexCurr = 0
        while True:
            try:
                tmp = self.MODPARAM.forwardFun["Fun"](SAMPLES[indexCurr,:])
                break
            except:
                indexCurr += 1
                if indexCurr > nbSamples:
                    raise Exception('The forward modelling failed!')
        SAMPLESDATA = np.zeros((nbSamples,len(tmp)))
        if Parallelization[0]:
            # We create a partial function that has a fixed fowrard function. The remaining arguments are :
            #   - Model: a numpy array containing the model to compute
            # It returns the Forward Computed, either a list of None or a list of values corresponding to the forward
            functionParallel = partial(ForwardParallelFun, function=self.MODPARAM.forwardFun["Fun"], nbVal=len(tmp))
            inputs = [SAMPLES[i,:] for i in range(nbSamples)]
            if Parallelization[1] is not None:
                pool = Parallelization[1]
                terminatePool = False
                # pool.restart()
            else:
                pool = pp.ProcessPool(mp.cpu_count()) # Create the pool for paralelization
                terminatePool = True
            outputs = pool.map(functionParallel,inputs)
            SAMPLESDATA = np.vstack(outputs) #ForwardParallel
            notComputed = [i for i in range(nbSamples) if SAMPLESDATA[i,0] is None]
            SAMPLES = np.array(np.delete(SAMPLES,notComputed,0),dtype=np.float64)
            SAMPLESDATA = np.array(np.delete(SAMPLESDATA,notComputed,0),dtype=np.float64)
            newSamplesNb = np.size(SAMPLES,axis=0) # Get the number of models remaining
            if terminatePool:
                pool.terminate()
        else:
            notComputed = []
            for i in range(nbSamples):
                # print(i)
                try:
                    SAMPLESDATA[i,:] = self.MODPARAM.forwardFun["Fun"](SAMPLES[i,:])
                except:
                    SAMPLESDATA[i,:] = [None]*len(tmp)
                    notComputed.append(i)
            # Getting the uncomputed models and removing them:
            SAMPLES = np.delete(SAMPLES,notComputed,0)
            SAMPLESDATA = np.delete(SAMPLESDATA,notComputed,0)
            newSamplesNb = np.size(SAMPLES,axis=0) # Get the number of models remaining
        if self.MODPARAM.method == "DC":
            # In the case of surface waves, the forward model sometimes provide datasets that have a sharp
            # transition that is not possible in practice. We therefore need to remove those models. They 
            # are luckily easy to identify. Their maximum variabilty is way larger than the other models.
            VariabilityMax = np.max(np.abs(SAMPLESDATA[:,1:]-SAMPLESDATA[:,:-1]),axis=1)
            from scipy.special import erfcinv # https://github.com/PyCQA/pylint/issues/3744 pylint: disable=no-name-in-module
            c = -1/(mt.sqrt(2)*erfcinv(3/2))
            VariabilityMaxAuthorized = np.median(VariabilityMax) + 3 * c*np.median(np.abs(VariabilityMax-np.median(VariabilityMax)))
            isOutlier = np.greater(np.abs(VariabilityMax),VariabilityMaxAuthorized)
            SAMPLES = np.delete(SAMPLES,np.where(isOutlier),0)
            SAMPLESDATA = np.delete(SAMPLESDATA,np.where(isOutlier),0)
            newSamplesNb = np.size(SAMPLES,axis=0) # Get the number of models remaining
            pass
        if verbose:
            print('{} models remaining after forward modelling!\nThe forward modelling was done in {} seconds'.format(newSamplesNb, time.time()-tInit))
        if OtherModels is None:
            self.nbSamples = newSamplesNb
            self.SAMPLES = SAMPLES
            self.SAMPLESDATA = SAMPLESDATA
        return SAMPLES, SAMPLESDATA
    
    def runRejection(self, NoiseModel=None, Parallelization=[False,None], verbose:bool=False):
        '''RUNREJECTION is a method that uses models sampled from the posterior
        model space and applies a Metropolis sampler to them in order to reject
        the most unlikely models.

        The algorithm takes as inputs:
            - NoiseModel (np.array): an array containing the noise level for every 
                                     datapoints (same size as the dataset).
            - Parallelization (list): parallelization instructions
                    - [False, _]: no parallel runs (default)
                    - [True, None]: parallel runs without pool provided
                    - [True, pool]: parallel runs with pool (defined by pathos.pools) 
                                    provided
            - verbose (bool): output progresses messages (True) or not (False - default)
        
        The function will return two numpy arrays:
            - ModelsAccepted (np.ndarray): the ensemble of accepted models
            - DataAccepted (np.ndarray): the corresponding data.
        '''
        if NoiseModel is None:
            raise Exception('No noise model provided. Impossible to compute the likelihood')
        if len(NoiseModel) != len(self.DATA['True'][0,:]):
            raise Exception('NoiseModel should have the same size as the dataset')
        timeIn = time.time()
        self.DataPost(Parallelization=Parallelization, verbose=verbose)
        Likelihood = np.zeros(len(self.SAMPLESDATA),)
        if verbose:
            print('Computing likelyhood . . .')
        for i, SynData in enumerate(self.SAMPLESDATA):
            FieldError = NoiseModel
            DataDiff = self.DATA['True'] - SynData
            A = np.divide(1,np.sqrt(2*np.pi*np.power(FieldError,2)))
            B = np.exp(-1/2 * np.power(np.divide(DataDiff,FieldError),2))
            Likelihood[i] = np.prod(np.multiply(A, B))
        Order = random.permutation(len(Likelihood))
        LikelihoodOrder = Likelihood[Order]
        Accepted = [Order[0]]
        LikeLast = LikelihoodOrder[0]
        if verbose:
            print('Running Metropolis sampler . . .')
        nbRejected = 0
        for i, Like in enumerate(LikelihoodOrder[1:]):
            ratio = Like/LikeLast
            if ratio > np.random.uniform(0,1):
                Accepted.append(Order[i+1])
                LikeLast = Like
                nbRejected = 0
            else:
                nbRejected += 1
                if nbRejected > 20: # To avoid staying in the same area all the time
                    Accepted.append(Order[i+1])
                    LikeLast = Like
                    nbRejected = 0
        ModelsAccepted = self.SAMPLES[Accepted,:]
        DataAccepted = self.SAMPLESDATA[Accepted,:]
        if verbose:
            print(f'Rejection sampling ({len(ModelsAccepted)} out of {len(LikelihoodOrder)} kept) on POSTBEL executed in {time.time()-timeIn} seconds.')
        return ModelsAccepted, DataAccepted

    def ShowPost(self, prior:bool=False, priorOther=None, TrueModel=None):
        '''SHOWPOST shows the posterior parameter distributions (uncorrelated).

        The optional arguments are:
            - prior (bool): Show the prior model space (True) or not
                            (False - default).
            - priorOther (np.array): The initial prior. Optional.
            - TrueModel (np.array): an array containing the benchmark model.
        '''
        nbParam = self.SAMPLES.shape[1]
        nbLayers = self.MODPARAM.nbLayer
        nbParamUnique = int((nbParam+1)/nbLayers)
        fig, axes = pyplot.subplots(nbLayers, nbParamUnique)
        mask = [True]*nbParamUnique*nbLayers
        mask[nbLayers-1] = False
        idY = np.repeat(np.arange(nbParamUnique),nbLayers)
        idY = idY[mask]
        idX = np.tile(np.arange(nbLayers),nbParamUnique)
        idX = idX[mask]
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        for i in range(nbParam):
            ax = axes[idX[i],idY[i]]
            if i != nbParam-1:
                if prior:
                    if priorOther is None:
                        ax.hist(self.MODELS[:,i], density=True, alpha=0.5, label='_Prior')
                    else:
                        ax.hist(priorOther[:,i], density=True, alpha=0.5, label='_Prior')
                ax.hist(self.SAMPLES[:,i], density=True, alpha=0.5, label='_Posterior')
                if TrueModel is not None:
                    ax.plot([TrueModel[i], TrueModel[i]], np.asarray(ax.get_ylim()), 'k', label='_True')
            else:
                if priorOther is None:
                    ax.hist(self.MODELS[:,i], density=True, alpha=0.5, label='Prior')
                else:
                    ax.hist(priorOther[:,i], density=True, alpha=0.5, label='Prior')
                ax.hist(self.SAMPLES[:,i], density=True, alpha=0.5, label='Posterior')
                if TrueModel is not None:
                    ax.plot([TrueModel[i], TrueModel[i]], np.asarray(ax.get_ylim()), 'k', label='True')
            ax.set_xlabel(self.MODPARAM.paramNames["NamesFU"][i])
        axLeg = axes[nbLayers-1, 0]
        axLeg.set_visible(False)
        pyplot.tight_layout()
        fig.legend(loc='lower left')
        pyplot.show(block=False)
    
    def ShowPostCorr(self,TrueModel=None, OtherMethod=None, OtherInFront=False, alpha=[1, 1], OtherModels=None):
        '''SHOWPOSTCORR shows the posterior parameter distributions (correlated).

        The optional arguments are:
            - TrueModel (np.array): an array containing the benchmark model
            - OtherMethod (np.array): an array containing an ensemble of models
            - OtherInFront (bool): Show the other in front (True) or at the back (False)
            - alpha (list or int): Transparancy value for the points. Default is 1 for
                                   both BEL1D and OtherMethod. ([BEL1D, OtherMethod])
            - OtherModels (np.array): an array that replaces the resultst from BEL1D in 
                                      the graph, for example, a rejection run after BEL1D
        '''
        # Adding the graph with correlations: 
        nbParam = self.SAMPLES.shape[1]
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        if (OtherMethod is not None) and (OtherMethod.shape[1]!=nbParam):
            print('OtherMethod is not a valid argument! Argument ignored . . .')
            OtherMethod = None
        if (OtherModels is not None) and (OtherModels.shape[1]!=nbParam):
            print('OtherModels is not a valid argument! Argument ignored . . .')
            OtherModels = None
        elif not(isinstance(alpha, list)):
            alpha = [alpha, alpha]
        fig = pyplot.figure(figsize=[10,10])# Creates the figure space
        axs = fig.subplots(nbParam, nbParam)
        namesVar =  "NamesSU"
        if nbParam > 8:
            namesVar =  "NamesS"
        for i in range(nbParam):
            for j in range(nbParam):
                if i == j: # Diagonal
                    if i != nbParam-1:
                        axs[i,j].get_shared_x_axes().join(axs[i,j],axs[-1,j])# Set the xaxis limit
                    if OtherInFront:
                        if OtherModels is not None:
                            axs[i,j].hist(OtherModels[:,j],color='b',density=True) # Plot the histogram for the given variable
                        else:
                            axs[i,j].hist(self.SAMPLES[:,j],color='b',density=True) # Plot the histogram for the given variable
                        if OtherMethod is not None:
                            axs[i,j].hist(OtherMethod[:,j],color='y',density=True)
                    else:
                        if OtherMethod is not None:
                            axs[i,j].hist(OtherMethod[:,j],color='y',density=True)
                        if OtherModels is not None:
                            axs[i,j].hist(OtherModels[:,j],color='b',density=True) # Plot the histogram for the given variable
                        else:
                            axs[i,j].hist(self.SAMPLES[:,j],color='b',density=True) # Plot the histogram for the given variable
                    if TrueModel is not None:
                        axs[i,j].plot([TrueModel[i],TrueModel[i]],np.asarray(axs[i,j].get_ylim()),'r')
                    if nbParam > 8:
                        axs[i,j].set_xticks([])
                        axs[i,j].set_yticks([])
                elif i > j: # Below the diagonal -> Scatter plot
                    if i != nbParam-1:
                        axs[i,j].get_shared_x_axes().join(axs[i,j],axs[-1,j])# Set the xaxis limit
                    if j != nbParam-1:
                        if i != nbParam-1:
                            axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-1])# Set the yaxis limit
                        else:
                            axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-2])# Set the yaxis limit
                    if OtherModels is not None:
                        axs[i,j].plot(OtherModels[:,j],OtherModels[:,i],'.b',alpha=alpha[0], markeredgecolor='none')
                    else:
                        axs[i,j].plot(self.SAMPLES[:,j],self.SAMPLES[:,i],'.b',alpha=alpha[0], markeredgecolor='none')
                    if TrueModel is not None:
                        axs[i,j].plot(TrueModel[j],TrueModel[i],'or')
                    if nbParam > 8:
                        axs[i,j].set_xticks([])
                        axs[i,j].set_yticks([])
                elif OtherMethod is not None:
                    if i != nbParam-1:
                        axs[i,j].get_shared_x_axes().join(axs[i,j],axs[-1,j])# Set the xaxis limit
                    if j != nbParam-1:
                        if i != 0:
                            axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-1])# Set the yaxis limit
                        else:
                            axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-2])# Set the yaxis limit
                    axs[i,j].plot(OtherMethod[:,j],OtherMethod[:,i],'.y',alpha=alpha[1], markeredgecolor='none')
                    if TrueModel is not None:
                        axs[i,j].plot(TrueModel[j],TrueModel[i],'or')
                    if nbParam > 8:
                        axs[i,j].set_xticks([])
                        axs[i,j].set_yticks([])
                else:
                    axs[i,j].set_visible(False)
                if j == 0: # First column of the graph
                    if ((i==0)and(j==0)) or not(i==j):
                        axs[i,j].set_ylabel(r'${}$'.format(self.MODPARAM.paramNames[namesVar][i]))
                if i == nbParam-1: # Last line of the graph
                    axs[i,j].set_xlabel(r'${}$'.format(self.MODPARAM.paramNames[namesVar][j]))
                if j == nbParam-1:
                    if not(i==j):
                        axs[i,j].yaxis.set_label_position("right")
                        axs[i,j].yaxis.tick_right()
                        axs[i,j].set_ylabel(r'${}$'.format(self.MODPARAM.paramNames[namesVar][i]))
                if i == 0:
                    axs[i,j].xaxis.set_label_position("top")
                    axs[i,j].xaxis.tick_top()
                    axs[i,j].set_xlabel(r'${}$'.format(self.MODPARAM.paramNames[namesVar][j]))
        # fig.suptitle("Posterior model space visualization")
        for ax in axs.flat:
            ax.label_outer()
        pyplot.tight_layout()
        pyplot.show(block=False)
    
    def ShowPostModels(self,TrueModel=None, RMSE:bool=False, Best:int=None, Parallelization=[False,None], NoiseModel=None, OtherModels=None, OtherData=None, OtherRMSE=False):
        '''SHOWPOSTMODELS shows the sampled posterior models.

        The optional argument are:
            - TrueModel (np.array): an array containing the benchmark model.
            - RMSE (bool):  show the RMSE (True) or not (False)
                            (default=False)
            - Best (int): only show the X best models (X is the argument)
            - Parallelization (list): parallelization instructions
                    - [False, _]: no parallel runs (default)
                    - [True, None]: parallel runs without pool provided
                    - [True, pool]: parallel runs with pool (defined bypathos.pools) 
                                    provided
            - NoiseModel (np.ndarray): an array containing the estimated noise for 
                                       every datapoints. If provided, we are using
                                       a wheigted RMSE (chi2) instead of RMSE.
            - OtherModels (np.ndarray): an array containing an other set of models
            - OtherData (np.ndarray): an array containing the simulated data for the
                                      other set of models
            - OtherRMSE (bool): use the Postbel RMSE (False) or the OtherModels RMSE
                                (True). Default is False.
        '''
        from matplotlib import colors
        nbParam = self.SAMPLES.shape[1]
        nbLayer = self.MODPARAM.nbLayer
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        if RMSE and len(self.SAMPLESDATA)==0:
            print('Computing the forward model for the posterior!')
            self.DataPost(Parallelization=Parallelization)
        if OtherModels is not None:
            if (RMSE is True) and (OtherData is None):
                raise Exception('No data provided for the other set of models')
        else:
            OtherData = None
        if RMSE:
            TrueData = self.DATA['True']
            if NoiseModel is None:
                NoiseEstimation = np.ones(TrueData.shape)
            else:
                NoiseEstimation = NoiseModel
            if OtherData is not None:
                RMS = np.sqrt(np.square(np.divide(np.subtract(TrueData,OtherData), NoiseEstimation)).mean(axis=-1))
                if OtherRMSE:
                    RMS_scale = RMS
                else:
                    RMS_scale = np.sqrt(np.square(np.divide(np.subtract(TrueData,self.SAMPLESDATA), NoiseEstimation)).mean(axis=-1))
            else:
                RMS = np.sqrt(np.square(np.divide(np.subtract(TrueData,self.SAMPLESDATA), NoiseEstimation)).mean(axis=-1))
                RMS_scale = RMS
            quantiles = np.divide([stats.percentileofscore(RMS_scale,a,'strict') for a in RMS],100)
            sortIndex = np.argsort(RMS)
            sortIndex = np.flip(sortIndex)
        else:
            sortIndex = np.arange(self.nbSamples)
        if Best is not None:
            Best = int(Best)
            sortIndex = sortIndex[-Best:]
        if nbLayer is not None:# If the model can be displayed as layers
            if self.MODPARAM.thicknessesFixed is None:
                nbParamUnique = int(np.ceil(nbParam/nbLayer))-1 # Number of parameters minus the thickness
            else:
                nbParamUnique = int(np.ceil(nbParam/nbLayer))
            fig = pyplot.figure(figsize=[4*nbParamUnique,10])
            Param = list()
            if OtherModels is not None:
                ModelsPlot = OtherModels
            else:
                ModelsPlot = self.SAMPLES
            if self.MODPARAM.thicknessesFixed is None:
                Param.append(np.cumsum(ModelsPlot[:,0:nbLayer-1],axis=1))
                for i in range(nbParamUnique):
                    Param.append(ModelsPlot[:,(i+1)*nbLayer-1:(i+2)*nbLayer-1])
            else:
                Param.append(np.repeat(np.atleast_2d(np.cumsum(self.MODPARAM.thicknessesFixed[:-1])), ModelsPlot.shape[0], axis=0))
                for i in range(nbParamUnique):
                    Param.append(ModelsPlot[:,(i)*nbLayer:(i+1)*nbLayer])
            if TrueModel is not None:
                TrueMod = list()
                if self.MODPARAM.thicknessesFixed is None:
                    TrueMod.append(np.cumsum(TrueModel[0:nbLayer-1]))
                else:
                    TrueMod.append(np.cumsum(self.MODPARAM.thicknessesFixed[:-1]))
                for i in range(nbParamUnique):
                    if self.MODPARAM.thicknessesFixed is None:
                        TrueMod.append(TrueModel[(i+1)*nbLayer-1:(i+2)*nbLayer-1])
                    else:
                        TrueMod.append(TrueModel[(i)*nbLayer:(i+1)*nbLayer])
                
            maxDepth = np.max(Param[0][:,-1])*1.5
            if RMSE:
                colormap = matplotlib.cm.get_cmap('viridis')
                # Set the space for the colorbar below:
                gridSpec = fig.add_gridspec(9,nbParamUnique) # One graph per parameter
                axes = []
                for k in range(nbParamUnique):
                    currAx = fig.add_subplot(gridSpec[:-1, k])
                    axes.append(currAx)
                axesCbar = fig.add_subplot(gridSpec[-1,:])
                for j in range(nbParamUnique):
                    for i in sortIndex:
                        axes[j].step(np.append(Param[j+1][i,:], Param[j+1][i,-1]),np.append(np.append(0, Param[0][i,:]), maxDepth),where='pre',color=colormap(quantiles[i]))
                    if TrueModel is not None:
                        axes[j].step(np.append(TrueMod[j+1][:], TrueMod[j+1][-1]),np.append(np.append(0, TrueMod[0][:]), maxDepth),where='pre',color='gray')
                    axes[j].invert_yaxis()
                    axes[j].set_ylim(bottom=maxDepth,top=0.0)
                    axes[j].set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][j+1]),fontsize=14)
                    axes[j].set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][0]),fontsize=14)
            else:
                axes = fig.subplots(1,nbParamUnique) # One graph per parameter
                if nbParamUnique > 1:
                    for j in range(nbParamUnique):
                        for i in sortIndex:
                            axes[j].step(np.append(Param[j+1][i,:], Param[j+1][i,-1]),np.append(np.append(0, Param[0][i,:]), maxDepth),where='pre',color='gray')
                        if TrueModel is not None:
                            axes[j].step(np.append(TrueMod[j+1][:], TrueMod[j+1][-1]),np.append(np.append(0, TrueMod[0][:]), maxDepth),where='pre',color='k')
                        axes[j].invert_yaxis()
                        axes[j].set_ylim(bottom=maxDepth,top=0.0)
                        axes[j].set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][j+1]),fontsize=14)
                        if j < 1 : # the first parameter
                            axes[j].set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][0]),fontsize=14)
                else:
                    j = 0 # Unique parameter
                    for i in sortIndex:
                        axes.step(np.append(Param[j+1][i,:], Param[j+1][i,-1]),np.append(np.append(0, Param[0][i,:]), maxDepth),where='pre',color='gray')
                    if TrueModel is not None:
                        axes.step(np.append(TrueMod[j+1][:], TrueMod[j+1][-1]),np.append(np.append(0, TrueMod[0][:]), maxDepth),where='pre',color='k')
                    axes.invert_yaxis()
                    axes.set_ylim(bottom=maxDepth,top=0.0)
                    axes.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][j+1]),fontsize=14)
                    axes.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][0]),fontsize=14)
                    fig.subplots_adjust(left=0.2)
        # if nbParamUnique > 1:
        #     for ax in axes: # .flat:
        #         ax.label_outer()
        
        if RMSE:
            # fig.subplots_adjust(bottom=0.25)
            ax_colorbar = axesCbar # fig.add_axes([0.15, 0.10, 0.70, 0.05])
            nb_inter = 1000
            color_for_scale = colormap(np.linspace(0,1,nb_inter,endpoint=True))
            cmap_scale = colors.ListedColormap(color_for_scale)
            scale = [stats.scoreatpercentile(RMS_scale,a,limit=(np.min(RMS_scale),np.max(RMS_scale)),interpolation_method='lower') for a in np.linspace(0,100,nb_inter,endpoint=True)]
            norm = colors.BoundaryNorm(scale,len(color_for_scale))
            data = np.atleast_2d(np.linspace(np.min(RMS_scale),np.max(RMS_scale),nb_inter,endpoint=True))
            ax_colorbar.imshow(data, aspect='auto',cmap=cmap_scale,norm=norm)
            if NoiseModel is None:
                ax_colorbar.set_xlabel('Root Mean Square Error {}'.format(self.MODPARAM.paramNames["DataUnits"]),fontsize=12)
            else:
                ax_colorbar.set_xlabel('Noise Weighted Root Mean Square Error [/]',fontsize=12)
            ax_colorbar.yaxis.set_visible(False)
            nbTicks = 5
            ax_colorbar.set_xticks(ticks=np.linspace(0,nb_inter,nbTicks,endpoint=True))
            ax_colorbar.set_xticklabels(labels=round_to_n([stats.scoreatpercentile(RMS_scale,a,limit=(np.min(RMS_scale),np.max(RMS_scale)),interpolation_method='lower') for a in np.linspace(0,100,nbTicks,endpoint=True)],n=2),rotation=30,ha='right')


        # fig.suptitle("Posterior model visualization",fontsize=16)
        # pyplot.tight_layout()
        pyplot.show(block=False)
    
    def ShowDataset(self,RMSE:bool=False,Prior:bool=False,Best:int=None,Parallelization=[False, None], OtherData=None):
        '''SHOWPOSTMODELS shows the sampled posterior models.

        The optional argument are:
            - RMSE (bool):  show the RMSE (True) or not (False)
                            (default=False)
            - Prior (bool): show the sampled prior datasets below (True) or not (False)
                            (default=False)
            - Best (int): only show the X best models (X is the argument)
            - Parallelization (list): parallelization instructions
                    - [False, _]: no parallel runs (default)
                    - [True, None]: parallel runs without pool provided
                    - [True, pool]: parallel runs with pool (defined bypathos.pools) 
                                    provided
            - OtherData (np.ndarray): an array containing the simulated data for the
                                      other set of models
        '''
        from matplotlib import colors
        # Model the dataset (if not already done)
        if OtherData is None:
            if len(self.SAMPLESDATA)==0:
                print('Computing the forward model for the posterior!')
                self.DataPost(Parallelization=Parallelization)
        if RMSE:
            TrueData = self.DATA['True']
            if OtherData is not None:
                RMS = np.sqrt(np.square(np.subtract(TrueData,OtherData)).mean(axis=-1))
                RMS_scale = np.sqrt(np.square(np.subtract(TrueData,self.SAMPLESDATA)).mean(axis=-1))
            else:
                RMS = np.sqrt(np.square(np.subtract(TrueData,self.SAMPLESDATA)).mean(axis=-1))
                RMS_scale = RMS
            quantiles = np.divide([stats.percentileofscore(RMS_scale,a,'strict') for a in RMS],100)
            sortIndex = np.argsort(RMS)
            sortIndex = np.flip(sortIndex)
        else:
            sortIndex = np.arange(self.nbSamples)
        if Best is not None:
            Best = int(Best)
            sortIndex = sortIndex[-Best:]# Select then best models
        fig = pyplot.figure()
        if RMSE:
            gs = fig.add_gridspec(5,1)
            ax = fig.add_subplot(gs[:,:]) # [:-1,:])
        else:
            ax = fig.add_subplot(111)
        if Prior:
            for j in range(self.nbModels):
                ax.plot(self.MODPARAM.forwardFun["Axis"],np.squeeze(self.FORWARD[j,:len(self.MODPARAM.forwardFun["Axis"])]),color='gray')
        if OtherData is not None:
            PlotData = OtherData
        else:
            PlotData = self.SAMPLESDATA
        if RMSE:
            colormap = matplotlib.cm.get_cmap('viridis')
            for j in sortIndex:
                ax.plot(self.MODPARAM.forwardFun["Axis"],np.squeeze(PlotData[j,:len(self.MODPARAM.forwardFun["Axis"])]),color=colormap(quantiles[j]))
            ax.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["DataAxis"]),fontsize=14)
            ax.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["DataName"]),fontsize=14)
        else:
            for j in sortIndex:
                ax.plot(self.MODPARAM.forwardFun["Axis"],np.squeeze(PlotData[j,:len(self.MODPARAM.forwardFun["Axis"])]),color='gray')
            ax.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["DataAxis"]),fontsize=14)
            ax.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["DataName"]),fontsize=14)
        if RMSE:
            # fig.subplots_adjust(bottom=0.30)
            ax_colorbar = fig.add_subplot(gs[-1,:]) #fig.add_axes([0.10, 0.15, 0.80, 0.05])
            nb_inter = 1000
            color_for_scale = colormap(np.linspace(0,1,nb_inter,endpoint=True))
            cmap_scale = colors.ListedColormap(color_for_scale)
            scale = [stats.scoreatpercentile(RMS_scale,a,limit=(np.min(RMS_scale),np.max(RMS_scale)),interpolation_method='lower') for a in np.linspace(0,100,nb_inter,endpoint=True)]
            norm = colors.BoundaryNorm(scale,len(color_for_scale))
            data = np.atleast_2d(np.linspace(np.min(RMS_scale),np.max(RMS_scale),nb_inter,endpoint=True))
            ax_colorbar.imshow(data, aspect='auto',cmap=cmap_scale,norm=norm)
            ax_colorbar.set_xlabel('Root Mean Square Error {}'.format(self.MODPARAM.paramNames["DataUnits"]),fontsize=12)
            ax_colorbar.yaxis.set_visible(False)
            nbTicks = 5
            ax_colorbar.set_xticks(ticks=np.linspace(0,nb_inter,nbTicks,endpoint=True))
            ax_colorbar.set_xticklabels(labels=round_to_n([stats.scoreatpercentile(RMS_scale,a,limit=(np.min(RMS_scale),np.max(RMS_scale)),interpolation_method='lower') for a in np.linspace(0,100,nbTicks,endpoint=True)],n=5),rotation=15,ha='center')
        pyplot.tight_layout()
        pyplot.show(block=False)

    def GetStats(self):
        '''GETSTATS is a method that returns the means and standard deviations of the 
        parameters distributions.
        '''
        means = np.mean(self.SAMPLES,axis=0)
        stds = np.std(self.SAMPLES,axis=0)
        return means, stds
    
class StatsResults:
    def __init__(self, means=None, stds=None, timing=None, distance=None):
        self.means = means
        self.stds = stds
        self.timing = timing
        self.distance = distance
    def saveStats(self,Filename='Stats'):
        import dill
        file_write = open(Filename+'.stats','wb')
        dill.dump(self,file_write)
        file_write.close()

def loadStats(Filename):
    import dill
    file_read = open(Filename,'rb')
    Stats = dill.load(file_read)
    file_read.close()
    return Stats
    
# Saving/loading operations:
def SavePREBEL(CurrentPrebel:PREBEL, Filename='PREBEL_Saved'):
    '''SavePREBEL is a function that saves the current prebel class object.

    It requieres as input:
        - CurrentPrebel: a PREBEL class object
        - FileName: a string with the name of the file to save

    The function will create de file "Filename.prebel" in the current directory
     (or the directory stated in the Filename input)
    '''
    import dill
    file_write = open(Filename+'.prebel','wb')
    dill.dump(CurrentPrebel,file_write)
    file_write.close()

def LoadPREBEL(Filename='PREBEL_Saved.prebel'):
    '''LoadPREBEL is a function that loads the prebel class object stored in Filename.

    It requieres as input:
        - FileName: a string with the name of the saved file

    The function returns the loaded PREBEL object.
    '''
    import dill
    file_read = open(Filename,'rb')
    PREBEL = dill.load(file_read)
    file_read.close()
    return PREBEL

def SavePOSTBEL(CurrentPostbel:POSTBEL, Filename='PREBEL_Saved'):
    '''SavePOSTBEL is a function that saves the current postbel class object.

    It requieres as input:
        - CurrentPostbel: a POSTBEL class object
        - FileName: a string with the name of the file to save

    The function will create de file "Filename.postbel" in the current directory
     (or the directory stated in the Filename input)
    '''
    import dill
    file_write = open(Filename+'.postbel','wb')
    dill.dump(CurrentPostbel,file_write)
    file_write.close()

def LoadPOSTBEL(Filename='POSTBEL_Saved.prebel'):
    '''LoadPOSTBEL is a function that loads the postbel class object stored in Filename.

    It requieres as input:
        - FileName: a string with the name of the saved file

    The function returns the loaded POSTBEL object.
    '''
    import dill
    file_read = open(Filename,'rb')
    POSTBEL = dill.load(file_read)
    file_read.close()
    return POSTBEL

def SaveSamples(CurrentPostbel:POSTBEL, Data=False, Filename='Models_Sampled'):
    '''SaveSamples is a function that saves the sampled models from a POSTBEL class object.

    It requieres as input:
        - CurrentPostbel: a POSTBEL class object
        - Data: a boolean (False=not saved, True=saved)
        - FileName: a string with the name of the file to save

    The function will create de file "Filename.models" (and optionaly "Filename.datas") 
    in the current directory (or the directory stated in the Filename input). The files 
    are classical ascii files
    '''
    if len(CurrentPostbel.SAMPLES)==0:
        raise EnvironmentError('No samples in current POSTBEL object!')
    if Data:
        if len(CurrentPostbel.SAMPLESDATA)==0:
            print('Computing the forward model for the posterior!')
            CurrentPostbel.DataPost() # By default not parallelized
        np.savetxt(Filename+'.datas',CurrentPostbel.SAMPLESDATA,delimiter='\t')
    np.savetxt(Filename+'.models',CurrentPostbel.SAMPLES,delimiter='\t')

# Iterative prior resampling:
def defaultMixing(iter:int) -> float:
    return 1
    
def IPR(MODEL:MODELSET, Dataset=None, NoiseEstimate=None, Parallelization:list=[False, None],
        nbModelsBase:int=1000, nbModelsSample:int=None, stats:bool=False, saveIters:bool=False, 
        saveItersFolder:str="IPR_Results", nbIterMax:int=100, reduceModels:bool=False, Rejection:float=0.0, 
        Mixing:Callable[[int], float]=defaultMixing, Graphs:bool=False, TrueModel=None, 
        PostbelConvert:POSTBEL=None, verbose:bool=False, PriorSampled=None):
    '''IPR (Iterative prior resampling) is a function that will compute the posterior 
    with iterative prior resampling for a given model defined via a MODELSET class object.

    It takes as arguments:
        - MODEL (MODELSET): the MODELSET class object defining the different elements for 
                            the computation of the forward model and the definition of the
                            prior.
        - Dataset: The true field dataset to be fitted
        - NoiseEstimate: The estimated noise level
        - Parallelization (list=[False,None]): parallelization instructions
                - [False, _]: no parallel runs (default)
                - [True, None]: parallel runs without pool provided
                - [True, pool]: parallel runs with pool (defined by pathos.pools) 
                                provided
        - nbModelsBase (int=1000): the number of models to sample in the initial prior
        - nbModelsSample (int=None): the number of models sampled to build the posterior. 
                                     If None, the number is equal to nbModelBase
        - stats (bool=False): return (True) or not (False) the statistics about the 
                              different iterations.
        - saveIters (bool=False): save (True) or not (False) the intermediate results in 
                                  the saveItersFolder directory
        - saveItersFolder (str="IPR_Results"): The directory where the files will be stored
        - nbIterMax (int=100): Maximum number of iterations
        - reduceModels (bool=False): apply PCA reduction to the models.
        - Rejection (float=0.9): Maximum quantile for the RMSE of the accepted models in the
                                 posterior
        - Mixing (callable): Function that returns the mixing ratio at a given iteration. The 
                             default value is 0.5 whatever the iteration.
        - Graphs (bool=False): Show diagnistic graphs (True) or not (False)
        - TrueModel (np.array): an array containing the benchmark model.
        - PostbelConvert (POSTBEL - optional): a POSTBEL class object that will be used as a 
                                               prior generator for the different 
        - verbose (bool): output progresses messages (True) or not (False - default).
    
    It returns possibly 4 elements:
        - Prebel (PREBEL): a PREBEL class object containing the last prior model space.
        - Postbel (POSTBEL): a POSTBEL class object containing the last posterior model space.
        - PrebelInit (PREBEL): a PREBEL class object containing the initial prior model space.
        - statsReturn (list - optional): a list containing the statistics at the different
                                         iterations. The statistics are contained in a 
                                         StatsResults class object. This argument is only
                                         outputted if the stats input is set to *True*.
    '''
    # Loading some utilities:
    import numpy as np
    from .utilities.Tools import nSamplesConverge
    from copy import deepcopy
    # Initializing:
    if verbose:
        print('Initializing the system . . .')
    if nbModelsSample is None:
        nbModelsSample = nbModelsBase
    if MODEL.method == 'DC':
        nbModelsBase = int(nbModelsBase*1.2) # To make sure to have enough models after failed modllings.
    if Dataset is None:
        raise Exception('No Dataset provided!')
    if verbose:
        print('Starting iterations . . .')
    start = time.time()
    Prebel = PREBEL(MODPARAM=MODEL, nbModels=nbModelsBase)
    if PriorSampled is not None:
        Prebel.run(Parallelization=Parallelization, RemoveOutlier=True, reduceModels=reduceModels, verbose=verbose, PriorSampled=PriorSampled)
    else:
        Prebel.run(Parallelization=Parallelization, reduceModels=reduceModels, verbose=verbose)
    PrebelInit = Prebel
    ModelLastIter = Prebel.MODELS
    statsNotReturn = True
    if Graphs:
        stats = True
    if stats:
        statsNotReturn = False
        statsReturn = []
    if saveIters:
        import os
        if not(os.path.isdir(saveItersFolder)):
            # Create the dierctory if it does not exist:
            os.mkdir(saveItersFolder)
        elif len(os.listdir(saveItersFolder)) != 0:
            print('The given directory will be overwritten!')
            input('Press any key to continue...')
        SavePREBEL(Prebel,saveItersFolder + '/IPR')
    for it in range(nbIterMax):
        if verbose:
            print('\n\n\nIteration {} running.\n\n'.format(it))
        # Iterating:
        if Mixing is not None:
            # nbModPrebel = Prebel.nbModels
            MixingUsed = Mixing(it)
            nbPostAdd = int((MixingUsed*nbModelsBase)/(1-Rejection)) #ModPrebel/(1-Rejection)) # We need to sample at least this number of models to be able to add to the prior with mixing satisfied
            nbSamples = max([int(nbModelsSample/(1-Rejection)),nbPostAdd])
            if Prebel.MODPARAM.method is 'DC':
                nbSamples = int(nbSamples*1.2) # Additional 20% to reach the number with failed models
        else:
            nbSamples = int(nbModelsSample/(1-Rejection))
            if Prebel.MODPARAM.method is 'DC':
                nbSamples = int(nbSamples*1.2) # Additional 20% to reach the number with failed models
            nbPostAdd = nbSamples
        Postbel = POSTBEL(Prebel)
        Postbel.run(Dataset=Dataset, nbSamples=nbSamples, NoiseModel=NoiseEstimate, verbose=verbose)
        end = time.time() # End of the iteration - begining of the preparation for the next iteration (if needed):
        if Graphs:
            if it == 0:
                NoiseToLastPrebel = Tools.PropagateNoise(Postbel, NoiseLevel=NoiseEstimate, verbose=verbose)
                Postbel.KDE.ShowKDE(Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))), Noise=NoiseToLastPrebel)
        Postbel.DataPost(Parallelization=Parallelization, verbose=verbose)
        # Testing for convergence (5% probability of false positive):
        if len(ModelLastIter) > nSamplesConverge:
            nbConvergeSamp = nSamplesConverge
        else:
            nbConvergeSamp = len(ModelLastIter)
        threshold = 1.87*nbConvergeSamp**(-0.50)# Power law defined from the different tests
        diverge, distance = Tools.ConvergeTest(SamplesA=ModelLastIter,SamplesB=Postbel.SAMPLES, tol=threshold)
        if verbose:
            print('KS distance at iter {}: {} (threshold at {}).'.format(it, distance, threshold))
        if stats:
            means, stds = Postbel.GetStats()
            statsReturn.append(StatsResults(means, stds, end-start, distance))
        if saveIters:
            SavePOSTBEL(Postbel,saveItersFolder + '/IPR_{}'.format(it))
        if not(diverge):
            if verbose:
                print('Model has converged at iter {}.'.format(it))
            if Graphs:
                NoiseToLastPrebel = Tools.PropagateNoise(Postbel, NoiseLevel=NoiseEstimate, verbose=verbose)
                Postbel.KDE.KernelDensity(NoiseError=NoiseToLastPrebel, RemoveOutlier=True, verbose=verbose)
                Postbel.KDE.ShowKDE(Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))), Noise=NoiseToLastPrebel)
            break
        ModelLastIter = Postbel.SAMPLES
        # If not converged yet --> apply transforms to the sampled set for mixing and rejection
        PostbelAdd = deepcopy(Postbel)
        if Rejection > 0:
            RMSE = np.sqrt(np.square(np.subtract(Dataset,PostbelAdd.SAMPLESDATA)).mean(axis=-1))
            RMSE_max = np.quantile(RMSE,1-Rejection) # We reject the x% worst fit
            idxDelete = np.greater_equal(RMSE,RMSE_max)
            PostbelAdd.SAMPLES = np.delete(PostbelAdd.SAMPLES,np.where(idxDelete),0)
            PostbelAdd.SAMPLESDATA = np.delete(PostbelAdd.SAMPLESDATA,np.where(idxDelete),0)
            PostbelAdd.nbSamples = np.size(PostbelAdd.SAMPLES,axis=0)
            nbPostAdd = int(nbPostAdd*(1-Rejection)) # We update the number of samples needed (for mixing)
        if Mixing is not None:
            # From there on, we need:
            #   - the Postbel object with nbModelsSample samples inside
            #   - the Postbel object with nbPostAdd samples inside
            # if nbPostAdd < nbModelsSample:
            #   PostbelAdd = sampled down Postbel
            # else:
            #   PostbelAdd = Postbel
            #   Postbel = sampled down postbel
            # Convergence on Postbel (with nbModelsSample models inside)
            import random
            if (nbPostAdd < nbModelsSample) and (PostbelAdd.nbSamples > nbPostAdd):
                idxKeep = random.sample(range(PostbelAdd.nbSamples), nbPostAdd)
                PostbelAdd.SAMPLES = PostbelAdd.SAMPLES[idxKeep,:]
                PostbelAdd.SAMPLESDATA = PostbelAdd.SAMPLESDATA[idxKeep,:]
                PostbelAdd.nbSamples = np.size(PostbelAdd.SAMPLES,axis=0)
            elif (nbModelsSample < nbPostAdd) and (PostbelAdd.nbSamples > nbModelsSample):
                idxKeep = random.sample(range(PostbelAdd.nbSamples), nbModelsSample)
                # ModelLastIter = PostbelAdd.SAMPLES[idxKeep,:]
                # PostbelAdd.SAMPLESDATA = PostbelAdd.SAMPLESDATA[idxKeep,:]
                # PostbelAdd.nbSamples = np.size(PostbelAdd.SAMPLES,axis=0)
                # # ModelLastIter = Postbel.SAMPLES[idxKeep,:]
        # Preparing next iteration:
        Prebel = PREBEL.POSTBEL2PREBEL(PREBEL=Prebel,POSTBEL=PostbelAdd,Dataset=Dataset,NoiseModel=NoiseEstimate,Parallelization=Parallelization, reduceModels=reduceModels, verbose=verbose)
    if Graphs:
        # plot the different graphs for the analysis of the results:
        pyplot.figure()
        pyplot.plot(range(len(statsReturn)),[statsReturn[i].timing for i in range(len(statsReturn))])
        ax = pyplot.gca()
        ax.set_ylabel('Cumulative CPU time [sec]')
        ax.set_xlabel('Iteration nb.')
        nbParam = len(Prebel.MODPARAM.prior)
        for j in range(nbParam):
            fig = pyplot.figure()
            ax = fig.add_subplot()
            ax.plot(range(len(statsReturn)),[statsReturn[i].means[j] for i in range(len(statsReturn))],'b-')
            ax.plot(range(len(statsReturn)),[statsReturn[i].means[j]+statsReturn[i].stds[j] for i in range(len(statsReturn))],'b--')
            ax.plot(range(len(statsReturn)),[statsReturn[i].means[j]-statsReturn[i].stds[j] for i in range(len(statsReturn))],'b--')
            if TrueModel is not None:
                ax.plot([0, len(statsReturn)-1],[TrueModel[j], TrueModel[j]],'r')
            ax.set_xlim(0, len(statsReturn)-1)
            ax.set_title(r'${}$'.format(Prebel.MODPARAM.paramNames["NamesFU"][j]))
            ax.set_ylabel('Posterior distribution')
            ax.set_xlabel('Iteration nb.')
        pyplot.show(block=False)
    if verbose:
        print('Computation done in {} seconds!'.format(end-start))

    if not(statsNotReturn):
        return Prebel, Postbel, PrebelInit, statsReturn
    else:
        return Prebel, Postbel, PrebelInit