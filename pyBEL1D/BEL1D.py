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
#   - Comment the codes!
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
from scipy import stats                     # For the statistical distributions
from pathos import multiprocessing as mp    # For parallelization (No issues with pickeling)
from pathos import pools as pp              # For parallelization
from functools import partial               # For building parallelizable functions
import time                                 # For CPU time measurements
from numpy import random                    # For random sampling

# Forward models:
from pygimli.physics.sNMR import MRS, MRS1dBlockQTModelling # sNMR (from pyGIMLI: https://www.pygimli.org/)
from pysurf96 import surf96                                 # Dispersion Curves (from Github: https://github.com/hadrienmichel/pysurf96)

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

    ATTENTION: The function MUST be pickable by dill.

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
                o "Fun" (callable lambda function): the forward model function for a given 
                                                    model
                o "Axis" (np.array): the X axis along which the computation is done
        - paramNames (dictionary): a dictionary with multiple entries
                o "NamesFU" (list): Full names of all the parameters with units
                o "NamesSU" (list): Short names of all the parameters with units
                o "NamesS" (list): Short names of all the parameters without units
                o "NamesGlobal" (list): Full names of the global parameters (not layered)
                o "NamesGlobalS" (list): Short names of the global parameters (not layered)
                o "DataUnits" (string): Units for the dataset,
                o "DataName" (string): Name of the Y-axis of the dataset (result from the 
                                        forward model)
                o "DataAxis" (string): Name of the X-axis of the dataset
        - nbLayer (int): the number of layers for the model (None if not layered)
    '''

    def __init__(self, prior=None, cond=None, method=None, forwardFun=None, paramNames=None, nbLayer=None):
        if (prior is None) or (method is None) or (forwardFun is None) or (paramNames is None):
            self.prior = []
            self.method = []
            self.forwardFun = []
            self.paramNames = []
            self.nbLayer = nbLayer # If None -> Model with parameters and no layers (not geophy?)
            self.cond = cond
        else:
            self.prior = prior
            self.method = method
            self.forwardFun = forwardFun
            self.cond = cond
            self.paramNames = paramNames
            self.nbLayer = nbLayer
    
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
            Kernel = "Data/sNMR/KernelTest.mrsk"
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
        NShort = ["e_{", "W_{", "T_{2,"]
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
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)

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
        NShort = ["e_{", "Vs_{", "Vp_{", "\\rho_{"]
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
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)

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
            o 'Data': a sklearn.decompostion.PCA object with the PCA
                      decomposition for the data dimensions.
            o 'Models': a sklearn.decompostion.PCA object with the PCA
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

    def run(self, Parallelization:list=[False, None], RemoveOutlier:bool=False, reduceModels:bool=False, verbose:bool=False):
        """The RUN method runs all the computations for the preparation of BEL1D

        It is an instance method that does not need any arguments.
        Howerev, optional arguments are:
            - Parallelization (list): instructions for parallelization
                o [False, ?]: no parallel runs
                o [True, None]: parallel runs without pool provided
                o [True, pool]: parallel runs with pool (defined bypathos.pools) 
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
        if self.nbModels is None: # Normally, we should never enter this
            self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS)
            self.nbModels = 1000
        else:
            self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS,self.nbModels)
        
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
            m_h = pca_model.fit_transform(self.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            # n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            d_h = pca_data.fit_transform(self.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions ({}) than the models ({})!'.format(n_CompPCA_Data, n_CompPCA_Mod))
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                d_h = pca_data.fit_transform(self.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            self.PCA = {'Data':pca_data,'Model':pca_model}
        else:
            m_h = self.MODELS # - np.mean(self.MODELS,axis=0)
            n_CompPCA_Mod = m_h.shape[1]
            #n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            d_h = pca_data.fit_transform(self.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions ({}) than the models ({})!'.format(n_CompPCA_Data, n_CompPCA_Mod))
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
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
    def POSTBEL2PREBEL(cls, PREBEL, POSTBEL, Dataset=None, NoiseModel=None, Parallelization:list=[False,None], reduceModels:bool=False, verbose:bool=False):
        ''' POSTBEL2PREBEL is a class method that converts a POSTBEL object to a PREBEL one.

        It takes as arguments:
            - PREBEL (PREBEL): The previous PREBEL object
            - POSTBEL (POSTBEL): the current POSTBEL object
        And optional arguments are:
            - Dataset (np.array): the field dataset
            - NoiseModel (list): the list defining the noise model (see dedicated functions)
            - Parallelization (list): parallelization instructions
                    o [False, _]: no parallel runs (default)
                    o [True, None]: parallel runs without pool provided
                    o [True, pool]: parallel runs with pool (defined by pathos.pools) 
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
        # 3) PCA on data (and optionally model):
        varRepresented = 0.90
        if verbose:
            print('Reducing the dimensionality . . .')
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=varRepresented) # Keeping 90% of the variance
            m_h = pca_model.fit_transform(PrebelNew.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            # n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            d_h = pca_data.fit_transform(PrebelNew.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions than the models!')
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                d_h = pca_data.fit_transform(PrebelNew.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            PrebelNew.PCA = {'Data':pca_data,'Model':pca_model}
        else:
            m_h = PrebelNew.MODELS # - np.mean(self.MODELS,axis=0)
            n_CompPCA_Mod = m_h.shape[1]
            #n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            d_h = pca_data.fit_transform(PrebelNew.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                if verbose:
                    print('The data space can be represented with fewer dimensions than the models!')
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
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
    
    def runMCMC(self, Dataset=None, NoiseModel=None, nbSamples:int=50000, nbChains:int=10, verbose:bool=False):
        ''' RUNMCMC is a class method that runs a simple metropolis McMC algorithm
        on the prior model space (PREBEL). 

        It takes as arguments:
            - Dataset (np.array): the field dataset
            - NoiseModel (np.array): the list defining the noise model
            - nbSamples (int): the number of models to sample per chains (larger for larger 
                               priors). The default value is 50000
            - nbChains (int): the number of chains to run. The larger, the better to avoid 
                              remaining in a local optimum. The default value is 10.
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
        timeIn = time.time()
        nbParam = len(self.MODPARAM.prior)
        accepted = np.zeros((nbChains, nbSamples, nbParam))
        acceptedData = np.zeros((nbChains, nbSamples, len(Dataset)))
        for j in range(nbChains):
            if verbose:
                print('Running chain {} out of {}. . .'.format(j, nbChains))
            rejectedNb = 0
            i = 0
            LikelihoodLast = 1e-50 # First sample most likely accepted
            Covariance = 0.01*np.cov(self.MODELS.T) # Compute the initial covariance from the prior distribution
            passed = False
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
                    acceptedData[j,i,:] = SynData
                    i += 1
                    passed = False
                else:
                    rejectedNb += 1
                if np.mod(i,50) == 0 and not(passed):
                    if verbose:
                        print('{} models sampled (out of {}) in chain {}.'.format(i, nbSamples, j))
                    # LikelihoodLast = 1e-50
                    AcceptanceRatio = i/(rejectedNb+i)
                    if AcceptanceRatio < 0.75 and i < nbSamples/2:
                        if verbose:
                            print('Acceptance ratio too low, reducing covariance.')
                        Covariance *= 0.8 # We are reducing the covariance to increase the acceptance rate
                    elif AcceptanceRatio > 0.85 and i < nbSamples/2:
                        if verbose:
                            print('Acceptance ratio too high, increasing covariance.')
                        Covariance *= 1.2 # We are increasing the covariance to decrease the acceptance rate
                    # print(f'{i} models drawn out of {nbSamples} in chain {j} - Acceptance rate = {AcceptanceRatio}')
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
        for ax in axes.flat:
            ax.label_outer()

        fig.suptitle("Prior model visualization",fontsize=16)
        pyplot.show()

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
        pyplot.show()
        
class POSTBEL:
    """Object that is used to store the POSTBEL elements:
    
    For a given PREBEL set (see PREBEL class), the POSTBEL class
    enables all the computations that takes place aftre the  
    field data acquisition. It takes as argument:
        - PREBEL (PREBEL class object): the PREBEL object from 
                                        PREBEL class
    """
    def __init__(self,PREBEL:PREBEL):
        self.nbModels = PREBEL.nbModels
        self.nbSamples = 1000 # Default number of sampled models
        self.FORWARD = PREBEL.FORWARD # Forward from the prior
        self.KDE = PREBEL.KDE
        self.PCA = PREBEL.PCA
        self.CCA = PREBEL.CCA
        self.MODPARAM = PREBEL.MODPARAM
        self.DATA = dict()
        self.SAMPLES = []
        self.SAMPLESDATA = []

    def run(self,Dataset,nbSamples:int=1000,Graphs:bool=False,NoiseModel:list=None):
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


        '''
        self.nbSamples = nbSamples
        # Transform dataset to CCA space:
        Dataset = np.reshape(Dataset,(1,-1))# Convert for reverse transform
        d_obs_h = self.PCA['Data'].transform(Dataset)
        d_obs_c = self.CCA.transform(d_obs_h)
        self.DATA = {'True':Dataset,'PCA':d_obs_h,'CCA':d_obs_c}
        # Propagate Noise:
        if NoiseModel is not None:
            Noise = np.sqrt(Tools.PropagateNoise(self,NoiseModel))
        else:
            Noise = None
        # Obtain corresponding distribution (KDE)
        if (self.KDE.Dist[0] is None):
            self.KDE.GetDist(Xvals=d_obs_c,Noise=Noise)
        if Graphs:
            self.KDE.ShowKDE(Xvals=d_obs_c)
        # Sample models:
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
                    raise Exception('Impossible to sample models in the current prior under reasonable timings!')
            self.SAMPLES = Samples
    
    def runMCMC(self, NoiseModel=None, nbSamples=10000, nbChains=10, verbose:bool=False):
        ''' RUNMCMC is a class method that runs a simple metropolis McMC algorithm
        on the last posterior model space (POSTBEL). 

        It takes as arguments:
            - NoiseModel (np.array): the list defining the noise model
            - nbSamples (int): the number of models to sample per chains (larger for larger 
                               priors). The default value is 50000
            - nbChains (int): the number of chains to run. The larger, the better to avoid 
                              remaining in a local optimum. The default value is 10.
            - verbose (bool): output progresses messages (True) or not (False - default)

        It returns 2 arrays containing the samples models and the associated datasets.
        '''
        if NoiseModel is None:
            raise Exception('No noise model provided. Impossible to compute the likelihood')
        if len(NoiseModel) != len(self.DATA['True'][0,:]):
            raise Exception('NoiseModel should have the same size as the dataset')
        timeIn = time.time()
        nbParam = len(self.MODPARAM.prior)
        accepted = np.zeros((nbChains, nbSamples, nbParam))
        acceptedData = np.zeros((nbChains, nbSamples, len(self.DATA['True'][0,:])))
        for j in range(nbChains):
            rejectedNb = 0
            i = 0
            LikelihoodLast = 1e-50 # First sample most likely accepted
            Covariance = 0.01*np.cov(self.SAMPLES.T) # Compute the initial covariance from the prior distribution
            passed = False
            while i < nbSamples:
                if i == 0:
                    ## Sampling a random model from the posterior distribution
                    samples_CCA = self.KDE.SampleKDE(nbSample=1)
                    # Back transform models to original space:
                    samples_PCA = np.matmul(samples_CCA,self.CCA.y_loadings_.T)
                    samples_PCA *= self.CCA.y_std_
                    samples_PCA += self.CCA.y_mean_
                    # samples_PCA = self.CCA.inverse_transform(samples_CCA)
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
                else:
                    rejectedNb += 1
                if np.mod(i,50) == 0 and not(passed):
                    # LikelihoodLast = 1e-50
                    AcceptanceRatio = i/(rejectedNb+i)
                    if AcceptanceRatio < 0.75 and i < nbSamples/2:
                        Covariance *= 0.8 # We are reducing the covariance to increase the acceptance rate
                    elif AcceptanceRatio > 0.85 and i < nbSamples/2:
                        Covariance *= 1.2 # We are increasing the covariance to decrease the acceptance rate
                    # print(f'{i} models drawn out of {nbSamples} in chain {j} - Acceptance rate = {AcceptanceRatio}')
                    passed = True
                LikelihoodLast = Likelihood
        if verbose:
            print(f'MCMC on POSTBEL executed in {time.time()-timeIn} seconds.')
        return np.asarray(accepted), np.asarray(acceptedData)

    def DataPost(self, Parallelization=[False,None],verbose:bool=False, OtherModels=None):
        '''DATAPOST is a function that computes the forward model for all the 
        models sampled from the posterior.

        The optional arguments are:
            - Parallelization (list): parallelization instructions
                    o [False, _]: no parallel runs (default)
                    o [True, None]: parallel runs without pool provided
                    o [True, pool]: parallel runs with pool (defined by pathos.pools) 
                                    provided 
            - verbose (bool): output progresses messages (True) or not (False - default)
        '''
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
            # pool.close()
            # pool.join()
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
            print('{} models remaining after forward modelling!'.format(newSamplesNb))
        if OtherModels is None:
            self.nbSamples = newSamplesNb
            self.SAMPLES = SAMPLES
            self.SAMPLESDATA = SAMPLESDATA
        return SAMPLESDATA
    
    def runRejection(self, NoiseModel=None, Parallelization=[False,None], verbose:bool=False):
        if NoiseModel is None:
            raise Exception('No noise model provided. Impossible to compute the likelihood')
        if len(NoiseModel) != len(self.DATA['True'][0,:]):
            raise Exception('NoiseModel should have the same size as the dataset')
        timeIn = time.time()
        self.DataPost(Parallelization=Parallelization)
        Likelihood = np.zeros(len(self.SAMPLESDATA),)
        for i, SynData in enumerate(self.SAMPLESDATA):
            FieldError = NoiseModel
            DataDiff = self.DATA['True'] - SynData
            A = np.divide(1,np.sqrt(2*np.pi*np.power(FieldError,2)))
            B = np.exp(-1/2 * np.power(np.divide(DataDiff,FieldError),2))
            Likelihood[i] = np.prod(np.multiply(A, B))
        Order = random.permutation(len(Likelihood))
        LikelihoodOrder = Likelihood[Order]
        LikelihoodRatio = LikelihoodOrder[1:]/LikelihoodOrder[:-1]
        Accepted = [Order[0]]
        LikeLast = LikelihoodOrder[0]
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
            print(f'Rejection sampling on POSTBEL executed in {time.time()-timeIn} seconds.')
        return ModelsAccepted, DataAccepted

    def ShowPost(self,TrueModel=None):
        '''SHOWPOST shows the posterior parameter distributions (uncorrelated).

        The optional argument TrueModel (np.array) is an array containing the benchmark model.
        '''
        nbParam = self.SAMPLES.shape[1]
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        for i in range(nbParam):
            _, ax = pyplot.subplots()
            ax.hist(self.SAMPLES[:,i])
            ax.set_title("Posterior histogram")
            ax.set_xlabel(self.MODPARAM.paramNames["NamesFU"][i])
            if TrueModel is not None:
                ax.plot([TrueModel[i],TrueModel[i]],np.asarray(ax.get_ylim()),'r')
            pyplot.show(block=False)
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
                        axs[i,j].set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesSU"][i]))
                if i == nbParam-1: # Last line of the graph
                    axs[i,j].set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesSU"][j]))
                if j == nbParam-1:
                    if not(i==j):
                        axs[i,j].yaxis.set_label_position("right")
                        axs[i,j].yaxis.tick_right()
                        axs[i,j].set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesSU"][i]))
                if i == 0:
                    axs[i,j].xaxis.set_label_position("top")
                    axs[i,j].xaxis.tick_top()
                    axs[i,j].set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesSU"][j]))
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
                    o [False, _]: no parallel runs (default)
                    o [True, None]: parallel runs without pool provided
                    o [True, pool]: parallel runs with pool (defined bypathos.pools) 
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
            nbParamUnique = int(np.ceil(nbParam/nbLayer))-1 # Number of parameters minus the thickness
            fig = pyplot.figure(figsize=[4*nbParamUnique,10])
            Param = list()
            if OtherModels is not None:
                ModelsPlot = OtherModels
            else:
                ModelsPlot = self.SAMPLES
            Param.append(np.cumsum(ModelsPlot[:,0:nbLayer-1],axis=1))
            for i in range(nbParamUnique):
                Param.append(ModelsPlot[:,(i+1)*nbLayer-1:(i+2)*nbLayer-1])
            if TrueModel is not None:
                TrueMod = list()
                TrueMod.append(np.cumsum(TrueModel[0:nbLayer-1]))
                for i in range(nbParamUnique):
                    TrueMod.append(TrueModel[(i+1)*nbLayer-1:(i+2)*nbLayer-1])
                
            maxDepth = np.max(Param[0][:,-1])*1.5
            if RMSE:
                colormap = matplotlib.cm.get_cmap('viridis')
                axes = fig.subplots(1,nbParamUnique) # One graph per parameter
                if nbParamUnique > 1:
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
                    j = 0
                    for i in sortIndex:
                        axes.step(np.append(Param[j+1][i,:], Param[j+1][i,-1]),np.append(np.append(0, Param[0][i,:]), maxDepth),where='pre',color=colormap(quantiles[i]))
                    if TrueModel is not None:
                        axes.step(np.append(TrueMod[j+1][:], TrueMod[j+1][-1]),np.append(np.append(0, TrueMod[0][:]), maxDepth),where='pre',color='gray')
                    axes.invert_yaxis()
                    axes.set_ylim(bottom=maxDepth,top=0.0)
                    axes.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][j+1]),fontsize=14)
                    axes.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["NamesGlobalS"][0]),fontsize=14)
                    fig.subplots_adjust(left=0.2)
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
        if nbParamUnique > 1:
            for ax in axes.flat:
                ax.label_outer()
        
        if RMSE:
            fig.subplots_adjust(bottom=0.25)
            ax_colorbar = fig.add_axes([0.15, 0.10, 0.70, 0.05])
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


        fig.suptitle("Posterior model visualization",fontsize=16)
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
                    o [False, _]: no parallel runs (default)
                    o [True, None]: parallel runs without pool provided
                    o [True, pool]: parallel runs with pool (defined bypathos.pools) 
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
        ax = fig.add_subplot(1, 1, 1)
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
            fig.subplots_adjust(bottom=0.30)
            ax_colorbar = fig.add_axes([0.10, 0.15, 0.80, 0.05])
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
from typing import Callable
def defaultMixing(iter:int) -> float:
    return 1
    
def IPR(MODEL:MODELSET, Dataset=None, NoiseEstimate=None, Parallelization:list=[False, None],
        nbModelsBase:int=1000, nbModelsSample:int=None, stats:bool=False, saveIters:bool=False, 
        saveItersFolder:str="IPR_Results", nbIterMax:int=100, Rejection:float=0.0, 
        Mixing:Callable[[int], float]=defaultMixing, Graphs:bool=False, TrueModel=None,
        verbose:bool=False):
    '''IPR (Iterative prior resampling) is a function that will compute the posterior 
    with iterative prior resampling for a given model defined via a MODELSET class object.

    It takes as arguments:
        - MODEL (MODELSET): the MODELSET class object defining the different elements for 
                            the computation of the forward model and the definition of the
                            prior.
        - Dataset: The true field dataset to be fitted
        - NoiseEstimate: The estimated noise level
        - Parallelization (list=[False,None]): parallelization instructions
                    o [False, _]: no parallel runs (default)
                    o [True, None]: parallel runs without pool provided
                    o [True, pool]: parallel runs with pool (defined by pathos.pools) 
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
        - Rejection (float=0.9): Maximum quantile for the RMSE of the accepted models in the
                                 posterior
        - Mixing (callable): Function that returns the mixing ratio at a given iteration. The 
                             default value is 0.5 whatever the iteration.
        - Graphs (bool=False): Show diagnistic graphs (True) or not (False)
    '''
    import numpy as np
    from .utilities.Tools import nSamplesConverge
    from copy import deepcopy
    if nbModelsSample is None:
        nbModelsSample = nbModelsBase
    if Dataset is None:
        raise Exception('No Dataset provided!')
    start = time.time()
    Prebel = PREBEL(MODPARAM=MODEL, nbModels=nbModelsBase)
    Prebel.run(Parallelization=Parallelization)
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
        # Iterating:
        if Mixing is not None:
            nbModPrebel = Prebel.nbModels
            MixingUsed = Mixing(it)
            nbPostAdd = int(MixingUsed*nbModPrebel/(1-Rejection)) # We need to sample at least this number of models to be able to add to the prior with mixing satisfied
            nbSamples = max([int(nbModelsSample/(1-Rejection)),nbPostAdd])
        else:
            nbSamples = int(nbModelsSample/(1-Rejection))
            nbPostAdd = nbSamples
        Postbel = POSTBEL(Prebel)
        Postbel.run(Dataset=Dataset, nbSamples=nbSamples, NoiseModel=NoiseEstimate)
        end = time.time() # End of the iteration - begining of the preparation for the next iteration (if needed):
        if Graphs:
            if it == 0:
                Postbel.KDE.ShowKDE(Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
        Postbel.DataPost(Parallelization=Parallelization)
        # Testing for convergence (5% probability of false positive):
        if len(ModelLastIter) > nSamplesConverge:
            nbConvergeSamp = nSamplesConverge
        else:
            nbConvergeSamp = len(ModelLastIter)
        threshold = 1.87*nbConvergeSamp**(-0.50)# Power law defined from the different tests
        diverge, distance = Tools.ConvergeTest(SamplesA=ModelLastIter,SamplesB=Postbel.SAMPLES, tol=threshold)
        if stats:
            means, stds = Postbel.GetStats()
            statsReturn.append(StatsResults(means, stds, end-start, distance))
        if saveIters:
            SavePOSTBEL(Postbel,saveItersFolder + '/IPR_{}'.format(it))
        if not(diverge):
            if verbose:
                print('Model has converged at iter {}.'.format(it))
            if Graphs:
                NoiseToLastPrebel = Tools.PropagateNoise(Postbel, NoiseLevel=NoiseEstimate)
                Postbel.KDE.KernelDensity(NoiseError=NoiseToLastPrebel, RemoveOutlier=True)#,Parallelization=Parallelization)
                Postbel.KDE.ShowKDE(Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
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
                PostbelAdd.SAMPLES = PostbelAdd.SAMPLES[idxKeep,:]
                PostbelAdd.SAMPLESDATA = PostbelAdd.SAMPLESDATA[idxKeep,:]
                PostbelAdd.nbSamples = np.size(PostbelAdd.SAMPLES,axis=0)
        # Preparing next iteration:
        Prebel = PREBEL.POSTBEL2PREBEL(PREBEL=Prebel,POSTBEL=PostbelAdd,Dataset=Dataset,NoiseModel=NoiseEstimate,Parallelization=Parallelization,verbose=verbose)
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
    if verbose and stats:
        print('Computation done in {} seconds!'.format(statsReturn[-1].timing))

    if not(statsNotReturn):
        return Prebel, Postbel, PrebelInit, statsReturn
    else:
        return Prebel, Postbel, PrebelInit