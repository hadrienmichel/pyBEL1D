# Importing custom libraries
from .utilities import Tools
from .utilities.KernelDensity import KDE
#Importing common libraries
import numpy as np 
import math as mt 
import matplotlib
from matplotlib import pyplot
import sklearn
from sklearn import decomposition, cross_decomposition
from scipy import stats
from pathos import multiprocessing as mp # No issues with pickeling with this
from pathos import pools as pp
from functools import partial
import time

def round_to_5(x,n=1): 
    # Modified from: https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    tmp = [(round(a, -int(mt.floor(mt.log10(abs(a)))) + (n-1)) if a != 0.0 else 0.0) for a in x]
    return tmp

# Parralelization fucntions:
def ForwardParallelFun(Model, function, nbVal):
    try:
        ForwardComputed = function(Model)
    except:
        ForwardComputed = [None]*nbVal
    return ForwardComputed

# TODO/DONE:
#   - (Done on 24/04/2020) Add conditions (function for checking that samples are within a given space)
#   - (Done on 13/04/2020) Add Noise propagation (work in progress 29/04/20 - OK for SNMR 30/04/20 - DC OK) -> Noise impact is always very low???
#   - (Done on 11/05/2020) Add DC example (uses pysurf96 for forward modelling: https://github.com/miili/pysurf96 - compiled with msys2 for python)
#   - (Done on 18/05/2020) Add postprocessing (partially done - need for True model visualization on top and colorscale of graphs)
#   - (Done on 12/05/2020) Speed up kernel density estimator (vecotization?) - result: speed x4
#   - (Done on 13/05/2020) Add support for iterations
#   - Parallelization of the computations:
#       - KDE (one core/thread per dimension) -> Most probable gain
#       - (Done on 14/07/2020) Forward modelling (all cores/threads operating) -> Needed for more complex forward models
#       - (Not possible to parallelize (same seed for different workers))Sampling and checking conditions (all cores/thread operating) -> Can be usefull - not priority
#   - Add iteration convergence critereon!
#   - Lower the memory needs (how? not urgent)
#   - Comment the codes!
#   - Check KDE behaviour whit outliers (too long computations and useless?)

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
        from pygimli.physics import sNMR
        import numpy.matlib
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
        KFile = sNMR.MRS()
        KFile.loadKernel(Kernel)
        ModellingMethod = sNMR.MRS1dBlockQTModelling(nlay=nLayer,K=KFile.K,zvec=KFile.z,t=Timing)
        forwardFun = lambda model: ModellingMethod.response(model)
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
        from pysurf96 import surf96
        import numpy.matlib
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
    """
    def __init__(self,MODPARAM:MODELSET,nbModels:int=1000):
        # PRIOR: a list of scipy.stats distributions describing 
        # the prior model space for all the parameters
        self.PRIOR = MODPARAM.prior
        # CONDITIONS: a list of lambda functions that are imbeded in a cond class (custom)
        # The compute method of the class must return a list of boolean values
        self.CONDITIONS = MODPARAM.cond
        # nbModels: the number of sampled models
        self.nbModels = nbModels
        # MODPARAM: a dictionnary with
        #   - 'method': the name of the geophysical method ('sNMR' or 'DC')
        #   - 'forward': a lambda function taking as argument model, a vector with the model's
        #                parameters
        #   - 'parmNames': a dictionnary with the instances of names for the parameters 
        #                  (full = full name with untis, units = reduced name with units 
        #                  and nounits = reduced name) contained in Lists
        self.MODPARAM = MODPARAM
        # MODELS: a numpy array containing the models parameters values.
        #           - the number of columns (second dimension) is the number of parameters
        #           - the number of lines (first dimension) is the number of models
        self.MODELS = []
        # FORWARD: a numpy array containing the forward response of each model
        #           - the number of column (second dimension) is the number of simulated points
        #           - the number of lines (first dimension) is the number of models
        self.FORWARD = []
        # PCA: a dictionnary containing the PCA reduction and their mathematical definitions
        self.PCA = dict()
        # CCA: a class object containing the CCA reduction and their mathematical definitions
        self.CCA = []
        # KDE: a class pobject KDE (custom)
        self.KDE = []

    def run(self, RemoveOutlier:bool=False, Parallelization=[False, None]):
        """The RUN method runs all the computations for the preparation of BEL1D

        It is an instance method that does not need any arguments.
        The optional argument RemoveOutlier (bool) is given to simplifie the KDE
        computation (default=False)
        If the argument Parallelization (list) is given it can either be:
            - [False, ?]: no parallel runs
            - [True, None]: parallel runs without pool provided
            - [True, pool]: parallel runs with pool (defined bypathos.pools) 
                            provided
        The default is no parallel runs.
        """

        # 1) Sampling (if not done already):
        if self.nbModels is None:
            self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS)
            self.nbModels = 1000
        else:
            self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS,self.nbModels)
        # 2) Running the forward model
        # For DC, sometimes, the code will return an error --> need to remove the model from the prior
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
            # pool.close()
            # pool.join()
            self.FORWARD = np.vstack(outputs) #ForwardParallel
            notComputed = [i for i in range(self.nbModels) if self.FORWARD[i,0] is None]
            self.MODELS = np.array(np.delete(self.MODELS,notComputed,0),dtype=np.float64)
            self.FORWARD = np.array(np.delete(self.FORWARD,notComputed,0),dtype=np.float64)
            newModelsNb = np.size(self.MODELS,axis=0) # Get the number of models remaining
            timeEnd = time.time()
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
            print('The Unparallelized Forward Modelling took {} seconds.'.format(timeEnd-timeBegin))
        print('{} models remaining after forward modelling!'.format(newModelsNb))
        self.nbModels = newModelsNb
        # 3) PCA on data (and optionally model):
        reduceModels = False
        varRepresented = 0.99
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=varRepresented) # Keeping 90% of the variance
            m_h = pca_model.fit_transform(self.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            # n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            d_h = pca_data.fit_transform(self.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
                print('The data space can be represented with fewer dimensions than the models!')
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
                print('The data space can be represented with fewer dimensions than the models!')
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                d_h = pca_data.fit_transform(self.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            self.PCA = {'Data':pca_data,'Model':None}
        # 4) CCA:
        cca_transform = sklearn.cross_decomposition.CCA(n_components=n_CompPCA_Mod)
        d_c,m_c = cca_transform.fit_transform(d_h,m_h)
        self.CCA = cca_transform
        # 5) KDE:
        self.KDE = KDE(d_c,m_c)
        self.KDE.KernelDensity(RemoveOutlier=RemoveOutlier,Parallelization=Parallelization)
        if Parallelization[0] and terminatePool:
            pool.terminate()
    
    @classmethod
    def POSTBEL2PREBEL(cls,PREBEL,POSTBEL,Dataset=None,NoiseModel=None,Simplified:bool=False,RemoveOutlier:bool=False,nbMax:int=100000,MixingRatio:float=None,Rejection=False,Parallelization:list=[False,None]):
        ''' POSTBEL2PREBEL is a class method that converts a POSTBEL object to a PREBEL one.

        It takes as arguments:
            - PREBEL (PREBEL): The previous PREBEL object
            - POSTBEL (POSTBEL): the current POSTBEL object
        And optional arguments are:
            - Dataset (np.array): the field dataset
            - NoiseModel (list): the list defining the noise model (see dedicated functions)
            - Simplified (bool): ramdom sampling of a number of models from componded prior or not
                                 (default=False)
            - RemoveOutlier (bool): remove ourliers for the KDE computation (default=False)
            - nbMax (int): the number of random samples to keep if Simplified=True (default=100000)
            - MixingRatio (float): mixing proportion for the prior and posterior (keep constant 
                                   accross iterations, default = 1.0)
            - Parallelization (list): parallelization instructions
                    o [False, _]: no parallel runs (default)
                    o [True, None]: parallel runs without pool provided
                    o [True, pool]: parallel runs with pool (defined bypathos.pools) 
                                    provided 

        '''
        # if (Dataset is None) and (NoiseModel is not None):
        #     NoiseModel = None 
        # 1) Initialize the Prebel class object
        Modelset = POSTBEL.MODPARAM # A MODELSET class object
        PrebelNew = cls(Modelset)
        # 2) Inject the samples from postbel:
        ModelsKeep = POSTBEL.SAMPLES
        # 2) Running the forward model
        if not(len(POSTBEL.SAMPLESDATA) != 0):
            # For DC, sometimes, the code will return an error --> need to remove the model from the prior
            indexCurr = 0
            while True:
                try:
                    tmp = PrebelNew.MODPARAM.forwardFun["Fun"](ModelsKeep[indexCurr,:])
                    break
                except:
                    indexCurr += 1
                    if indexCurr > PrebelNew.nbModels:
                        raise Exception('The forward modelling failed!')
            
            if Parallelization[0]:
                # We create a partial function that has a fixed fowrard function. The remaining arguments are :
                #   - Model: a numpy array containing the model to compute
                # It returns the Forward Computed, either a list of None or a list of values corresponding to the forward
                functionParallel = partial(ForwardParallelFun, function=PrebelNew.MODPARAM.forwardFun["Fun"], nbVal=len(tmp))
                inputs = [ModelsKeep[i,:] for i in range(np.size(ModelsKeep,axis=0))]
                if Parallelization[1] is not None:
                    pool = Parallelization[1]
                    terminatePool = False
                else:
                    pool = pp.ProcessPool(mp.cpu_count()) # Create the pool for paralelization
                    Parallelization[1] = pool
                    terminatePool = True
                pool = mp.Pool(mp.cpu_count()) # Create the pool for paralelization
                outputs = pool.map(functionParallel,inputs)
                # pool.close()
                # pool.join()
                ForwardKeep = np.vstack(outputs) #ForwardParallel
                notComputed = [i for i in range(np.size(ModelsKeep,axis=0)) if ForwardKeep[i,0] is None]
                ModelsKeep = np.delete(ModelsKeep,notComputed,0)
                ForwardKeep = np.delete(ForwardKeep,notComputed,0)
                newModelsNb = np.size(ModelsKeep,axis=0) # Get the number of models remaining
            else:
                ForwardKeep = np.zeros((np.size(ModelsKeep,axis=0),len(tmp)))
                notComputed = []
                for i in range(np.size(ModelsKeep,axis=0)):
                    # print(i)
                    try:
                        ForwardKeep[i,:] = PrebelNew.MODPARAM.forwardFun["Fun"](ModelsKeep[i,:])
                    except:
                        ForwardKeep[i,:] = [None]*len(tmp)
                        notComputed.append(i)
                # Getting the uncomputed models and removing them:
                ModelsKeep = np.delete(ModelsKeep,notComputed,0)
                ForwardKeep = np.delete(ForwardKeep,notComputed,0)
                newModelsNb = np.size(ModelsKeep,axis=0) # Get the number of models remaining
            print('{} models remaining after forward modelling!'.format(newModelsNb))
            PrebelNew.MODELS = np.append(ModelsKeep,PREBEL.MODELS,axis=0)
            PrebelNew.FORWARD = np.append(ForwardKeep,PREBEL.FORWARD,axis=0)
            PrebelNew.nbModels = np.size(PrebelNew.MODELS,axis=0) # Get the number of sampled models
        else: # The posterior datasets had already been computed
            PrebelNew.MODELS = np.append(POSTBEL.SAMPLES,PREBEL.MODELS,axis=0)
            PrebelNew.FORWARD = np.append(POSTBEL.SAMPLESDATA,PREBEL.FORWARD,axis=0)
            PrebelNew.nbModels = np.size(PrebelNew.MODELS,axis=0) # Get the number of sampled models
        # if Rejection and Dataset is not None: # Rejection only if true dataset known
        #     # Compute the RMSE
        #     RMSE = np.sqrt(np.square(np.subtract(Dataset,PrebelNew.FORWARD)).mean(axis=-1))
        #     RMSE_max = np.quantile(RMSE,0.95) # We reject the 1% worst fit
        #     idxDelete = np.greater_equal(RMSE,RMSE_max)
        #     PrebelNew.MODELS = np.delete(PrebelNew.MODELS,np.where(idxDelete),0)
        #     PrebelNew.FORWARD = np.delete(PrebelNew.FORWARD,np.where(idxDelete),0)
        #     PrebelNew.nbModels = np.size(PrebelNew.MODELS,axis=0)
        if Simplified and (PrebelNew.nbModels>nbMax):
            import random
            # Using the mixing ratio to ensure a correct representation of all the models in the prior
            if MixingRatio is not None:
                # We need to keep half of the models from the first half of the prior and half from the second half
                samp1 = np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))])
                samp2 = nbMax-np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))])
                print('Sampling 1: {} \nSampling 2: {}'.format(samp1,samp2))
                if samp2 > PrebelNew.nbModels-int(np.ceil(PrebelNew.nbModels/2)):
                    samp2 = PrebelNew.nbModels-int(np.ceil(PrebelNew.nbModels/2))
                    samp1 = nbMax - samp2
                    if samp1 > int(np.ceil(PrebelNew.nbModels/2)):
                        raise Exception('Impossible to use this proportion with current sample set!')
                # print('Nb To Sample 1 : {} out of {}'.format(np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))]),int(np.ceil(PrebelNew.nbModels/2))))
                # if np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))]) > int(np.ceil(PrebelNew.nbModels/2)):
                #     raise Exception('Error')
                idxKeep1 = np.asarray(random.sample(range(int(np.ceil(PrebelNew.nbModels/2))), samp1))#np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))])))
                # print('Nb To Sample 2 : {} out of {}'.format(nbMax-np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))]),PrebelNew.nbModels-int(np.ceil(PrebelNew.nbModels/2))))
                # if nbMax-np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))]) > PrebelNew.nbModels-int(np.ceil(PrebelNew.nbModels/2)):
                #     raise Exception('Error')
                idxKeep2 = np.asarray(random.sample(range(PrebelNew.nbModels-int(np.ceil(PrebelNew.nbModels/2))), samp2))  + int(np.ceil(PrebelNew.nbModels/2))#nbMax-np.min([int(np.ceil(MixingRatio*nbMax)),int(np.ceil(PrebelNew.nbModels/2))]))) + int(np.ceil(PrebelNew.nbModels/2))
                idxKeep = np.concatenate((idxKeep1,idxKeep2))
            else:
                idxKeep = random.sample(range(PrebelNew.nbModels), nbMax)
            PrebelNew.MODELS = PrebelNew.MODELS[idxKeep,:]
            PrebelNew.FORWARD = PrebelNew.FORWARD[idxKeep,:]
            PrebelNew.nbModels = np.size(PrebelNew.MODELS,axis=0) # Get the number of sampled models
            print('Prior simplified to {} random samples'.format(nbMax))
        # 3) PCA on data (and optionally model):
        reduceModels = False
        varRepresented = 0.99
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=varRepresented) # Keeping 90% of the variance
            m_h = pca_model.fit_transform(PrebelNew.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            # n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=varRepresented)
            d_h = pca_data.fit_transform(PrebelNew.FORWARD)
            n_CompPCA_Data = d_h.shape[1]
            if n_CompPCA_Data < n_CompPCA_Mod:
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
                print('The data space can be represented with fewer dimensions than the models!')
                pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)# Ensure at least the same number of dimensions
                d_h = pca_data.fit_transform(PrebelNew.FORWARD)
                n_CompPCA_Data = d_h.shape[1]
            PrebelNew.PCA = {'Data':pca_data,'Model':None}
        # 4) CCA:
        cca_transform = sklearn.cross_decomposition.CCA(n_components=n_CompPCA_Mod)
        d_c,m_c = cca_transform.fit_transform(d_h,m_h)
        PrebelNew.CCA = cca_transform
        # 5-pre) If dataset already exists:
        if Dataset is not None:
            Dataset = np.reshape(Dataset,(1,-1))# Convert for reverse transform
            d_obs_h = PrebelNew.PCA['Data'].transform(Dataset)
            d_obs_c = PrebelNew.CCA.transform(d_obs_h)
            if NoiseModel is not None:
                Noise = np.sqrt(Tools.PropagateNoise(PrebelNew,NoiseModel,DatasetIn=Dataset))
            else:
                Noise = None
        # 5) KDE:
        PrebelNew.KDE = KDE(d_c,m_c)
        if Dataset is None:
            PrebelNew.KDE.KernelDensity(RemoveOutlier=RemoveOutlier,Parallelization=Parallelization)
        else:
            PrebelNew.KDE.KernelDensity(XTrue=np.squeeze(d_obs_c), NoiseError=Noise,RemoveOutlier=RemoveOutlier,Parallelization=Parallelization)
        if Parallelization[0] and terminatePool:
            pool.terminate()
        return PrebelNew
    
    def ShowPreModels(self,TrueModel=None):
        '''SHOWPREMODELS is a function that displays the models sampled from the prior model space.

        The optional argument TrueModel (np.array) is an array containing the benchmark model.
        '''
        from matplotlib import colors
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
            self.SAMPLES = Samples

    def DataPost(self, Parallelization=[False,None]):
        '''DATAPOST is a function that computes the forward model for all the 
        models sampled from the posterior.

        The optional argument Parallelization (list) can take as values:
            - [False, _]: no parallel runs (default)
            - [True, None]: parallel runs without pool provided
            - [True, pool]: parallel runs with pool (defined by pathos.pools) 
                            provided 
        '''
        # TODO: add option to parallelize in calling functions
        if len(self.SAMPLESDATA)!=0:# The dataset is already simulated
            print('Forward modelling already conducted!')
            return self.SAMPLESDATA
        indexCurr = 0
        while True:
            try:
                tmp = self.MODPARAM.forwardFun["Fun"](self.SAMPLES[indexCurr,:])
                break
            except:
                indexCurr += 1
                if indexCurr > self.nbModels:
                    raise Exception('The forward modelling failed!')
        self.SAMPLESDATA = np.zeros((self.nbSamples,len(tmp)))
        if Parallelization[0]:
            # We create a partial function that has a fixed fowrard function. The remaining arguments are :
            #   - Model: a numpy array containing the model to compute
            # It returns the Forward Computed, either a list of None or a list of values corresponding to the forward
            functionParallel = partial(ForwardParallelFun, function=self.MODPARAM.forwardFun["Fun"], nbVal=len(tmp))
            inputs = [self.SAMPLES[i,:] for i in range(self.nbSamples)]
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
            self.SAMPLESDATA = np.vstack(outputs) #ForwardParallel
            notComputed = [i for i in range(self.nbSamples) if self.SAMPLESDATA[i,0] is None]
            self.SAMPLES = np.array(np.delete(self.SAMPLES,notComputed,0),dtype=np.float64)
            self.SAMPLESDATA = np.array(np.delete(self.SAMPLESDATA,notComputed,0),dtype=np.float64)
            newSamplesNb = np.size(self.SAMPLES,axis=0) # Get the number of models remaining
            if terminatePool:
                pool.terminate()
        else:
            notComputed = []
            for i in range(self.nbSamples):
                # print(i)
                try:
                    self.SAMPLESDATA[i,:] = self.MODPARAM.forwardFun["Fun"](self.SAMPLES[i,:])
                except:
                    self.SAMPLESDATA[i,:] = [None]*len(tmp)
                    notComputed.append(i)
            # Getting the uncomputed models and removing them:
            self.SAMPLES = np.delete(self.SAMPLES,notComputed,0)
            self.SAMPLESDATA = np.delete(self.SAMPLESDATA,notComputed,0)
            newSamplesNb = np.size(self.SAMPLES,axis=0) # Get the number of models remaining
        print('{} models remaining after forward modelling!'.format(newSamplesNb))
        self.nbSamples = newSamplesNb
        return self.SAMPLESDATA

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
    
    def ShowPostCorr(self,TrueModel=None,OtherMethod=None):
        '''SHOWPOSTCORR shows the posterior parameter distributions (correlated).

        The optional arguments are:
            - TrueModel (np.array): an array containing the benchmark model
            - OtherMethod (np.array): an array containing an ensemble of models
        '''
        # Adding the graph with correlations: 
        nbParam = self.SAMPLES.shape[1]
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        if (OtherMethod is not None) and (OtherMethod.shape[1]!=nbParam):
            print('OtherMethod is not a valid argument!')
            OtherMethod = None
        fig = pyplot.figure(figsize=[10,10])# Creates the figure space
        axs = fig.subplots(nbParam, nbParam)
        for i in range(nbParam):
            for j in range(nbParam):
                if i == j: # Diagonal
                    if i != nbParam-1:
                        axs[i,j].get_shared_x_axes().join(axs[i,j],axs[-1,j])# Set the xaxis limit
                    if OtherMethod is not None:
                        axs[i,j].hist(OtherMethod[:,j],color='y',density=True)
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
                    axs[i,j].plot(self.SAMPLES[:,j],self.SAMPLES[:,i],'.b')
                    if TrueModel is not None:
                        axs[i,j].plot(TrueModel[j],TrueModel[i],'.r')
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
                    axs[i,j].plot(OtherMethod[:,j],OtherMethod[:,i],'.y')
                    if TrueModel is not None:
                        axs[i,j].plot(TrueModel[j],TrueModel[i],'.r')
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
        fig.suptitle("Posterior model space visualization")
        for ax in axs.flat:
            ax.label_outer()
        pyplot.show(block=False)
    
    def ShowPostModels(self,TrueModel=None, RMSE:bool=False, Best:int=None, Parallelization=[False,None]):
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
        '''
        from matplotlib import colors
        nbParam = self.SAMPLES.shape[1]
        nbLayer = self.MODPARAM.nbLayer
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        if RMSE and len(self.SAMPLESDATA)==0:
            print('Computing the forward model for the posterior!')
            self.DataPost(Parallelization=Parallelization)
        if RMSE:
            TrueData = self.DATA['True']
            RMS = np.sqrt(np.square(np.subtract(TrueData,self.SAMPLESDATA)).mean(axis=-1))
            quantiles = np.divide([stats.percentileofscore(RMS,a,'strict') for a in RMS],100)
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
            Param.append(np.cumsum(self.SAMPLES[:,0:nbLayer-1],axis=1))
            for i in range(nbParamUnique):
                Param.append(self.SAMPLES[:,(i+1)*nbLayer-1:(i+2)*nbLayer-1])
            if TrueModel is not None:
                TrueMod = list()
                TrueMod.append(np.cumsum(TrueModel[0:nbLayer-1]))
                for i in range(nbParamUnique):
                    TrueMod.append(TrueModel[(i+1)*nbLayer-1:(i+2)*nbLayer-1])
                
            maxDepth = np.max(Param[0][:,-1])*1.25
            if RMSE:
                colormap = matplotlib.cm.get_cmap('jet')
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
        if nbParamUnique > 1:
            for ax in axes.flat:
                ax.label_outer()
        
        if RMSE:
            fig.subplots_adjust(bottom=0.25)
            ax_colorbar = fig.add_axes([0.15, 0.10, 0.70, 0.05])
            nb_inter = 1000
            color_for_scale = colormap(np.linspace(0,1,nb_inter,endpoint=True))
            cmap_scale = colors.ListedColormap(color_for_scale)
            scale = [stats.scoreatpercentile(RMS,a,limit=(np.min(RMS),np.max(RMS)),interpolation_method='lower') for a in np.linspace(0,100,nb_inter,endpoint=True)]
            norm = colors.BoundaryNorm(scale,len(color_for_scale))
            data = np.atleast_2d(np.linspace(np.min(RMS),np.max(RMS),nb_inter,endpoint=True))
            ax_colorbar.imshow(data, aspect='auto',cmap=cmap_scale,norm=norm)
            ax_colorbar.set_xlabel('Root Mean Square Error {}'.format(self.MODPARAM.paramNames["DataUnits"]),fontsize=12)
            ax_colorbar.yaxis.set_visible(False)
            nbTicks = 5
            ax_colorbar.set_xticks(ticks=np.linspace(0,nb_inter,nbTicks,endpoint=True))
            ax_colorbar.set_xticklabels(labels=round_to_5([stats.scoreatpercentile(RMS,a,limit=(np.min(RMS),np.max(RMS)),interpolation_method='lower') for a in np.linspace(0,100,nbTicks,endpoint=True)],n=5),rotation=30,ha='right')


        fig.suptitle("Posterior model visualization",fontsize=16)
        pyplot.show(block=False)
    
    def ShowDataset(self,RMSE:bool=False,Prior:bool=False,Best:int=None,Parallelization=[False, None]):
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
        '''
        from matplotlib import colors
        # Model the dataset (if not already done)
        if len(self.SAMPLESDATA)==0:
            print('Computing the forward model for the posterior!')
            self.DataPost(Parallelization=Parallelization)
        if RMSE:
            TrueData = self.DATA['True']
            RMS = np.sqrt(np.square(np.subtract(TrueData,self.SAMPLESDATA)).mean(axis=-1))
            quantiles = np.divide([stats.percentileofscore(RMS,a,'strict') for a in RMS],100)
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
        if RMSE:
            colormap = matplotlib.cm.get_cmap('jet')
            for j in sortIndex:
                ax.plot(self.MODPARAM.forwardFun["Axis"],np.squeeze(self.SAMPLESDATA[j,:len(self.MODPARAM.forwardFun["Axis"])]),color=colormap(quantiles[j]))
            ax.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["DataAxis"]),fontsize=14)
            ax.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["DataName"]),fontsize=14)
        else:
            for j in sortIndex:
                ax.plot(self.MODPARAM.forwardFun["Axis"],np.squeeze(self.SAMPLESDATA[j,:len(self.MODPARAM.forwardFun["Axis"])]),color='gray')
            ax.set_xlabel(r'${}$'.format(self.MODPARAM.paramNames["DataAxis"]),fontsize=14)
            ax.set_ylabel(r'${}$'.format(self.MODPARAM.paramNames["DataName"]),fontsize=14)
        if RMSE:
            fig.subplots_adjust(bottom=0.30)
            ax_colorbar = fig.add_axes([0.10, 0.15, 0.80, 0.05])
            nb_inter = 1000
            color_for_scale = colormap(np.linspace(0,1,nb_inter,endpoint=True))
            cmap_scale = colors.ListedColormap(color_for_scale)
            scale = [stats.scoreatpercentile(RMS,a,limit=(np.min(RMS),np.max(RMS)),interpolation_method='lower') for a in np.linspace(0,100,nb_inter,endpoint=True)]
            norm = colors.BoundaryNorm(scale,len(color_for_scale))
            data = np.atleast_2d(np.linspace(np.min(RMS),np.max(RMS),nb_inter,endpoint=True))
            ax_colorbar.imshow(data, aspect='auto',cmap=cmap_scale,norm=norm)
            ax_colorbar.set_xlabel('Root Mean Square Error {}'.format(self.MODPARAM.paramNames["DataUnits"]),fontsize=12)
            ax_colorbar.yaxis.set_visible(False)
            nbTicks = 5
            ax_colorbar.set_xticks(ticks=np.linspace(0,nb_inter,nbTicks,endpoint=True))
            ax_colorbar.set_xticklabels(labels=round_to_5([stats.scoreatpercentile(RMS,a,limit=(np.min(RMS),np.max(RMS)),interpolation_method='lower') for a in np.linspace(0,100,nbTicks,endpoint=True)],n=5),rotation=15,ha='center')
        pyplot.show(block=False)

    def GetStats(self):
        '''GETSTATS is a method that returns the means and standard deviations of the 
        parameters distributions.
        '''
        means = np.mean(self.SAMPLES,axis=0)
        stds = np.std(self.SAMPLES,axis=0)
        return means, stds
    
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
