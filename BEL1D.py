# Importing custom libraries
from utilities import Tools
from utilities.KernelDensity import KDE
from utilities import ForwardModelling
#Importing common libraries
import numpy as np 
import math as mt 
import matplotlib
import sklearn
from scipy import stats

class MODELSET:

    def __init__(self,prior=None,method=None,forwardFun=None):
        if (prior is None) or (method is None) or (forwardFun is None):
            self.prior = []
            self.method = []
            self.forwardFun = []
        else:
            self.prior = prior
            self.method = method
            self.forwardFun = forwardFun
    
    @classmethod
    def SNMR(cls):
        prior = [stats.uniform(loc=20,scale=50), stats.uniform(loc=10,scale=30)]
        method = "SNMR"
        forwardFun = lambda model,parameters: model*5
        return cls(prior=prior,method=method,forwardFun=forwardFun)


class PREBEL:
    """Object that is used to store the PREBEL elements:
    
    For a given model set (see MODELSET class), the PREBEL class
    enables all the computations taht takes place previous to any 
    field data knowledge. It takes as argument:
        - method (str): the name of the geophysical method
        - prior (list of stats): the prior description
        - nbModels (int): the number of models to sample in the prior
    """
    def __init__(self,method='SNMR',prior=None,nbModels=None):
        # PRIOR: a list of scipy.stats distributions describing 
        # the prior model space for all the parameters
        self.PRIOR = prior
        # CONDITIONS: a list of lambda functions that are imbeded in a cond class (custom)
        # The compute method of the class must return a list of boolean values
        self.CONDITIONS = None
        # nbModels: the number of sampled models
        self.nbModels = nbModels
        # MODPARAM: a dictionnary with
        #   - 'method': the name of the geophysical method ('sNMR' or 'DC')
        #   - 'forward': a dictionnary containing the elements for the forward model 
        #                (kernel, timings, frequencies, etc.)
        #   - 'parmNames': a dictionnary with the instances of names for the parameters 
        #                  (full = full name with untis, units = reduced name with units 
        #                  and nounits = reduced name) contained in Lists
        self.MODPARAM = {'method':method,'forward':dict(),'paramNames':dict()}
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

    def run(self):
        """The RUN method runs all the computations for the preparation of BEL1D

        It is an instance method that does not need any arguments.
        """
        # 1) Sampling (if not done already):
        if self.nbModels is None:
            self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS)
            self.nbModels = 1000
        else:
            self.MODELS = Tools.Sampling(self.PRIOR,self.CONDITIONS,self.nbModels)
        # 2) Running the forward model
        geophMethod = self.MODPARAM['method']
        if geophMethod is 'SNMR':
            self.FORWARD = ForwardModelling.SNMR(self.MODELS,self.MODPARAM['forward'])
        elif geophMethod is 'DC':
            self.FORWARD = ForwardModelling.DC(self.MODELS,self.MODPARAM['forward'])
        else:
            self.FORWARD = ForwardModelling.Pendulum(self.MODELS,self.MODPARAM['forward'])
        # 3) PCA on data (and optionally model):
        reduceModels = True
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=0.9) # Keeping 90% of the variance
            m_h = pca_model.fit_transform(self.MODELS)
            n_CompPCA_Mod = m_h.size
            n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)
            d_h = pca_data.fit_transform(self.FORWARD)
            self.PCA = {'Data':pca_data,'Model':pca_model}
        else:
            m_h = self.MODELS - np.mean(self.MODELS,axis=0)
            n_CompPCA_Mod = m_h.size
            n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)
            d_h = pca_data.fit_transform(self.FORWARD)
            self.PCA = {'Data':pca_data,'Model':None}
        # 4) CCA:
        cca_transform = sklearn.cross_decomposition.CCA(n_components=n_CompPCA_Mod)
        d_c,m_c = cca_transform.fit_transform(d_h,m_h)
        self.CCA = cca_transform
        # 5) KDE:
        self.KDE = KDE(d_c,m_c)
        self.KDE.KernelDensity()
        
class POSTBEL:
    """Object that is used to store the POSTBEL elements:
    
    For a given PREBEL set (see PREBEL class), the POSTBEL class
    enables all the computations that takes place aftre the  
    field data acquisition. It takes as argument:
        - PREBEL (PREBEL class object): the PREBEL object from 
                                        PREBEL class
    """
    def __init__(self,PREBEL:PREBEL):
        self.KDE = PREBEL.KDE
        self.PCA = PREBEL.PCA
        self.CCA = PREBEL.CCA
        self.DATA = dict()
        self.SAMPLES = dict()

    def run(self,Dataset,nbSamples=1000,Graphs=False,NoiseModel=None):
        # Transform dataset to CCA space:
        d_obs_h = self.PCA['Data'].transform(Dataset)
        d_obs_c = self.CCA.transform(d_obs_h)
        self.DATA = {'True':Dataset,'PCA':d_obs_h,'CCA':d_obs_c}
        # Propagate Noise:
        if NoiseModel is not None:
            Noise = Tools.PropagateNoise(self,NoiseModel)
        else:
            Noise = None
        # Obtain corresponding distribution (KDE)
        self.KDE.GetDist(Xvals=d_obs_c,Noise=Noise)
        if Graphs:
            self.KDE.ShowKDE(Xvals=d_obs_c)
        # Sample models:
        samples_CCA = self.KDE.SampleKDE(nbSample=1000)
        # Back transform models to original space:
        samples_PCA = np.matmul(samples_CCA,self.CCA.y_loadings_.T)
        samples_PCA *= self.CCA.y_std_
        samples_PCA += self.CCA.y_mean_
        samples_PCA = self.CCA.inverse_transform(samples_CCA)
        if self.PCA['Model'] is None:
            samples_Init = samples_PCA 
        else:
            samples_Init = self.PCA['Model'].inverse_transform(samples_PCA)
        self.SAMPLES = samples_Init

    