# Importing custom libraries
from utilities import Tools
from utilities.KernelDensity import KDE
from utilities import ForwardModelling
#Importing common libraries
import numpy as np 
import math as mt 
import matplotlib
from matplotlib import pyplot
import sklearn
from sklearn import decomposition, cross_decomposition
from scipy import stats

# TODO:
#   - Add conditions (function for checking that samples are within a given space)
#   - Add Noise propagation
#   - Add DC example
#   - Add posprocessing

class MODELSET:

    def __init__(self,prior=None,cond=None,method=None,forwardFun=None,paramNames=None):
        if (prior is None) or (method is None) or (forwardFun is None) or (paramNames is None):
            self.prior = []
            self.method = []
            self.forwardFun = []
            self.paramNames = []
            self.cond = cond
        else:
            self.prior = prior
            self.method = method
            self.forwardFun = forwardFun
            self.cond = cond
            self.paramNames = paramNames
    
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
        """
        from pygimli.physics import sNMR
        import numpy.matlib
        if prior is None:
            prior = np.array([[2.5, 7.5, 3.5, 10, 5, 350], [0, 0, 10, 30, 5, 350]])
            Kernel = "Data/sNMR/KernelTest.mrsk"
            Timing = np.arange(0.005,0.5,0.001)
        nLayer, nParam = prior.shape
        nParam /= 2
        nParam = int(nParam)
        prior = np.multiply(prior,np.matlib.repmat(np.array([1, 1, 1/100, 1/100, 1/1000, 1/1000]),nLayer,1))
        ListPrior = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesFullUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShort = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        Units = [" [m]", " [/]", " [s]"]
        NFull = ["Thickness","Water Content","Relaxation Time"]
        NShort = ["e_{", "W_{", "T_{2,"]
        ident = 0
        for j in range(nParam):
            for i in range(nLayer):
                if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                    NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                    NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                    NamesShort[ident] = NShort[j] + str(i+1) + "}"
                    ident += 1
        method = "SNMR"
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["e_i", "W_i", "T_{2,i}"]}
        KFile = sNMR.MRS()
        KFile.loadKernel(Kernel)
        ModellingMethod = sNMR.MRS1dBlockQTModelling(nlay=nLayer,K=KFile.K,zvec=KFile.z,t=Timing)
        forwardFun = lambda model: ModellingMethod.response(model)
        cond = None
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forwardFun,paramNames=paramNames)


class PREBEL:
    """Object that is used to store the PREBEL elements:
    
    For a given model set (see MODELSET class), the PREBEL class
    enables all the computations taht takes place previous to any 
    field data knowledge. It takes as argument:
        - method (str): the name of the geophysical method
        - prior (list of stats): the prior description
        - nbModels (int): the number of models to sample in the prior
    """
    def __init__(self,MODPARAM:MODELSET,nbModels=1000):
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
        tmp = self.MODPARAM.forwardFun(self.MODELS[0,:])
        self.FORWARD = np.zeros((self.nbModels,len(tmp)))
        for i in range(self.nbModels):
            self.FORWARD[i,:] = self.MODPARAM.forwardFun(self.MODELS[i,:])
        # 3) PCA on data (and optionally model):
        reduceModels = False
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=0.9) # Keeping 90% of the variance
            m_h = pca_model.fit_transform(self.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            # n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)
            d_h = pca_data.fit_transform(self.FORWARD)
            self.PCA = {'Data':pca_data,'Model':pca_model}
        else:
            m_h = self.MODELS # - np.mean(self.MODELS,axis=0)
            n_CompPCA_Mod = m_h.shape[1]
            #n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
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
        self.MODPARAM = PREBEL.MODPARAM
        self.DATA = dict()
        self.SAMPLES = []

    def run(self,Dataset,nbSamples=1000,Graphs=False,NoiseModel=None):
        # Transform dataset to CCA space:
        Dataset = np.reshape(Dataset,(1,-1))# Convert for reverse transform
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
        # samples_PCA = self.CCA.inverse_transform(samples_CCA)
        if self.PCA['Model'] is None:
            samples_Init = samples_PCA 
        else:
            samples_Init = self.PCA['Model'].inverse_transform(samples_PCA)
        self.SAMPLES = samples_Init

    def ShowPost(self,TrueModel=None):
        nbParam = self.SAMPLES.shape[1]
        if (TrueModel is not None) and (len(TrueModel)!=nbParam):
            TrueModel = None
        for i in range(nbParam):
            fig, ax = pyplot.subplots()
            ax.hist(self.SAMPLES[:,i])
            ax.set_title("Posterior histogram")
            ax.set_xlabel(self.MODPARAM.paramNames["NamesFU"][i])
            if TrueModel is not None:
                ax.plot([TrueModel[i],TrueModel[i]],np.asarray(ax.get_ylim()),'r')
            pyplot.show(block=False)
        pyplot.show()
    