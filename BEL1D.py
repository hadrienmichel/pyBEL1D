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
import multiprocessing as mp 


# TODO/DONE:
#   - (Done on 24/04/2020) Add conditions (function for checking that samples are within a given space)
#   - (Done on 13/04/2020) Add Noise propagation (work in progress 29/04/20 - OK for SNMR 30/04/20 - DC OK) -> Noise impact is always very low???
#   - (Done on 11/05/2020) Add DC example (uses pysurf96 for forward modelling: https://github.com/miili/pysurf96 - compiled with msys2 for python)
#   - Add postprocessing (partially done - need for models viewer)
#   - (Done on 12/05/2020) Speed up kernel density estimator (vecotization?) - result: speed x4
#   - (Done on 13/05/2020) Add support for iterations
#   - Add iteration convergence critereon!
#   - Lower the memory needs (how? not urgent)
#   - Comment the codes!

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
        NFull = ["Thickness","Water Content","Relaxation Time"]
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
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["e_i", "W_i", "T_{2,i}"]}
        KFile = sNMR.MRS()
        KFile.loadKernel(Kernel)
        ModellingMethod = sNMR.MRS1dBlockQTModelling(nlay=nLayer,K=KFile.K,zvec=KFile.z,t=Timing)
        forwardFun = lambda model: ModellingMethod.response(model)
        forward = {"Fun":forwardFun,"Axis":Timing}
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames)

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
        Units = [" [m]", " [km/s]", " [km/s]", " [T/m^3]"]
        NFull = ["Thickness","s-Wave velocity","p-Wave velocity", "Density"]
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
        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits, "NamesS":NamesShort, "NamesGlobal":NFull, "NamesGlobalS":["e_i", "Vs_i", "Vp_i", "\\rho_i"]}
        forwardFun = lambda model: surf96(thickness=np.squeeze([model[0:nLayer-1], [0]]),vp=model[2*nLayer-1:3*nLayer-1],vs=model[nLayer-1:2*nLayer-1],rho=model[3*nLayer-1:4*nLayer-1],periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        forward = {"Fun":forwardFun,"Axis":Periods}
        def PoissonRatio(model):
            vp=model[2*nLayer-1:3*nLayer-1]
            vs=model[nLayer-1:2*nLayer-1]
            ratio = 1/2 * (np.power(vp,2) - 2*np.power(vs,2))/(np.power(vp,2)-np.power(vs,2))
            return ratio
        RatioMin = [0.1]*nLayer
        RatioMax = [0.5]*nLayer
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all() and (np.logical_and(np.greater_equal(PoissonRatio(model),RatioMin),np.less_equal(PoissonRatio(model),RatioMax))).all()
        return cls(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames)

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
        # For DC, sometimes, the code will return an error --> need to remove the model from the prior
        indexCurr = 0
        while True:
            try:
                tmp = self.MODPARAM.forwardFun["Fun"](self.MODELS[indexCurr,:])
                break
            except:
                indexCurr += 1
        self.FORWARD = np.zeros((self.nbModels,len(tmp)))
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
        print('{} models remaining after forward modelling!'.format(newModelsNb))
        self.nbModels = newModelsNb
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
    
    @classmethod
    def POSTBEL2PREBEL(cls,POSTBEL):
        # 1) Initialize the Prebel class object
        Modelset = POSTBEL.MODPARAM # A MODELSET class object
        PrebelNew = cls(Modelset)
        # 2) Inject the samples from postbel
        PrebelNew.MODELS = POSTBEL.SAMPLES
        PrebelNew.nbModels = np.size(POSTBEL.SAMPLES,axis=0) # Get the number of sampled models
        # 2) Running the forward model
        # For DC, sometimes, the code will return an error --> need to remove the model from the prior
        indexCurr = 0
        while True:
            try:
                tmp = PrebelNew.MODPARAM.forwardFun["Fun"](PrebelNew.MODELS[indexCurr,:])
                break
            except:
                indexCurr += 1
        PrebelNew.FORWARD = np.zeros((PrebelNew.nbModels,len(tmp)))
        notComputed = []
        for i in range(PrebelNew.nbModels):
            # print(i)
            try:
                PrebelNew.FORWARD[i,:] = PrebelNew.MODPARAM.forwardFun["Fun"](PrebelNew.MODELS[i,:])
            except:
                PrebelNew.FORWARD[i,:] = [None]*len(tmp)
                notComputed.append(i)
        # Getting the uncomputed models and removing them:
        PrebelNew.MODELS = np.delete(PrebelNew.MODELS,notComputed,0)
        PrebelNew.FORWARD = np.delete(PrebelNew.FORWARD,notComputed,0)
        newModelsNb = np.size(PrebelNew.MODELS,axis=0) # Get the number of models remaining
        print('{} models remaining after forward modelling!'.format(newModelsNb))
        PrebelNew.nbModels = newModelsNb
        # 3) PCA on data (and optionally model):
        reduceModels = False
        if reduceModels:
            pca_model = sklearn.decomposition.PCA(n_components=0.9) # Keeping 90% of the variance
            m_h = pca_model.fit_transform(PrebelNew.MODELS)
            n_CompPCA_Mod = m_h.shape[1]
            # n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)
            d_h = pca_data.fit_transform(PrebelNew.FORWARD)
            PrebelNew.PCA = {'Data':pca_data,'Model':pca_model}
        else:
            m_h = PrebelNew.MODELS # - np.mean(self.MODELS,axis=0)
            n_CompPCA_Mod = m_h.shape[1]
            #n_CompPCA_Mod = n_CompPCA_Mod[1] # Second dimension is the number of components
            pca_data = sklearn.decomposition.PCA(n_components=n_CompPCA_Mod)
            d_h = pca_data.fit_transform(PrebelNew.FORWARD)
            PrebelNew.PCA = {'Data':pca_data,'Model':None}
        # 4) CCA:
        cca_transform = sklearn.cross_decomposition.CCA(n_components=n_CompPCA_Mod)
        d_c,m_c = cca_transform.fit_transform(d_h,m_h)
        PrebelNew.CCA = cca_transform
        # 5) KDE:
        PrebelNew.KDE = KDE(d_c,m_c)
        PrebelNew.KDE.KernelDensity()
        return PrebelNew
        
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
        self.FORWARD_PRIOR = PREBEL.FORWARD
        self.KDE = PREBEL.KDE
        self.PCA = PREBEL.PCA
        self.CCA = PREBEL.CCA
        self.MODPARAM = PREBEL.MODPARAM
        self.DATA = dict()
        self.SAMPLES = []
        self.SAMPLESDATA = []

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

    def DataPost(self):
        indexCurr = 0
        while True:
            try:
                tmp = self.MODPARAM.forwardFun["Fun"](self.SAMPLES[indexCurr,:])
                break
            except:
                indexCurr += 1
        self.SAMPLESDATA = np.zeros((self.nbModels,len(tmp)))
        notComputed = []
        for i in range(self.nbModels):
            # print(i)
            try:
                self.SAMPLESDATA[i,:] = self.MODPARAM.forwardFun["Fun"](self.SAMPLES[i,:])
            except:
                self.SAMPLESDATA[i,:] = [None]*len(tmp)
                notComputed.append(i)
        # Getting the uncomputed models and removing them:
        self.SAMPLES = np.delete(self.SAMPLES,notComputed,0)
        self.SAMPLESDATA = np.delete(self.SAMPLESDATA,notComputed,0)
        newModelsNb = np.size(self.SAMPLES,axis=0) # Get the number of models remaining
        print('{} models remaining after forward modelling!'.format(newModelsNb))
        self.nbModels = newModelsNb
        return self.SAMPLESDATA

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
    
    def ShowPostCorr(self,TrueModel=None,OtherMethod=None):
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
                    axs[i,j].hist(self.SAMPLES[:,j]) # Plot the histogram for the given variable
                    if TrueModel is not None:
                        axs[i,j].plot([TrueModel[i],TrueModel[i]],np.asarray(axs[i,j].get_ylim()),'r')
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
                else:
                    axs[i,j].set_visible(False)
                if j == 0: # First column of the graph
                    if not(i==j):
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
        fig.suptitle("Posterior model space visualtization")
        for ax in axs.flat:
            ax.label_outer()
        pyplot.show()

    def GetStats(self):
        means = np.mean(self.SAMPLES,axis=0)
        stds = np.std(self.SAMPLES,axis=0)
        return means, stds
    
