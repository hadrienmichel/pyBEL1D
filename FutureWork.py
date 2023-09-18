'''In this script, we will produce preliminary results for the ideas that are 
emitted in the disrtation entiteled "A new Bayesian framework for the 
interpretation of geophysical data". 
Those ideas are:
1) Building a prior with a fixed, large number of layers
2) Propagating the posterior model space from close-by points
3) Providing insight on models for falsification
'''
from scipy import stats                         # To build the prior model space
def buildMODELSET_MASW():
    '''BUILDMODELSET is a function that will build the benchmark model.
    It does not take any arguments. '''
    # Values for the benchmark model parameters: 
    TrueModel1 = np.asarray([0.01, 0.05, 0.120, 0.280, 0.600])   # Thickness and Vs for the 3 layers (variable of the problem)
    TrueModel2 = np.asarray([0.0125, 0.0525, 0.120, 0.280, 0.600])
    Vp = np.asarray([0.300, 0.750, 1.5])                        # Vp for the 3 layers
    rho = np.asarray([1.5, 1.9, 2.2])                           # rho for the 3 layers
    nLayer = 3                                                  # Number of layers in the model
    Frequency = np.logspace(0.1,1.5,50)                         # Frequencies at which the signal is simulated
    Periods = np.divide(1,Frequency)                            # Corresponding periods
    # Forward modelling using surf96:
    Dataset1 = surf96(thickness=np.append(TrueModel1[0:nLayer-1], [0]),vp=Vp,vs=TrueModel1[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    Dataset2 = surf96(thickness=np.append(TrueModel2[0:nLayer-1], [0]),vp=Vp,vs=TrueModel2[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    # Building the noise model (Boaga et al., 2011)
    ErrorModelSynth = [0.075, 20]
    NoiseEstimate = np.asarray(np.divide(ErrorModelSynth[0]*Dataset1*1000 + np.divide(ErrorModelSynth[1],Frequency),1000)) # Standard deviation for all measurements in km/s
    RMSE_Noise = np.sqrt(np.square(NoiseEstimate).mean(axis=-1))
    print('The RMSE for the dataset with 1 times the standard deviation is: {} km/s'.format(RMSE_Noise))
    # Define the prior model space:
    # Find min and max Vp for each layer in the range of Poisson's ratio [0.2, 0.45]:
    # For Vp1=0.3, the roots are : 0.183712 and 0.0904534 -> Vs1 = [0.1, 0.18]
    # For Vp2=0.75, the roots are : 0.459279 and 0.226134 -> Vs2 = [0.25, 0.45]
    # For Vp3=1.5, the roots are : 0.918559 and 0.452267 -> Vs2 = [0.5, 0.9]
    prior = np.array([[0.001, 0.03, 0.1, 0.18],[0.01, 0.1, 0.25, 0.45],[0.0, 0.0, 0.5, 0.9]])# Thicknesses min and max, Vs min and max for each layers.
    # Defining names of the variables (for graphical outputs).
    nParam = 2 # e and Vs
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
    paramNames = {"NamesFU":NamesFullUnits, 
                    "NamesSU":NamesShortUnits, 
                    "NamesS":NamesShort,
                    "NamesGlobal":NFull,
                    "NamesGlobalS":["Depth\\ [km]", "Vs\\ [km/s]", "Vp\\ [km/s]", "\\rho\\ [T/m^3]"],
                    "DataUnits":"[km/s]",
                    "DataName":"Phase\\ velocity\\ [km/s]",
                    "DataAxis":"Periods\\ [s]"}
    # Defining the forward modelling function
    def funcSurf96(model):
        import numpy as np
        from pysurf96 import surf96
        Vp = np.asarray([0.300, 0.750, 1.5])    # Defined again inside the function for parallelization
        rho = np.asarray([1.5, 1.9, 2.2])       # Idem
        nLayer = 3                              # Idem
        Frequency = np.logspace(0.1,1.5,50)     # Idem
        Periods = np.divide(1,Frequency)        # Idem
        return surf96(thickness=np.append(model[0:nLayer-1], [0]),  # The 2 first values of the model are the thicknesses
                        vp=Vp,                                        # Fixed value for Vp
                        vs=model[nLayer-1:2*nLayer-1],                # The 3 last values of the model are the Vs
                        rho=rho,                                      # Fixed value for rho
                        periods=Periods,                              # Periods at which to compute the model
                        wave="rayleigh",                              # Type of wave to simulate
                        mode=1,                                       # Only compute the fundamental mode
                        velocity="phase",                             # Use phase velocity and not group velocity
                        flat_earth=True)                              # Local model where the flat-earth hypothesis makes sens

    forwardFun = funcSurf96
    forward = {"Fun":forwardFun,"Axis":Periods}
    # Building the function for conditions (here, just checks that a sampled model is inside the prior)
    cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
    # Initialize the model parameters for BEL1D
    ModelSynthetic = BEL1D.MODELSET(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)
    return TrueModel1, TrueModel2, Periods, Dataset1, Dataset2, NoiseEstimate, ModelSynthetic

if __name__ == '__main__':
    import numpy as np
    from pyBEL1D import BEL1D

    from pathos import multiprocessing as mp 
    from pathos import pools as pp 

    from matplotlib import pyplot as plt

    from pysurf96 import surf96                     # Code for the forward modelling of dispersion curves
    ### Parameters for the computation:
    RunFixedLayers = True
    RunPostPropag = False
    ParallelComputing = True
    RandomSeed = False

    if not(RandomSeed):
        np.random.seed(0) # For reproductibilty
        from random import seed
        seed(0)

    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count())# Create the parallel pool with at most the number of available CPU cores
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing

    '''1) Building a prior with fixed, large number of layers'''
    
    if RunFixedLayers:
        ### Building the synthetic benchmark:
        Kernel = "Data/sNMR/MRS2021.mrsk"
        Timing = np.arange(0.005, 0.5, 0.005)
        SyntheticBenchmarkSNMR = np.asarray([0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.15, 0.05, 0.05, 0.06, 0.07, 0.08, 0.12, 0.16, 0.20, 0.24, 0.25]) # 3-layers model
        ### Building the prior/forward model class (MODELSET)
        InitialModel = BEL1D.MODELSET.sNMR_logLayers(Kernel=Kernel, Timing=Timing, logUniform=False ,nbLayers=10, maxThick=10)
        ### Computing the model:
        DatasetBenchmark = InitialModel.forwardFun["Fun"](SyntheticBenchmarkSNMR)
        Noise = np.mean(DatasetBenchmark)/20
        print('The noise level is {} nV'.format(Noise))
        DatasetBenchmark += np.random.normal(scale=Noise, size=DatasetBenchmark.shape)
        ## Creating the BEL1D instances and IPR:
        Prebel, Postbel, PrebelInit , stats = BEL1D.IPR(MODEL=InitialModel, Dataset=DatasetBenchmark, NoiseEstimate=Noise*1e9, Parallelization=ppComp,
            nbModelsBase=10000, nbModelsSample=10000, stats=True, reduceModels=True, Mixing=(lambda x: 1), Graphs=False, saveIters=False, verbose=True)
        # Displaying the results:
        Postbel.ShowPostModels(TrueModel=SyntheticBenchmarkSNMR, RMSE=True, Parallelization=ppComp)
        plt.tight_layout()
        plt.savefig('./10LayersModel_Test.png',transparent=True, dpi=300)
        plt.show()

    '''2) Propagating the posterior model from space from close-by points'''
    if RunPostPropag:
        ### Defining the synthetic bechmarks:
        TrueModel1, TrueModel2, Periods, Dataset1, Dataset2, NoiseEstimate, ModelSynthetic = buildMODELSET_MASW()
        ### Creating the firts BEL1D instance:
        nbModelsBase = 1000
        def MixingFunc(iter:int) -> float:
            return 1# Always keeping the same proportion of models as the initial prior (see paper for argumentation).
        
        Prebel1, Postbel1, PrebelInit1, statsCompute1 = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset1,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,
                                                           nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True, Mixing=MixingFunc,
                                                           Graphs=False, TrueModel=TrueModel1, verbose=True, nbIterMax=10)

        Prebel2, Postbel2, PrebelInit2, statsCompute2 = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset2,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,
                                                           nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True, Mixing=MixingFunc,
                                                           Graphs=False, TrueModel=TrueModel1, verbose=True, PriorSampled=PrebelInit1.MODELS, nbIterMax=10)

        Postbel1.ShowPostModels(TrueModel=TrueModel1, RMSE=True, Parallelization=ppComp)
        fig = plt.gcf()
        ax = fig.get_axes()[0]
        ax.set_ylim(bottom=0.140, top=0.0)
        ax.set_xlim(left=0.0, right=0.8)
        ax.set_title('Initial Prior: Model 1', fontsize=16)
        plt.tight_layout()
        plt.savefig('./PostPropag_Model1_Init.png',transparent=True, dpi=300)
        Postbel2.ShowPostModels(TrueModel=TrueModel2, RMSE=True, Parallelization=ppComp)
        fig = plt.gcf()
        ax = fig.get_axes()[0]
        ax.set_ylim(bottom=0.140, top=0.0)
        ax.set_xlim(left=0.0, right=0.8)
        ax.set_title('Initial Prior: Model 2', fontsize=16)
        plt.tight_layout()
        plt.savefig('./PostPropag_Model2_Init.png',transparent=True, dpi=300)

        ### Creating a new instance with mixing of initial prior and posterior 1 form dataset 2:
        sharePost = 1/4
        ModelsPrior = PrebelInit1.MODELS[:int(PrebelInit1.nbModels*(1-sharePost)),:]
        ModelsPosterior = Postbel1.SAMPLES[:int(Postbel1.nbSamples*sharePost),:]
        MixedPrior = np.vstack((ModelsPrior, ModelsPosterior))
        Prebel2_bis, Postbel2_bis, PrebelInit2_bis, statsCompute2_bis = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset2,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,
                                                           nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True, Mixing=MixingFunc,
                                                           Graphs=False, TrueModel=TrueModel1, verbose=True, PriorSampled=MixedPrior, nbIterMax=10)

        Postbel2_bis.ShowPostModels(TrueModel=TrueModel2, RMSE=True, Parallelization=ppComp)
        fig = plt.gcf()
        ax = fig.get_axes()[0]
        ax.set_ylim(bottom=0.140, top=0.0)
        ax.set_xlim(left=0.0, right=0.8)
        ax.set_title('Propagated Posterior: Model 2', fontsize=16)
        plt.tight_layout()
        plt.savefig('./PostPropag_Model2_Propag.png',transparent=True, dpi=300)

        plt.show()
        
    if ParallelComputing:
        pool.terminate()