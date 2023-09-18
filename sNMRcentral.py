if __name__=="__main__": # To prevent recomputation when in parallel
    #########################################################################################
    ###           Import the different libraries that are used in the script              ###
    #########################################################################################
    ## Common libraries:
    import numpy as np                              # For matrix-like operations and storage
    import os                                       # For files structures and read/write operations
    from os import listdir                          # To retreive elements from a folder
    from os.path import isfile, join                # Common files operations
    from matplotlib import pyplot, colors                   # For graphics on post-processing
    import matplotlib
    pyplot.rcParams['font.size'] = 12
    pyplot.rcParams['figure.autolayout'] = True
    pyplot.rcParams['xtick.labelsize'] = 8
    pyplot.rcParams['ytick.labelsize'] = 8
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import time                                     # For simple timing measurements
    from copy import deepcopy

    ## Libraries for parallel computing:
    from pathos import multiprocessing as mp        # Multiprocessing utilities (get CPU cores info)
    from pathos import pools as pp                  # Building the pool to use for computations

    ## BEL1D requiered libraries:
    from scipy import stats                         # To build the prior model space
    from pyBEL1D import BEL1D                       # The main code for BEL1D
    from pyBEL1D.utilities import Tools             # For further post-processing
    from pyBEL1D.utilities.Tools import multiPngs   # For saving the figures as png

    from scipy.io import loadmat

    ### For reproductibility - Random seed fixed
    RandomSeed = False # If True, use true random seed, else (False), fixed for reproductibility
    if not(RandomSeed):
        np.random.seed(0) # For reproductibilty
        from random import seed
        seed(0)
    ### End random seed fixed

    #########################################################################################
    ###                            Initilizing the parallel pool                          ###
    #########################################################################################
    ParallelComputing = True
    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count()) # Create the parallel pool with at most the number of dimensions
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing

    modelTrue = np.asarray([2.5, 45.0, 0.25, 0.03, 0.01, 0.25, 0.07, 0.05])
    priorSNMR = np.array([[0.0, 10.0, 0.10, 0.40, 0.0, 0.400],
                          [20.0, 60.0, 0.0, 0.10, 0.0, 0.400],
                          [0.0, 0.0, 0.0, 0.10, 0.0, 0.400]])
    Timings = np.arange(0.005, 0.5, 0.001)
    stdUniform = lambda a,b: (b-a)/np.sqrt(12)
    stdTrue = np.asarray([stdUniform(priorSNMR[0,0],priorSNMR[0,1]), stdUniform(priorSNMR[0,2],priorSNMR[0,3]), stdUniform(priorSNMR[1,2],priorSNMR[1,3]), stdUniform(priorSNMR[0,4],priorSNMR[0,5]), stdUniform(priorSNMR[1,4],priorSNMR[1,5])])
    KernelRx50 = "MultiCentral/Tx50L1_Rx50L1.mrsk"
    KernelRx25 = "MultiCentral/Tx50L1_Rx25L2.mrsk"
    KernelRx10 = "MultiCentral/Tx50L1_Rx10L4.mrsk"
    KernelMultiCentral = "MultiCentral/Tx50L1_Rx50Rx25Rx10.mrsk"
    Kernels = [KernelRx50, KernelRx25, KernelRx10]
    KSizes = [50, 25, 10]
    Kareas = [np.pi*(d/2)**2 for d in KSizes]
    tmp = [loadmat(kern, struct_as_record=False, squeeze_me=True)['kdata'] for kern in Kernels]
    Kernels = tmp
    
    ###### Present the kernels:
    # titles = ['Tx50/Rx50', 'Tx50/Rx25', 'Tx50/Rx10']
    # fig, ax = pyplot.subplots(nrows=1, ncols=3, sharey=True, figsize=(10,5))
    # i = 0
    # X, Y = np.meshgrid(Kernels[0].measure.pm_vec , Kernels[0].model.z)
    # im = []
    # for kern in Kernels:
    #     im.append(ax[i].contourf(X, Y, np.abs(kern.K.transpose()/Kareas[i]), cmap='jet'))
    #     ax[i].plot(np.sum(np.abs(kern.K.transpose()), axis=1)*(10/np.max(np.sum(np.abs(kern.K.transpose()), axis=1))), Kernels[0].model.z, ':w', linewidth=2.5)
    #     ax[i].invert_yaxis()
    #     ax[i].set_xlabel('Pulse moment (As)')
    #     ax[i].grid()
    #     ax[i].set_title(titles[i])
    #     i += 1
    # ax[0].set_ylabel('Depth (m)')
    # pyplot.show(block=True)

    ### Classical case:
    Coincident = BEL1D.MODELSET().SNMR(prior=priorSNMR, Kernel=KernelRx50, Timing=Timings)
    PrebelC = BEL1D.PREBEL(Coincident, nbModels=10000)
    PrebelC.run(Parallelization=ppComp)
    # Simulate the dataset:
    DataSetC = PrebelC.MODPARAM.forwardFun["Fun"](modelTrue)
    # Postbel run:
    PostbelC = BEL1D.POSTBEL(PrebelC)
    PostbelC.run(Dataset=DataSetC, nbSamples=10000, NoiseModel=5)
    # Show results:
    # PostbelC.ShowPostCorr(TrueModel=modelTrue, OtherMethod=PrebelC.MODELS)
    
    ### Multi-central case:
    MultiCentral = BEL1D.MODELSET().SNMR(prior=priorSNMR, Kernel=KernelMultiCentral, Timing=Timings)
    PrebelM = BEL1D.PREBEL(MultiCentral, nbModels=10000)
    PrebelM.run(Parallelization=ppComp)
    # Simulate the dataset:
    DataSetM = PrebelM.MODPARAM.forwardFun["Fun"](modelTrue)
    # Postbel run:
    PostbelM = BEL1D.POSTBEL(PrebelM)
    PostbelM.run(Dataset=DataSetM, nbSamples=10000, NoiseModel=5)
    # Show results:
    # PostbelM.ShowPostCorr(TrueModel=modelTrue, OtherMethod=PrebelM.MODELS)

    PostbelM.ShowPostCorr(TrueModel=modelTrue, OtherMethod=PostbelC.SAMPLES, alpha=[0.05,0.05])
    PostbelM.ShowPost(prior=True, priorOther=PostbelC.SAMPLES, TrueModel=modelTrue)

    pyplot.show(block=True)



