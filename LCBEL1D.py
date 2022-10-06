'''
In this script, we will try to implement BEL1D for the interpretation of profiles (2D) or even full 3D models
based on the usage of multiple 1D models. 
The idea is to use information from previously solved profiles and to propaget the learned information to adjacent models.

We will investigate the impact of the departure point, the direction of the operation, etc. on a synthetic 2D profile.

Later, we will apply the same tehchnique to a field dataset (E-Test?)
'''
from matplotlib import pyplot


if __name__ == '__main__':
    # Importing common libraries:
    import numpy as np
    from pysurf96 import surf96 # For the forward modelling
    from pyBEL1D import BEL1D

    from scipy.stats import pearsonr
    import pandas as pd

    ## Libraries for parallel computing:
    from pathos import multiprocessing as mp        # Multiprocessing utilities (get CPU cores info)
    from pathos import pools as pp                  # Building the pool to use for computations

    RandomSeed = False
    ParallelComputing = True
    runBEL1D = True
    # Description of the synthetic model:
    '''The synthetic model will contain a simple geology with dipping layers and a fault.
    Therefore, the algorithm should be efficient to transfer information from one profile
    to the other when the fault is not present, but the fault will result in a transition 
    that is not smooth with adjacent models and thus the behaviour is unpredictable.'''
    dippingAngle = 5  # Dipping angle compared to the surface [degree] (in the anti-trigonometric direction)
    hasFault = True     # Define if the model has a fault in it
    faultCrossing = 1.200 # Position of the crossing between the fault and the surface [m]
    faultAngle = 10    # Dipping angle of the fault plane compared to the surface [degree] (trigonometric direction)
    faultDisp = -0.030      # Vertical displacement of the fault. If positive, Normal fault, otherwise, Reverse fault
    nbLayers = 5        # Numbers of layers in the model
    layersDepthAt0 = np.asarray([-0.010, 0.020, 0.070, 0.080, 100.000]) # The last layer is a half-space (100000 is not taken into account)
    layersVs = np.asarray([0.500, 0.900, 1.500, 2.000, 2.500])
    PoissonRatio = 0.25
    RhoTypical = 2.0
    ErrorModelSynth = [0.075, 20]
    # This set of parameters works for the forward modelling of all the informations!
    # I cannot warant the fact that it will work with any set of parameters!

    def findModel(position):
        '''findModel gives the 1D model for a given position along the profile given 
        the data above.
        '''
        modelLeftFault = layersDepthAt0 + np.tan(2*np.pi/360 * dippingAngle) * position
        if hasFault:
            depthOfFault = (position - faultCrossing)/np.tan(2*np.pi/360 * faultAngle)
            if depthOfFault > 0: # The fault impacts our model
                if faultAngle >= 0: # Hangging wall to the right of fault
                    modelTopFault = modelLeftFault + faultDisp # If faultDisp is positive, Normal fault, otherwise Reverse fault
                    modelBottomFault = modelLeftFault
                else: # Hangging wall to the left of the fault
                    modelBottomFault = modelLeftFault - faultDisp
                    modelTopFault = modelLeftFault
                nbLayerFaultTop = np.sum(modelTopFault < depthOfFault)
                nbLayersFaultBottom = np.sum(modelBottomFault < depthOfFault)
                if nbLayersFaultBottom != nbLayerFaultTop: # The fault connects different layers (visible in profile)
                    nbLayersTop = np.sum(modelTopFault < depthOfFault)
                    nbLayersTopAbove = np.sum(modelTopFault > 0)-1
                    modelVs = np.hstack((layersVs[nbLayersTopAbove:nbLayersTop+1], layersVs[modelBottomFault > depthOfFault]))
                    modelTopFault = modelTopFault[np.logical_and((modelTopFault > 0), (modelTopFault < depthOfFault))]
                    modelBottomFault = modelBottomFault[modelBottomFault > depthOfFault]
                    modelThick = np.diff(np.hstack(([0], modelTopFault, [depthOfFault], modelBottomFault)))[:-1]
                    modelNbLayers = len(modelThick) + 1
                else: # The fault connects the same layer (not visible in profile)
                    modelVs = np.hstack((layersVs[np.logical_and((modelTopFault > 0), (modelTopFault < depthOfFault))], layersVs[modelBottomFault > depthOfFault]))
                    modelTopFault = modelTopFault[np.logical_and((modelTopFault > 0), (modelTopFault < depthOfFault))]
                    modelBottomFault = modelBottomFault[modelBottomFault > depthOfFault]
                    modelThick = np.diff(np.hstack(([0], modelTopFault, modelBottomFault)))[:-1]
                    modelNbLayers = len(modelThick) + 1
            else:
                modelVs = layersVs[modelLeftFault > 0]
                modelNbLayers = len(modelVs)
                modelLeftFault = np.hstack(([0], modelLeftFault[modelLeftFault > 0]))
                modelThick = np.diff(modelLeftFault)[:-1]
        else:
            modelVs = layersVs[modelLeftFault > 0]
            modelNbLayers = len(modelVs)
            modelLeftFault = np.hstack(([0], modelLeftFault[modelLeftFault > 0]))
            modelThick = np.diff(modelLeftFault)[:-1]
        return modelNbLayers, modelThick, modelVs

    '''
    We build the model with one model every 10 meters along a 200m profile.
    '''
    Frequency = np.logspace(np.log10(1),np.log10(100),60)
    Periods = np.divide(1,Frequency)
    '''Defining the datasets'''
    Positions = np.arange(0.0, 2.00001, 0.01)
    Datasets = []
    Noises = []
    trueModels = []
    nbLayers = []
    for pos in Positions:
        modelNbLayers, modelThick, modelVs = findModel(pos)
        trueModels.append(np.hstack((modelThick, modelVs)))
        nbLayers.append(modelNbLayers)
        print(f'Model at {pos} [km]:\n\t-Thickness [km]:{modelThick}\n\t-Vs [km/s]:{modelVs}\n\t-Vp [km/s]:{np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(modelVs,2))}')
        data = None
        try: 
            data = surf96(thickness=np.append(modelThick, [0]),vp=np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(modelVs,2)),vs=modelVs,rho=np.ones((modelNbLayers,))*RhoTypical,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
            print(f'\t-Dataset [km/s]: {data}')
        except:
            print(f'\t-Dataset [km/s]: UNABLE TO COMPUTE')
        Datasets.append(data)
        if data is not None:
            NoiseEstimate = np.asarray(np.divide(ErrorModelSynth[0]*data*1000 + np.divide(ErrorModelSynth[1],Frequency),1000)) # Standard deviation for all measurements in km/s
            Noises.append(NoiseEstimate)
        else:
            Noises.append(None)

    data = {
        'Position': Positions,
        'nbLayers': nbLayers,
        'TrueModel': trueModels,
        'Dataset': Datasets,
        'Noise': Noises}
    
    df = pd.DataFrame(data)
    
    correlation = np.ones_like(Positions)
    fig, ax = pyplot.subplots()
    i = 0
    mask = []
    for data in Datasets:
        if data is None:
            correlation[i] = 0
            mask.append(1)
        else:
            mask.append(0)
            ax.plot(Periods, data)
            if i > 0:
                dataPrevious = Datasets[i-1]
                if dataPrevious is not None:
                    correlation[i], _ = pearsonr(data, dataPrevious)
                else:
                    correlation[i] = 0
        i += 1
    ax.set_xlabel('Periods [s]')
    ax.set_ylabel('Phase velocity [km/s]')
    
    fig, ax = pyplot.subplots()
    ax.plot(Positions, correlation*100)
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Correlation [%]')

    if sum(mask) == 0:
        corrMatrix = np.ma.corrcoef(Datasets)
        X, Y = np.meshgrid(Positions, Positions)
        fig, ax = pyplot.subplots()
        im = ax.contourf(X, Y, corrMatrix, vmin=0, vmax=1)
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Distance [km]')
        cb = fig.colorbar(im)

    pyplot.show(block=True)

    if runBEL1D:
        ### For reproductibility - Random seed fixed
        if not(RandomSeed):
            np.random.seed(0) # For reproductibilty
            from random import seed
            seed(0)
        ### End random seed fixed
        '''Defining the prior model space:'''
        # The prior has 6 layers, all with the same prior model space
        # Thicknesses are between 0.5 m and 100 m
        # Vs are between 200 m/s and 3000 m/s
        prior = np.repeat(np.asarray([[0.0005, 0.1, 0.2, 3.0]]),10,axis=0)
        InitialModel = BEL1D.MODELSET.DCVs_logLayers(Frequency = Frequency, nbLayers=50)

        #########################################################################################
        ###                            Initilizing the parallel pool                          ###
        #########################################################################################
        if ParallelComputing:
            pool = pp.ProcessPool(mp.cpu_count()) # Create the parallel pool with at most the number of dimensions
            ppComp = [True, pool]
        else:
            ppComp = [False, None] # No parallel computing
        def MixingFunc(iter:int) -> float:
            return 1# Always keeping the same proportion of models as the initial prior (see paper for argumentation).
        Prebel, Postbel, PrebelInit, statsCompute = BEL1D.IPR(MODEL=InitialModel,Dataset=data,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,
                                                                nbModelsBase=10000000,nbModelsSample=100000, reduceModels=False, stats=True, Mixing=MixingFunc,
                                                                Graphs=False, verbose=True, nbIterMax=1) # , TrueModel=np.hstack((modelThick, modelVs))
        
        Models, Datasets = Postbel.runRejection(NoiseModel=NoiseEstimate, Parallelization=ppComp)
        Postbel.ShowPostModels(Parallelization=ppComp, OtherModels=Models, OtherData=Datasets, OtherRMSE=True, RMSE=True, NoiseModel=NoiseEstimate)#, TrueModel=np.hstack((modelThick, modelVs)))
        # Postbel.ShowPostCorr(OtherMethod=PrebelInit.MODELS, OtherModels=Models, alpha=[0.01, 0.1])#, TrueModel=np.hstack((modelThick, modelVs)))
        print(f'Model at {pos} [km]:\n\t-Thickness [km]:{modelThick}\n\t-Vs [km/s]:{modelVs}\n\t-Vp [km/s]:{np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(modelVs,2))}')
        
        ####
        ## Second localization (100 meters further)
        dataOK = False # To be changed latter
        while dataOK:
            pos += 0.025
            modelNbLayers, modelThick, modelVs = findModel(pos)
            print(f'Model at {pos} [km]:\n\t-Thickness [km]:{modelThick}\n\t-Vs [km/s]:{modelVs}\n\t-Vp [km/s]:{np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(modelVs,2))}')
            try: 
                data = surf96(thickness=np.append(modelThick, [0]),vp=np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(modelVs,2)),vs=modelVs,rho=np.ones((modelNbLayers,))*RhoTypical,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
                print(f'\t-Dataset [km/s]: {data}')
            except:
                dataOK = False
                print(f'\t-Dataset [km/s]: UNABLE TO COMPUTE')

            ErrorModelSynth = [0.075, 20]
            NoiseEstimate = np.asarray(np.divide(ErrorModelSynth[0]*data*1000 + np.divide(ErrorModelSynth[1],Frequency),1000)) # Standard deviation for all measurements in km/s

            Prebel = BEL1D.PREBEL.POSTBEL2PREBEL(Prebel, Postbel, Dataset=data, reduceModels=True, Parallelization=ppComp)
            Postbel = BEL1D.POSTBEL(Prebel)
            Postbel.run(Dataset=data, nbSamples=Prebel.nbModels, NoiseModel=NoiseEstimate, verbose=True)
            Models, Datasets = Postbel.runRejection(NoiseModel=NoiseEstimate, Parallelization=ppComp)
            Postbel.ShowPostModels(Parallelization=ppComp, OtherModels=Models, OtherData=Datasets, OtherRMSE=True, RMSE=True, NoiseModel=NoiseEstimate)#, TrueModel=np.hstack((modelThick, modelVs)))
            Postbel.ShowPostCorr(OtherMethod=PrebelInit.MODELS, OtherModels=Models, alpha=[0.01, 0.1])#, TrueModel=np.hstack((modelThick, modelVs)))
            print(f'Model at {pos} [km]:\n\t-Thickness [km]:{modelThick}\n\t-Vs [km/s]:{modelVs}\n\t-Vp [km/s]:{np.sqrt((2*PoissonRatio-2)/(2*PoissonRatio-1) * np.power(modelVs,2))}')

        # BEL1D.pyplot.show()

        if ParallelComputing:
            pool.terminate()