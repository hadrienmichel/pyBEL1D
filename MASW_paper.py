'''In this script, all the operations that are presented in the paper called 
"Using Iterative Prior Resampling to improve Bayesian Evidential Learning 1D 
imaging (BEL1D) accuracy: the case of surface waves" are performed and explained.

The different graphs that are originating from the python script are also 
outputted here.
'''
if __name__=="__main__": # To prevent recomputation when in parallel

    from pyBEL1D import BEL1D
    import cProfile # For debugging and timings measurements
    import time # For simple timing measurements
    import numpy as np # For the initialization of the parameters
    from matplotlib import pyplot # For graphics on post-processing
    from pyBEL1D.utilities import Tools # For further post-processing
    from os import listdir
    from os.path import isfile, join
    from pathos import multiprocessing as mp
    from pathos import pools as pp

    #########################################################################################
    ###                         Synthetic case for Vs and e only                          ###
    #########################################################################################
    from pysurf96 import surf96
    from scipy import stats

    # Define the model:
    TrueModel = np.asarray([0.01, 0.05, 0.120, 0.280, 0.600])#Thickness and Vs only
    Vp = np.asarray([0.300, 0.750, 1.5])
    rho = np.asarray([1.5, 1.9, 2.2])
    nLayer = 3
    Frequency = np.logspace(0.1,1.5,50)
    Periods = np.divide(1,Frequency)
    #model = model.append(rho)
    # Forward modelling using surf96:
    DatasetClean = surf96(thickness=np.append(TrueModel[0:nLayer-1], [0]),vp=Vp,vs=TrueModel[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    ErrorModelSynth = [0.075, 20]
    NoiseEstimate = np.asarray(np.divide(ErrorModelSynth[0]*DatasetClean*1000 + np.divide(ErrorModelSynth[1],Frequency),1000)) # Standard deviation for all measurements in km/s
    # np.save('NoiseEstimateTest.npy',NoiseEstimate)
    randVal = 0#np.random.randn(1)
    print("The dataset is shifted by {} times the NoiseLevel".format(randVal))
    Dataset = DatasetClean + randVal*NoiseEstimate
    # Define the prior:
    # Find min and max Vp for each layer in the range of Poisson's ratio [0.2, 0.45]:
    # For Vp1=0.3, the roots are : 0.183712 and 0.0904534 -> Vs1 = [0.1, 0.18]
    # For Vp2=0.75, the roots are : 0.459279 and 0.226134 -> Vs2 = [0.25, 0.45]
    # For Vp3=1.5, the roots are : 0.918559 and 0.452267 -> Vs2 = [0.5, 0.9]
    prior = np.array([[0.001, 0.03, 0.1, 0.18],[0.01, 0.1, 0.25, 0.45],[0.0, 0.0, 0.5, 0.9]])
    nParam = 2 # e and Vs
    ListPrior = [None] * ((nLayer*nParam)-1)# Half space at bottom
    NamesFullUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
    NamesShort = [None] * ((nLayer*nParam)-1)# Half space at bottom
    NamesShortUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
    Mins = np.zeros(((nLayer*nParam)-1,))
    Maxs = np.zeros(((nLayer*nParam)-1,))
    Units = ["\\ [km]", "\\ [km/s]"]
    NFull = ["Thickness\\ ","s-Wave\\ velocity\\ "]
    NShort = ["e_{", "Vs_{"]
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
    forwardFun = lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    forward = {"Fun":forwardFun,"Axis":Periods}
    cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
    # Initialize the model parameters for BEL1D
    nbModelsBase = 1000
    start = time.time()
    ModelSynthetic = BEL1D.MODELSET(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)
    # Compute the operations prior to the knowledge of field data:
    PrebelSynthetic = BEL1D.PREBEL(MODPARAM=ModelSynthetic, nbModels=nbModelsBase)
    # pool = pp.ProcessPool(mp.cpu_count())
    PrebelSynthetic.run()#Parallelization=[True,pool])
    PreModsSynth=PrebelSynthetic.MODELS
    # Compute the operations posterior to the knowledge of the dataset:
    PostbelSynthetic = BEL1D.POSTBEL(PREBEL=PrebelSynthetic)
    PostbelSynthetic.run(Dataset=Dataset,nbSamples=nbModelsBase,NoiseModel=NoiseEstimate)
    end = time.time()
    # BEL1D.SavePOSTBEL(CurrentPostbel=PostbelSynthetic,Filename='BaseTesting')
    # Show the results:
    PostbelSynthetic.ShowDataset(RMSE=True,Prior=True)#,Parallelization=[True,pool])
    CurrentGraph = pyplot.gcf()
    CurrentGraph = CurrentGraph.get_axes()[0]
    CurrentGraph.plot(Periods, DatasetClean+NoiseEstimate,'k--')
    CurrentGraph.plot(Periods, DatasetClean-NoiseEstimate,'k--')
    CurrentGraph.plot(Periods, DatasetClean+2*NoiseEstimate,'k:')
    CurrentGraph.plot(Periods, DatasetClean-2*NoiseEstimate,'k:')
    CurrentGraph.plot(Periods, Dataset,'k')
    PostbelSynthetic.ShowPostCorr(TrueModel=TrueModel,OtherMethod=PreModsSynth)
    PostbelSynthetic.ShowPostModels(TrueModel=TrueModel,RMSE=True)#,Parallelization=[True, pool])
    # Graph for the CCA space parameters loads
    _, ax = pyplot.subplots()
    B = PostbelSynthetic.CCA.y_loadings_
    B = np.divide(np.abs(B).T,np.repeat(np.reshape(np.sum(np.abs(B),axis=0),(1,B.shape[0])),B.shape[0],axis=0).T)
    ind =  np.asarray(range(B.shape[0]))+1
    ax.bar(x=ind,height=B[0],label=r'${}$'.format(PostbelSynthetic.MODPARAM.paramNames["NamesSU"][0]))
    for i in range(B.shape[0]+1)[1:-1]:
        ax.bar(x=ind,height=B[i],bottom=np.reshape(np.sum(B[0:i],axis=0),(B.shape[0],)),label=r'${}$'.format(PostbelSynthetic.MODPARAM.paramNames["NamesSU"][i]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3)
    ax.set_ylabel('Relative contribution')
    ax.set_xlabel('CCA dimension')

    PostbelSynthetic.KDE.ShowKDE(Xvals=PostbelSynthetic.CCA.transform(PostbelSynthetic.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
    pyplot.show(block=False)

    # Iterate:
    print("The posterior still overestimates the uncertainty!")
    nbIter = 100
    nbParam = len(TrueModel)
    means = np.zeros((nbIter,nbParam))
    stds = np.zeros((nbIter,nbParam))
    timings = np.zeros((nbIter,))
    diverge = True
    distancePrevious = 1e10
    MixingUpper = 0
    MixingLower = 1
    for idxIter in range(nbIter):
        if idxIter == 0: # Initialization: already done (see 2 and 3)
            # PostbelTest.KDE.ShowKDE(Xvals=PostbelTest.CCA.transform(PostbelTest.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
            means[idxIter,:], stds[idxIter,:] = PostbelSynthetic.GetStats()
            timings[idxIter] = end-start
            ModLastIter = PrebelSynthetic.MODELS
        else:
            ModLastIter = PostbelSynthetic.SAMPLES
            PrebelLast = PrebelSynthetic
            PostbelLast = PostbelSynthetic
            MixingUpper += 1
            MixingLower += 1
            Mixing = MixingUpper/MixingLower
            # Here, we will use the POSTBEL2PREBEL function that adds the POSTBEL from previous iteration to the prior (Iterative prior resampling)
            # However, the computations are longer with a lot of models, thus you can opt-in for the "simplified" option which randomely select up to 10 times the numbers of models
            PrebelSynthetic = BEL1D.PREBEL.POSTBEL2PREBEL(PREBEL=PrebelSynthetic,POSTBEL=PostbelSynthetic,Dataset=Dataset,NoiseModel= NoiseEstimate,Simplified=False,nbMax=nbModelsBase,MixingRatio=Mixing)#,Parallelization=[False,None])
            # Since when iterating, the dataset is known, we are not computing the full relationship but only the posterior distributions directly to gain computation timing
            print(idxIter+1)
            PostbelSynthetic = BEL1D.POSTBEL(PrebelSynthetic)
            PostbelSynthetic.run(Dataset,nbSamples=nbModelsBase,NoiseModel=NoiseEstimate)
            means[idxIter,:], stds[idxIter,:] = PostbelSynthetic.GetStats()
            end = time.time()
            timings[idxIter] = end-start
        # The distance is computed on the normalized distributions. Therefore, the tolerance is relative.
        diverge, distance = Tools.ConvergeTest(SamplesA=ModLastIter,SamplesB=PostbelSynthetic.SAMPLES, tol=1e-3)
        print('Wasserstein distance: {}'.format(distance))
        if not(diverge):# or (abs((distancePrevious-distance)/distancePrevious)*100<0.25):
            # Convergence acheived if:
            # 1) Distance below threshold
            # 2) Distance does not vary significantly (less than 2.5%)
            print('Model has converged at iter {}!'.format(idxIter+1))
            break
        distancePrevious = distance
        start = time.time()
    timings = timings[:idxIter+1]
    means = means[:idxIter+1,:]
    stds = stds[:idxIter+1,:]

    # PostbelSynthetic.run(Dataset,nbSamples=10000, NoiseModel=NoiseEstimate)
    # Graphs for the iterations:
    PostbelSynthetic.ShowDataset(RMSE=True,Prior=True)#,Parallelization=[True,pool])
    CurrentGraph = pyplot.gcf()
    CurrentGraph = CurrentGraph.get_axes()[0]
    CurrentGraph.plot(Periods, DatasetClean+NoiseEstimate,'k--')
    CurrentGraph.plot(Periods, DatasetClean-NoiseEstimate,'k--')
    CurrentGraph.plot(Periods, DatasetClean+2*NoiseEstimate,'k:')
    CurrentGraph.plot(Periods, DatasetClean-2*NoiseEstimate,'k:')
    CurrentGraph.plot(Periods, Dataset,'k')
    PostbelSynthetic.ShowPostCorr(TrueModel=TrueModel,OtherMethod=PreModsSynth)
    PostbelSynthetic.ShowPostModels(TrueModel=TrueModel,RMSE=True)#,Parallelization=[True, pool])
    # Graph for the CCA space parameters loads
    _, ax = pyplot.subplots()
    B = PostbelSynthetic.CCA.y_loadings_
    B = np.divide(np.abs(B).T,np.repeat(np.reshape(np.sum(np.abs(B),axis=0),(1,B.shape[0])),B.shape[0],axis=0).T)
    ind =  np.asarray(range(B.shape[0]))+1
    ax.bar(x=ind,height=B[0],label=r'${}$'.format(PostbelSynthetic.MODPARAM.paramNames["NamesSU"][0]))
    for i in range(B.shape[0]+1)[1:-1]:
        ax.bar(x=ind,height=B[i],bottom=np.reshape(np.sum(B[0:i],axis=0),(B.shape[0],)),label=r'${}$'.format(PostbelSynthetic.MODPARAM.paramNames["NamesSU"][i]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3)
    ax.set_ylabel('Relative contribution')
    ax.set_xlabel('CCA dimension')
    pyplot.show(block=False)

    # Add KDE graph at last iteration:
    # PrebelGraph = BEL1D.PREBEL.POSTBEL2PREBEL(PREBEL=PrebelLast,POSTBEL=PostbelLast, NoiseModel=NoiseEstimate)
    # PostbelGraph = BEL1D.POSTBEL(PREBEL=PrebelGraph)
    # print('Postbel for graphs initialized')
    # PostbelGraph.run(Dataset=Dataset, nbSamples=10000)
    # print('Printing KDE Graphs')
    # PostbelGraph.KDE.ShowKDE(Xvals=PostbelGraph.CCA.transform(PostbelGraph.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))

    print('Total computation time: {} seconds'.format(np.sum(timings)))

    # Compare the results to McMC results:
    McMC = np.load("MASW_Bench.npy")
    DREAM=McMC[:,:5]
    DREAM = np.unique(DREAM,axis=0)
    PostbelSynthetic.ShowPostCorr(TrueModel=TrueModel, OtherMethod=DREAM)

    pyplot.show(block=False)
    # figs = [pyplot.figure(n) for n in pyplot.get_fignums()]
    # for fig in figs:
    #     fig.savefig('Figure{}.png'.format(fig.number), format='png')
    
    pyplot.show()

    MIRANDOLA = False
    if MIRANDOLA:
        #########################################################################################
        ###                                1) Retreive the data                               ###
        #########################################################################################
        Dataset = np.loadtxt("Data/DC/Mirandola_InterPACIFIC/Average/Average_interp60_cuttoff.txt")
        FreqMIR = Dataset[:,0]
        Dataset = np.divide(Dataset[:,1],1000)# Phase velocity in km/s for the forward model
        ErrorModel = [0.075, 20] # Error model: see dedicated function for more informations
        ErrorFreq = np.asarray(np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000))# Errors in km/s for each Frequency

        # Graph 1: Error model:
        pyplot.plot(FreqMIR, Dataset,'b') # Base dataset
        # +/- standard deviations
        dataNoisyU = Dataset + np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
        dataNoisyU2 = Dataset + 2*np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
        dataNoisyL = Dataset - np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
        dataNoisyL2 = Dataset - 2*np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
        # Experts curves (resampled on the same space):
        DataPath = "Data/DC/Mirandola_InterPACIFIC/"
        files = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        for currFile in files:
            DatasetOther = np.loadtxt(DataPath+currFile)
            DatasetOther = np.divide(DatasetOther[:,1],1000) # Dataset for surf96 in km/s
            DatasetOther[DatasetOther==0] = np.nan
            pyplot.plot(FreqMIR, DatasetOther,'ko')
        pyplot.plot(FreqMIR, dataNoisyU,'r')
        pyplot.plot(FreqMIR, dataNoisyL,'r')
        pyplot.plot(FreqMIR, dataNoisyU2,'r--')
        pyplot.plot(FreqMIR, dataNoisyL2,'r--')
        pyplot.plot(FreqMIR, Dataset,'b')
        pyplot.xlabel("Frequency [Hz]")
        pyplot.ylabel("Phase velocity [km/s]")
        pyplot.xscale('log')
        pyplot.yscale('log')
        pyplot.show(block=False)

        #########################################################################################
        ###            2) Run PreBEL operations (up to the knowledge of the dataset)          ###
        #########################################################################################
        print('Initializing . . .')
        # Define the prior
        priorDC = np.array([[0.005, 0.05, 0.1, 0.5, 0.2, 4.0, 1.5, 3.5], [0.045, 0.145, 0.1, 0.8, 0.2, 4.0, 1.5, 3.5], [0, 0, 0.3, 2.5, 0.2, 4.0, 1.5, 3.5]]) # MIRANDOLA prior test case
        nbModPrior = 10000 # See discussion for more details
        nbSamplesPost = 10000
        # Creating the MODELSET object through the classmethod DC (dispersion curve)
        start = time.time()
        TestCase = BEL1D.MODELSET().DC(prior=priorDC, Frequency=FreqMIR)
        # Then, we build the "pre-bel" operations using the PREBEL function
        Prebel = BEL1D.PREBEL(TestCase,nbModels=nbModPrior)
        # We then run the prebel operations:
        print('Running PREBEL . . .')
        # Building a parallel pool for the computations:
        pool = pp.ProcessPool(mp.cpu_count())
        # Running the operations by themselve:
        Prebel.run(Parallelization=[True,pool])
        # We could run the KDE graphs now, but it is much easier toi run it afterwards sinc the
        # posterior in CCA space is also displayed.
        BEL1D.SavePREBEL(Prebel,Filename='Mirandola_Pre')# Save the object

        #########################################################################################
        ###           3) Run PostBEL operations (after the knowledge of the dataset)          ###
        #########################################################################################
        # Then, since we know the dataset, we can initialize the "post-bel" operations:
        Postbel = BEL1D.POSTBEL(Prebel)
        # Run the operations:
        print('Sampling posterior . . .')
        Postbel.run(Dataset=Dataset, nbSamples=nbSamplesPost, NoiseModel=ErrorFreq)
        end = time.time()
        BEL1D.SavePOSTBEL(Postbel,Filename='Mirandola_avg_Iter1')# Save the object

        #########################################################################################
        ###                 4) Graphs for the results of the first iteration                  ###
        #########################################################################################
        # 1) Show the CCA space (kernel-density estimated)
        Postbel.KDE.ShowKDE(dim=[0,4,10],Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
        # 2) Show the correlated posterior space compared to the prior:
        PreMods = Prebel.MODELS
        Postbel.ShowPostCorr(OtherMethod=PreMods)
        # 3) Show the posterior datasets:
        Postbel.ShowDataset(RMSE=True, Prior=True, Parallelization=[True,pool])
        CurrentGraph = pyplot.gcf()
        CurrentGraph = CurrentGraph.get_axes()[0]
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyU,'k--')
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyL,'k--')
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyU2,'k:')
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyL2,'k:')
        CurrentGraph.plot(np.divide(1,FreqMIR), Dataset,'k')
        # pyplot.show()

        # Graph for the CCA space parameters loads
        _, ax = pyplot.subplots()
        B = Postbel.CCA.y_loadings_
        B = np.divide(np.abs(B).T,np.repeat(np.reshape(np.sum(np.abs(B),axis=0),(1,B.shape[0])),B.shape[0],axis=0).T)
        ind =  np.asarray(range(B.shape[0]))+1
        ax.bar(x=ind,height=B[0],label=r'${}$'.format(Postbel.MODPARAM.paramNames["NamesSU"][0]))
        for i in range(B.shape[0]+1)[1:-1]:
            ax.bar(x=ind,height=B[i],bottom=np.reshape(np.sum(B[0:i],axis=0),(B.shape[0],)),label=r'${}$'.format(Postbel.MODPARAM.paramNames["NamesSU"][i]))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3)
        ax.set_ylabel('Relative contribution')
        ax.set_xlabel('CCA dimension')
        pyplot.show(block=False)

        # pyplot.show()

        print('As is seen, the uncertainty still is still largely overestimated')
        #########################################################################################
        ###                          5) Iterative prior resampling                            ###
        #########################################################################################
        nbIter = 25
        nbParam = int(priorDC.size/2 - 1)
        means = np.zeros((nbIter,nbParam))
        stds = np.zeros((nbIter,nbParam))
        timings = np.zeros((nbIter,))
        diverge = True
        distancePrevious = 1e10
        MixingUpper = 0
        MixingLower = 1
        for idxIter in range(nbIter):
            if idxIter == 0: # Initialization: already done (see 2 and 3)
                # PostbelTest.KDE.ShowKDE(Xvals=PostbelTest.CCA.transform(PostbelTest.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
                means[idxIter,:], stds[idxIter,:] = Postbel.GetStats()
                timings[idxIter] = end-start
                ModLastIter = Prebel.MODELS
            else:
                ModLastIter = Postbel.SAMPLES
                # Here, we will use the POSTBEL2PREBEL function that adds the POSTBEL from previous iteration to the prior (Iterative prior resampling)
                # However, the computations are longer with a lot of models, thus you can opt-in for the "simplified" option which randomely select up to 10 times the numbers of models
                MixingUpper += 1
                MixingLower += 1
                Mixing = MixingUpper/MixingLower
                Prebel = BEL1D.PREBEL.POSTBEL2PREBEL(PREBEL=Prebel,POSTBEL=Postbel,Dataset=Dataset,NoiseModel=ErrorFreq,Parallelization=[True,pool],Simplified=True,nbMax=nbSamplesPost,MixingRatio=Mixing)
                # Since when iterating, the dataset is known, we are not computing the full relationship but only the posterior distributions directly to gain computation timing
                print(idxIter+1)
                Postbel = BEL1D.POSTBEL(Prebel)
                Postbel.run(Dataset,nbSamples=nbSamplesPost,NoiseModel=ErrorFreq)
                means[idxIter,:], stds[idxIter,:] = Postbel.GetStats()
                end = time.time()
                timings[idxIter] = end-start
            # The distance is computed on the normalized distributions. Therefore, the tolerance is relative.
            diverge, distance = Tools.ConvergeTest(SamplesA=ModLastIter,SamplesB=Postbel.SAMPLES, tol=5e-4)
            print('Wasserstein distance: {}'.format(distance))
            if not(diverge) or (abs((distancePrevious-distance)/distancePrevious)*100<1):
                # Convergence acheived if:
                # 1) Distance below threshold
                # 2) Distance does not vary significantly (less than 2.5%)
                print('Model has converged at iter {}!'.format(idxIter+1))
                break
            distancePrevious = distance
            start = time.time()
        timings = timings[:idxIter+1]
        means = means[:idxIter+1,:]
        stds = stds[:idxIter+1,:]

        BEL1D.SavePOSTBEL(Postbel,Filename='Mirandola_avg_IterLast')# Save the object

        #########################################################################################
        ###                 6) Graphs for the results of the last iteration                   ###
        #########################################################################################
        # 1) Show the posterior datasets:
        Postbel.ShowDataset(RMSE=True, Prior=True, Parallelization=[True,pool])
        CurrentGraph = pyplot.gcf()
        CurrentGraph = CurrentGraph.get_axes()[0]
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyU,'k--')
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyL,'k--')
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyU2,'k:')
        CurrentGraph.plot(np.divide(1,FreqMIR), dataNoisyL2,'k:')
        CurrentGraph.plot(np.divide(1,FreqMIR), Dataset,'k')

        # 2) Show depth to bedrock
        Post = Postbel.SAMPLES
        PrebelInit = BEL1D.LoadPREBEL('Mirandola_Pre.prebel')
        Pre = PrebelInit.MODELS
        ThickBedrockPre = np.sum(Pre[:,0:2],axis=1)*1000
        ThickBedrockPost = np.sum(Post[:,0:2],axis=1)*1000
        fig = pyplot.figure()
        axes = fig.subplots()
        axes.hist(ThickBedrockPre,density=True,label='Prior')# Prior before iterations
        axes.hist(ThickBedrockPost,density=True,alpha=0.8,label='Posterior')# Posterior after iterations
        TrueDepth=118
        axes.plot([TrueDepth, TrueDepth],axes.get_ylim(),'k',label='Measured')
        axes.legend()
        axes.set_xlabel('Depth to bedrock [m]')
        axes.set_ylabel('Probability [/]')

        # 3) Show models with RMSE
        Postbel.ShowPostModels(RMSE=True)

        Postbel.ShowPostCorr(OtherMethod=PreMods)
        print('Total computation time: {} seconds'.format(np.sum(timings)))

        # pyplot.show()
        # We do not need the parallel pool anymore:
        pool.terminate()

    Discussion = False
    if Discussion:

        #########################################################################################
        ###                   7) Testing the prebel with different datasets                   ###
        #########################################################################################
        DataPath = "Data/DC/Mirandola_InterPACIFIC/"
        Succeded= []
        files = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        nbFiles = len(files)
        nbSamplesTest = 10000
        Models = np.zeros((nbFiles, nbSamplesTest, Pre.shape[1]))
        i = 0
        success = []
        for currFile in files:
            DatasetOther = np.loadtxt(DataPath+currFile)
            DatasetOther = np.divide(DatasetOther[:,1],1000) # Dataset for surf96 in km/s
            # DatasetOther[DatasetOther==0] = np.nan
            # Now, run the prediction (no iterations)
            try:
                Prebel = BEL1D.LoadPREBEL('Mirandola_Pre.prebel')
                Postbel = BEL1D.POSTBEL(Prebel)
                print('Sampling posterior . . .')
                start = time.time()
                Postbel.run(Dataset=DatasetOther, nbSamples=nbSamplesTest, NoiseModel=ErrorFreq)
                end = time.time()
                Models[i,:,:] = Postbel.SAMPLES
                # BEL1D.SaveSamples(CurrentPostbel=Postbel,Data=True,Filename=currFile)
                # Succeded.append(currFile)
                print('Dataset {} succeded in {} seconds!'.format(currFile,end-start))
                success.append(i)
                DatasetName = currFile.split('_')[0]
                BEL1D.SavePOSTBEL(CurrentPostbel=Postbel,Filename='TestingsDatasets/{}'.format(DatasetName))
            except:
                print('Dataset {} failed!'.format(currFile))
            i += 1
        # Graph of the most sensitive parameter Vs3:
        paramNb = 2
        fig = pyplot.figure()
        axes = fig.subplots()
        axes.hist(Prebel.MODELS[:,paramNb],density=True,label='Prior')# Prior before iterations
        for i in success:
            DatasetName = files[i].split('_')[0]
            axes.hist(Models [i,:,paramNb],density=True,label=DatasetName,alpha=0.25)#,fill=False,edgecolor='auto')# Posterior after iterations
        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width, box.height*0.8])
        axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3)
        axes.set_xlabel(r'${}$'.format(Postbel.MODPARAM.paramNames["NamesFU"][paramNb]))
        axes.set_ylabel('Probability [/]')
        pyplot.show(block=False)
        np.save('TestingsDatasets/ModelsSampled',Models)


        #########################################################################################
        ###                   8) Testing the number of models in the prior                    ###
        #########################################################################################
        if False:
            import os
            pool = pp.ProcessPool(6)# For reproductibility: pool of 6 cpus only
            Parallel = [True, pool]
            TestCase = BEL1D.MODELSET().DC(prior=priorDC, Frequency=FreqMIR)
            values_tested = list(np.logspace(2,6,10,dtype=np.int32))
            nbRepeat = 10
            timings = np.zeros((len(values_tested),nbRepeat))
            idxTested = 0
            for nbPre in values_tested:
                if not(os.path.exists('TestingsNbPre/{}_Pre'.format(nbPre))):
                    os.mkdir('TestingsNbPre/{}_Pre'.format(nbPre))
                for i in range(nbRepeat):
                    print('Test nb. {} on {} for value {} on {}'.format(i,nbRepeat,idxTested+1,10))
                    start = time.time()
                    Prebel = BEL1D.PREBEL(TestCase,nbModels=nbPre)
                    Prebel.run(Parallelization=Parallel)
                    Postbel = BEL1D.POSTBEL(Prebel)
                    Postbel.run(Dataset=Dataset, nbSamples=10000, NoiseModel=ErrorFreq)
                    end = time.time()
                    timings[idxTested,i] = end-start
                    # Postbel.DataPost(Parallelization=Parallel)
                    BEL1D.SavePOSTBEL(CurrentPostbel=Postbel,Filename='TestingsNbPre/{}_Pre/Attempt_{}'.format(nbPre,i))
                idxTested += 1
            pool.terminate()
            np.savetxt('TestingsNbPre/timings.txt',timings)
            # Graph of time vs standard deviations vs nbMods

    pyplot.show()
    
