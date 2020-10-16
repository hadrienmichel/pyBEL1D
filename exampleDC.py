# This file is an example on how to use the BEL1D codes using a simple 2-layer DC experiment (with noise)
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

    # Parameters for the tested model
    # modelTrue = np.asarray([5.0, 0.05, 0.25, 0.1, 0.2])
    priorDC = np.array([[0.005, 0.05, 0.1, 0.5, 0.2, 4.0, 1.5, 3.5], [0.045, 0.145, 0.1, 0.8, 0.2, 4.0, 1.5, 3.5], [0, 0, 0.3, 2.5, 0.2, 4.0, 1.5, 3.5]]) # MIRANDOLA prior test case
    nbParam = int(priorDC.size/2 - 1)
    nLayer, nParam = priorDC.shape
    nParam = int(nParam/2)
    stdPrior = [None]*nbParam
    meansPrior = [None]*nbParam
    stdUniform = lambda a,b: (b-a)/np.sqrt(12)
    meansUniform = lambda a,b: (b-a)/2
    ident = 0
    for j in range(nParam):
                for i in range(nLayer):
                    if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                        stdPrior[ident] = stdUniform(priorDC[i,j*2],priorDC[i,j*2+1])
                        meansPrior[ident] = meansUniform(priorDC[i,j*2],priorDC[i,j*2+1])
                        ident += 1
    Dataset = np.loadtxt("Data/DC/Mirandola_InterPACIFIC/Average/Average_interp60_cuttoff.txt")
    FreqMIR = Dataset[:,0]
    Dataset = np.divide(Dataset[:,1],1000)# Phase velocity in km/s for the forward model
    ErrorModel = [0.075, 20]
    ErrorFreq = np.asarray(np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000))# Errors in km/s for each Frequency

    # Attention, the maximum number of periods is 60 for the forward model! Keep the number of points low!

    # Show the noise model for the dataset:
    pyplot.plot(FreqMIR, Dataset,'b')
    dataNoisyU = Dataset + np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
    dataNoisyU2 = Dataset + 2*np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
    dataNoisyL = Dataset - np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
    dataNoisyL2 = Dataset - 2*np.divide(ErrorModel[0]*Dataset*1000 + np.divide(ErrorModel[1],FreqMIR),1000)
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
    pyplot.show()
    ErrorModel = ErrorFreq # Error model for every frequency

    # Function to test the most direct approach:
    def test(nbModPre=1000):
        # To first declare the parameters, we call the constructor MODELSET().SNMR() with the right parameters
        print('Initializing . . .')
        TestCase = BEL1D.MODELSET().DC(prior=priorDC, Frequency=FreqMIR)
        # Then, we build the "pre-bel" operations using the PREBEL function
        Prebel = BEL1D.PREBEL(TestCase,nbModels=nbModPre)
        # We then run the prebel operations:
        print('Running PREBEL . . .')
        pool = pp.ProcessPool(mp.cpu_count())
        Prebel.run(Parallelization=[True,pool])
        # You can observe the relationship using:
        # Prebel.KDE.ShowKDE()

        # Then, since we know the dataset, we can initialize the "post-bel" operations:
        Postbel = BEL1D.POSTBEL(Prebel)
        # Run the operations:
        print('Sampling posterior . . .')
        Postbel.run(Dataset=Dataset, nbSamples=nbModPre, NoiseModel=ErrorModel)

        # All the operations are done, now, you just need to analyze the results (or run the iteration process - see next example)
        # Show the models parameters uncorrelated:
        Postbel.ShowPost()
        # Show the models parameters correlated with also the prior samples (Prebel.MODELS):
        Postbel.ShowPostCorr(OtherMethod=Prebel.MODELS)
        # Show the depth distributions of the parameters with the RMSE
        Postbel.ShowPostModels(RMSE=True,Parallelization=[True,pool])
        Postbel.ShowDataset(RMSE=True,Parallelization=[True,pool])
        Postbel.KDE.ShowKDE(Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
        
        Prebel.ShowPreModels()
        Prebel.ShowPriorDataset()
        Postbel.ShowPostModels(RMSE=True,Best=1)
        Postbel.ShowDataset(RMSE=True,Best=1)
        # Get key statistics
        means, stds = Postbel.GetStats()
        pool.terminate()
        # TODO: debug save functions
        BEL1D.SavePREBEL(CurrentPrebel=Prebel,Filename="TestSave")
        BEL1D.SavePOSTBEL(CurrentPostbel=Postbel,Filename="TestSave")
        BEL1D.SaveSamples(CurrentPostbel=Postbel,Data=True,Filename="TestSave")
        return means, stds

    # Now, let's see how to iterate:
    def testIter(nbIter=5):
        nbModPre = 10000
        means = np.zeros((nbIter,nbParam))
        stds = np.zeros((nbIter,nbParam))
        timings = np.zeros((nbIter,))
        start = time.time()
        pool = pp.ProcessPool(mp.cpu_count())
        diverge = True
        distancePrevious = 1e10
        MixingUpper = 0
        MixingLower = 1
        for idxIter in range(nbIter):
            if idxIter == 0: # Initialization
                TestCase = BEL1D.MODELSET().DC(prior=priorDC, Frequency=FreqMIR)
                PrebelIter = BEL1D.PREBEL(TestCase,nbModPre)
                PrebelIter.run(Parallelization=[True,pool])
                ModLastIter = PrebelIter.MODELS
                print(idxIter+1)
                PostbelTest = BEL1D.POSTBEL(PrebelIter)
                PostbelTest.run(Dataset=Dataset,nbSamples=nbModPre,NoiseModel=ErrorModel)
                # PostbelTest.KDE.ShowKDE(Xvals=PostbelTest.CCA.transform(PostbelTest.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
                means[idxIter,:], stds[idxIter,:] = PostbelTest.GetStats()
                end = time.time()
                timings[idxIter] = end-start
            else:
                ModLastIter = PostbelTest.SAMPLES
                # Here, we will use the POSTBEL2PREBEL function that adds the POSTBEL from previous iteration to the prior (Iterative prior resampling)
                # However, the computations are longer with a lot of models, thus you can opt-in for the "simplified" option which randomely select up to 10 times the numbers of models
                MixingUpper += 1
                MixingLower += 1
                Mixing = MixingUpper/MixingLower
                PrebelIter = BEL1D.PREBEL.POSTBEL2PREBEL(PREBEL=PrebelIter,POSTBEL=PostbelTest,Dataset=Dataset,NoiseModel=ErrorModel,Parallelization=[True,pool],Simplified=True,nbMax=nbModPre,MixingRatio=Mixing)
                # Since when iterating, the dataset is known, we are not computing the full relationship but only the posterior distributions directly to gain computation timing
                print(idxIter+1)
                PostbelTest = BEL1D.POSTBEL(PrebelIter)
                PostbelTest.run(Dataset,nbSamples=nbModPre,NoiseModel=ErrorModel)
                means[idxIter,:], stds[idxIter,:] = PostbelTest.GetStats()
                end = time.time()
                timings[idxIter] = end-start
            diverge, distance = Tools.ConvergeTest(SamplesA=ModLastIter,SamplesB=PostbelTest.SAMPLES, tol=1e-4)
            print('Wasserstein distance: {}'.format(distance))
            if not(diverge) or (abs((distancePrevious-distance)/distancePrevious)*100<1):
                # Convergence acheived if:
                # 1) Distance below threshold
                # 2) Distance does not vary significantly (less than 2.5%)
                print('Model has converged at iter {}!'.format(idxIter+1))
                break
            distancePrevious = distance
            start = time.time()
        PostbelTest.ShowPostCorr(OtherMethod=PrebelIter.MODELS)
        PostbelTest.ShowPostModels(RMSE=True, Parallelization=[True,pool])
        PostbelTest.ShowDataset(RMSE=True,Prior=False, Parallelization=[True,pool])
        timings = timings[:idxIter+1]
        means = means[:idxIter+1,:]
        stds = stds[:idxIter+1,:]
        paramnames = PostbelTest.MODPARAM.paramNames["NamesS"] # For the legend of the futur graphs
        pool.terminate()
        return timings, means, stds, paramnames

    IterTest = True

    if IterTest:
        nbIter = 100
        timings, means, stds, names = testIter(nbIter=nbIter)
        print('Total time: {} seconds'.format(np.sum(timings)))
        pyplot.plot(np.arange(len(timings)),timings)
        pyplot.ylabel('Computation Time [sec]')
        pyplot.xlabel('Iteration nb.')
        pyplot.show()
        pyplot.plot(np.arange(len(timings)),np.divide(means,meansPrior))
        pyplot.ylabel('Normalized means [/]')
        pyplot.xlabel('Iteration nb.')
        pyplot.legend(names)
        pyplot.show()
        pyplot.plot(np.arange(len(timings)),np.divide(stds,stdPrior))
        pyplot.ylabel('Normalized standard deviations [/]')
        pyplot.xlabel('Iteration nb.')
        pyplot.legend(names)
        pyplot.show()

    if not(IterTest):
        test(nbModPre=10000)

    # nbModPre = 10000
    # print('Initializing . . .')
    # TestCase = BEL1D.MODELSET().DC(prior=priorDC, Frequency=FreqMIR)
    # # Then, we build the "pre-bel" operations using the PREBEL function
    # Prebel = BEL1D.PREBEL(TestCase,nbModels=nbModPre)
    # # We then run the prebel operations:
    # print('Running PREBEL . . .')
    # Prebel.run()
    # # You can observe the relationship using:
    # # Prebel.KDE.ShowKDE()

    # # Then, since we know the dataset, we can initialize the "post-bel" operations:
    # Postbel = BEL1D.POSTBEL(Prebel)
    # # Run the operations:
    # for idxUsed in range(len(files)):
    #     print('Current file: {}'.format(files[idxUsed]))
    #     DatasetOther = np.loadtxt(DataPath+files[idxUsed])
    #     DatasetOther = np.divide(DatasetOther[:,1],1000)
    #     Dataset = DatasetOther # Testing if works with other dataset than the mean (not the same span)
    #     Prebel.KDE.ShowKDE(Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(Dataset,(1,-1)))))
    #     try:
    #         print('Sampling posterior . . .')
    #         start = time.time()
    #         Postbel.run(Dataset=Dataset, nbSamples=10000, NoiseModel=ErrorModel)
    #         end = time.time()
    #         print('Dataset {} succeded in {} seconds!'.format(idxUsed+1,end-start))
    #     except:
    #         print('Dataset {} failed!'.format(idxUsed+1))