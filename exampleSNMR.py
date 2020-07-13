# This file is an example on how to use the BEL1D codes using a simple 2-layer SNMR experiment (with noise)
from pyBEL1D import BEL1D
import cProfile # For debugging and timings measurements
import time # For simple timing measurements
import numpy as np # For the initialization of the parameters
from matplotlib import pyplot # For graphics on post-processing
from pyBEL1D.utilities import Tools # For further post-processing

# Parameters for the tested model
modelTrue = np.asarray([5.0, 0.05, 0.25, 0.1, 0.2])
priorSNMR = np.array([[2.5, 7.5, 0.035, 0.10, 0.005, 0.350], [0, 0, 0.10, 0.30, 0.005, 0.350]])
stdUniform = lambda a,b: (b-a)/np.sqrt(12)
stdTrue = np.asarray([stdUniform(priorSNMR[0,0],priorSNMR[0,1]), stdUniform(priorSNMR[0,2],priorSNMR[0,3]), stdUniform(priorSNMR[1,2],priorSNMR[1,3]), stdUniform(priorSNMR[0,4],priorSNMR[0,5]), stdUniform(priorSNMR[1,4],priorSNMR[1,5])])
Kernel = "Data/sNMR/KernelTest.mrsk"
Timings = np.arange(0.005, 0.5, 0.001)

# Function to test the most direct approach:
def test(nbModPre=1000):
    # To first declare the parameters, we call the constructor MODELSET().SNMR() with the right parameters
    TestCase = BEL1D.MODELSET().SNMR(prior=priorSNMR,Kernel=Kernel, Timing=np.arange(0.005, 0.5, 0.001))
    # Then, we build the "pre-bel" operations using the PREBEL function
    Prebel = BEL1D.PREBEL(TestCase,nbModels=nbModPre)
    # We then run the prebel operations:
    Prebel.run()
    # You can observe the relationship using:
    Prebel.KDE.ShowKDE()
    
    # Then, we generate the synthetic benchmark dataset (with noise)
    DatasetSim = Prebel.MODPARAM.forwardFun["Fun"](modelTrue)
    # To add noise (sNMR -> Gaussian Noise) [in this case, we add 10nV = 10*1e-9V]:
    DatasetSim += np.random.normal(loc=0, scale=10*1e-9,size=DatasetSim.shape)

    # Then, since we know the dataset, we can initialize the "post-bel" operations:
    Postbel = BEL1D.POSTBEL(Prebel)
    # Run the operations:
    Postbel.run(Dataset=DatasetSim, nbSamples=1000, NoiseModel=10)

    # All the operations are done, now, you just need to analyze the results (or run the iteration process - see next example)
    # Show the models parameters uncorrelated:
    Postbel.ShowPost(TrueModel=modelTrue)
    # Show the models parameters correlated with also the prior samples (Prebel.MODELS):
    Postbel.ShowPostCorr(TrueModel=modelTrue, OtherMethod=Prebel.MODELS)
    # Show the depth distributions of the parameters with the RMSE
    Postbel.ShowPostModels(TrueModel=modelTrue, RMSE=True)
    # Get key statistics
    means, stds = Postbel.GetStats()
    return means, stds

# Now, let's see how to iterate:
def testIter(nbIter=5):
    nbModPre = 1000
    means = np.zeros((nbIter,len(modelTrue)))
    stds = np.zeros((nbIter,len(modelTrue)))
    timings = np.zeros((nbIter,))
    start = time.time()
    diverge = True
    for idxIter in range(nbIter):
        if idxIter == 0: # Initialization
            TestCase = BEL1D.MODELSET().SNMR(prior=priorSNMR, Kernel=Kernel, Timing = Timings)
            PrebelIter = BEL1D.PREBEL(TestCase,nbModPre)
            PrebelIter.run()
            ModLastIter = PrebelIter.MODELS
            # Compute benchmark dataset:
            Dataset = PrebelIter.MODPARAM.forwardFun["Fun"](modelTrue) 
            Dataset += np.random.normal(loc=0, scale=10*1e-9,size=Dataset.shape)
            print(idxIter+1)
            PostbelTest = BEL1D.POSTBEL(PrebelIter)
            PostbelTest.run(Dataset,nbSamples=nbModPre,NoiseModel=10)#NoiseModel=[0.005,100])
            means[idxIter,:], stds[idxIter,:] = PostbelTest.GetStats()
            end = time.time()
            timings[idxIter] = end-start
        else:
            ModLastIter = PostbelTest.SAMPLES
            # Here, we will use the POSTBEL2PREBEL function that adds the POSTBEL from previous iteration to the prior (Iterative prior resampling)
            # However, the computations are longer with a lot of models, thus you can opt-in for the "simplified" option which randomely select up to 10 times the numbers of models
            PrebelIter = BEL1D.PREBEL.POSTBEL2PREBEL(PREBEL=PrebelIter,POSTBEL=PostbelTest,Dataset=Dataset,NoiseModel=10,Simplified=True,nbMax=10*nbModPre)
            # Since when iterating, the dataset is known, we are not computing the full relationship but only the posterior distributions directly to gain computation timing
            print(idxIter+1)
            PostbelTest = BEL1D.POSTBEL(PrebelIter)
            PostbelTest.run(Dataset,nbSamples=nbModPre,NoiseModel=None)
            means[idxIter,:], stds[idxIter,:] = PostbelTest.GetStats()
            end = time.time()
            timings[idxIter] = end-start
        diverge, distance = Tools.ConvergeTest(SamplesA=ModLastIter,SamplesB=PostbelTest.SAMPLES, tol=1e-5)
        print('Wasserstein distance: {}'.format(distance))
        if not(diverge):
            print('Model has converged at iter {}!'.format(idxIter+1))
            break
        start = time.time()
    PostbelTest.ShowPostCorr(TrueModel=modelTrue,OtherMethod=PrebelIter.MODELS)
    PostbelTest.ShowPostModels(TrueModel=modelTrue, RMSE=True)
    timings = timings[:idxIter+1]
    means = means[:idxIter+1,:]
    stds = stds[:idxIter+1,:]
    paramnames = PostbelTest.MODPARAM.paramNames["NamesS"] # For the legend of the futur graphs
    return timings, means, stds, paramnames

IterTest = False

if IterTest:
    nbIter = 5
    timings, means, stds, names = testIter(nbIter=nbIter)
    pyplot.plot(np.arange(len(timings)),timings)
    pyplot.ylabel('Computation Time [sec]')
    pyplot.xlabel('Iteration nb.')
    pyplot.show()
    pyplot.plot(np.arange(len(timings)),np.divide(means,modelTrue))
    pyplot.ylabel('Normalized means [/]')
    pyplot.xlabel('Iteration nb.')
    pyplot.legend(names)
    pyplot.show()
    pyplot.plot(np.arange(len(timings)),np.divide(stds,stdTrue))
    pyplot.ylabel('Normalized standard deviations [/]')
    pyplot.xlabel('Iteration nb.')
    pyplot.legend(names)
    pyplot.show()

if not(IterTest):
    test()







