from pyBEL1D import BEL1D
import numpy as np
from matplotlib import pyplot as plt
## Libraries for parallel computing:
from pathos import multiprocessing as mp        # Multiprocessing utilities (get CPU cores info)
from pathos import pools as pp                  # Building the pool to use for computations
if __name__ == '__main__':
    ### For reproductibility - Random seed fixed
    RandomSeed = False # If True, use true random seed, else (False), fixed for reproductibility
    if not(RandomSeed):
        np.random.seed(0) # For reproductibilty
        from random import seed
        seed(0)
    ### End random seed fixed

    ParallelComputing = True
    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count()) # Create the parallel pool with at most the number of dimensions
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing

    # Building the prebel instance:
    modelSet = BEL1D.MODELSET.pendulum()

    # Building the synthetic dataset:
    benchmarkModel = np.asarray([3, 7.5, 40])
    benchData = modelSet.forwardFun['Fun'](benchmarkModel) + np.random.randn(111)*0.1

    prebel, postbel, prebelInit, statsCompute = BEL1D.IPR(MODEL=modelSet,Dataset=benchData,NoiseEstimate=np.ones_like(benchData)*0.1,Parallelization=ppComp,
                                                           nbModelsBase=10000,nbModelsSample=10000,stats=True,
                                                           Graphs=False, TrueModel=benchmarkModel, verbose=True)

    # prebel = BEL1D.PREBEL(modelSet, 10000)

    # prebel.run(Parallelization=ppComp, verbose=True)

    # # Building the synthetic dataset:
    # benchmarkModel = np.asarray([3, 7.5, 40])
    # benchData = modelSet.forwardFun['Fun'](benchmarkModel) + np.random.randn(111)*0.1

    # # Runing BEL1D:
    # postbel = BEL1D.POSTBEL(prebel)
    # postbel.run(benchData, 2000, NoiseModel=np.ones_like(benchData)*0.1, verbose=True)

    nbParam = 3
    fig, ax = plt.subplots(nbParam, nbParam)#, sharex=True, sharey=True)
    for i in range(nbParam):
        for j in range(nbParam):
            if i == j:
                ax[i,j].hist(postbel.MODELS[:,i], bins=20, density=True)
                ax[i,j].hist(postbel.SAMPLES[:,i], bins=20, density=True)
            elif i > j:
                ax[i,j].scatter(postbel.MODELS[:,j], postbel.MODELS[:,i], s=0.1)
                ax[i,j].scatter(postbel.SAMPLES[:,j], postbel.SAMPLES[:,i], s=0.1)

    fig.show()

    if ParallelComputing:
        pool.terminate()