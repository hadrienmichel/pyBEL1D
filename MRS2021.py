from pyBEL1D import BEL1D
from pyBEL1D.BEL1D import LoadPOSTBEL, LoadPREBEL
import numpy as np
from matplotlib import pyplot as plt
from pathos import multiprocessing as mp
from pathos import pools as pp
from os import listdir



### For reproductibility - Random seed fixed
RandomSeed = False # If True, use true random seed, else (False), fixed for reproductibility
if not(RandomSeed):
    np.random.seed(0) # For reproductibilty
    from random import seed
    seed(0)
### End random seed fixed

if __name__ == '__main__':
    # Parameters of the prior model space:
    SyntheticBenchmarkSNMR = np.asarray([25, 25, 0.05, 0.25, 0.10, 0.1, 0.2, 0.05]) # 3-layers model
    priorSNMRBenchmark = np.asarray([[1.0, 50.0, 0.01, 0.50, 0.005, 0.500], [1.0, 50.0, 0.01, 0.50, 0.005, 0.500], [1.0, 50.0, 0.01, 0.50, 0.005, 0.500]])
    KernelBenchmark = "Data/sNMR/MRS2021.mrsk" # Kernel file for the benchmark case.
    TimingsBenchmark = np.arange(0.005, 0.5, 0.002)

    RunBenchmark = False
    if RunBenchmark:
        ParallelComputing = True
        if ParallelComputing:
            pool = pp.ProcessPool(mp.cpu_count())# Create the parallel pool with at most the number of available CPU cores
            ppComp = [True, pool]
        else:
            ppComp = [False, None] # No parallel computing
        # Realizing BEL1D on the benchmark dataset
        ## Initializing the models parameters and the prior:
        ModelSet = BEL1D.MODELSET().SNMR(prior=priorSNMRBenchmark, Kernel=KernelBenchmark, Timing= TimingsBenchmark)
        DatasetBenchmark = ModelSet.forwardFun["Fun"](SyntheticBenchmarkSNMR)
        Noise = np.mean(DatasetBenchmark)/5
        print("Max signal: {} nV \nMean signal: {} nV \nStandard deviation of noise (S/N = 5): {} nV".format(np.max(DatasetBenchmark)*1e9, np.mean(DatasetBenchmark)*1e9, Noise*1e9))
        NoisyDatasetBenchmark = DatasetBenchmark + np.random.randn(len(DatasetBenchmark))*Noise
        ## Creating the BEL1D instances and IPR:
        Prebel, Postbel, PrebelInit , stats = BEL1D.IPR(MODEL=ModelSet, Dataset=DatasetBenchmark, NoiseEstimate=Noise*1e9, Parallelization=ppComp,
            nbModelsBase=1000, nbModelsSample=1000, stats=True, Mixing=(lambda x: 1), Graphs=False, saveIters=True, verbose=True)
    
    RunFigs = True
    if RunFigs:
        # Get the number of iterations:
        nbIter = len(listdir('./IPR_Results/'))
        PostbelLast = LoadPOSTBEL('./IPR_Results/IPR_{}.postbel'.format(nbIter-2))# Prebel and 0
        PrebelInit = LoadPREBEL('./IPR_Results/IPR.prebel')
        PostbelLast.ShowPostModels(TrueModel=SyntheticBenchmarkSNMR, RMSE=True)#, Best=100)
        PostbelLast.ShowPostCorr(TrueModel=SyntheticBenchmarkSNMR, OtherMethod=PrebelInit.MODELS, alpha=[0.05, 0.5])
        plt.tight_layout()

        ## Download the 1st iteration results and display the graphs:
        Postbel1 = LoadPOSTBEL('./IPR_Results/IPR_0.postbel')
        Postbel3 = LoadPOSTBEL('./IPR_Results/IPR_3.postbel')
        # Postbel1.ShowPostCorr(TrueModel=SyntheticBenchmarkSNMR, OtherMethod=PrebelInit.MODELS)

        nbParam = PostbelLast.SAMPLES.shape[1]
        nbLayers = int((nbParam+1)/3)
        fig, axes = plt.subplots(nbLayers,3)
        idAxesX = [0, 0, 1, 1, 1, 2, 2, 2]
        idAxesY = [0, 1, 0, 1, 2, 0, 1, 2]
        for i in range(nbParam):
            ax = axes[idAxesY[i], idAxesX[i]]
            if i != nbParam-1:
                ax.hist(PrebelInit.MODELS[:,i], density=True, alpha=0.5, label='_Prior')
                ax.hist(Postbel1.SAMPLES[:,i], density=True, alpha=0.5, label='_Iter 1')
                ax.hist(Postbel3.SAMPLES[:,i], density=True, alpha=0.5, label='_Iter 3')
                ax.hist(PostbelLast.SAMPLES[:,i], density=True, alpha=0.5, label='_Last iter')
                ax.plot([SyntheticBenchmarkSNMR[i],SyntheticBenchmarkSNMR[i]],np.asarray(ax.get_ylim()),'r',label='_Benchmark')
            else:
                ax.hist(PrebelInit.MODELS[:,i], density=True, alpha=0.5, label='Prior')
                ax.hist(Postbel1.SAMPLES[:,i], density=True, alpha=0.5, label='Iter 1')
                ax.hist(Postbel3.SAMPLES[:,i], density=True, alpha=0.5, label='Iter 3')
                ax.hist(PostbelLast.SAMPLES[:,i], density=True, alpha=0.5, label='Last iter')
                ax.plot([SyntheticBenchmarkSNMR[i],SyntheticBenchmarkSNMR[i]],np.asarray(ax.get_ylim()),'r',label='Benchmark')
            ax.set_xlabel(PostbelLast.MODPARAM.paramNames["NamesFU"][i])
            ax.get_yaxis().set_ticks([])
        axLeg = axes[2,0]
        axLeg.set_visible(False)
        plt.tight_layout()
        fig.legend(loc='lower left')

        totalWater = SyntheticBenchmarkSNMR[0]*SyntheticBenchmarkSNMR[2] + SyntheticBenchmarkSNMR[1]*SyntheticBenchmarkSNMR[3] + 25*SyntheticBenchmarkSNMR[4]
        _, ax = plt.subplots(2,1)
        ax[0].hist(np.multiply(PrebelInit.MODELS[:,0],PrebelInit.MODELS[:,2])+np.multiply(PrebelInit.MODELS[:,1],PrebelInit.MODELS[:,3])+np.multiply(25,PrebelInit.MODELS[:,4]), density=True, alpha=0.5, label='Prior')
        ax[0].hist(np.multiply(Postbel1.SAMPLES[:,0],Postbel1.SAMPLES[:,2])+np.multiply(Postbel1.SAMPLES[:,1],Postbel1.SAMPLES[:,3])+np.multiply(25,Postbel1.SAMPLES[:,4]), density=True, alpha=0.5, label='Iteration 1')
        ax[0].hist(np.multiply(Postbel3.SAMPLES[:,0],Postbel3.SAMPLES[:,2])+np.multiply(Postbel3.SAMPLES[:,1],Postbel3.SAMPLES[:,3])+np.multiply(25,Postbel3.SAMPLES[:,4]), density=True, alpha=0.5, label='Iteration 3')
        ax[0].hist(np.multiply(PostbelLast.SAMPLES[:,0],PostbelLast.SAMPLES[:,2])+np.multiply(PostbelLast.SAMPLES[:,1],PostbelLast.SAMPLES[:,3])+np.multiply(25,PostbelLast.SAMPLES[:,4]), density=True, alpha=0.5, label='Last iteration')
        ax[0].plot([totalWater, totalWater],np.asarray(ax[0].get_ylim()),'r',label='Benchmark')
        ax[0].set_xlabel(r'Total water content estimation [$m^3/m^2$]')
        ax[0].set_ylabel('Probability density estimation')
        ax[0].set_title('Correlated variables')
        ax[0].legend()
        ax[0].set_xlim(0,40)
        nbWaters = 1000000
        idxs = np.random.choice(np.arange(len(PrebelInit.MODELS[:,0])),(nbWaters,5))
        totalWaterDistribUncorrPrior = np.multiply(PrebelInit.MODELS[idxs[:,0],0],PrebelInit.MODELS[idxs[:,1],2])+np.multiply(PrebelInit.MODELS[idxs[:,2],1],PrebelInit.MODELS[idxs[:,3],3])+np.multiply(25,PrebelInit.MODELS[idxs[:,4],4])
        idxs = np.random.choice(np.arange(len(Postbel1.SAMPLES[:,0])),(nbWaters,5))
        totalWaterDistribUncorr1 = np.multiply(Postbel1.SAMPLES[idxs[:,0],0],Postbel1.SAMPLES[idxs[:,1],2])+np.multiply(Postbel1.SAMPLES[idxs[:,2],1],Postbel1.SAMPLES[idxs[:,3],3])+np.multiply(25,Postbel1.SAMPLES[idxs[:,4],4])
        idxs = np.random.choice(np.arange(len(Postbel3.SAMPLES[:,0])),(nbWaters,5))
        totalWaterDistribUncorr3 = np.multiply(Postbel3.SAMPLES[idxs[:,0],0],Postbel3.SAMPLES[idxs[:,1],2])+np.multiply(Postbel3.SAMPLES[idxs[:,2],1],Postbel3.SAMPLES[idxs[:,3],3])+np.multiply(25,Postbel3.SAMPLES[idxs[:,4],4])
        idxs = np.random.choice(np.arange(len(PostbelLast.SAMPLES[:,0])),(nbWaters,5))
        totalWaterDistribUncorrLast = np.multiply(PostbelLast.SAMPLES[idxs[:,0],0],PostbelLast.SAMPLES[idxs[:,1],2])+np.multiply(PostbelLast.SAMPLES[idxs[:,2],1],PostbelLast.SAMPLES[idxs[:,3],3])+np.multiply(25,PostbelLast.SAMPLES[idxs[:,4],4])
        ax[1].hist(totalWaterDistribUncorrPrior, density=True, alpha=0.5, label='Prior')
        ax[1].hist(totalWaterDistribUncorr1, density=True, alpha=0.5, label='Iteration 1')
        ax[1].hist(totalWaterDistribUncorr3, density=True, alpha=0.5, label='Iteration 3')
        ax[1].hist(totalWaterDistribUncorrLast, density=True, alpha=0.5, label='Last iteration')
        ax[1].plot([totalWater, totalWater],np.asarray(ax[1].get_ylim()),'r',label='Benchmark')
        ax[1].set_xlabel(r'Total water content estimation [$m^3/m^2$]')
        ax[1].set_ylabel('Probability density estimation')
        ax[1].set_title('Uncorrelated variables')
        ax[1].legend()
        ax[1].set_xlim(0,40)
        plt.tight_layout()
        print('The cross-correation is : \n{}'.format(np.corrcoef(PostbelLast.SAMPLES, rowvar=False)))
        plt.show(block=True)