from pyBEL1D import BEL1D
from pyBEL1D.BEL1D import LoadPOSTBEL, LoadPREBEL
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot, colors                   # For graphics on post-processing
import matplotlib
pyplot.rcParams['font.size'] = 18
pyplot.rcParams['figure.autolayout'] = True
pyplot.rcParams['xtick.labelsize'] = 16
pyplot.rcParams['ytick.labelsize'] = 16
from pathos import multiprocessing as mp
from pathos import pools as pp
from os import listdir

'''
MRS2021 is a script that runs all the computations for the different results 
presented at the MRS2021 conference (Strasbourg, France).

***Application of BEL1D for sNMR data interpretation***

It runs at first the numerical benchmark for a dataset that is created directly.
    - Creating the dataset 
    - Running BEL1D (initialization + first iteration)
    - Presenting graphs of the results
    - Applying IPR
    - Presenting graphs of the improved result

Author: 
Hadrien MICHEL
ULiège, UGent, F.R.S.-FNRS
hadrien[dot]michel[at]uliege[dot]be
(c) October 2021
'''

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
    # Realizing BEL1D on the benchmark dataset
    ## Initializing the models parameters and the prior:
    ModelSet = BEL1D.MODELSET().SNMR(prior=priorSNMRBenchmark, Kernel=KernelBenchmark, Timing= TimingsBenchmark)
    DatasetBenchmark = ModelSet.forwardFun["Fun"](SyntheticBenchmarkSNMR)
    Noise = np.mean(DatasetBenchmark)/5
    # print("Max signal: {} nV \nMean signal: {} nV \nStandard deviation of noise (S/N = 5): {} nV".format(np.max(DatasetBenchmark)*1e9, np.mean(DatasetBenchmark)*1e9, Noise*1e9))
    # NoisyDatasetBenchmark = DatasetBenchmark + np.random.randn(len(DatasetBenchmark))*Noise

    # RunBenchmark = True
    ParallelComputing = True
    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count())# Create the parallel pool with at most the number of available CPU cores
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing
    # if RunBenchmark:
    #     ## Creating the BEL1D instances and IPR:
    #     Prebel, Postbel, PrebelInit , stats = BEL1D.IPR(MODEL=ModelSet, Dataset=NoisyDatasetBenchmark, NoiseEstimate=Noise*1e9, Parallelization=ppComp,
    #         nbModelsBase=1000, nbModelsSample=1000, stats=True, Mixing=(lambda x: 1), Graphs=False, saveIters=True, verbose=True)
    
    RunFigs = False
    if RunFigs:
        # Get the number of iterations:
        nbIter = len(listdir('./IPR_Results/'))
        PostbelLast = LoadPOSTBEL('./IPR_Results/IPR_{}.postbel'.format(nbIter-2))# Prebel and 0
        PrebelInit = LoadPREBEL('./IPR_Results/IPR.prebel')
        # PostbelLast.ShowPostModels(TrueModel=SyntheticBenchmarkSNMR,  RMSE=True)#, Best=100)
        # plt.savefig('./MRS2021Figs/PostModels_Iter10.png',transparent=True, dpi=300)
        # ModelsRejection, _ = PostbelLast.runRejection(NoiseModel=np.ones_like(DatasetBenchmark)*Noise, verbose=True)
        PostbelLast.ShowPostCorr(TrueModel=SyntheticBenchmarkSNMR, OtherMethod=PrebelInit.MODELS, alpha=[0.5*PrebelInit.nbModels/PostbelLast.nbSamples, 0.5]) #, OtherModels=ModelsRejection)
        plt.tight_layout()
        # plt.savefig('./MRS2021Figs/PostCorr_Iter10.png',transparent=True, dpi=300)
        # plt.tight_layout()
        # ## Download the 1st iteration results and display the graphs:
        Postbel1 = LoadPOSTBEL('./IPR_Results/IPR_0.postbel')
        # Postbel1.ShowPost(prior=True, TrueModel=SyntheticBenchmarkSNMR)
        # plt.savefig('./MRS2021Figs/Post_Iter0.png',transparent=True, dpi=300)
        Postbel1.ShowPostCorr(TrueModel=SyntheticBenchmarkSNMR, OtherMethod=PrebelInit.MODELS, alpha=[0.5, 0.5])
        plt.tight_layout()
        # plt.savefig('./MRS2021Figs/PostCorr_Iter0.png',transparent=True, dpi=300)
        # MCMC_Models, _ = PrebelInit.runMCMC(Dataset=NoisyDatasetBenchmark, NoiseModel=np.ones(DatasetBenchmark.shape)*Noise, nbSamples=10000000, nbChains=10, noData=True, verbose=True)
        # np.save('MRS2021Figs/MCMC_MRS2021_Benchmark_10000000M_100hCPU.npy',MCMC_Models)
        # MCMC_Models = np.load('MRS2021Figs/MCMC_MRS2021_Benchmark_10000000M_100hCPU.npy')
        # MCMC = []
        # for i in range(MCMC_Models.shape[0]):
        #     for j in np.arange(int(MCMC_Models.shape[1]/100),MCMC_Models.shape[1]): # Burn in of 99% (very large prior)
        #         MCMC.append(np.squeeze(MCMC_Models[i,j,:]))
        # Postbel1.ShowPostCorr(TrueModel=SyntheticBenchmarkSNMR, OtherMethod=np.asarray(MCMC), OtherInFront=True, alpha=[0.25, 0.05]) # They are 3 times more models for BEL1D than DREAM
        # plt.savefig('./MRS2021Figs/MCMC_MRS2021_Benchmark_1000000M_10hCPU.png',transparent=True, dpi=300)
        Postbel3 = LoadPOSTBEL('./IPR_Results/IPR_5.postbel')

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
                ax.hist(Postbel3.SAMPLES[:,i], density=True, alpha=0.5, label='_Iter 5')
                ax.hist(PostbelLast.SAMPLES[:,i], density=True, alpha=0.5, label='_Last iter')
                ax.plot([SyntheticBenchmarkSNMR[i],SyntheticBenchmarkSNMR[i]],np.asarray(ax.get_ylim()),'r',label='_Benchmark')
            else:
                ax.hist(PrebelInit.MODELS[:,i], density=True, alpha=0.5, label='Prior')
                ax.hist(Postbel1.SAMPLES[:,i], density=True, alpha=0.5, label='Iter 1')
                ax.hist(Postbel3.SAMPLES[:,i], density=True, alpha=0.5, label='Iter 5')
                ax.hist(PostbelLast.SAMPLES[:,i], density=True, alpha=0.5, label='Last iter')
                ax.plot([SyntheticBenchmarkSNMR[i],SyntheticBenchmarkSNMR[i]],np.asarray(ax.get_ylim()),'r',label='Benchmark')
            ax.set_xlabel(PostbelLast.MODPARAM.paramNames["NamesFU"][i])
            ax.get_yaxis().set_ticks([])
        axLeg = axes[2,0]
        axLeg.set_visible(False)
        plt.tight_layout()
        fig.legend(loc='lower left')
        # plt.savefig('./MRS2021Figs/Evolution.png',transparent=True, dpi=300)

        totalWater = SyntheticBenchmarkSNMR[0]*SyntheticBenchmarkSNMR[2] + SyntheticBenchmarkSNMR[1]*SyntheticBenchmarkSNMR[3] + (75-SyntheticBenchmarkSNMR[0]-SyntheticBenchmarkSNMR[1])*SyntheticBenchmarkSNMR[4]
        _, ax = plt.subplots(2,1)
        ax[0].hist(np.multiply(PrebelInit.MODELS[:,0],PrebelInit.MODELS[:,2])+np.multiply(PrebelInit.MODELS[:,1],PrebelInit.MODELS[:,3])+np.multiply(25,PrebelInit.MODELS[:,4]), density=True, alpha=0.5, label='Prior')
        ax[0].hist(np.multiply(Postbel1.SAMPLES[:,0],Postbel1.SAMPLES[:,2])+np.multiply(Postbel1.SAMPLES[:,1],Postbel1.SAMPLES[:,3])+np.multiply(25,Postbel1.SAMPLES[:,4]), density=True, alpha=0.5, label='Iteration 1')
        ax[0].hist(np.multiply(Postbel3.SAMPLES[:,0],Postbel3.SAMPLES[:,2])+np.multiply(Postbel3.SAMPLES[:,1],Postbel3.SAMPLES[:,3])+np.multiply(25,Postbel3.SAMPLES[:,4]), density=True, alpha=0.5, label='Iteration 5')
        ax[0].hist(np.multiply(PostbelLast.SAMPLES[:,0],PostbelLast.SAMPLES[:,2])+np.multiply(PostbelLast.SAMPLES[:,1],PostbelLast.SAMPLES[:,3])+np.multiply(25,PostbelLast.SAMPLES[:,4]), density=True, alpha=0.5, label='Last iteration')
        ax[0].plot([totalWater, totalWater],np.asarray(ax[0].get_ylim()),'r',label='Benchmark')
        ax[0].set_xlabel(r'Total water content estimation [$m^3/m^2$]')
        ax[0].set_ylabel('Probability density estimation')
        ax[0].set_title('Correlated variables')
        ax[0].legend()
        ax[0].set_xlim(0,40)
        nbWaters = 1000000
        idxs = np.random.choice(np.arange(len(PrebelInit.MODELS[:,0])),(nbWaters,5))
        totalWaterDistribUncorrPrior = np.multiply(PrebelInit.MODELS[idxs[:,0],0],PrebelInit.MODELS[idxs[:,1],2])+np.multiply(PrebelInit.MODELS[idxs[:,2],1],PrebelInit.MODELS[idxs[:,3],3])+np.multiply((75-PrebelInit.MODELS[idxs[:,0],0]-PrebelInit.MODELS[idxs[:,2],1]),PrebelInit.MODELS[idxs[:,4],4])
        idxs = np.random.choice(np.arange(len(Postbel1.SAMPLES[:,0])),(nbWaters,5))
        totalWaterDistribUncorr1 = np.multiply(Postbel1.SAMPLES[idxs[:,0],0],Postbel1.SAMPLES[idxs[:,1],2])+np.multiply(Postbel1.SAMPLES[idxs[:,2],1],Postbel1.SAMPLES[idxs[:,3],3])+np.multiply((75-Postbel1.SAMPLES[idxs[:,0],0]-Postbel1.SAMPLES[idxs[:,2],1]),Postbel1.SAMPLES[idxs[:,4],4])
        idxs = np.random.choice(np.arange(len(Postbel3.SAMPLES[:,0])),(nbWaters,5))
        totalWaterDistribUncorr3 = np.multiply(Postbel3.SAMPLES[idxs[:,0],0],Postbel3.SAMPLES[idxs[:,1],2])+np.multiply(Postbel3.SAMPLES[idxs[:,2],1],Postbel3.SAMPLES[idxs[:,3],3])+np.multiply((75-Postbel3.SAMPLES[idxs[:,0],0]-Postbel3.SAMPLES[idxs[:,2],1]),Postbel3.SAMPLES[idxs[:,4],4])
        idxs = np.random.choice(np.arange(len(PostbelLast.SAMPLES[:,0])),(nbWaters,5))
        totalWaterDistribUncorrLast = np.multiply(PostbelLast.SAMPLES[idxs[:,0],0],PostbelLast.SAMPLES[idxs[:,1],2])+np.multiply(PostbelLast.SAMPLES[idxs[:,2],1],PostbelLast.SAMPLES[idxs[:,3],3])+np.multiply((75-PostbelLast.SAMPLES[idxs[:,0],0]-PostbelLast.SAMPLES[idxs[:,2],1]),PostbelLast.SAMPLES[idxs[:,4],4])
        ax[1].hist(totalWaterDistribUncorrPrior, density=True, alpha=0.5, label='Prior')
        ax[1].hist(totalWaterDistribUncorr1, density=True, alpha=0.5, label='Iteration 1')
        ax[1].hist(totalWaterDistribUncorr3, density=True, alpha=0.5, label='Iteration 5')
        ax[1].hist(totalWaterDistribUncorrLast, density=True, alpha=0.5, label='Last iteration')
        ax[1].plot([totalWater, totalWater],np.asarray(ax[1].get_ylim()),'r',label='Benchmark')
        ax[1].set_xlabel(r'Total water content estimation [$m^3/m^2$]')
        ax[1].set_ylabel('Probability density estimation')
        ax[1].set_title('Uncorrelated variables')
        ax[1].legend()
        ax[1].set_xlim(0,40)
        plt.tight_layout()
        # plt.savefig('./MRS2021Figs/CorrelationAnalysis.png',transparent=True, dpi=300)
        corr = np.corrcoef(PostbelLast.SAMPLES, rowvar=False)
        print('The cross-correation is : \n{}'.format(corr))
        print(corr[3,1])
        plt.show(block=True)

    MtRigi = True
    if MtRigi:
        from pygimli.physics import sNMR
        Dataset = "Data/sNMR/SEG2020_MtRigi.mrsd"
        Kernel = "Data/sNMR/SEG2020_MtRigi.mrsk"
        ModelParam = sNMR.MRS()
        sNMR.MRS.loadKernel(ModelParam,Kernel)
        sNMR.MRS.loadMRSI(ModelParam,Dataset)
        FieldData = np.ravel(ModelParam.dcube)
        TimingField = ModelParam.t
        Noise = 18 #nV
        # Initialize BEL1D:
        nbSampled = 10000
        priorMtRigi = np.asarray([[0.0, 7.5, 0.30, 0.80, 0.0, 0.200], [0, 0, 0.0, 0.15, 0.100, 0.400]])
        MODEL_MtRigi = BEL1D.MODELSET().SNMR(prior=priorMtRigi,Kernel=Kernel, Timing=TimingField)
        PrebelMtRigi, PostbelMtRigi, PrebelInitMtRigi = BEL1D.IPR(MODEL_MtRigi, FieldData, NoiseEstimate=18, Parallelization=ppComp, nbModelsBase=10000, nbModelsSample=10000, verbose=True)

        PostbelMtRigi.ShowPost(prior=True, priorOther=PrebelInitMtRigi.MODELS)
        plt.savefig('./MRS2021Figs/MtRigi_Post.png',transparent=True, dpi=300)
        PostbelMtRigi.ShowPostCorr(OtherMethod=PrebelInitMtRigi.MODELS, alpha=[0.2, 0.5])
        plt.savefig('./MRS2021Figs/MtRigi_PostCorr.png',transparent=True, dpi=300)
        

        fig = plt.figure(figsize=[10,10])# Creates the figure space
        ax = fig.subplots()
        ax.plot(PrebelInitMtRigi.MODELS[:,0],PrebelInitMtRigi.MODELS[:,1],'.y',alpha=1, markersize=20, markeredgecolor='none', label='Prior')
        ax.plot(PostbelMtRigi.SAMPLES[:,0],PostbelMtRigi.SAMPLES[:,1],'.b',alpha=0.2, markersize=20, markeredgecolor='none', label='Posterior')
        ax.set_xlabel(r'$e_{1} [m]$')
        ax.set_ylabel(r'$W_{1} [/]$')
        plt.tight_layout()
        plt.savefig('./MRS2021Figs/MtRigi_PostCorrZoom.png',transparent=True, dpi=300)

        PostbelMtRigi.ShowPostModels(RMSE=True, Parallelization=ppComp)

        plt.show(block=True)

    # pool.terminate()