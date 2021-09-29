'''In this script, all the operations that are presented in the paper called 
"Using Iterative Prior Resampling to improve Bayesian Evidential Learning 1D 
imaging (BEL1D) accuracy: the case of surface waves" are performed and explained.

The different graphs that are originating from the python script are also 
outputted here.
'''
from os import pardir
from matplotlib.lines import Line2D
import numpy as np

### For reproductibility - Random seed fixed
RandomSeed = False # If True, use true random seed, else (False), fixed for reproductibility (seed=0)
### End random seed fixed


if __name__=="__main__": # To prevent recomputation when in parallel

    from pyBEL1D import BEL1D
    import cProfile # For debugging and timings measurements
    import time # For simple timing measurements
    from matplotlib import pyplot # For graphics on post-processing
    from pyBEL1D.utilities import Tools # For further post-processing
    import os
    from os import listdir
    from os.path import isfile, join

    from pathos import multiprocessing as mp
    from pathos import pools as pp

    from pysurf96 import surf96
    from scipy import stats

    def multipage(filename, figs=None, dpi=300):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        pp = PdfPages(filename)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
    
    def multiPngs(folder, figs=None, dpi=300):
        import matplotlib.pyplot as plt
        from os.path import join
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        i = 1
        for fig in figs:
            fig.savefig(join(folder,'Figure{}.png'.format(i)),dpi=dpi, format='png')
            i += 1

    Graphs = True
    ParallelComputing = True
    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count())# Create the parallel pool with at most the number of dimensions
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing
    #########################################################################################
    ###                         Synthetic case for Vs and e only                          ###
    #########################################################################################
    Test1=True
    if Test1:
        ### For reproductibility - Random seed fixed
        if not(RandomSeed):
            np.random.seed(0) # For reproductibilty
            from random import seed
            seed(0)
        ### End random seed fixed
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
        # np.savetxt('MASW_BenchmarkDataset.txt', DatasetClean)
        # np.savetxt('MASW_BenchamrkNoise.txt', NoiseEstimate)
        # np.save('NoiseEstimateTest.npy',NoiseEstimate)
        # randVal = 0#np.random.randn(1)
        # print("The dataset is shifted by {} times the NoiseLevel".format(randVal))
        # Dataset = DatasetClean + randVal*NoiseEstimate
        Dataset = DatasetClean
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
        def funcSurf96(model):
            import numpy as np
            from pysurf96 import surf96
            Vp = np.asarray([0.300, 0.750, 1.5])
            rho = np.asarray([1.5, 1.9, 2.2])
            nLayer = 3
            Frequency = np.logspace(0.1,1.5,50)
            Periods = np.divide(1,Frequency)
            return surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)

        forwardFun = funcSurf96 #lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        forward = {"Fun":forwardFun,"Axis":Periods}
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
        # Initialize the model parameters for BEL1D
        nbModelsBase = 1000
        ModelSynthetic = BEL1D.MODELSET(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)

        stats = True
        def MixingFunc(iter:int) -> float:
            return 1# Always keeping the same proportion of models as the initial prior
        if stats:
            Prebel, Postbel, PrebelInit, stats = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True, Mixing=MixingFunc,Graphs=Graphs, verbose=True)
        else:
            Prebel, Postbel, PrebelInit = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,Mixing=None,Graphs=Graphs)
        if Graphs:

            # Show final results analysis:
            if True: # First iteration results?
                PostbelInit = BEL1D.POSTBEL(PrebelInit)
                PostbelInit.run(Dataset=Dataset, nbSamples=nbModelsBase,NoiseModel=NoiseEstimate)
                PostbelInit.DataPost(Parallelization=ppComp)
                PostbelInit.ShowPostCorr(TrueModel=TrueModel, OtherMethod=PrebelInit.MODELS, alpha=[0.5, 1])
                PostbelInit.ShowDataset(RMSE=True, Prior=True)
                CurrentGraph = pyplot.gcf()
                CurrentGraph = CurrentGraph.get_axes()[0]
                CurrentGraph.plot(Periods, DatasetClean+NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, DatasetClean-NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, DatasetClean+2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, DatasetClean-2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, Dataset,'k')
                PostbelInit.ShowPostModels(TrueModel=TrueModel, RMSE=True)
            if True: # Comparison iterations?
                # Graphs for the iterations:
                Postbel.ShowDataset(RMSE=True,Prior=True)#,Parallelization=[True,pool])
                CurrentGraph = pyplot.gcf()
                CurrentGraph = CurrentGraph.get_axes()[0]
                CurrentGraph.plot(Periods, DatasetClean+NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, DatasetClean-NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, DatasetClean+2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, DatasetClean-2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, Dataset,'k')
                Postbel.ShowPostCorr(TrueModel=TrueModel,OtherMethod=PrebelInit.MODELS, alpha=[0.05, 1])
                Postbel.ShowPostModels(TrueModel=TrueModel,RMSE=True)#,Parallelization=[True, pool])
                # Graph for the CCA space parameters loads
                _, ax = pyplot.subplots()
                B = PrebelInit.CCA.y_loadings_
                B = np.divide(np.abs(B).T,np.repeat(np.reshape(np.sum(np.abs(B),axis=0),(1,B.shape[0])),B.shape[0],axis=0).T)
                ind =  np.asarray(range(B.shape[0]))+1
                ax.bar(x=ind,height=B[0],label=r'${}$'.format(PrebelInit.MODPARAM.paramNames["NamesSU"][0]))
                for i in range(B.shape[0]+1)[1:-1]:
                    ax.bar(x=ind,height=B[i],bottom=np.reshape(np.sum(B[0:i],axis=0),(B.shape[0],)),label=r'${}$'.format(PrebelInit.MODPARAM.paramNames["NamesSU"][i]))
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3)
                ax.set_ylabel('Relative contribution')
                ax.set_xlabel('CCA dimension')
                ax.set_title('First iteration')
                pyplot.show(block=False)

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
                ax.set_title('Last iteration')
                pyplot.show(block=False)

                # Compare the results to McMC results:
                McMC = np.load("./Data/DC/SyntheticBenchmark/DREAM_MASW.npy")
                # We consider a burn-in period of 50%:
                DREAM=McMC[int(len(McMC)/2):,:5] # The last 2 columns are the likelihood and the log-likelihood, which presents no interest here
                # DREAM = np.unique(DREAM,axis=0)
                print('Number of models in the postrior: \n\t-BEL1D: {}\n\t-DREAM: {}'.format(len(Postbel.SAMPLES[:,1]),len(DREAM[:,1])))
                Postbel.ShowPostCorr(TrueModel=TrueModel, OtherMethod=DREAM, OtherInFront=True, alpha=[0.02, 0.06]) # They are 3 times more models for BEL1D than DREAM
            
            if False: # Comparison MCMC/rejection?
                ### For reproductibility - Random seed fixed
                if not(RandomSeed):
                    np.random.seed(0) # For reproductibilty
                    from random import seed
                    seed(0)
                ### End random seed fixed
                ## Testing the McMC algorithm after BEL1D with IPR:
                print('Executing MCMC on PREBEL . . .')
                ## Executing MCMC on the prior:
                MCMC_Init, MCMC_Init_Data = PrebelInit.runMCMC(Dataset=Dataset, nbChains=20, NoiseModel=NoiseEstimate)# 10 independant chains of 50000 models
                ## Extracting the after burn-in models (last 50%)
                MCMC = []
                MCMC_Data = []
                for i in range(MCMC_Init.shape[0]):
                    for j in np.arange(int(MCMC_Init.shape[1]/4*3),MCMC_Init.shape[1]):
                        MCMC.append(np.squeeze(MCMC_Init[i,j,:]))
                        MCMC_Data.append(np.squeeze(MCMC_Init_Data[i,j,:]))
                MCMC_Init = np.asarray(MCMC)
                MCMC_Init_Data = np.asarray(MCMC_Data)
                # Postbel.ShowPostCorr(TrueModel=TrueModel, OtherMethod=MCMC_Init)
                ## Exectuing MCMC on the posterior:
                print('Executing MCMC on POSTBEL . . .')
                MCMC_Final, MCMC_Final_Data = Postbel.runMCMC(nbChains=20, NoiseModel=NoiseEstimate)# 10 independant chains of 10000 models
                ## Extracting the after burn-in models (last 50%)
                MCMC = []
                MCMC_Data = []
                for i in range(MCMC_Final.shape[0]):
                    for j in np.arange(int(MCMC_Final.shape[1]/4*3),MCMC_Final.shape[1]):
                        MCMC.append(np.squeeze(MCMC_Final[i,j,:]))
                        MCMC_Data.append(np.squeeze(MCMC_Final_Data[i,j,:]))
                MCMC_Final = np.asarray(MCMC)
                MCMC_Final_Data = np.asarray(MCMC_Data)
                # Postbel.ShowPostCorr(TrueModel=TrueModel, OtherMethod=MCMC_Final)
                
                print('Executing rejection on the BEL1D models . . .')
                ModelsRejection, DataRejection = Postbel.runRejection(Parallelization=ppComp,NoiseModel=NoiseEstimate)
                # Postbel.ShowPostCorr(TrueModel=TrueModel, OtherMethod=ModelsRejection)
                Postbel.ShowPostModels(TrueModel=TrueModel, RMSE=True, OtherModels=ModelsRejection, OtherData=DataRejection)

                # Adding the graph with correlations: 
                nbParam = Postbel.SAMPLES.shape[1]
                fig = pyplot.figure(figsize=[10,10])# Creates the figure space
                axs = fig.subplots(nbParam, nbParam)
                for i in range(nbParam):
                    for j in range(nbParam):
                        if i == j: # Diagonal
                            if i != nbParam-1:
                                axs[i,j].get_shared_x_axes().join(axs[i,j],axs[-1,j])# Set the xaxis limit
                            axs[i,j].hist(PrebelInit.MODELS[:,j], color='gold',density=True)
                            axs[i,j].hist(Postbel.SAMPLES[:,j],color='royalblue',density=True,alpha=0.75) # Plot the histogram for the given variable
                            axs[i,j].hist(MCMC_Init[:,j],color='darkorange',density=True,alpha=0.75)
                            axs[i,j].hist(MCMC_Final[:,j],color='limegreen',density=True,alpha=0.75)
                            axs[i,j].hist(ModelsRejection[:,j],color='peru',density=True,alpha=0.75)
                            if TrueModel is not None:
                                axs[i,j].plot([TrueModel[i],TrueModel[i]],np.asarray(axs[i,j].get_ylim()),'r')
                            if nbParam > 8:
                                axs[i,j].set_xticks([])
                                axs[i,j].set_yticks([])
                        elif i > j: # Below the diagonal -> Scatter plot
                            if i != nbParam-1:
                                axs[i,j].get_shared_x_axes().join(axs[i,j],axs[-1,j])# Set the xaxis limit
                            if j != nbParam-1:
                                if i != nbParam-1:
                                    axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-1])# Set the yaxis limit
                                else:
                                    axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-2])# Set the yaxis limit
                            axs[i,j].plot(PrebelInit.MODELS[:,j], PrebelInit.MODELS[:,i], color='gold',marker= '.', linestyle='None', markeredgecolor='none')
                            axs[i,j].plot(Postbel.SAMPLES[:,j],Postbel.SAMPLES[:,i],color = 'royalblue', marker = '.', linestyle='None', alpha=0.2, markeredgecolor='none')
                            axs[i,j].plot(ModelsRejection[:,j],ModelsRejection[:,i],color='peru', marker = '.', linestyle='None', alpha=0.2, markeredgecolor='none')
                            if TrueModel is not None:
                                axs[i,j].plot(TrueModel[j],TrueModel[i],'or')
                            if nbParam > 8:
                                axs[i,j].set_xticks([])
                                axs[i,j].set_yticks([])
                        elif MCMC_Init is not None:
                            if i != nbParam-1:
                                axs[i,j].get_shared_x_axes().join(axs[i,j],axs[-1,j])# Set the xaxis limit
                            if j != nbParam-1:
                                if i != 0:
                                    axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-1])# Set the yaxis limit
                                else:
                                    axs[i,j].get_shared_y_axes().join(axs[i,j],axs[i,-2])# Set the yaxis limit
                            axs[i,j].plot(PrebelInit.MODELS[:,j], PrebelInit.MODELS[:,i], color='gold',marker= '.', linestyle='None', markeredgecolor='none')
                            axs[i,j].plot(MCMC_Init[:,j],MCMC_Init[:,i],color='darkorange', marker = '.', linestyle='None', alpha=0.2, markeredgecolor='none')
                            axs[i,j].plot(MCMC_Final[:,j],MCMC_Final[:,i],color='limegreen', marker = '.', linestyle='None', alpha=0.2, markeredgecolor='none')
                            if TrueModel is not None:
                                axs[i,j].plot(TrueModel[j],TrueModel[i],'or')
                            if nbParam > 8:
                                axs[i,j].set_xticks([])
                                axs[i,j].set_yticks([])
                        else:
                            axs[i,j].set_visible(False)
                        if j == 0: # First column of the graph
                            if ((i==0)and(j==0)) or not(i==j):
                                axs[i,j].set_ylabel(r'${}$'.format(Postbel.MODPARAM.paramNames["NamesSU"][i]))
                        if i == nbParam-1: # Last line of the graph
                            axs[i,j].set_xlabel(r'${}$'.format(Postbel.MODPARAM.paramNames["NamesSU"][j]))
                        if j == nbParam-1:
                            if not(i==j):
                                axs[i,j].yaxis.set_label_position("right")
                                axs[i,j].yaxis.tick_right()
                                axs[i,j].set_ylabel(r'${}$'.format(Postbel.MODPARAM.paramNames["NamesSU"][i]))
                        if i == 0:
                            axs[i,j].xaxis.set_label_position("top")
                            axs[i,j].xaxis.tick_top()
                            axs[i,j].set_xlabel(r'${}$'.format(Postbel.MODPARAM.paramNames["NamesSU"][j]))
                # fig.suptitle("Posterior model space visualization")
                import matplotlib.patches as mpatches
                patch0 = mpatches.Patch(facecolor='red', edgecolor='#000000')
                patch1 = mpatches.Patch(facecolor='royalblue', edgecolor='#000000') #this will create a red bar with black borders, you can leave out edgecolor if you do not want the borders
                patch2 = mpatches.Patch(facecolor='darkorange', edgecolor='#000000')
                patch3 = mpatches.Patch(facecolor='limegreen', edgecolor='#000000')
                patch4 = mpatches.Patch(facecolor='peru', edgecolor='#000000')
                patch5 = mpatches.Patch(facecolor='gold', edgecolor='#000000')
                fig.legend(handles=[patch0, patch5, patch1, patch2, patch3, patch4],labels=["Benchmark", "Prior", "BEL1D + IPR", "MCMc", "BEL1D + IPR + MCMc", "BEL1D + IPR + Rejection"], loc="upper center", ncol=6)
                for ax in axs.flat:
                    ax.label_outer()
                pyplot.tight_layout(rect=(0,0,1,0.975))
                pyplot.show(block=False)

            # Stop execution to display the graphs:
            # multipage('Benchmark.pdf',dpi=300)
            multiPngs('BenchmarkFigs')
            pyplot.show()
        ##########
        # Testing the model with more layers (4, 5 and 6)
        ##########
        # We need to rebuild the MODELSET structure since the forward cannot be exctly the same (more layers means that the fixed parameters must change as well)
        TestOtherNbLayers = False
        if TestOtherNbLayers:
            ### For reproductibility - Random seed fixed
            if not(RandomSeed):
                np.random.seed(0) # For reproductibilty
                from random import seed
                seed(0)
            ### End random seed fixed
            Postbel.ShowPostModels(TrueModel=TrueModel,RMSE=True)
            CurrentGraph = pyplot.gcf()
            CurrentAxes = CurrentGraph.get_axes()[0]
            nbLayer = 3
            TrueMod = list()
            TrueMod.append(np.cumsum(TrueModel[0:nbLayer-1]))
            TrueMod.append(TrueModel[nbLayer-1:2*nbLayer-1])
            CurrentAxes.step(np.append(TrueMod[1][:], TrueMod[1][-1]),np.append(np.append(0, TrueMod[0][:]), 0.150),where='pre',color=[0.5, 0.5, 0.5])   
            CurrentAxes.set_xlim(left=0,right=1)
            CurrentAxes.set_ylim(bottom=0.100, top=0.0)
            from scipy import stats
            prior4 = np.array([[0.0005, 0.015, 0.1, 0.18],[0.0005, 0.015, 0.1, 0.18],[0.01, 0.1, 0.25, 0.45],[0.0, 0.0, 0.5, 0.9]])
            def funcSurf96_4(model):
                import numpy as np
                from pysurf96 import surf96
                Vp = np.asarray([0.300, 0.300, 0.750, 1.5])
                rho = np.asarray([1.5, 1.5, 1.9, 2.2])
                nLayer = 4
                Frequency = np.logspace(0.1,1.5,50)
                Periods = np.divide(1,Frequency)
                return surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)

            prior5 = np.array([[0.0005, 0.015, 0.1, 0.18],[0.0005, 0.015, 0.1, 0.18],[0.005, 0.05, 0.25, 0.45],[0.005, 0.05, 0.25, 0.45],[0.0, 0.0, 0.5, 0.9]])
            def funcSurf96_5(model):
                import numpy as np
                from pysurf96 import surf96
                Vp = np.asarray([0.300, 0.300, 0.750, 0.750, 1.5])
                rho = np.asarray([1.5, 1.5, 1.9, 1.9, 2.2])
                nLayer = 5
                Frequency = np.logspace(0.1,1.5,50)
                Periods = np.divide(1,Frequency)
                return surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)

            prior6 = np.array([[0.00033, 0.01, 0.1, 0.18],[0.00033, 0.01, 0.1, 0.18],[0.00033, 0.01, 0.1, 0.18],[0.005, 0.05, 0.25, 0.45],[0.005, 0.05, 0.25, 0.45],[0.0, 0.0, 0.5, 0.9]])
            def funcSurf96_6(model):
                import numpy as np
                from pysurf96 import surf96
                Vp = np.asarray([0.300, 0.300, 0.300, 0.750, 0.750, 1.5])
                rho = np.asarray([1.5, 1.5, 1.5, 1.9, 1.9, 2.2])
                nLayer = 6
                Frequency = np.logspace(0.1,1.5,50)
                Periods = np.divide(1,Frequency)
                return surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)

            prior7 = np.array([[0.00033, 0.01, 0.1, 0.18],[0.00033, 0.01, 0.1, 0.18],[0.00033, 0.01, 0.1, 0.18],[0.0033, 0.033, 0.25, 0.45],[0.0033, 0.033, 0.25, 0.45],[0.0033, 0.033, 0.25, 0.45],[0.0, 0.0, 0.5, 0.9]])
            def funcSurf96_7(model):
                import numpy as np
                from pysurf96 import surf96
                Vp = np.asarray([0.300, 0.300, 0.300, 0.750, 0.750, 0.750, 1.5])
                rho = np.asarray([1.5, 1.5, 1.5, 1.9, 1.9, 1.9, 2.2])
                nLayer = 7
                Frequency = np.logspace(0.1,1.5,50)
                Periods = np.divide(1,Frequency)
                return surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)

            def MixingFunc(iter:int) -> float:
                return 1# Always keeping the same proportion of models as the initial prior

            for nLayer in np.arange(4,7+1):
                if nLayer == 4:
                    nbModelsBase = 2000
                    prior = prior4
                    forwardFun = funcSurf96_4
                elif nLayer == 5:
                    nbModelsBase = 4000
                    prior = prior5
                    forwardFun = funcSurf96_5
                elif nLayer == 6:
                    nbModelsBase = 8000
                    prior = prior6
                    forwardFun = funcSurf96_6
                else:
                    nbModelsBase = 16000
                    prior = prior7
                    forwardFun = funcSurf96_7
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
                forward = {"Fun":forwardFun,"Axis":Periods}
                cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
                # Initialize the model parameters for BEL1D
                ModelSynthetic = BEL1D.MODELSET(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)
                timeIn = time.time()
                Prebel, Postbel, PrebelInit = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=False, Mixing=MixingFunc,Graphs=False, verbose=True)
                timeOut = time.time()
                print(f'Run for {nLayer} layers done in {timeOut-timeIn} seconds')
                Postbel.ShowPostModels(TrueModel=TrueModel, RMSE=True)
                CurrentGraph = pyplot.gcf()
                CurrentAxes = CurrentGraph.get_axes()[0]
                nbLayer = 3
                CurrentAxes.step(np.append(TrueMod[1][:], TrueMod[1][-1]),np.append(np.append(0, TrueMod[0][:]), 0.150),where='pre',color=[0.5, 0.5, 0.5])
                CurrentAxes.set_xlim(left=0,right=1)
                CurrentAxes.set_ylim(bottom=0.100, top=0.0)
            pyplot.show()
    #########################################################################################
    ###                                 Mirandola test case                               ###
    #########################################################################################
    Test2 = False
    if Test2:
        ### For reproductibility - Random seed fixed
        if not(RandomSeed):
            np.random.seed(0) # For reproductibilty
            from random import seed
            seed(0)
        ### End random seed fixed
        ### Mirandola test case - 3 layers:
        priorMIR = np.array([[0.005, 0.05, 0.1, 0.5, 0.2, 4.0, 1.5, 3.5], [0.045, 0.145, 0.1, 0.8, 0.2, 4.0, 1.5, 3.5], [0, 0, 0.3, 2.5, 0.2, 4.0, 1.5, 3.5]]) # MIRANDOLA prior test case
        nbParam = int(priorMIR.size/2 - 1)
        nLayer, nParam = priorMIR.shape
        nParam = int(nParam/2)
        stdPrior = [None]*nbParam
        meansPrior = [None]*nbParam
        stdUniform = lambda a,b: (b-a)/np.sqrt(12)
        meansUniform = lambda a,b: (b-a)/2
        ident = 0
        for j in range(nParam):
                    for i in range(nLayer):
                        if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                            stdPrior[ident] = stdUniform(priorMIR[i,j*2],priorMIR[i,j*2+1])
                            meansPrior[ident] = meansUniform(priorMIR[i,j*2],priorMIR[i,j*2+1])
                            ident += 1
        Dataset = np.loadtxt("Data/DC/Mirandola_InterPACIFIC/Average/Average_interp60_cuttoff.txt")
        FreqMIR = Dataset[:,0]
        DatasetMIR = np.divide(Dataset[:,1],1000)# Phase velocity in km/s for the forward model
        ErrorModel = [0.075, 20]
        ModelSetMIR = BEL1D.MODELSET.DC(prior=priorMIR, Frequency=FreqMIR)
        MixingFunc = lambda iter: 1 #Return 1 whatever the iteration
        NoiseEstimate = np.asarray(np.divide(ErrorModel[0]*DatasetMIR*1000 + np.divide(ErrorModel[1],FreqMIR),1000)) # Standard deviation for all measurements in km/s
        nbModelsBase = 10000
        Prebel, Postbel, PrebelInit, stats = BEL1D.IPR(MODEL=ModelSetMIR,Dataset=DatasetMIR,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True, Mixing=MixingFunc,Graphs=False, verbose=True)
        Postbel.ShowPostCorr(OtherMethod=PrebelInit.MODELS)
        Postbel.ShowPostModels(RMSE=True)
        Postbel.ShowDataset(RMSE=True, Prior=True)
        fig = pyplot.gcf()
        ax = fig.axes[0]
        DataPath = "Data/DC/Mirandola_InterPACIFIC/"
        files = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        for currFile in files:
            DatasetOther = np.loadtxt(DataPath+currFile)
            DatasetOther = np.divide(DatasetOther[:,1],1000) # Dataset for surf96 in km/s
            DatasetOther[DatasetOther==0] = np.nan
            ax.plot(np.divide(1,FreqMIR), DatasetOther,':k')
        ax.plot(np.divide(1,FreqMIR),DatasetMIR,'k',linewidth=2) # Adding the field dataset on to of the graph
        fig, ax = pyplot.subplots()
        ax.hist(np.sum(PrebelInit.MODELS[:,:2],axis=1)*1000,density=True,label='Prior')
        ax.hist(np.sum(Postbel.SAMPLES[:,:2],axis=1)*1000,density=True,label='Posterior')
        ylim = ax.get_ylim()
        dBedrock = 118
        ax.plot([dBedrock, dBedrock],ylim,'k',label='Measured')
        ax.set_xlabel('Depth to bedrock [m]')
        ax.set_ylabel('Probability estimation [/]')
        ax.legend()
        RejectionModels, RejectionData = Postbel.runRejection(NoiseModel=NoiseEstimate)
        Postbel.ShowDataset(RMSE=True, Prior=True, OtherData=RejectionData)
        fig = pyplot.gcf()
        ax = fig.axes[0]
        DataPath = "Data/DC/Mirandola_InterPACIFIC/"
        files = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        for currFile in files:
            DatasetOther = np.loadtxt(DataPath+currFile)
            DatasetOther = np.divide(DatasetOther[:,1],1000) # Dataset for surf96 in km/s
            DatasetOther[DatasetOther==0] = np.nan
            ax.plot(np.divide(1,FreqMIR), DatasetOther,':w')
        ax.plot(np.divide(1,FreqMIR),DatasetMIR,'w',linewidth=2) # Adding the field dataset on to of the graph

        MCMC_Final, MCMC_Final_Data = Postbel.runMCMC(nbChains=10, NoiseModel=NoiseEstimate)
        ## Extracting the after burn-in models (last 50%)
        MCMC = []
        MCMC_Data = []
        for i in range(MCMC_Final.shape[0]):
            for j in np.arange(int(MCMC_Final.shape[1]/2),MCMC_Final.shape[1]):
                MCMC.append(np.squeeze(MCMC_Final[i,j,:]))
                MCMC_Data.append(np.squeeze(MCMC_Final_Data[i,j,:]))
        MCMC_Final = np.asarray(MCMC)
        MCMC_Final_Data = np.asarray(MCMC_Data)
        Postbel.ShowDataset(RMSE=True, Prior=True, OtherData=MCMC_Final_Data)
        fig = pyplot.gcf()
        ax = fig.axes[0]
        DataPath = "Data/DC/Mirandola_InterPACIFIC/"
        files = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        for currFile in files:
            DatasetOther = np.loadtxt(DataPath+currFile)
            DatasetOther = np.divide(DatasetOther[:,1],1000) # Dataset for surf96 in km/s
            DatasetOther[DatasetOther==0] = np.nan
            ax.plot(np.divide(1,FreqMIR), DatasetOther,':k')
        ax.plot(np.divide(1,FreqMIR),DatasetMIR,'k',linewidth=2) # Adding the field dataset on to of the graph

        pyplot.show(block=False)
        # multipage('Mirandola.pdf',dpi=300)
        multiPngs('MirandolaFigs')
        pyplot.show()
        
    if ParallelComputing:
        pool.terminate()

    Discussion = False
    if Discussion:
        '''
        First, we test only with the same model for every cases. The dataset is noisy!

        For this test, we try different values for the main parameters:
            - NbModelsPrior = NbModelsPosterior --> from 100 to 25000 with 10 values on a log scale
            - Mixing ratio --> from 0.1 to 2 with 4 values on a linear space + no considerations on Mixing (None)
            - Rejection --> from 0 (no rejection) to 0.9 (90% rejection) with 5 values on a linear space

        Each test is repeated 10 times (with different random noise added to the model). 
        After each pass, all the results are saved.
        '''
        ##################
        # Now that it works, we will test the different input parameters of the function:
        #   nbModelsBase
        #   nbModelsSample
        #   Mixing
        #   (Rejection)
        # For each case: testing variations whithin given range + repeat 100 times -> analysis of only the statistics
        ###################
        ### For reproductibility - Random seed fixed
        if not(RandomSeed):
            np.random.seed(0) # For reproductibilty
            from random import seed
            seed(0)
        ### End random seed fixed
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
        # np.savetxt('MASW_BenchmarkDataset.txt', DatasetClean)
        # np.savetxt('MASW_BenchamrkNoise.txt', NoiseEstimate)
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
        def funcSurf96(model):
            import numpy as np
            from pysurf96 import surf96
            Vp = np.asarray([0.300, 0.750, 1.5])
            rho = np.asarray([1.5, 1.9, 2.2])
            nLayer = 3
            Frequency = np.logspace(0.1,1.5,50)
            Periods = np.divide(1,Frequency)
            return surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)

        forwardFun = funcSurf96 #lambda model: surf96(thickness=np.append(model[0:nLayer-1], [0]),vp=Vp,vs=model[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        forward = {"Fun":forwardFun,"Axis":Periods}
        cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
        # Initialize the model parameters for BEL1D
        nbModelsBase = 1000
        ModelSynthetic = BEL1D.MODELSET(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)

        #from wrapt_timeout_decorator import timeout
        timeMax = 60*60 #Number of seconds before timeout
        #@timeout(timeMax)
        def testIPR(ModelSynthetic,Dataset,NoiseEstimate,nbModelsBase,Rejection,MixingFuncTest,ParallelParam):
            from pyBEL1D import BEL1D
            import numpy as np
            try:
                _, _, _, stats = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset,NoiseEstimate=NoiseEstimate,nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True,Rejection=Rejection,Mixing=MixingFuncTest,Graphs=False,Parallelization=ParallelParam)
            except Exception as e:
                print(e)
                stats = None
            return stats
        pool = pp.ProcessPool(mp.cpu_count())# Create the parallel pool with at most the number of dimensions
        nbTestN, nbTestM, nbTestR, nbRepeat = (10, 5, 5, 10)
        valTestModels = np.logspace(np.log10(100),np.log10(25000),nbTestN,dtype=np.int) # Tests between 100 and 100000 models in the initial prior/sampleing
        valTestMixing = np.linspace(0.1,2.0,nbTestM-1) # Tests between 0.1 and 2 for the mixing of prior/posterior
        valTestMixing = np.append(valTestMixing,None)
        valTestRejection = np.linspace(0,0.9,nbTestR) # Tests between 0 and 0.9 for the probability of rejection (only keeping the best fit)
        # Initialize the lists with the values:
        TrueModelTest = Tools.Sampling(ModelSynthetic.prior,ModelSynthetic.cond,1)
        LenModels = np.shape(TrueModelTest)[1]
        print('\n\nBeginning testings . . . \n\n')
        nbIter = np.empty((nbTestN, nbTestM, nbTestR, nbRepeat))
        cpuTime = np.empty((nbTestN, nbTestM, nbTestR, nbRepeat))
        meansEnd = np.empty((nbTestN, nbTestM, nbTestR, nbRepeat, LenModels))
        stdsEnd = np.empty((nbTestN, nbTestM, nbTestR, nbRepeat, LenModels))
        distEnd = np.empty((nbTestN, nbTestM, nbTestR, nbRepeat))
        TrueModels = np.empty((nbTestN, nbTestM, nbTestR, nbRepeat, LenModels))
        randVals = np.empty((nbTestN, nbTestM, nbTestR, nbRepeat))
        k = 0
        for repeat in range(nbRepeat):
            for idxNbModels, nbModelsBase in enumerate(valTestModels):
                for idxMixing, MixingParam in enumerate(valTestMixing):
                    for idxReject, Rejection in enumerate(valTestRejection):
                        k += 1
                        print('Test {} on {} (valTest: nbModels = {}, Mixing = {}, Rejection = {})'.format(k,nbTestN*nbTestM*nbTestR*nbRepeat,nbModelsBase, MixingParam, Rejection))
                        # thresholdValue = 0.2
                        # while True:
                        #     TrueModelTest = Tools.Sampling(ModelSynthetic.prior,ModelSynthetic.cond,1)
                        #     try:
                        #         Dataset = ModelSynthetic.forwardFun["Fun"](TrueModelTest[0,:])
                        #         VariabilityMax = np.max(np.diff(Dataset))
                        #         if VariabilityMax > thresholdValue:
                        #             pass
                        #         else:
                        #             NoiseEstimate = np.asarray(np.divide(ErrorModelSynth[0]*Dataset*1000 + np.divide(ErrorModelSynth[1],Frequency),1000))
                        #             break
                        #     except:
                        #         pass
                        TrueModelTest = TrueModel
                        randVal = 0 #np.random.randn(1) --> To better compare to the results from DREAM
                        Dataset = DatasetClean + randVal*NoiseEstimate
                        if MixingParam is not None:
                            def MixingFuncTest(iter:int) -> float:
                                return MixingParam # Always keeping the same proportion of models as the initial prior
                        else:
                            MixingFuncTest = None
                        try:
                            stats = testIPR(ModelSynthetic,Dataset,NoiseEstimate,nbModelsBase,Rejection,MixingFuncTest,[True, pool])
                            # Processing of the results:
                            if stats is not None:
                                nbIter[idxNbModels,idxMixing,idxReject,repeat] = len(stats)
                                cpuTime[idxNbModels,idxMixing,idxReject,repeat] = stats[-1].timing
                                meansEnd[idxNbModels,idxMixing,idxReject,repeat,:] = stats[-1].means
                                stdsEnd[idxNbModels,idxMixing,idxReject,repeat,:] = stats[-1].stds
                                distEnd[idxNbModels,idxMixing,idxReject,repeat] = stats[-1].distance
                                TrueModels[idxNbModels,idxMixing,idxReject,repeat,:] = TrueModelTest # [0,:]
                                randVals[idxNbModels,idxMixing,idxReject,repeat] = randVal
                                print('Finished in {} iterations ({} seconds).'.format(len(stats),stats[-1].timing))
                            else:
                                nbIter[idxNbModels,idxMixing,idxReject,repeat] = np.nan
                                cpuTime[idxNbModels,idxMixing,idxReject,repeat] = np.nan
                                stdsNaN = TrueModelTest #[0,:]
                                stdsNaN[:] = np.nan
                                meansEnd[idxNbModels,idxMixing,idxReject,repeat,:] = stdsNaN
                                stdsEnd[idxNbModels,idxMixing,idxReject,repeat,:] = stdsNaN
                                distEnd[idxNbModels,idxMixing,idxReject,repeat] = np.nan
                                TrueModels[idxNbModels,idxMixing,idxReject,repeat,:] = TrueModelTest # [0,:]
                                randVals[idxNbModels,idxMixing,idxReject,repeat] = randVal
                                print('Did not finish! (ERROR)')
                        except Exception as e:
                            print(e)
                            nbIter[idxNbModels,idxMixing,idxReject,repeat] = np.nan
                            cpuTime[idxNbModels,idxMixing,idxReject,repeat] = np.nan
                            stdsNaN = TrueModelTest #[0,:]
                            stdsNaN[:] = np.nan
                            meansEnd[idxNbModels,idxMixing,idxReject,repeat,:] = stdsNaN
                            stdsEnd[idxNbModels,idxMixing,idxReject,repeat,:] = stdsNaN
                            distEnd[idxNbModels,idxMixing,idxReject,repeat] = np.nan
                            TrueModels[idxNbModels,idxMixing,idxReject,repeat,:] = TrueModelTest #[0,:]
                            randVals[idxNbModels,idxMixing,idxReject,repeat] = randVal
                            print('Did not finish! (TIMEOUT after 1 hour)')
            # Savingf the results after each pass:
            print('\n \n \n \t Pass {} of {} over! \n Moving on . . . \n \n \n'.format(repeat+1, nbRepeat))
            cwd = os.getcwd()
            directory = os.path.join(cwd,'testingInitModelsNoNoiseAdded/{}'.format(repeat))
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save(os.path.join(directory,'nbIter'),nbIter)
            np.save(os.path.join(directory,'cpuTime'),cpuTime)
            np.save(os.path.join(directory,'meansEnd'),meansEnd)
            np.save(os.path.join(directory,'stdsEnd'),stdsEnd)
            np.save(os.path.join(directory,'distEnd'),distEnd)
            np.save(os.path.join(directory,'TrueModels'),TrueModels)
            np.save(os.path.join(directory,'randVals'),randVals)
        # pool.terminate()
        # Final pass saving of the results
        cwd = os.getcwd()
        directory = os.path.join(cwd,'testingInitModelsNoNoiseAdded/final')
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory,'nbIter'),nbIter)
        np.save(os.path.join(directory,'cpuTime'),cpuTime)
        np.save(os.path.join(directory,'meansEnd'),meansEnd)
        np.save(os.path.join(directory,'stdsEnd'),stdsEnd)
        np.save(os.path.join(directory,'distEnd'),distEnd)
        np.save(os.path.join(directory,'TrueModels'),TrueModels)
        np.save(os.path.join(directory,'randVals'),randVals)
