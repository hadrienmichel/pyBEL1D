'''In this script, all the operations that are presented in the paper called 
"Using Iterative Prior Resampling to improve Bayesian Evidential Learning 1D
imaging (BEL1D): application to surface waves" are performed and explained.

The different graphs that are originating from the python script are also 
outputted here.
'''
if __name__=="__main__": # To prevent recomputation when in parallel
    #########################################################################################
    ###           Import the different libraries that are used in the script              ###
    #########################################################################################
    ## Common libraries:
    import numpy as np                              # For matrix-like operations and storage
    import os                                       # For files structures and read/write operations
    from os import listdir                          # To retreive elements from a folder
    from os.path import isfile, join                # Common files operations
    from matplotlib import pyplot                   # For graphics on post-processing
    import time                                     # For simple timing measurements

    ## Libraries for parallel computing:
    from pathos import multiprocessing as mp        # Multiprocessing utilities (get CPU cores info)
    from pathos import pools as pp                  # Building the pool to use for computations

    ## BEL1D requiered libraries:
    from scipy import stats                         # To build the prior model space
    from pyBEL1D import BEL1D                       # The main code for BEL1D
    from pyBEL1D.utilities import Tools             # For further post-processing
    from pyBEL1D.utilities.Tools import multiPngs   # For saving the figures as png

    ## Forward modelling code:
    from pysurf96 import surf96                     # Code for the forward modelling of dispersion curves
    #########################################################################################
    ###                    Flags for the different computation possible                   ###
    #########################################################################################
    '''
    For reproductibility of the results, we can fix the random seed.
    To fix the random seed, set RamdomSeed to False. Otherwise, the
    seed will be provided by the operating system.
    Note that the results exposed in the publication are performed
    under Windows 10 running python 3.7.6 (numpy=1.16.5, scikit-
    learn=0.23.1 and scipy=1.5.0).
    We observed that the random function does not necesseraly produce
    exactly the same results under other environments (and python 
    versions)!
    '''
    RandomSeed = False          # If True, use true random seed, else (False), fixed for reproductibility (seed=0)
    '''
    Some input parameters, to obtain some results or others.
    Eventhough the computations are relativelly fast, producing the 
    different graphs might be very cumbersome (matplotlib produces 
    nice figures, but is very slow).
    '''
    Graphs = True               # Obtain all the graphs?
    ParallelComputing = True    # Use parallel computing whenever possible?
    BenchmarkCompute = True     # Compute the results for the benchmark model?
    TestOtherNbLayers = False   # Testing the benchmark model with more layers than what is really in the model. Only active if BenchmarkCompute is.
    MirandolaCompute = False    # Compute the results for the Mirandola case study?
    DiscussionCompute = False   # Compute the necessary results for the discussion? WARNING: VERY LONG COMPUTATIONS
    verbose = True              # Output all the details about the current progress of the computations
    stats = True                # Parameter for the computation/retrun of statistics along with the iterations.
    #########################################################################################
    ###                            Initilizing the parallel pool                          ###
    #########################################################################################
    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count()) # Create the parallel pool with at most the number of dimensions
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing
    #########################################################################################
    ###                           Defining the benchmark model                            ###
    #########################################################################################
    def buildMODELSET():
        '''BUILDMODELSET is a function that will build the benchmark model.
        It does not take any arguments. '''
        # Values for the benchmark model parameters: 
        TrueModel = np.asarray([0.01, 0.05, 0.120, 0.280, 0.600])   # Thickness and Vs for the 3 layers (variable of the problem)
        Vp = np.asarray([0.300, 0.750, 1.5])                        # Vp for the 3 layers
        rho = np.asarray([1.5, 1.9, 2.2])                           # rho for the 3 layers
        nLayer = 3                                                  # Number of layers in the model
        Frequency = np.logspace(0.1,1.5,50)                         # Frequencies at which the signal is simulated
        Periods = np.divide(1,Frequency)                            # Corresponding periods
        # Forward modelling using surf96:
        Dataset = surf96(thickness=np.append(TrueModel[0:nLayer-1], [0]),vp=Vp,vs=TrueModel[nLayer-1:2*nLayer-1],rho=rho,periods=Periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
        # Building the noise model (Boaga et al., 2011)
        ErrorModelSynth = [0.075, 20]
        NoiseEstimate = np.asarray(np.divide(ErrorModelSynth[0]*Dataset*1000 + np.divide(ErrorModelSynth[1],Frequency),1000)) # Standard deviation for all measurements in km/s
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
        return TrueModel, Periods, Dataset, NoiseEstimate, ModelSynthetic
    #########################################################################################
    ###                         Synthetic case for Vs and e only                          ###
    #########################################################################################
    if BenchmarkCompute:
        print('\n\n\nComputing for the benchmark model!\n\n\n')
        ### For reproductibility - Random seed fixed
        if not(RandomSeed):
            np.random.seed(0) # For reproductibilty
            from random import seed
            seed(0)
        ### End random seed fixed
        
        # Initializing the model:
        TrueModel, Periods, Dataset, NoiseEstimate, ModelSynthetic = buildMODELSET()
        nbModelsBase = 1000
        def MixingFunc(iter:int) -> float:
            return 1# Always keeping the same proportion of models as the initial prior
        if stats:
            Prebel, Postbel, PrebelInit, stats = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,
                                                           nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True, Mixing=MixingFunc,
                                                           Graphs=Graphs, TrueModel=TrueModel, verbose=verbose)
        else:
            Prebel, Postbel, PrebelInit = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,
                                                    nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,Mixing=None,Graphs=Graphs, TrueModel=TrueModel)
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
                CurrentGraph.plot(Periods, Dataset+NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, Dataset-NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, Dataset+2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, Dataset-2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, Dataset,'k')
                PostbelInit.ShowPostModels(TrueModel=TrueModel, RMSE=True) #, NoiseModel=NoiseEstimate)
                CurrentGraph = pyplot.gcf()
                CurrentAxes = CurrentGraph.get_axes()[0]
                CurrentAxes.set_xlim(left=0,right=1)
                CurrentAxes.set_ylim(bottom=0.150, top=0.0)
                CurrentGraph.suptitle("BEL1D",fontsize=16)
            if True: # Comparison iterations?
                # Graphs for the iterations:
                Postbel.ShowDataset(RMSE=True,Prior=True)#,Parallelization=[True,pool])
                CurrentGraph = pyplot.gcf()
                CurrentGraph = CurrentGraph.get_axes()[0]
                CurrentGraph.plot(Periods, Dataset+NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, Dataset-NoiseEstimate,'k--')
                CurrentGraph.plot(Periods, Dataset+2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, Dataset-2*NoiseEstimate,'k:')
                CurrentGraph.plot(Periods, Dataset,'k')
                Postbel.ShowPostCorr(TrueModel=TrueModel,OtherMethod=PrebelInit.MODELS, alpha=[0.05, 1])
                Postbel.ShowPostModels(TrueModel=TrueModel,RMSE=True) #, NoiseModel=NoiseEstimate)#,Parallelization=[True, pool])
                CurrentGraph = pyplot.gcf()
                CurrentAxes = CurrentGraph.get_axes()[0]
                CurrentAxes.set_xlim(left=0,right=1)
                CurrentAxes.set_ylim(bottom=0.150, top=0.0)
                CurrentGraph.suptitle("BEL1D + IPR",fontsize=16)
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
                # We consider a burn-in period of 75%:
                DREAM=McMC[int(len(McMC)*3/4):,:5] # The last 2 columns are the likelihood and the log-likelihood, which presents no interest here
                # DREAM = np.unique(DREAM,axis=0)
                print('Number of models in the postrior: \n\t-BEL1D: {}\n\t-DREAM: {}'.format(len(Postbel.SAMPLES[:,1]),len(DREAM[:,1])))
                Postbel.ShowPostCorr(TrueModel=TrueModel, OtherMethod=DREAM, OtherInFront=True, alpha=[0.02, 0.06]) # They are 3 times more models for BEL1D than DREAM
                DREAM_Models, DREAM_Data = Postbel.DataPost(Parallelization=ppComp, OtherModels=DREAM)
                Postbel.ShowPostModels(TrueModel=TrueModel, RMSE=True, OtherModels=DREAM, OtherData=DREAM_Data, OtherRMSE=True)
                CurrentGraph = pyplot.gcf()
                CurrentAxes = CurrentGraph.get_axes()[0]
                CurrentAxes.set_xlim(left=0,right=1)
                CurrentAxes.set_ylim(bottom=0.100, top=0.0)
                CurrentGraph.suptitle("DREAM",fontsize=16)
            
            if True: # Comparison MCMC/rejection?
                ### For reproductibility - Random seed fixed
                if not(RandomSeed):
                    np.random.seed(0) # For reproductibilty
                    from random import seed
                    seed(0)
                ### End random seed fixed
                ## Testing the McMC algorithm after BEL1D with IPR:
                print('Executing MCMC on PREBEL . . .')
                ## Executing MCMC on the prior:
                MCMC_Init, MCMC_Init_Data = PrebelInit.runMCMC(Dataset=Dataset, nbChains=20, NoiseModel=NoiseEstimate, verbose=verbose)# 10 independant chains of 50000 models
                ## Extracting the after burn-in models (last 75%)
                MCMC = []
                MCMC_Data = []
                for i in range(MCMC_Init.shape[0]):
                    for j in np.arange(int(MCMC_Init.shape[1]/4*3),MCMC_Init.shape[1]):
                        MCMC.append(np.squeeze(MCMC_Init[i,j,:]))
                        MCMC_Data.append(np.squeeze(MCMC_Init_Data[i,j,:]))
                MCMC_Init = np.asarray(MCMC)
                MCMC_Init_Data = np.asarray(MCMC_Data)
                Postbel.ShowPostModels(TrueModel=TrueModel, RMSE=True, OtherModels=MCMC_Init, OtherData=MCMC_Init_Data) #, NoiseModel=NoiseEstimate)
                CurrentGraph = pyplot.gcf()
                CurrentAxes = CurrentGraph.get_axes()[0]
                CurrentAxes.set_xlim(left=0,right=1)
                CurrentAxes.set_ylim(bottom=0.100, top=0.0)
                CurrentGraph.suptitle("BEL1D + IPR + Rejection",fontsize=16)
                
                ## Exectuing MCMC on the posterior:
                print('Executing MCMC on POSTBEL . . .')
                MCMC_Final, MCMC_Final_Data = Postbel.runMCMC(nbChains=20, NoiseModel=NoiseEstimate, verbose=verbose)# 10 independant chains of 10000 models
                ## Extracting the after burn-in models (last 75%)
                MCMC = []
                MCMC_Data = []
                for i in range(MCMC_Final.shape[0]):
                    for j in np.arange(int(MCMC_Final.shape[1]/4*3),MCMC_Final.shape[1]):
                        MCMC.append(np.squeeze(MCMC_Final[i,j,:]))
                        MCMC_Data.append(np.squeeze(MCMC_Final_Data[i,j,:]))
                MCMC_Final = np.asarray(MCMC)
                MCMC_Final_Data = np.asarray(MCMC_Data)
                Postbel.ShowPostModels(TrueModel=TrueModel, RMSE=True, OtherModels=MCMC_Final, OtherData=MCMC_Final_Data) #, NoiseModel=NoiseEstimate)
                CurrentGraph = pyplot.gcf()
                CurrentAxes = CurrentGraph.get_axes()[0]
                CurrentAxes.set_xlim(left=0,right=1)
                CurrentAxes.set_ylim(bottom=0.100, top=0.0)
                CurrentGraph.suptitle("BEL1D + IPR + McMC",fontsize=16)
                
                print('Executing rejection on the BEL1D models . . .')
                ModelsRejection, DataRejection = Postbel.runRejection(Parallelization=ppComp, NoiseModel=NoiseEstimate, verbose=verbose)
                Postbel.ShowPostModels(TrueModel=TrueModel, RMSE=True, OtherModels=ModelsRejection, OtherData=DataRejection) #, NoiseModel=NoiseEstimate)
                CurrentGraph = pyplot.gcf()
                CurrentAxes = CurrentGraph.get_axes()[0]
                CurrentAxes.set_xlim(left=0,right=1)
                CurrentAxes.set_ylim(bottom=0.100, top=0.0)
                CurrentGraph.suptitle("BEL1D + IPR + Rejection",fontsize=16)

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
                fig.legend(handles=[patch0, patch5, patch1, patch2, patch3, patch4],
                           labels=["Benchmark", "Prior", "BEL1D + IPR", "McMC", "BEL1D + IPR + McMC", "BEL1D + IPR + Rejection"], 
                           loc="upper center", ncol=6)
                for ax in axs.flat:
                    ax.label_outer()
                pyplot.tight_layout(rect=(0,0,1,0.975))
                pyplot.show(block=False)

            # Stop execution to display the graphs:
            multiPngs('BenchmarkFigs')
            pyplot.show()
        ##########
        # Testing the model with more layers (4, 5 and 6)
        ##########
        # We need to rebuild the MODELSET structure since the forward cannot be exctly the same (more layers means that the fixed parameters must change as well)
        if TestOtherNbLayers:
            ### For reproductibility - Random seed fixed
            if not(RandomSeed):
                np.random.seed(0) # For reproductibilty
                from random import seed
                seed(0)
            ### End random seed fixed
            Frequency = np.logspace(0.1,1.5,50)
            Postbel.ShowPostModels(TrueModel=TrueModel,RMSE=True) #, NoiseModel=NoiseEstimate)
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
                paramNames = {"NamesFU":NamesFullUnits,
                              "NamesSU":NamesShortUnits, 
                              "NamesS":NamesShort, 
                              "NamesGlobal":NFull, 
                              "NamesGlobalS":["Depth\\ [km]", "Vs\\ [km/s]", "Vp\\ [km/s]", "\\rho\\ [T/m^3]"],
                              "DataUnits":"[km/s]",
                              "DataName":"Phase\\ velocity\\ [km/s]",
                              "DataAxis":"Periods\\ [s]"}
                forward = {"Fun":forwardFun,"Axis":Periods}
                cond = lambda model: (np.logical_and(np.greater_equal(model,Mins),np.less_equal(model,Maxs))).all()
                # Initialize the model parameters for BEL1D
                ModelSynthetic = BEL1D.MODELSET(prior=ListPrior,cond=cond,method=method,forwardFun=forward,paramNames=paramNames,nbLayer=nLayer)
                timeIn = time.time()
                Prebel, Postbel, PrebelInit = BEL1D.IPR(MODEL=ModelSynthetic,Dataset=Dataset,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,
                                                        nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=False, Mixing=MixingFunc,Graphs=False, verbose=verbose)
                timeOut = time.time()
                print(f'Run for {nLayer} layers done in {timeOut-timeIn} seconds')
                Postbel.ShowPostModels(TrueModel=TrueModel, RMSE=True) #, NoiseModel=NoiseEstimate)
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
    if MirandolaCompute:
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
        RMSE_Noise = np.sqrt(np.square(NoiseEstimate).mean(axis=-1))
        print('The RMSE for the clean dataset with 1 times the standard deviation is: {} km/s'.format(RMSE_Noise))
        nbModelsBase = 10000
        Prebel, Postbel, PrebelInit, stats = BEL1D.IPR(MODEL=ModelSetMIR,Dataset=DatasetMIR,NoiseEstimate=NoiseEstimate,Parallelization=ppComp,nbModelsBase=nbModelsBase,nbModelsSample=nbModelsBase,stats=True, Mixing=MixingFunc,Graphs=False, verbose=verbose)
        Postbel.ShowPostCorr(OtherMethod=PrebelInit.MODELS, alpha=[0.05, 1])
        Postbel.ShowPostModels(RMSE=True) #, NoiseModel=NoiseEstimate)
        Postbel.ShowDataset(RMSE=True, Prior=True)
        fig = pyplot.gcf()
        ax = fig.axes[0]
        DataPath = "Data/DC/Mirandola_InterPACIFIC/"
        files = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        for currFile in files:
            DatasetOther = np.loadtxt(DataPath+currFile)
            DatasetOther = np.divide(DatasetOther[:,1],1000) # Dataset for surf96 in km/s
            DatasetOther[DatasetOther==0] = np.nan
            ax.plot(np.divide(1,FreqMIR), DatasetOther,color='w',marker= '.', linestyle='None', markeredgecolor='none')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR+NoiseEstimate,'k--')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR-NoiseEstimate,'k--')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR+2*NoiseEstimate,'k:')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR-2*NoiseEstimate,'k:')
        ax.plot(np.divide(1,FreqMIR),DatasetMIR,'k',linewidth=2) # Adding the field dataset on top of the graph
        

        ## Running the rejection sampling:
        Rejection, RejectionData = Postbel.runRejection(Parallelization=ppComp, NoiseModel=NoiseEstimate, verbose=True)
        Postbel.ShowPostModels(RMSE=True, OtherModels=Rejection, OtherData=RejectionData) #, NoiseModel=NoiseEstimate)
        Postbel.ShowDataset(RMSE=True, Prior=True, OtherData=RejectionData)
        fig = pyplot.gcf()
        ax = fig.axes[0]
        DataPath = "Data/DC/Mirandola_InterPACIFIC/"
        files = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        for currFile in files:
            DatasetOther = np.loadtxt(DataPath+currFile)
            DatasetOther = np.divide(DatasetOther[:,1],1000) # Dataset for surf96 in km/s
            DatasetOther[DatasetOther==0] = np.nan
            ax.plot(np.divide(1,FreqMIR), DatasetOther,color='w',marker= '.', linestyle='None', markeredgecolor='none')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR+NoiseEstimate,'k--')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR-NoiseEstimate,'k--')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR+2*NoiseEstimate,'k:')
        ax.plot(np.divide(1,FreqMIR), DatasetMIR-2*NoiseEstimate,'k:')
        ax.plot(np.divide(1,FreqMIR),DatasetMIR,'k',linewidth=2) # Adding the field dataset on top of the graph

        Postbel.ShowPostCorr(OtherMethod=PrebelInit.MODELS, alpha=[0.05, 1], OtherModels=Rejection)

        fig, ax = pyplot.subplots()
        ax.hist(np.sum(PrebelInit.MODELS[:,:2],axis=1)*1000,density=True,label='Prior', alpha=0.5)
        ax.hist(np.sum(Postbel.SAMPLES[:,:2],axis=1)*1000,density=True,label='Posterior (BEL1D+IPR)', alpha=0.5)
        ax.hist(np.sum(Rejection[:,:2],axis=1)*1000,density=True,label='Posterior (BEL1D+IPR+Rejection)', alpha=0.5)
        ylim = ax.get_ylim()
        dBedrock = 118
        ax.plot([dBedrock, dBedrock],ylim,'k',label='Measured')
        ax.set_xlabel('Depth to bedrock [m]')
        ax.set_ylabel('Probability estimation [/]')
        ax.legend()

        pyplot.show(block=False)
        multiPngs('MirandolaFigs')
        pyplot.show()
    #########################################################################################
    ###                               Discussion on benchmark                             ###
    #########################################################################################    
    if DiscussionCompute:
        '''
        First, we test only with the same model for every cases. The dataset is noisy!

        For this test, we try different values for the main parameters:
            - NbModelsPrior = NbModelsPosterior --> from 100 to 25000 with 10 values on a log scale
            - Mixing ratio --> from 0.1 to 2 with 4 values on a linear space + no considerations on Mixing (None)
            - Rejection --> from 0 (no rejection) to 0.9 (90% rejection) with 5 values on a linear space

        Each test is repeated 10 times. 
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

        TrueModel, Periods, Dataset, NoiseEstimate, ModelSynthetic = buildMODELSET()
        nbModelsBase = 1000

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
                        TrueModelTest = TrueModel
                        randVal = 0
                        Dataset = Dataset + randVal*NoiseEstimate
                        if MixingParam is not None:
                            def MixingFuncTest(iter:int) -> float:
                                return MixingParam # Always keeping the same proportion of models as the initial prior
                        else:
                            MixingFuncTest = None
                        try:
                            stats = testIPR(ModelSynthetic,Dataset,NoiseEstimate,nbModelsBase,Rejection,MixingFuncTest,ppComp)
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
    #########################################################################################
    ###                               Closing the parallel pool                           ###
    #########################################################################################
    if ParallelComputing:
        pool.terminate()