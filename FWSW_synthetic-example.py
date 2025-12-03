'''Here we show the application on the synthetic benchmark of Tokimatsu et al. (1992)
for model 1 to 3.
'''
import matplotlib.pyplot as plt

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
    # try:
    #     from pysurf96 import surf96                     # Code for the forward modelling of dispersion curves
    # except:
    #     pass
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
    ParallelComputing = False   # Use parallel computing whenever possible?
    FWImage = True              # Projection of dispersion images
    verbose = True              # Output all the details about the current progress of the computations
    stats = True                # Parameter for the computation/retrun of statistics along with the iterations.
    #########################################################################################
    ###                            Initializing the parallel pool                          ###
    #########################################################################################
    if ParallelComputing:
        pool = pp.ProcessPool(mp.cpu_count()) # Create the parallel pool with at most the number of dimensions
        ppComp = [True, pool]
    else:
        ppComp = [False, None] # No parallel computing

    #########################################################################################
    ###                         FW image on synthetic benchmark                           ###
    #########################################################################################
    if FWImage:

        ## PRIORS
        # uniform priors are structured as follows:
        # prior = np.array([[th1_min, th1_max, Vs1_min, Vs1_max, Vp1_min, Vp1_max], #rho1_min, rho1_max, Qalphas1_min, Qalphas2_max, Qbetas1_min, Qbetas1_max
        #                   [th2_min, th2_max, Vs2_min, ...],
        #                   [th3_min, ...],
        #                   ...
        #                   ])

        # # UNIFORM prior SYNTHETIC (Tokimatsu et al. 1992)
        prior = np.array([[0.001, 0.003, 0.05, 0.25, 0.1, 0.5], #, 1., 2.5],
                          [0.002, 0.006, 0.05, 0.25, 0.5, 1.5], # 1., 2.5],
                          [0.005, 0.01, 0.05, 0.25, 1.0, 2.0], # 1., 2.5],
                          [0, 0, 0.2, 0.5, 1.0, 2.0]]) #, 1., 2.5]])
        # with attenuation parameters Qalphas (Qp) and Qbetas (Qs)
        # prior_Q = np.array([[0.001, 0.003, 0.05, 0.25, 0.1, 0.5, 20, 60, 10, 30],  # , 1., 2.5],
        #                        [0.002, 0.006, 0.05, 0.25, 0.5, 1.5, 20, 60, 10, 30],  # , 1., 2.5],
        #                        [0.005, 0.01, 0.05, 0.25, 1.0, 2.0, 20, 60, 10, 30],  # , 1., 2.5],
        #                        [0, 0, 0.2, 0.5, 1.0, 2.0, 20, 60, 10, 30]])  # , 1., 2.5]])

        # # GAUSSIAN prior SYNTHETIC (Tokimatsu et al. 1992)
        # # here, values are not min and max, but mu and sigma
        # prior = np.array([[0.002, 0.0005, 0.18, 0.05, 0.3, 0.1], #, 1.8, .5],
        #                   [0.004, 0.001, 0.18, 0.05, 1., 0.25], #, 1.8, .5],
        #                   [0.0075, 0.001, 0.18, 0.05, 1.5, 0.25], #, 1.8, .5],
        #                   [0, 0, 0.35, 0.1, 1.5, 0.25]])#, 1.8, .5]])




        ## MODELS
        ## SYNTHETIC BENCHMARKS

        # # benchmark model 9-c1 (Tokimatsu et al. 1992 - case 1, normal)
        # Thickness = np.asarray([0.002, 0.004, 0.008])
        # Vs = np.asarray([0.08, 0.12, 0.18, 0.36])
        # Vp = np.asarray([0.3, 1.0, 1.4, 1.4])
        # # rho = np.asarray([1.8, 1.8, 1.8, 1.8])
        # nLayer = 4

        # benchmark model 9-c2 (Tokimatsu et al. 1992 - case 2, inverse)
        Thickness = np.asarray([0.002, 0.004, 0.008])
        Vs = np.asarray([0.18, 0.12, 0.18, 0.36])
        Vp = np.asarray([0.3, 1.0, 1.4, 1.4])
        # rho = np.asarray([1.8, 1.8, 1.8, 1.8])
        # Qalphas = np.asarray([50, 50, 50, 50])
        # Qbetas = np.asarray([20, 20, 20, 20])
        nLayer = 4

        # benchmark model 9-c3 (Tokimatsu et al. 1992 - case 3, irregular)
        # Thickness = np.asarray([0.002, 0.004, 0.008])
        # Vs = np.asarray([0.08, 0.18, 0.12, 0.36])
        # Vp = np.asarray([0.3, 1.0, 1.4, 1.4])
        # # rho = np.asarray([1.8, 1.8, 1.8, 1.8])
        # nLayer = 4

        model = np.hstack((Thickness, Vs, Vp))#, Qalphas, Qbetas)) #, rho))



        ## TEST RUNS
        ## ENVIRONMENT COMPOSTI FOR DC_FW_IMAGE

        from composti.src.sourcefunction import SourceFunctionGenerator
        from composti.src.utils import create_frequencyvector, create_timevector, convert_freq_to_time
        from composti.src import reflectivityCPP
        import scipy.interpolate as interp
        from swprocess import Masw, Sensor1C, Source, Array1D


        # Create COMPOSTI options:
        options = np.zeros(4, dtype=np.int32)
        options[0] = 1  # Compute the direct wave
        options[1] = 1  # Compute multiple reflections
        options[2] = 0  # Use frequency windowing
        options[3] = 1  # Return type: 0 - displacement, 1 - velocity, 2 - acceleration


        # RUNS - SYNTHETIC BENCHMARK

        # Parameters for BEL1D forward modelling:

        # #########
        # IF SYNTHETIC (SYNTHETIC Tokimatsu et al. 1992):
        dx = 1  # space between receivers
        array = 52  # length of array (without offset)
        xReceivers = np.arange(5, array + dx, dx).astype(np.float64)  # position of traces (offset in x-direction)
        np.savetxt('xReceivers.txt', xReceivers)
        Tacq = 1 # acquisition time of seismic traces

        fMax = 1000
        fMaxImage = 120 # maximum frequency used to calculate image
        vMax = 350 # maximum velocity used to calculate image

        freq, dt = create_frequencyvector(T_end=Tacq, f_max_requested=fMax, f_min_requested=0.1)

        source_generator = SourceFunctionGenerator(freq)

        source_sw = Source(x=0, y=0, z=0)

        # # Min-max frequency for dispersion image plot
        fmin, fmax = 0.5, 120 #(synthetic model 2)
        # fmin, fmax = 0.5, 80 #(synthetic model 1 and 3)

        # # Selection of trial velocities (velocity in m/s) with minimum, maximum, number of steps, and space {"linear", "log"}.
        vmin, vmax, nvel, vspace = 50, 350, 100, "linear" #(synthetic model 1-3)
        # ##########


       # Ricker source wavelet of 80 Hz
        source = source_generator.Ricker(1, 80)


        # SETTINGS FOR SW PROCESS

        # (wavefield transform here FK, if want to change -> in BEL1D.py):
        # Weighting for "fdbf" {"sqrt", "invamp", "none"} (ignored for all other wavefield transforms). "sqrt" is recommended.
        fdbf_weighting = "sqrt"

        # # Steering vector for "fdbf" {"cylindrical", "plane"} (ignored for all other wavefield transforms). "cylindrical" is recommended.
        fdbf_steering = "cylindrical"


        settings = Masw.create_settings_dict(fmin=fmin, fmax=fmax, vmin=vmin, vmax=vmax, nvel=nvel, vspace=vspace,
                                             weighting=fdbf_weighting, steering=fdbf_steering)



        ModelsetImage = BEL1D.MODELSET.DC_FW_image(prior=prior, priorDist='Uniform', priorBound=None,
                                                   fMaxCalc=fMax, fMaxImage=fMaxImage, vMax=vMax, xReceivers=xReceivers,
                                                   source_sw=source_sw, Tacq=Tacq, settingsSW=settings,
                                                   Q=False, Qalphas_fixed=False,
                                                   rho_fixed=True, rho_val=2500, propagate_noise=True)#, add_noise_coherent=False)

        dataModel, fSyn, vSyn, u_z_time_cpp_Levin = BEL1D.ForwardFWIMAGE(model=model, nLayer=nLayer, freqCalc=freq,
                                                                         xReceivers=xReceivers, source=source, source_sw=source_sw,
                                                                         options=options, Tacq=Tacq, dt=dt, settingsSW=settings,
                                                                         normalization='frequency-maximum', Q=False, Qalphas_fixed=False,
                                                                         showIm=True, returnAxis=True, rho_fixed=True, rho_val=2500,
                                                                         return_raw=True)#, add_noise=1e-9)#, add_noise_coherent=False)

        np.savetxt('fSyn.txt', fSyn) # needed for plots in read_postbel.py
        np.savetxt('vSyn.txt', vSyn) # needed for plots in read_postbel.py
        np.savetxt('dataModel.txt', dataModel)
        np.savetxt('u_z_time_cpp_Levin.txt', u_z_time_cpp_Levin)

        # fSyn = np.loadtxt('fSyn.txt')
        # vSyn = np.loadtxt('vSyn.txt')

        # if forward model already run (comment ForwardFWIMAGE):
        # dataModel, u_z_time_cpp_Levin = ModelsetImage.forwardFun['Fun'](model)

        ###########
        # IF SYNTHETIC:
        dataSyn = dataModel
        ###########



        # # RUN PREBEL AND POSTBEL
        Prebel = BEL1D.PREBEL(MODPARAM=ModelsetImage, nbModels=1000)      # activate if not prebel saved

        # # We then run the prebel operations:
        Prebel.run(verbose=verbose)       # activate if not prebel saved


        BEL1D.SavePREBEL(Prebel, Filename='prebel_model')     # activate if not prebel saved
        # path_prebel = 'YOURPATH'
        # Prebel = BEL1D.LoadPREBEL(path_prebel + 'prebel_model.prebel')



        # Then, since we know the dataset, we can initialize the "post-bel" operations:
        Postbel = BEL1D.POSTBEL(Prebel)

        # Run the operations:
        Postbel.run(Dataset=dataSyn, nbSamples=1000, verbose=verbose) #, NoiseModel=1e-12)

        Postbel.ShowPostModels(TrueModel=model, RMSE=False)
        Postbel.ShowPostModels(TrueModel=model, RMSE=True)


        BEL1D.SavePOSTBEL(Postbel, Filename='postbel_model')
        BEL1D.SaveSamples(Postbel, Data=True, Filename='postbel_samples')


        # # Optional plots
        ## For FW images prior and datasets, showing the mean prior
        prior_data = np.reshape(Prebel.FORWARD, (Prebel.nbModels, len(vSyn), len(fSyn)))
        mean = np.mean(prior_data, axis=0)
        std = np.std(prior_data, axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(mean, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(std, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()


        Postbel.KDE.ShowKDE(
            Xvals=Postbel.CCA.transform(Postbel.PCA['Data'].transform(np.reshape(dataSyn, (1, -1)))))
        pyplot.show()

        post_data = np.reshape(Postbel.SAMPLESDATA, (Postbel.nbSamples, len(vSyn), len(fSyn)))
        mean = np.mean(post_data, axis=0)
        std = np.std(post_data, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(mean, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(std, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()


    #########################################################################################
    ###                               Closing the parallel pool                           ###
    #########################################################################################
    if ParallelComputing:
        pool.terminate()
