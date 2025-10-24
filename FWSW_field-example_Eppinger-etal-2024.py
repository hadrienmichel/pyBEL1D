'''
FWSW: Full wavefield surface wave analysis
Here we show the application on a real dataset (from Eppinger et al., 2024).
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
    FWImage = True  # Projection of dispersion images
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
    ###                     FW image on synthetic and real data                           ###
    #########################################################################################
    if FWImage:

        ## PRIORS
        # uniform priors are structured as follows:
        # prior = np.array([[th1_min, th1_max, Vs1_min, Vs1_max, Vp1_min, Vp1_max], #rho1_min, rho1_max, Qalphas1_min, Qalphas2_max, Qbetas1_min, Qbetas1_max
        #                   [th2_min, th2_max, Vs2_min, ...],
        #                   [th3_min, ...],
        #                   ...
        #                   ])

        # # prior (Eppinger et al., 2024)
        Eppinger
        prior = np.array([[0.005, 0.015, 0.15, 1., 0.3, 1.5, 5, 100],
                               [0.005, 0.02, 0.015, 1.5, 1.8, 3.5, 5, 100],
                               [0.005, 0.02, 0.75, 2.2, 2., 4.5, 5, 100],
                               [0, 0, 1.8, 3., 3., 6., 5, 100]])

        ## MODELS
        ## FIELD MODELS

        # # model (Eppinger et al., 2024)
        Thickness = np.asarray([0.01, 0.012, 0.012])
        Vs = np.asarray([0.5, 1.2, 1.9, 2.5])
        Vp = np.asarray([1., 2.5, 3.2, 4.1])
        nLayer = 4

        model = np.hstack((Thickness, Vs, Vp))

        ## TEST RUNS
        ## ENVIRONMENT COMPOSTI FOR DC_FW_IMAGE

        from composti.src.sourcefunction import SourceFunctionGenerator
        from composti.src.utils import create_frequencyvector, create_timevector, convert_freq_to_time
        from composti.src import reflectivityCPP
        import scipy.interpolate as interp
        from swprocess import Masw, Sensor1C, Source, Array1D
        from swprocess.wavefieldtransforms import SlantStack, FK, FDBF, PhaseShift
        from matplotlib.pyplot import GridSpec

        path = 'YOURPATH'

        # Create COMPOSTI options:
        options = np.zeros(4, dtype=np.int32)
        options[0] = 1  # Compute the direct wave
        options[1] = 1  # Compute multiple reflections
        options[2] = 0  # Use frequency windowing
        options[3] = 1  # Return type: 0 - displacement, 1 - velocity, 2 - acceleration


        # RUNS - SYNTHETIC OR REAL DATA

        # Parameters for BEL1D forward modelling:


        ###########
        # IF FIELD DATA:
        # Define headers

        real_data_ascii = np.loadtxt(path + '117_eppinger.ascii')
        real_data_ascii = np.fliplr(real_data_ascii)
        source_sw = Source(x=0, y=0, z=0)
        N = 5980  # nr of samples
        T = 0.00025  # dt
        dx = 1  # space between receivers
        array = 180  # 117.sgy

        xReceivers = np.arange(1, array + dx, dx).astype(np.float64)  # position of traces (offset in x-direction)

        Tacq = N * T  # acquisition time of seismic traces
        print(Tacq)
        print(len(xReceivers))
        # print(xReceivers)

        ##########

        # Ricker source wavelet of 20 Hz
        source = source_generator.Ricker(1, 20)



        # SETTINGS FOR SW PROCESS
        # (wavefield transform here FK, if want to change -> in BEL1D.py):
        # Weighting for "fdbf" {"sqrt", "invamp", "none"} (ignored for all other wavefield transforms). "sqrt" is recommended.
        fdbf_weighting = "sqrt"
        # # Steering vector for "fdbf" {"cylindrical", "plane"} (ignored for all other wavefield transforms). "cylindrical" is recommended.
        fdbf_steering = "cylindrical"
        #
        settings = Masw.create_settings_dict(fmin=fmin, fmax=fmax, vmin=vmin, vmax=vmax, nvel=nvel, vspace=vspace,
                                             weighting=fdbf_weighting, steering=fdbf_steering)



        #########
        ### IF FIELD DATA:
        sensors = []
        for i in range(real_data_ascii.shape[1]):
            sensors.append(Sensor1C(real_data_ascii[:,i], dt=dt, x=xReceivers[i], y=0, z=0))
        array = Array1D(sensors, source_sw)

        # choose here your wavefield transform method and normalization
        # for modelled dispersion images during ForwardFWImage, change in BEL1D.py
        wavefieldTransform = FDBF.from_array(array=array, settings=settings["processing"])
        f = wavefieldTransform.frequencies
        v = wavefieldTransform.velocities

        # choose between 'absolute-maximum' (similar to "spectrum power" in geopsy, used for field test)
        # or 'frequency-maximum' (similar to "maximum beam power" in geopsy, used for synthetic tests)
        wavefieldTransform.normalize(by='absolute-maximum')
        img = wavefieldTransform.power

        # Plot seismic traces
        timeVec = create_timevector(Tacq, T)
        min_rec_distance = np.min(np.diff(xReceivers))
        fig = plt.figure(figsize=(7.0, 5.0))
        ax = fig.add_subplot(111)
        for rec in range(len(xReceivers)):
            uzTimeNorm = real_data_ascii[:, rec] * (0.6 / np.max(abs(real_data_ascii[:, rec]))) * min_rec_distance
            ax.plot(xReceivers[rec] + uzTimeNorm, timeVec, '-k')
        ax.grid('on')
        ax.axis('tight')
        ax.set_ylim((timeVec[-1], timeVec[0]))  # Set tight limits and reverse y-axis
        # ax.set_title('Seismograms measured on the surface (z-component)')
        ax.set_ylabel('Time (s)', fontsize=14)
        ax.set_xlabel('Offset (m)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.imshow(img, aspect='auto', origin='lower', extent=(f[0], f[-1], v[0], v[-1]),
                  interpolation='bilinear', cmap='gist_rainbow')

        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        # ax.set_xscale('log')
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.show(block=True)
        ############


        ModelsetImage = BEL1D.MODELSET.DC_FW_image(prior=prior, priorDist='Uniform', priorBound=None, fMaxCalc=fMax,
                                                   fMaxImage=fMaxImage, vMax=vMax, xReceivers=xReceivers, source_sw=source_sw,
                                                   Tacq=Tacq, settingsSW=settings,
                                                   Q=False, Qalphas_fixed=False,
                                                   rho_fixed=True, rho_val=1800, propagate_noise=True)
        dataModel, fSyn, vSyn, u_z_time_cpp_Levin = BEL1D.ForwardFWIMAGE(model=model, nLayer=nLayer, freqCalc=freq,
                                                                         xReceivers=xReceivers, source=source, source_sw=source_sw,
                                                                         options=options, Tacq=Tacq, dt=dt, settingsSW=settings,
                                                                         Q=False, Qalphas_fixed=False, normalization='absolute-maximum',
                                                                         showIm=True, returnAxis=True, rho_fixed=True, rho_val=1800,
                                                                         return_raw=True)

        np.savetxt('fSyn.txt', fSyn) # needed for plots in read_postbel.py
        np.savetxt('vSyn.txt', vSyn) # needed for plots in read_postbel.py
        np.savetxt('dataModel.txt', dataModel)
        np.savetxt('u_z_time_cpp_Levin.txt', u_z_time_cpp_Levin)

        # fSyn = np.loadtxt('fSyn.txt')
        # vSyn = np.loadtxt('vSyn.txt')

        # if forward model already run (comment ForwardFWIMAGE):
        # dataModel, u_z_time_cpp_Levin = ModelsetImage.forwardFun['Fun'](model)


        ############
        # IF FIELD DATA:
        interpolator = interp.RectBivariateSpline(v, f, img, kx=1, ky=1)
        Upf = interpolator(vSyn, fSyn)  # interpolate to get the sames axes as for the forward model from ForwardFWIMAGE

        # # apply filter on dispersion image with a threshold value of, e.g., 0.5
        # Upf[np.less(Upf, 0.5)] = 0.0

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.imshow(Upf, aspect='auto', origin='lower', extent=(fSyn[0], fSyn[-1], vSyn[0], vSyn[-1]),
                  interpolation='bilinear', cmap='gist_rainbow')

        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        # ax.set_xscale('log')
        ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        fig.tight_layout()
        plt.show(block=True)

        dataSyn = Upf.flatten()
        np.savetxt('dataSyn_field.txt', dataSyn)
        ############


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
        # vSyn = np.divide(1, pSyn)
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
