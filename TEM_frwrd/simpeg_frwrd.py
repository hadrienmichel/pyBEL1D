# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:46:17 2020

wrapper class of simpeg electromagnetics to create a TEM forward calc
currently only TEMfast frwrd solution available, could be expanded to other systems

@author: lukas
"""
# %% modules
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # aimed at discretize warnings

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import SimPEG.electromagnetics.time_domain as tdem

from SimPEG.electromagnetics.time_domain.receivers import PointMagneticFluxTimeDerivative as point_dbdt
from SimPEG.electromagnetics.time_domain import simulation

from discretize import CylMesh
from SimPEG import maps

from SimPEG import SolverLU as Solver


# %%
class simpeg_frwrd(object):
    """
    """
    def __init__(self, setup_device, setup_simpeg,
                 device='TEMfast', filter_times=None,
                 nlayer=3, nparam=2):
        """
        Constructor for the frwrd solution with simpeg.
        Parameters
        ----------
        setup_device : dictionary
            sounding settings of TEM device. have to fit to device.
        coredepth : int, optional
            depth of core mesh. The default is 100 in (m).
        csz : float, optional
            cell size in z direction of core mesh. The default is 1 in (m).
        relerr : float, optional
            relative error for frwrd solution. The default is 0.001 (%/100).
        abserr : float, optional
            absolute error level  for frwrd solution. The default is 1e-15 (V/m²).
        device : string, optional
            Name of tem device. The default is 'TEMfast'.
        filter_times : tuple (minT, maxT) in (s), optional
            if not None it will be used to filter the times at which to 
            compute the frwrd sol (minT, maxT)
        nlayer : int, optional
            number of layers in the model.
        nparam : int, optional
            number of parameters in the model. The default is 

        Returns
        -------
        None.

        """

        self.setup_device = setup_device
        self.coredepth = setup_simpeg['coredepth']
        self.csz = setup_simpeg['csz']
        self.relerr = setup_simpeg['relerr']
        self.abserr = setup_simpeg['abserr']
        self.device = device
        self.nlayer = nlayer
        self.nparam = nparam
        self.model = None
        self.response = None
        self.mesh = None
        self.properties_snd = None
        self.times_rx = None
        self.model_map = None
        self.model_vec = None
        self.filter_times = None
        
        if self.device == 'TEMfast':
            print('creating tem fast properties')
            self.create_props_temfast(show_rampInt=False)
        else:
            print('device/name not available ... ')
            print('currently available: TEMfast')
            print('!! forward solution not calculated !!')
            sys.exit(0)
        
        self.create_cylmesh(show_mesh=False)

    @staticmethod
    def getR_fromSquare(a):
        return np.sqrt((a*a) / np.pi)

    @staticmethod
    def reshape_model(model, nLayer, nParam):
        mdlRSHP = np.zeros((nLayer,nParam))
        i = 0
        for col in range(0, nParam):
            for row in range(0, nLayer):
                if col == 0 and row == nLayer-1:
                    pass
                else:
                    mdlRSHP[row, col] = model[i]
                    i += 1
        
        model = mdlRSHP
        return model

    @staticmethod
    def get_lowerandupper_cellborder(depth, csz=1):
        lower = depth - (depth % csz)
        upper = lower + csz
        return lower, upper

    def create_props_temfast(self, show_rampInt=True):
        """
        This method creates the device properties of the TEMfast device.
        Necessary to calculate the forward solution. It sets the class attributes
        times_rx and properties_snd according to the selected device settings.

        Parameters
        ----------
        show_rampInt : boolean, optional
            To decide wether to show the ramp interpolation. The default is False.

        Returns
        -------
        properties_snd : dictionary
            {"radiustx": radius of transmitter (equal area as the square),
             "timesrx": times in (s) at which the signal should be sampled,
             "pulsel": length of the dc pulse (s),
             "rampoff": length of the turn-off ramp in (s),
             "current": injected current in (A)}.

        """

        time_key = self.setup_device["timekey"]
        tx_loop = self.setup_device["txloop"]
        current = self.setup_device["current"]
        filter_powerline = self.setup_device["filter_powerline"]

        # input check
        available_timekeys = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        available_currents = [1, 4]
        if not time_key in available_timekeys:
            print("You chose a time key that is not available for the TEMfast instrument.")
            print("Please select a time key between 1 and 9.")
            print("Creation of properties not succesful!!!")
            exit(0)

        if not current in available_currents:
            print("You chose a current that is not available for the TEMfast instrument.")
            print("Please select either 1 or 4 A of current.")
            print("Creation of properties not succesful!!!")
            exit(0)

        if filter_powerline == 50:                                                     # in Hz
            times_onoff = np.r_[0.31,0.63,1.25,2.50,5.00,10.00,30.00,50.00,90.00]      # in milliseconds
            times_on = times_onoff / 4 * 3                                             # aka pulse lengths
            times_off = times_onoff / 4
        elif filter_powerline == 60:
            times_onoff = np.r_[0.26,0.52,1.04,2.08,4.17,8.33,25.00,41.67,75.00]       # in milliseconds
            times_on = times_onoff / 4 * 3
            times_off = times_onoff / 4
        else:
            print('Choose either 50 or 60 as Frequency (Hz) for the powergrid filter!')
            print("Creation of properties not succesful!!!")
            exit(0)

        # generate properties of transmitter
        # necessary to chose adequate reciever params from gates
        timekeys = np.arange(1,10)
        all_gates = np.arange(16,52,4);
        maxTime = 2**(timekeys+5);
        analogStack = 2**(timekeys[::-1]+1)

        propDict = {"timeKeys": timekeys,
                    "maxTimes": maxTime,
                    "allGates": all_gates,
                    "analogStack": analogStack,
                    "ONOFFtimes": times_onoff,
                    "ONtimes": times_on,
                    "OFFtimes": times_off}
        properties_device = pd.DataFrame(propDict)

        if not 'TEM_frwrd' in os.getcwd():  # hack to make also the test_frwrdclass script work...
            path2csv = './TEM_frwrd/TEMfast_props/'
        else:
            path2csv = './TEMfast_props/'
        timegates = pd.read_csv(path2csv + 'TEMfast_timegates.csv',
                                delimiter=',', skiprows=1)
        rampdata = pd.read_csv(path2csv + f'TEMfast_rampOFF_{int(current)}A.csv',
                               delimiter=',', skiprows=1)

        # convert square side length to radius of a ring loop with the same area
        r_Tx = self.getR_fromSquare(tx_loop) # in m

        # get timegates of tem fast device and combine with the adequate key
        gates = properties_device.allGates[properties_device.timeKeys == time_key].values[0]
        times_rx = np.asarray(timegates.centerT[0:gates] * 1e-6) # from mus to s

        # create parameters of source and waveform:
        pulse_length = properties_device.ONtimes[properties_device.timeKeys == time_key].values[0]
        pulse_length = pulse_length * 1e-3 # from ms to s
        rampf = interp1d(rampdata.side, rampdata.rampOFF,
                         kind='linear',
                         fill_value='extrapolate')
        ramp_off = rampf(tx_loop) * 1e-6 # from mus to s

        if show_rampInt:
            plt.plot(rampdata.side, rampdata.rampOFF, '--dk')
            plt.plot(tx_loop, ramp_off, 'dr')
        else:
            print('You did not want to see the interp. of the turn-off ramp.')

        if not self.filter_times is None:
            mask = [(times_rx > self.filter_times[0]) &
                    (times_rx < self.filter_times[1])]
            times_rx = times_rx[mask]
            self.times_rx = times_rx

        properties_snd = {"radiustx": r_Tx,
                          "timesrx": times_rx,
                          "pulsel":pulse_length,
                          "rampoff": ramp_off,
                          "current": current}

        self.properties_snd = properties_snd
        self.times_rx = times_rx

        return properties_snd


    def create_cylmesh(self, show_mesh=False):
        """
        Method to initialize the cylindrical mesh for the forward calculation.

        Parameters
        ----------
        show_mesh : boolean, optional
            To decide whether to show the mesh. The default is False.

        Returns
        -------
        mesh : TYPE
            DESCRIPTION.

        """
        
        depth = self.coredepth
        csz = self.csz  # core cell size in the z-direction
        csx = csz * 3  # core cell size in the x-direction
        domainx = depth * 3  # coremesh in x

        # padding parameters
        npadx, npadz = 30, 30  # number of padding cells
        pfx = 1.3  # expansion factor for the padding to infinity in the x-direction
        pfz = 1.3  # expansion factor for the padding to infinity in the z-direction

        ncx = int(domainx / csx)
        ncz = int(depth / csz)  # number of z cells in the core region

        hx = [(csx, ncx), (csx, npadx, pfx)]
        hz = [(csz, npadz, -pfz), (csz, ncz), (csz, npadz, pfz)]

        # create the cyl mesh
        mesh = CylMesh([hx, 1, hz], "0CC")
        # put the origin at the top of the target
        mesh.x0 = [0, 0, -mesh.h[2][:npadz + ncz].sum()]

        if show_mesh:
            figMesh, axMesh = plt.subplots(1,1) # show mesh (empty)
            mesh.plotGrid(ax=axMesh, grid=True)
            axMesh.set_title('cylindrical meshgrid')

        self.mesh = mesh
        return mesh


    def infer_layer2mesh(self, model, show_mesh=False):
        """
        method to add layer information to mesh

        Parameters
        ----------
        model : np.array (2x2) with floats
            [thk, res]. units: [(m), (Ohmm)]
        show_mesh : boolean, optional
            To decide wether to show the mesh with params. The default is False.

        Returns
        -------
        None.

        """
        self.model = model
        
        if model.ndim == 1:
            print('\n\nfound a one dimensional model')
            print(model)
            print('reshaping to [thk, res] assuming ...')
            print(('thk_l_0, thk_l_1, ..., thk_l_n-1, ' + 
                   'res_l_0, res_l_1, ..., res_l_n'))
            print(' ... structure, bottom thk will be set to 0\n')
            mdlrshp = self.reshape_model(self.model, self.nlayer, self.nparam)
            self.model = mdlrshp
            print(self.model)
        else:
            pass

        air_value = 1e9     # in Ohmm
        ind_active = self.mesh.gridCC[:, 2] < 0  # active part of mesh, below 0
        # set active part to mesh
        model_map = maps.InjectActiveCells(self.mesh, ind_active, air_value)

        rho_half = self.model[-1, 1]
        model_vec = rho_half * np.ones(ind_active.sum())  # vector with all active cells

        # add layers
        nlayers = len(self.model) - 1

        top = 0
        bot = -self.model[0,0]
        for i in range(0,nlayers):
            print('layer_', i+1, '___top__bot: ', top, bot)
            layer = ((self.mesh.gridCC[ind_active, 2] < top) &
                     (self.mesh.gridCC[ind_active, 2] >= bot))

            print('mapping in range: ', top, bot)
            print('res: ', self.model[i,1])
            model_vec[layer] = self.model[i,1] # add resistivity
            # TODO linear interpolation between two adjacent layer with changing res
            # for layer boundaries which are in between mesh elements
            print('------------------\n')#

            if bot % self.csz != 0:
                print(bot % self.csz)
                print('encountering layer boundary within core cell size...')
                rest = bot % self.csz
                lower, upper = self.get_lowerandupper_cellborder(depth=bot,
                                                                     csz=self.csz)
                # lyr_to = bot - rest
                av_lyr = ((self.mesh.gridCC[ind_active, 2] < upper) &
                          (self.mesh.gridCC[ind_active, 2] >= lower))
                
                dpth_i = np.r_[upper, lower]
                res_i = np.r_[self.model[i,1], self.model[i+1,1]]
                print('values for interpol (thk, res):')
                print(dpth_i, '\n', res_i)
                print('-----------------')

                res_f = interp1d(dpth_i, res_i,
                                 kind='linear',
                                 fill_value='extrapolate')
                print('interpolating to: ', bot)
                av_res = res_f(bot)
                print('interpolated resistivity: ', av_res)
                
                # av_res = do a linear interpolation here!!!
                print('mapping in range: ', upper, lower)
                print('res: ', av_res)
                model_vec[av_lyr] = av_res # add resistivity
                
                # adjust layer boundaries
                top = lower
                bot = upper - rest
                print('bot + rest', bot)
                print('------------------\n')

            else:
                top = bot

            bot = bot - self.model[i+1,0]
            print('bot with new thk', bot)

        self.model_map = model_map
        self.model_vec = model_vec

        if show_mesh:  # plotting part
            figMdl, axMdl = plt.subplots(nrows=1, ncols=1) # show mesh (empty)
            imag = self.mesh.plotImage(model_map * model_vec,
                                  ax=axMdl, grid=True, mirror=True)[0]
            imag.set_clim((min(self.model_vec),max(self.model_vec)))
            imag.set_cmap('viridis')
            cb = plt.colorbar(imag)
            cb.set_label(r'$\rho$ ($\Omega$m)', fontsize=12)

            axMdl.axis('equal')
            # axMdl.set_xlim((-20, 20))
            # axMdl.set_ylim((-self.coredepth/2, 5))
            axMdl.set_xlim((-20, 20))
            axMdl.set_ylim((-25, 5))


    def calc_response(self, model):
        """
        method to calculate the response of a given thickness/resistivity model,
        forwards the model to the infer_layer2mesh function and runs all
        necessary parts. sets the response attribute to the calculated signal

        Parameters
        ----------
        model : np.array (n x 2) with floats
            [thk, res]. units: [(m), (Ohmm)]

        Returns
        -------
        forward response (V/m²)

        """
        self.infer_layer2mesh(model, show_mesh=False)

        pulse_length = self.properties_snd["pulsel"]
        rampOff_len = self.properties_snd["rampoff"]
        r_Tx = self.properties_snd["radiustx"]
        current = self.properties_snd["current"]

        # rampOn_len = 3e-6
        # ramp_on = np.r_[0., rampOn_len]
        # ramp_off = pulse_length + np.r_[rampOn_len, rampOff_len]
        # time_cutoff = pulse_length + rampOn_len
        # print('wrong trap pulse?')
        # print(ramp_on, ramp_off)
        
        rampOn_len = 3e-6
        time_cutoff = pulse_length + rampOn_len
        ramp_on = np.r_[0., rampOn_len]
        ramp_off = np.r_[time_cutoff, time_cutoff+rampOff_len]
        print('corr trap pulse!!')
        print('[ramp_on_min, ramp_on_max] [ramp_off_min, ramp_off_max,')
        print(ramp_on, ramp_off)

        trapezoid = tdem.sources.TrapezoidWaveform(ramp_on=ramp_on,
                                                   ramp_off=ramp_off,
                                                   eps=pulse_length,
                                                   offTime=time_cutoff)

        # creating reciever object
        # TODO adapt to actual single loop config
        rxLoc = np.zeros((1,3))
        Rx = point_dbdt(locs=rxLoc,                              # position in m
                        times=self.times_rx + time_cutoff,                     # have to be in s
                        orientation='z')

        # creating transmitter object
        Tx = tdem.sources.CircularLoop([Rx],                     # receiver object
                                       waveform=trapezoid,       # type of waveform
                                       loc=rxLoc,                # location of transmitter (here same as rx)
                                       radius=r_Tx,              # in m
                                       current=current)
        survey = tdem.Survey([Tx])

        # %% setup of the forward simulation
        # TODO improve time discretization to decrease run time, but keep accuracy
        # timeSteps = [(time_cutoff/10, 10), (1e-7, 100), (1e-06, 100), (1e-05, 100), (1e-04, 100), (1e-3, 100)]
        # timeSteps = [(time_cutoff/50, 50), (3.1e-7, 10), (1e-7, 200), (3e-7, 50), (1e-06, 200), (3e-6, 50), (1e-05, 200)]
        timeSteps = [(time_cutoff/100, 100), (3.1e-7, 10), (1e-7, 200), (3e-7, 50), (1e-06, 200), (3e-6, 50), (1e-05, 200)] # , (3e-5, 50)

        simulate = simulation.Simulation3DElectricField(self.mesh, survey=survey,
                                                        rhoMap=self.model_map,
                                                        Solver=Solver,
                                                        t0=0,
                                                        time_steps=timeSteps)

        # %% calculate forwrds response and visualize the synthetic data
        # create synthetic data
        print('\nabout to make synthetic data with times:')
        print(self.times_rx)
        data = simulate.make_synthetic_data(self.model_vec,
                                            relative_error=self.relerr,
                                            noise_floor=self.abserr,
                                            f=None, add_noise=True)
        self.response = -data.dobs # from data get the dobs attribute and set it to attribute of self
        
        return self.response
