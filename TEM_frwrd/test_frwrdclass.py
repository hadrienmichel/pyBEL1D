# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:03:31 2020

script to test simpeg_frwrd class
used to update mapping to include layer averaging

@author: lukas
"""
import numpy as np
from simpeg_frwrd import simpeg_frwrd
import matplotlib.pyplot as plt


# %% setup
device = 'TEMfast'

setup_device = {"timekey": 5,
                "txloop": 12.5,
                "rxloop": 12.5,
                "current": 1,
                "filter_powerline": 50}

setup_simpeg = {'coredepth': 100,
                'csz': 0.8,
                'relerr': 0.001,
                'abserr': 1e-15}

# coredepth = 100
# csz = 0.5
# relerr = 0.001
# abserr = 1e-15


# %% forward calc
thk = [5, 10.3, 2.5, 0]
res = [20, 5, 20, 5]

thk = [3.5, 3.2, 3.5, 0]
# res = [100, 5, 100, 5]
res = [5, 100, 5, 100]

model1 = np.column_stack((thk, res))

forward = simpeg_frwrd(setup_device=setup_device,
                       setup_simpeg=setup_simpeg,
                       device=device,
                       nlayer=model1.shape[0], nparam=model1.shape[1])


forward.infer_layer2mesh(model1, show_mesh=True)
# forward.calc_response(model1)

# %% plot
# fig, ax = plt.subplots(1, 1, figsize=(6,6))
# ax.loglog(forward.times_rx, forward.response, '-ok',
#           label='mdl-res: [20, 5, 50]')


# res = [50, 50, 50]
# thk = [5, 10, 0]
# model2 = np.column_stack((thk, res))

# forward.calc_response(model2)

# plt.loglog(forward.times_rx, forward.response, '-or',
#            label='mdl-res: [50, 50, 50]')
# plt.legend(loc="best")
# plt.show()


def get_lowerandupper_cellborder(depth, csz=1):
    lower = depth - (depth % csz)
    upper = lower + csz
    return lower, upper