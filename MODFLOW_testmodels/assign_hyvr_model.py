# -*- coding: utf-8 -*-
"""

Read MODFLOW model and assign properties based on a HYVR model


@author: vcant
"""


# Assumes model is unrotated.

dirc = 'test1'

import os
import pickle
import sys
import numpy as np
import flopy


sys.path.append('../../sedimentary_structure_generator/')


from hysim.objects.channel import Channel
from hysim.objects.sheet import Sheet
from hysim.objects.trough import Trough


sim = flopy.mf6.MFSimulation.load(sim_name = 'test1.nam', sim_ws=dirc, )
gwf = sim.gwf[0]
grid = gwf.modelgrid
centers = grid.xyzcellcenters

X = centers[0]
Y = centers[1]
Z = centers[2]

X = np.broadcast_to(X, Z.shape)
Y = np.broadcast_to(Y, Z.shape)

## Creating and assigning facies to the model:

object_list = []

facies_list = np.arange(0,7)
facies_names = ['fine_silt','silty_sand','loamy_sand','lag_clay','coarse_sand','clay_body','gravel']

base_sheet = Sheet(-50., -30., np.median(X), np.median(Y), np.array([0,1]), dip_direction=np.array([90,90]), dip = [8,8], dipsets='dip',
                   facies_ordering='alternating', dipset_dist= 1.5)

object_list.append(base_sheet)

# contact_surface_kwargs = {'var': 3**2, 'corlx': 70, 'corly': 50}
# contact_surface_base = contact_surface(grid, z = -30., mode = 'random', random_mode_args=contact_surface_kwargs)
# contact_surface_base.max()
# contact_surface_base = np.expand_dims(contact_surface_base, axis=2)

facies = np.empty(X.shape)
dip_dir = np.empty(X.shape)
dip = np.empty(X.shape)

output = base_sheet.assign_points_to_object(X[Z <= -30.],Y[Z <=  -30.],Z[Z <= -30.])

facies[Z <= -30.] = output[0]

dip_dir[Z <= -30.] = output[1]

dip[Z <= -30.] = output[2]

middle_sheet = Sheet(-30., -10, np.median(X), np.median(Y), np.array([2]))

layer_zone = (Z> -30) & (Z< -10)
output = middle_sheet.assign_points_to_object(X[layer_zone],Y[layer_zone],Z[layer_zone])

facies[layer_zone] = output[0]

dip_dir[layer_zone] = output[1]

dip[layer_zone] = output[2]

middle_troughs = []
for i in range(10):
    x = np.random.choice(np.ravel(X))
    y = np.random.choice(np.ravel(Y))
    z_top = np.random.uniform(-30,-10.)
    trough = Trough(x, y, z_top, a = 80., b = 40., c = 5., facies = [4], lag = True, lag_facies = 3, lag_height = [1.,2.],)
    middle_troughs.append(trough)

# middle troughs
middle_troughs.sort(key=lambda x: x.z_top)


for i in range(len(middle_troughs)):
    output = middle_troughs[i].assign_points_to_object(X[layer_zone],Y[layer_zone],Z[layer_zone])

    facies[layer_zone] = np.where(output[0] != -1, output[0], facies[layer_zone])

    dip_dir[layer_zone] = np.where(output[1] != -1, output[1], dip_dir[layer_zone])

    dip[layer_zone] = np.where(output[2] != -1, output[2], dip[layer_zone])
    

upper_sheet = Sheet(-10., 0., np.median(X), np.median(Y), np.array([5]))
layer_zone = (Z>= -10.)
output = upper_sheet.assign_points_to_object(X[layer_zone],Y[layer_zone],Z[layer_zone])

facies[layer_zone] = output[0]

dip_dir[layer_zone] = output[1]

dip[layer_zone] = output[2]

channels = []

for i in range(2):
    y_start = 0 + 100*i
    x_start = 5
    z = np.random.uniform(-2, 0.)
    channel = Channel(facies = 6, z_top = z, width = 16., depth = 12., x_start = x_start, y_start = y_start, s_max = 800, k = 2*np.pi/500, h = 0.1,
                      eps_factor = (np.pi/6)**2, azimuth = [90., 90.])
    channels.append(channel)

channels.sort(key=lambda x: x.z_top)

for i in range(len(channels)):
    
    output = channels[i].assign_points_to_object(X[layer_zone],Y[layer_zone],Z[layer_zone])

    facies[layer_zone] = np.where(output[0] != -1, output[0], facies[layer_zone])

    dip_dir[layer_zone] = np.where(output[1] != -1, output[1], dip_dir[layer_zone])

    dip[layer_zone] = np.where(output[2] != -1, output[2], dip[layer_zone])
    
facies_names = ['fine_silt','silty_sand','loamy_sand','lag_clay','coarse_sand','clay_body','gravel']
k_choices = [1e-6, 1e-5, 5e-5,1e-7,1e-4,1e-7,1e-3]
k_facies = np.choose(facies.astype(np.int16), k_choices)

gwf.npf.k = k_facies
ims = sim.ims
wel = gwf.wel
#Setting new flow rates:
stress_period_data = wel.stress_period_data.get_data()
stress_period_data[0][0][1] = -100./86400
stress_period_data[0][1][1] = -100./86400

gwf.remove_package('wel')

wel = flopy.mf6.ModflowGwfwel(
    gwf,
    auxiliary=[('iface',)],
    stress_period_data=stress_period_data[0],
)

new_dir = 'hysim_model_test'
isExist = os.path.exists(new_dir)
if not isExist:

    # Create a new directory because it does not exist
    os.makedirs(new_dir)
sim.set_sim_path(new_dir)

sim.write_simulation()

print(sim.check())

success, buff = sim.run_simulation()
assert success, "MODFLOW 6 did not terminate normally."

'''
Running MODPATH
'''
# nlays = grid.nlay
# ncols = grid.ncol
# nrows = grid.nrow
# cell_centers = grid.xyzcellcenters
# head = gwf.output.head()
# head_array = head.get_data()
# X = np.ravel(np.tile(cell_centers[0], (nlays, 1, 1)))
# Y = np.ravel(np.tile(cell_centers[1], (nlays, 1, 1)))
# Z = cell_centers[2].flatten()
# layers = np.arange(0, nlays)
# layers = layers[:, np.newaxis, np.newaxis]
# layers = layers * np.ones(cell_centers[2].shape)
# rows = np.arange(0, ncols)
# rows = rows[np.newaxis, :, np.newaxis]
# rows = rows * np.ones(cell_centers[2].shape)
# cols = np.arange(0, ncols)
# cols = cols[np.newaxis, np.newaxis, :]
# cols = cols * np.ones(cell_centers[2].shape)
# layers = np.ravel(layers)
# rows = np.ravel(rows)
# cols = np.ravel(cols)
#
# particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))
# nodes = []
# for i in range(particle_centers.shape[0]):
#     lrc = tuple(particle_centers[i,3:].astype(np.int64))
#     node = grid.get_node(lrc)
#     nodes.append(node)
#
# # create modpath files
# name = 'test1'
# mpnamf = name + "_mp_backward"
# # create basic forward tracking modpath simulation
# mp = flopy.modpath.Modpath7.create_mp7(
#     modelname=mpnamf,
#     trackdir="backward",
#     flowmodel=gwf,
#     model_ws=new_dir,
#     rowcelldivisions=1,
#     columncelldivisions=1,
#     layercelldivisions=1,
#     exe_name='mp7',
#     nodes=nodes
# )
#
# #Create MPBAS package file:
# mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.3)
#
# # write modpath datasets
# mp.write_input()
# # run modpath
# mp.run_model()