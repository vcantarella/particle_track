import flopy
import os
import matplotlib.pyplot as plt
import numpy as np
from particle_track.particle_track import particle_track, trajectory
from particle_track.cumulate_relative_reactivity import cumulative_reactivity, travel_time_cum_reactivity, cumulate_react
from particle_track.cumulate_relative_reactivity import cumulative_gu, travel_time_cuda
from particle_track.preprocessing import prepare_arrays

cumulate_react.parallel_diagnostics(level=4)

name = 'mf6_verif_2'
model_directory = "MODFLOW_testmodels/modflow_verification_2"
mpnamf = name + "_mp_forward"

sim = flopy.mf6.MFSimulation.load(
    sim_ws=model_directory,
    exe_name='mf6',
    verbosity_level=0,
)
gwf = sim.get_model(name)
head = gwf.output.head()
head_array = head.get_data()
grid = gwf.modelgrid

#Get pathline:
p = flopy.utils.PathlineFile(os.path.join(model_directory,mpnamf + ".mppth"))
pf = p.get_alldata()

X, Y, Z = grid.xyzcellcenters
inds = []
for i in range(len(pf)):
    x = pf[i][0][0]
    y = pf[i][0][1]
    x,y = grid.get_coords(x, y) # convert from local coordinates to global coordinates
    z = pf[i][0][2]
    layer, row, col = grid.intersect(x,y,z)
    inds.append((x,y,z,layer,row,col))

inds = np.vstack(inds)
prts_loc = inds

reactivity = np.ones_like(head_array)
initial_location = prts_loc[0,:3]
initial_cell = prts_loc[0,3:].astype(np.int64)

xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(gwf, model_directory)

face_velocities = face_velocities*(-1)
gvs = gvs*(-1)

tr = travel_time_cum_reactivity(initial_location, initial_cell, face_velocities, gvs, xedges, yedges, z_lf, z_uf,
                                termination, reactivity)
initial_location = initial_location[np.newaxis,:]
initial_cell = initial_cell[np.newaxis,:]
tr2 = np.zeros_like(initial_location)
tr2 = travel_time_cuda(initial_location, initial_cell, face_velocities, gvs, xedges, yedges, z_lf, z_uf,
                                termination, reactivity, tr2)
inds, ts, xes, yes, zes = trajectory(initial_location, initial_cell, face_velocities, gvs, xedges, yedges, z_lf, z_uf,
                                termination)

print(tr)
print(np.sum(ts))

ct_results = cumulative_reactivity(gwfmodel = gwf, model_directory = model_directory, particles_starting_location = prts_loc, reactivity=np.ones_like(head_array))
pt_results = particle_track(gwfmodel = gwf, model_directory = model_directory, particles_starting_location = prts_loc, mode = 'backwards')
ct_res2 = cumulative_gu(gwfmodel = gwf, model_directory = model_directory, particles_starting_location = prts_loc, reactivity=np.ones_like(head_array))

ttnumbapath = []
for j in range(np.max(pt_results[:,0]+1).astype(np.int16)):
    results = pt_results[pt_results[:,0] == j, 1:]
    t = results[:,-1]
    x = results[-1,0]
    y = results[-1,1]
    total_t = np.sum(t)
    ttnumbapath.append(total_t)

print(np.isclose(ttnumbapath,ct_results[0]))
print(np.isclose(ttnumbapath,ct_results[1]))
print(np.isclose(ttnumbapath,ct_res2[:,0]))
print(np.isclose(ttnumbapath,ct_res2[:,1]))