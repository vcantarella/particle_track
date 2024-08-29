import flopy
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from particle_track.preprocessing import prepare_arrays
from particle_track import (
    cumulative_cuda,
    pollock_v2,
)
# Load the MODFLOW model
name = "ammer_V07"
ws = os.path.join("MODFLOW_testmodels","complex_model")
sim = flopy.mf6.MFSimulation.load(
    sim_name=name,
    version="mf6",
    exe_name="mf6",
    sim_ws=ws,
)
gwf = sim.get_model(name)
head = gwf.output.head()
head_array = head.get_data()
grid = gwf.modelgrid

# Porosity:
porosity_pairs = { #no unit
        2: 0.75,
        4: 0.75,
        6: 0.75,
        7: 0.75,
        8: 0.8,
        9: 0.8,
        10: 0.20,
        11: 0.7,
        12: 0.25,
        21: 1.0,
        31: 1.0,
    }
facies = np.load(os.path.join(ws,"facies.npy"))
vectorized_porosity = np.vectorize(porosity_pairs.get)
porosity = vectorized_porosity(facies)

# load modpath results
# First the backward model:
mpnamf = name + "_mp_backward"
#Get pathline:
p = flopy.utils.PathlineFile(os.path.join(ws,mpnamf + ".mppth"))
e = flopy.utils.EndpointFile(os.path.join(ws,mpnamf + ".mpend"))
pf = p.get_alldata()
ef = e.get_alldata()
# Second the forward model:
mpnamf = name + "_mp_forward"
#Get pathline:
p = flopy.utils.PathlineFile(os.path.join(ws,mpnamf + ".mppth"))
e = flopy.utils.EndpointFile(os.path.join(ws,mpnamf + ".mpend"))
pf2 = p.get_alldata()
ef2 = e.get_alldata()

#backward tracking:
ttmodpath = []
inds = []
for i in range(len(pf)):
    x = pf[i][0]["x"]
    y = pf[i][0]["y"]
    x, y = grid.get_coords(
        x, y
    )  # convert from local coordinates to global coordinates
    z = pf[i][0]["z"]
    node = pf[i][0]["node"]
    layer, row, col = grid.get_lrc([node])[0]
    inds.append((x, y, z, layer, row, col))
    time = pf[i]["time"]
    ttmodpath.append(time[-1])

inds = np.vstack(inds)
prts_loc = inds


pt_results_v2 = pollock_v2(
    gwfmodel=gwf,
    model_directory=ws,
    particles_starting_location=prts_loc,
    porosity=porosity,
    mode="backwards",
)
ttnumbapath_v2 = []
for j in range(np.max(pt_results_v2[:,0]+1).astype(np.int16)):
    results = pt_results_v2[pt_results_v2[:,0] == j, 1:]
    t = results[:,-1]
    total_t = np.sum(t)
    ttnumbapath_v2.append(total_t)
# Measuring the difference between modpath and out implementation.
ttmodpath
# check where the difference is large
diff2 = (np.array(ttnumbapath_v2) - np.array(ttmodpath))/np.array(ttmodpath)
plt.plot(diff2)
plt.show()
np.sum(np.abs(diff2)>0.01)/len(diff2)
# checking the weird results
pr = np.where(diff2 > 0.1)
pr = ([20],)
our = pt_results_v2[pt_results_v2[:,0] == pr[0][0], 1:]
theirs = pf[pr[0][0]][:][["x","y","z", "time"]]
theirs = np.array(theirs.tolist())
# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(our[:,0], our[:,1], our[:,2], label='model')
ax.plot(theirs[:,0], theirs[:,1], theirs[:,2], label='modpath')
ax.legend()
plt.show()

pr = np.where(diff2 < -0.1)
our = pt_results_v2[pt_results_v2[:,0] == pr[0][0], 1:]
theirs = pf[pr[0][0]][:][["x","y","z", "time"]]
theirs = np.array(theirs.tolist())
# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t_our = np.cumsum(our[:,3])
t_theirs = np.cumsum(theirs[:,3])
ax.plot(our[:,0], our[:,1], our[:,2], label='model',)
ax.plot(theirs[:,0], theirs[:,1], theirs[:,2], label='modpath')
ax.legend()
plt.show()


ct_2 = cumulative_cuda(
    gwfmodel=gwf,
    model_directory=ws,
    particles_starting_location=prts_loc,
    porosity=porosity,
    reactivity=np.ones_like(head_array),
)

ttcumulative_cuda = ct_2[:, 0]
np.assert_allclose(ttcumulative_cuda, ttnumbapath_v2, rtol=1e-7)

# Checking the solution of cumulative reactivity in all valid cells
centers = grid.xyzcellcenters
X = centers[0]
Y = centers[1]
Z = centers[2]
# Same size arrays:
X = np.broadcast_to(X, Z.shape)
Y = np.broadcast_to(Y, Z.shape)
nlay = gwf.modelgrid.nlay
nrow = gwf.modelgrid.nrow
ncol = gwf.modelgrid.ncol
# Extracting indexes:
layers = np.arange(0, nlay)
layers = layers[:, np.newaxis, np.newaxis]
layers = layers * np.ones(centers[2].shape)
rows = np.arange(0, nrow)
rows = rows[np.newaxis, :, np.newaxis]
rows = rows * np.ones(centers[2].shape)
cols = np.arange(0, ncol)
cols = cols[np.newaxis, np.newaxis, :]
cols = cols * np.ones(centers[2].shape)
layers = np.ravel(layers)
rows = np.ravel(rows)
cols = np.ravel(cols)
# Flattening arrays
X = np.ravel(X)
Y = np.ravel(Y)
Z = np.ravel(Z)
idomain = np.where(facies == 21, 0, np.where(facies == 31, 0, 1))
X = X[idomain.ravel() == 1]
Y = Y[idomain.ravel() == 1]
Z = Z[idomain.ravel() == 1]
layers = layers[idomain.ravel() == 1]
rows = rows[idomain.ravel() == 1]
cols = cols[idomain.ravel() == 1]
# Particle centers:
particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))
ct = cumulative_cuda(
    gwfmodel=gwf,
    model_directory=ws,
    particles_starting_location=particle_centers,
    porosity=porosity,
    reactivity=np.ones_like(head_array),
)
print("Cumulative Reactivity: DONE")
traveltimes = ct[:, 0]
tt_array = np.zeros((nlay, nrow, ncol))
tt_array[idomain == 1] = traveltimes
cum_react = ct[:, 1]
max_cum = np.max(cum_react)
cm_array = np.zeros((nlay, nrow, ncol))
cm_array[idomain == 1] = cum_react
err_array = np.zeros((nlay, nrow, ncol))
err_array[idomain == 1] = ct[:, 2]
prob_idx = np.where(err_array[idomain==1] > 0.0)
prob_idx = prob_idx[0]
layers[prob_idx]
rows[prob_idx]
cols[prob_idx]

xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(
    gwfmodel=gwf,
    model_directory=ws,
    porosity=porosity,
)
face_velocities[:,layers[prob_idx].astype(np.int32),
    rows[prob_idx].astype(np.int32),
    cols[prob_idx].astype(np.int32)]
col2 = np.where(cols[prob_idx].astype(np.int32) == 0,
    cols[prob_idx].astype(np.int32)+1,cols[prob_idx].astype(np.int32)-1)
idomain[layers[prob_idx].astype(np.int32),rows[prob_idx].astype(np.int32),col2]

