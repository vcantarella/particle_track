import numpy as np
import flopy
from particle_track.particle_track import particle_track
from particle_track.cumulate_relative_reactivity import cumulative_reactivity, cumulate_react, cumulative_gu
import os
import matplotlib.pyplot as plt


cumulate_react.parallel_diagnostics(level=4)


name = 'mf6_verif_3'
model_directory = "MODFLOW_testmodels/modflow_verification_3"
mpnamf = name + "_mp_backward"
sim = flopy.mf6.MFSimulation.load(
    sim_ws=model_directory,
    exe_name='mf6',
    verbosity_level=0,
)
gwf = sim.get_model(name)
head = gwf.output.head()
head_array = head.get_data()
grid = gwf.modelgrid
## reading modpath output

#Get pathline:
p = flopy.utils.PathlineFile(os.path.join(model_directory,mpnamf + ".mppth"))
e = flopy.utils.EndpointFile(os.path.join(model_directory,mpnamf + ".mpend"))
pf = p.get_alldata()

p_sample = []
ttmodpath = []
for i in range(len(pf)):
    x = pf[i]['x']
    y = pf[i]['y']
    time = pf[i]['time']
    p_sample.append((x,y))
    ttmodpath.append(time[-1])

#%% md
### 3.2. Getting starting particle coordinates to use as input in particle track code
#%%
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
#%% md
### 3.3. Runnning the Particle Track
#%%
pt_results = particle_track(gwfmodel = gwf, model_directory = model_directory, particles_starting_location = prts_loc, mode='backwards')
#%%
ct_results = cumulative_reactivity(gwfmodel = gwf, model_directory = model_directory, particles_starting_location = prts_loc, reactivity=np.ones_like(head_array))
#%%
ct_results

ct_2 = cumulative_gu(gwfmodel = gwf, model_directory = model_directory,
                     particles_starting_location = prts_loc, reactivity=np.ones_like(head_array))


#%% md

ttnumbapath = []
for j in range(np.max(pt_results[:,0]+1).astype(np.int16)):
    results = pt_results[pt_results[:,0] == j, 1:]
    t = results[:,-1]
    total_t = np.sum(t)
    ttnumbapath.append(total_t)
#%% md

#%%
ttnumbapath = np.array(ttnumbapath)
ttmodpath = np.array(ttmodpath)
ttcumulative_track = ct_results[0]
particle_index = np.arange(ttmodpath.size)

fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
ax.plot(particle_index,ttmodpath,'-*b', label = 'modpath')
ax.plot(particle_index,ttnumbapath,'-Dr', label='particle_track')
ax.plot(particle_index,ttcumulative_track,'--g', label='cumulative_react_time')
ax.plot(particle_index,ct_2[0])
ax.set_ylabel('travel time [days]')
ax.set_xlabel('particle index')
ax.set_title('Travel times')
ax.legend()
plt.show()