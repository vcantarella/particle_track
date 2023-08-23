#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:18:27 2023
Test flopy model to extract specific discharges and compute travel-time
@author: vcant
"""

import numpy as np
import flopy
import matplotlib.pyplot as plt
import os
import pickle


# Input parameters

h1 = 100
h2 = 90
nlay = 20
N = 101
L = 400.0
H = 50.0
k = 1e-4 #1.0/86400 # hydraulic conductivity
Q = -1000./86400 # pumping rate

# Create flopy object


mdir = 'test1' ##model directory
name = 'test1'
sim = flopy.mf6.MFSimulation(sim_name = name, exe_name = 'mf6', version = 'mf6', sim_ws = mdir)

## TDIS

tdis = flopy.mf6.ModflowTdis(sim, pname = 'tdis', time_units = 'seconds', nper = 1, perioddata= [(1., 1, 1.)])

## IMS

ims = flopy.mf6.ModflowIms(simulation = sim, pname = 'ims', complexity= 'SIMPLE',
                           linear_acceleration="BICGSTAB",
                           outer_maximum = 300,
                           inner_maximum = 500,
                           rcloserecord = [0.001, 'STRICT'])

## GWF

gwf = flopy.mf6.ModflowGwf(sim, save_flows = True, newtonoptions='NEWTON UNDER RELAXATION')

## DIS

bot = np.linspace(-H/nlay, -H, nlay)

delrow = delcol = L/(N-1)

dis = flopy.mf6.ModflowGwfdis(gwf, nlay = nlay,
                              nrow = N, ncol = N,
                              delr = delrow,
                              delc = delcol,
                              top=0.,
                              botm = bot)

## IC

start = h1*np.ones((nlay,N,N))
ic = flopy.mf6.ModflowGwfic(gwf, strt = start, pname = 'ic')


## npf

npf = flopy.mf6.ModflowGwfnpf(gwf,
                              icelltype= 1,
                              k = k,
                              save_specific_discharge=True)

## CHD

chd_rec = []
for layer in range(nlay):  
    for row_col in range(0, N):
        chd_rec.append(((layer, row_col, 0), h1, 1))
        chd_rec.append(((layer, row_col, N - 1), h2, 2))
        # if row_col != 0 and row_col != N - 1:
        #     chd_rec.append(((layer, 0, row_col), h1))
        #     chd_rec.append(((layer, N - 1, row_col), h1))
chd = flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=[('iface',)],
    stress_period_data=chd_rec,
)


## WEL

wel_rec = [(nlay - 1, int(N / 4), int(N / 4), Q),(nlay - 2, int(N / 4), int(N / 4), Q, 0)]
wel = flopy.mf6.ModflowGwfwel(
    gwf,
    auxiliary=[('iface',)],
    stress_period_data=wel_rec,
)

## OC

headfile = f"{name}.hds"
head_filerecord = [headfile]
budgetfile = f"{name}.cbb"
budget_filerecord = [budgetfile]
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    saverecord=saverecord,
    head_filerecord=head_filerecord,
    budget_filerecord=budget_filerecord,
    printrecord=printrecord,
)

sim.write_simulation()

print(sim.check())

success, buff = sim.run_simulation()
assert success, "MODFLOW 6 did not terminate normally."


## Get grid information, heads and specific discharge
grid = gwf.modelgrid
#edges = grid.xyzgrid
centers = grid.xyzcellcenters
head = gwf.output.head().get_data()
budget = gwf.output.budget()
print(budget)
specific_discharge = budget.get_data(text = 'DATA-SPDIS')[0]
flow_ja_face = budget.get_data(text = 'FLOW-JA-FACE')[0]
wel_flow = budget.get_data(text = 'WEL')
wel_flow = wel_flow[0]
wel_nodes = wel_flow['node']-1 #get lrc needs 0 array indices
wel_index = grid.get_lrc(wel_nodes) # 
chd_flow = budget.get_data(text = 'CHD')
chd_flow = chd_flow[0]
chd_nodes = chd_flow['node']-1 #get lrc needs 0 array indices
chd_q = chd_flow['q']
chd_index = grid.get_lrc(chd_nodes)

with open(os.path.join(mdir, 'chd_index.pickle'), 'wb') as filehandle:
    # Store the data as a binary data stream
    pickle.dump(chd_index, filehandle)
    
with open(os.path.join(mdir, 'wel_index.pickle'), 'wb') as filehandle:
    # Store the data as a binary data stream
    pickle.dump(wel_index, filehandle)
    
np.save(os.path.join(mdir, 'chd_discharge.npy'), chd_q)

specific_discharge = flopy.utils.postprocessing.get_specific_discharge(specific_discharge, gwf, head = head, position = 'centers')
structured_face_flows = flopy.mf6.utils.postprocessing.get_structured_faceflows(flow_ja_face, grb_file=os.path.join(mdir,'model.dis.grb'), verbose=True)
centers[0] = np.tile(centers[0], (10,1,1))
centers[1] = np.tile(centers[1], (10,1,1))

np.save(os.path.join(mdir,'specific_discharge.npy'), specific_discharge)
np.save(os.path.join(mdir,'centers.npy'), centers)
np.save(os.path.join(mdir,'heads.npy'), head)
np.save(os.path.join(mdir,'structured_flows.npy'), structured_face_flows)

