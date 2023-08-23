# -*- coding: utf-8 -*-
"""
Create and run the MODFLOW and MODPATH model and save the model files in folder 'modflow_verification_1'

@author: vcant
"""

# Running the MODFLOW model:
""" -----------------------------------------------------"""
import flopy
import os
import matplotlib.pyplot as plt
import numpy as np

# Model creation:
name = 'mf6_verif'
ws = 'modflow_verification_1'
isExist = os.path.exists(ws)
if not isExist:

    # Create a new directory because it does not exist
    os.makedirs(ws)

sim = flopy.mf6.MFSimulation(
    sim_name=name, exe_name="mf6", version="mf6", sim_ws=ws
)

# Simulation time:
tdis = flopy.mf6.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
)

# Nam file
model_nam_file = "{}.nam".format(name)

# Groundwater flow object:
gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname=name,
    model_nam_file=model_nam_file,
    save_flows=True,
)

# Grid properties:
Lx = 1000  # problem lenght [m]
Ly = 1000  # problem width [m]
H = 20  # aquifer height [m]
delx = 1  # block size x direction
dely = 1  # block size y direction
delz = 20  # block size z direction

nlay = 1

ncol = int(Lx/delx)  # number of columns

nrow = int(Ly/dely)  # number of layers

# Flopy discretization Objects (DIS)

dis = flopy.mf6.ModflowGwfdis(
    gwf,
    xorigin = -1.,
    yorigin = -500.5,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=dely,
    delc=delx,
    top=20.,
    botm=0.,
)

# Flopy initial Conditions

h0 = 18
start = h0 * np.ones((nlay, nrow, ncol))
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

# Node property flow

k = 3 # Model conductivity in m/d
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=1, #This we define the model as convertible (water table aquifer)
    k=k,
)

#boundary conditions:

## Constant head    
chd_rec = []

for row in range(0, nrow):
    #((layer,row,col),head,iface)
    chd_rec.append(((0, row, 0), h0, 1))

chd = flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=[('iface',)],
    stress_period_data=chd_rec,
    print_input=True,
    print_flows=True,
    save_flows=True,
)

##Constant flow
wel_rec = []
base_flow = 1
for row in range(0, nrow):
    #((layer,row,col),flow_rate,iface,boundaryname)
    wel_rec.append(((0, row, ncol-1),base_flow, 2, "base_flow"))

x_well = 30.5
y_well = 50
pump_rate = -250
grid = gwf.modelgrid

well_cell = grid.intersect(x_well,y_well)

print(well_cell)

#((layer,row,col),flow_rate,iface,boundaryname)
wel_rec.append(((0,well_cell[0],well_cell[1]), pump_rate, 0, "pump_wel"))

wel_obs = {
    "wel_flows.csv": [
        ("base_flow","WEL", "base_flow"),
        ("pump_wel", "WEL", "pump_wel"),
    ],
}

wel = flopy.mf6.ModflowGwfwel(
    gwf,
    boundnames = True,
    save_flows = True,
    auxiliary=[('iface',)],
    stress_period_data=wel_rec,
    observations = wel_obs,
)
wel.obs.print_input = True

#Output control and Solver


headfile = "{}.hds".format(name)
head_filerecord = [headfile]
budgetfile = "{}.cbb".format(name)
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
ims = flopy.mf6.ModflowIms(
    sim,
    pname="ims",
    complexity="SIMPLE",
    linear_acceleration="BICGSTAB",
    outer_maximum = 300,
    inner_maximum = 500,
    rcloserecord = [0.001, 'STRICT']
)

# Solving
sim.write_simulation()
sim.check()
success, buff = sim.run_simulation()
if not success:
    raise Exception("MODFLOW 6 did not terminate normally.")

## Reading output files

head = gwf.output.head()
head_array = head.get_data()

# Printing output:
fig, ax = plt.subplots(1, 1, figsize=(12, 12), constrained_layout=True)
# first subplot
contour_intervals = np.arange(10, 20, 0.5)

ax.set_title("Head Results")
modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, extent = (-1,100,0,100))
pa = modelmap.plot_array(head_array, vmin=14, vmax=18)
quadmesh = modelmap.plot_bc("CHD")
linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
contours = modelmap.contour_array(
    head_array,
    levels=contour_intervals,
    colors="black",
)
ax.clabel(contours, fmt="%2.1f")
cb = plt.colorbar(pa, shrink=0.5, ax=ax)
fig.savefig('verification_1_groundwater_heads.png', dpi=400)

## Create MODPATH simulation
inds = [rec[0] for rec in chd_rec]
nodes = grid.get_node(inds)

# create modpath files
mpnamf = name + "_mp_forward"
# create basic forward tracking modpath simulation
mp = flopy.modpath.Modpath7.create_mp7(
    modelname=mpnamf,
    trackdir="forward",
    flowmodel=gwf,
    model_ws=ws,
    rowcelldivisions=1,
    columncelldivisions=1,
    layercelldivisions=1,
    exe_name='mp7',
    nodes = nodes
)

#Create MPBAS package file:
mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.3)

# write modpath datasets
mp.write_input()
# run modpath
mp.run_model()


