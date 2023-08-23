# -*- coding: utf-8 -*-
"""
Create and run the MODFLOW and MODPATH model and save the model files in folder 'modflow_verification_2'

@author: vcant
"""

# Running the MODFLOW model:
""" -----------------------------------------------------"""
import flopy
import os
import matplotlib.pyplot as plt
import numpy as np


# Model creation:
name = 'mf6_verif_2'
model_directory = "modflow_verification_2"
isExist = os.path.exists(model_directory)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(model_directory)

sim = flopy.mf6.MFSimulation(
    sim_name=name, exe_name="mf6", version="mf6", sim_ws=model_directory
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
Lx = 2000 #problem lenght [m]
Ly = 1 #problem width [m]
H = 100  #aquifer height [m]
delx = 1 #block size x direction
dely = 1 #block size y direction
delz = 1 #block size z direction

nlay = 100

ncol = int(Lx/delx) # number of columns

nrow = int(Ly/dely) # number of layers

# Flopy Discretizetion Objects (DIS)
bottom_array = np.ones((nlay,nrow,ncol))
bottom_range = np.arange(99,-1,-1)
bottom_range = bottom_range[:,np.newaxis,np.newaxis]
bottom_array = bottom_array * bottom_range

dis = flopy.mf6.ModflowGwfdis(
    gwf,
    xorigin = 0.,
    yorigin = 0.,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=dely,
    delc=delx,
    top=100.,
    botm=bottom_array,
)

# Flopy initial Conditions

h0 = 200
start = h0 * np.ones((nlay, nrow, ncol))
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

# Node property flow

k = 1e-4 * np.ones((nlay, nrow, ncol)) # Model conductivity in m/s
k[9:30,:,300:1701] = 1e-8
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=0, #This we define the model as convertible (water table aquifer)
    k=k,
)

#boundary conditions:

## Constant head    
chd_rec = []

h = np.linspace(101, 110, ncol)
i = 0
for col in range(0, ncol):
    #((layer,row,col),head,iface)
    chd_rec.append(((0, 0, col), h[i], 6))
    i+=1

chd = flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=[('iface',)],
    stress_period_data=chd_rec,
    print_input=True,
    print_flows=True,
    save_flows=True,
)


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
    #linear_acceleration="BICGSTAB",
    outer_maximum = 10,
    inner_maximum = 1500,
    inner_dvclose=1e-3,
    rcloserecord = [0.01, 'STRICT']
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
fig, ax = plt.subplots(1, 1, figsize=(24, 6), constrained_layout=True)
# first subplot
contour_intervals = np.arange(101, 110, 0.5)

ax.set_title("Head Results")
modelmap = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line = {"row": 0})
pa = modelmap.plot_array(np.log(k))
quadmesh = modelmap.plot_bc("CHD")
linecollection = modelmap.plot_grid(lw=0.05, color="0.5")
contours = modelmap.contour_array(
    head_array,
    levels=contour_intervals,
    colors="black",
)
ax.clabel(contours, fmt="%2.1f")
cb = plt.colorbar(pa, shrink=0.5, ax=ax)
fig.savefig('verification_2_groundwater_heads.png', dpi = 400)
## Create MODPATH simulation
grid = gwf.modelgrid
inds = [rec[0] for rec in chd_rec]
nodes = grid.get_node(inds)

# create modpath files
mpnamf = name + "_mp_forward"
# create basic forward tracking modpath simulation
mp = flopy.modpath.Modpath7.create_mp7(
    modelname=mpnamf,
    trackdir="forward",
    flowmodel=gwf,
    model_ws=model_directory,
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