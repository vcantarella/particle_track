# -*- coding: utf-8 -*-
import flopy
import os
import matplotlib.pyplot as plt
import numpy as np
"""
Create and run the MODFLOW and MODPATH model and save the model files in folder 'modflow_verification_3'
3D Grounwater flow problem from Provost et al 2017 - Groundwater Whirls problem from Documentation of the XT3D document.
@author: vcant
"""

""" Running the MODFLOW model:"""
# Model creation:
name = 'mf6_verif_3'
model_directory = "modflow_verification_3"
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

length_units = "meters"
time_units = "days"


# Grid properties:
nper = 1  # Number of periods
nlay = 10  # Number of layers
nrow = 10  # Number of rows
ncol = 51  # Number of columns
delr = 100.0  # Spacing along rows ($m$)
delc = 100.0  # Spacing along columns ($m$)
top = 0.0  # Top of the model ($m$)
botm_str = "-100, -200, -300, -400, -500, -600, -700, -800, -900, -1000"  # Layer bottom elevations ($m$)
strt = 0.0  # Starting head ($m$)
icelltype = 0  # Cell conversion type
k11 = 1.0  # Hydraulic conductivity in the 11 direction ($m/d$)
k22 = 0.1  # Hydraulic conductivity in the 22 direction ($m/d$)
k33 = 1.0  # Hydraulic conductivity in the 33 direction ($m/d$)
angle1_str = "45, 45, 45, 45, 45, -45, -45, -45, -45, -45"  # Rotation of the hydraulic conductivity ellipsoid in the x-y plane
inflow_rate = 0.01  # Inflow rate ($m^3/d$)
botm = [float(value) for value in botm_str.split(",")]
angle1 = [float(value) for value in angle1_str.split(",")]


# Flopy Discretizetion Objects (DIS)

dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
)

# Flopy initial Conditions

h0 = 0.
start = h0 * np.ones((nlay, nrow, ncol))
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

# Node property flow

npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=0, #This we define the model as convertible (water table aquifer)
    k=k11,
    k22=k22,
    k33=k33,
    angle1=angle1,
    save_specific_discharge=True,
    xt3doptions=True,
)

#boundary conditions:

## Flow boundaries:

rate = np.zeros((nlay, nrow, ncol), dtype=np.float64)
rate[:, :, 0] = inflow_rate
rate[:, :, -1] = -inflow_rate
iface = np.where(rate > 0, 1, np.where(rate < 0, 2, 0)).astype(np.int64)
wellay, welrow, welcol = np.where(rate != 0.0)
wel_spd = {0:[
            ((k, i, j), rate[k, i, j], iface[k, i, j])
            for k, i, j in zip(wellay, welrow, welcol)
        ]}

wel = flopy.mf6.ModflowGwfwel(
    gwf,
    pname='wel',
    auxiliary=[('iface',)],
    stress_period_data=wel_spd,
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

nouter = 50
ninner = 100
hclose = 1e-9
rclose = 1e-6

ims = flopy.mf6.ModflowIms(
    sim,
    pname="ims",
    complexity="SIMPLE",
    linear_acceleration="BICGSTAB",
    outer_maximum=nouter,
    inner_maximum=ninner,
    inner_dvclose=hclose,
    rcloserecord = [rclose, 'STRICT']
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
# contour_intervals = np.arange(101, 110, 0.5)

ax.set_title("Head Results")
modelmap = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line = {"row": 0})
pa = modelmap.plot_array(head_array)
quadmesh = modelmap.plot_bc("WEL")
linecollection = modelmap.plot_grid(lw=0.05, color="0.5")
# contours = modelmap.contour_array(
#     head_array,
#     levels=contour_intervals,
#     colors="black",
# )
# ax.clabel(contours, fmt="%2.1f")
cb = plt.colorbar(pa, shrink=0.5, ax=ax)
fig.savefig('verification_3_groundwater_heads.png', dpi = 400)

## Create MODPATH simulation
grid = gwf.modelgrid
inflow_points = np.where(rate > 0.0)
outflow_points = np.where(rate < 0.0)
inflow_lrc = [(l, r, c) for l, r, c in zip(inflow_points[0], inflow_points[1], inflow_points[2])]
outflow_lrc = [(l, r, c) for l, r, c in zip(outflow_points[0], outflow_points[1], outflow_points[2])]

inflow_nodes = grid.get_node(inflow_lrc)
outflow_nodes = grid.get_node(outflow_lrc)

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
    nodes=inflow_nodes,
)

#Create MPBAS package file:
mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.3)

# write modpath datasets
mp.write_input()
# run modpath
mp.run_model()

# create modpath files
mpnamf = name + "_mp_backward"
# create basic forward tracking modpath simulation
mp = flopy.modpath.Modpath7.create_mp7(
    modelname=mpnamf,
    trackdir="backward",
    flowmodel=gwf,
    model_ws=model_directory,
    rowcelldivisions=1,
    columncelldivisions=1,
    layercelldivisions=1,
    exe_name='mp7',
    nodes=outflow_nodes,
)

#Create MPBAS package file:
mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.3)

# write modpath datasets
mp.write_input()
# run modpath
mp.run_model()