import flopy
import os
import matplotlib.pyplot as plt
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

fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))

ax = axs[0]
ax.set_aspect("equal")
ax.set_title("Particle pathlines mode: backward")
mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
mm.plot_grid(lw=0.1)
mm.plot_pathline(
    pf,
    layer="all",
    colors="blue",
    lw=1.0,
    linestyle=":",
    label="backward tracking",
)
mm.plot_endpoint(ef, direction="ending")  # , colorbar=True, shrink=0.5);
ax = axs[1]
ax.set_aspect("equal")
ax.set_title("Particle pathlines mode: forward")
mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
mm.plot_grid(lw=0.1)
mm.plot_pathline(
    pf2,
    layer="all",
    colors="darkgreen",
    lw=1.0,
    linestyle=":",
    label="forward tracking",
)
mm.plot_endpoint(ef2, direction="ending")  # , colorbar=True, shrink=0.5);
plt.tight_layout()
plt.show()
fig.savefig("tests/pathlines_complex_model.png")