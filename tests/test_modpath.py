import os
import subprocess

import flopy
import numpy as np

from particle_track import (
    cumulative_cuda,
    cumulative_gu,
    cumulative_reactivity,
    pollock,
)

# Running the MODFLOW and modpath models
subprocess.check_call(["python", "tests/verification_1.py"])
subprocess.check_call(["python", "tests/verification_2.py"])
subprocess.check_call(["python", "tests/verification_3.py"])


def test_verification1():
    """Testing the first verification case:"""
    name = "mf6_verif"
    r_folder = "MODFLOW_testmodels"
    model_directory = os.path.join(r_folder,"modflow_verification_1")
    mpnamf = name + "_mp_forward"
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=model_directory,
        exe_name="mf6",
        verbosity_level=0,
    )
    gwf = sim.get_model(name)
    grid = gwf.modelgrid
    # Well cell:
    x_well = 30.5
    y_well = 50
    well_cell = grid.intersect(x_well, y_well)
    # Get pathline:
    p = flopy.utils.PathlineFile(os.path.join(model_directory, mpnamf + ".mppth"))
    e = flopy.utils.EndpointFile(os.path.join(model_directory, mpnamf + ".mpend"))
    pw = p.get_destination_pathline_data(
        dest_cells=grid.get_node((0, well_cell[0], well_cell[1]))
    )
    ttmodpath = e.get_destination_endpoint_data(
        dest_cells=grid.get_node((0, well_cell[0], well_cell[1]))
    )["time"]
    # Getting the particles for the pollocking code:
    X, Y, Z = grid.xyzcellcenters
    inds = []
    for i in range(len(pw)):
        x = pw[i][0][0]
        y = pw[i][0][1]
        x, y = grid.get_coords(
            x, y
        )  # convert from local coordinates to global coordinates
        z = pw[i][0][2]
        layer, row, col = grid.intersect(x, y, z)
        inds.append((x, y, z, layer, row, col))

    inds = np.vstack(inds)
    prts_loc = inds
    pt_results = pollock(
        gwfmodel=gwf,
        model_directory=model_directory,
        particles_starting_location=prts_loc,
        porosity=0.3,
    )
    ttnumbapath = []
    for j in range(np.max(pt_results[:, 0] + 1).astype(np.int16)):
        results = pt_results[pt_results[:, 0] == j, 1:]
        t = results[:, -1]
        x = results[-1, 0]
        y = results[-1, 1]
        total_t = np.sum(t)
        ttnumbapath.append(total_t)
    ttnumbapath = np.array(ttnumbapath)
    # testing the model results
    np.testing.assert_allclose(ttnumbapath, ttmodpath, rtol = 1e-4, atol = 0.,)


def test_verification2():
    """Testing Verification 2"""
    name = "mf6_verif_2"
    r_folder = "MODFLOW_testmodels"
    model_directory = os.path.join(r_folder,"modflow_verification_2")
    mpnamf = name + "_mp_forward"
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=model_directory,
        exe_name="mf6",
        verbosity_level=0,
    )
    gwf = sim.get_model(name)
    grid = gwf.modelgrid
    # Get pathline:
    p = flopy.utils.PathlineFile(os.path.join(model_directory, mpnamf + ".mppth"))
    pf = p.get_alldata()
    p_sample = []
    ttmodpath = []
    for i in range(len(pf)):
        x = pf[i]["x"]
        z = pf[i]["z"]
        time = pf[i]["time"]
        p_sample.append((x, z))
        ttmodpath.append(time[-1])
    X, Y, Z = grid.xyzcellcenters
    inds = []
    for i in range(len(pf)):
        x = pf[i][0][0]
        y = pf[i][0][1]
        x, y = grid.get_coords(
            x, y
        )  # convert from local coordinates to global coordinates
        z = pf[i][0][2]
        layer, row, col = grid.intersect(x, y, z)
        inds.append((x, y, z, layer, row, col))
    inds = np.vstack(inds)
    prts_loc = inds
    pt_results = pollock(
        gwfmodel=gwf,
        model_directory=model_directory,
        particles_starting_location=prts_loc,
        porosity=0.3,
    )
    ttnumbapath = []
    for j in range(np.max(pt_results[:, 0] + 1).astype(np.int16)):
        results = pt_results[pt_results[:, 0] == j, 1:]
        t = results[:, -1]
        x = results[-1, 0]
        z = results[-1, 1]
        total_t = np.sum(t)
        ttnumbapath.append(total_t)
    ttnumbapath = np.array(ttnumbapath)
    ttmodpath = np.array(ttmodpath)
    np.testing.assert_allclose(ttnumbapath, ttmodpath, rtol = 1e-4, atol = 0.,)


def test_verification3():
    """Testing Verification 3"""
    name = "mf6_verif_3"
    r_folder = "MODFLOW_testmodels"
    model_directory = os.path.join(r_folder,"modflow_verification_3")
    mpnamf = name + "_mp_backward"
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=model_directory,
        exe_name="mf6",
        verbosity_level=0,
    )
    gwf = sim.get_model(name)
    head = gwf.output.head()
    head_array = head.get_data()
    grid = gwf.modelgrid
    ## reading modpath output
    # Get pathline:
    p = flopy.utils.PathlineFile(os.path.join(model_directory, mpnamf + ".mppth"))
    pf = p.get_alldata()
    p_sample = []
    ttmodpath = []
    for i in range(len(pf)):
        x = pf[i]["x"]
        y = pf[i]["y"]
        time = pf[i]["time"]
        p_sample.append((x, y))
        ttmodpath.append(time[-1])
    inds = []
    for i in range(len(pf)):
        x = pf[i][0][0]
        y = pf[i][0][1]
        x, y = grid.get_coords(
            x, y
        )  # convert from local coordinates to global coordinates
        z = pf[i][0][2]
        layer, row, col = grid.intersect(x, y, z)
        inds.append((x, y, z, layer, row, col))
    inds = np.vstack(inds)
    prts_loc = inds
    pt_results = pollock(
        gwfmodel=gwf,
        model_directory=model_directory,
        particles_starting_location=prts_loc,
        porosity=0.3,
        mode="backwards",
    )
    ct_results = cumulative_reactivity(
        gwfmodel=gwf,
        model_directory=model_directory,
        particles_starting_location=prts_loc,
        porosity=0.3,
        reactivity=np.ones_like(head_array),
    )
    ct_results_gu = cumulative_gu(
        gwfmodel=gwf,
        model_directory=model_directory,
        particles_starting_location=prts_loc,
        porosity=0.3,
        reactivity=np.ones_like(head_array),
    )
    ct_2 = cumulative_cuda(
        gwfmodel=gwf,
        model_directory=model_directory,
        particles_starting_location=prts_loc,
        porosity=0.3,
        reactivity=np.ones_like(head_array),
    )
    ttnumbapath = []
    for j in range(np.max(pt_results[:, 0] + 1).astype(np.int16)):
        results = pt_results[pt_results[:, 0] == j, 1:]
        t = results[:, -1]
        x = results[-1, 0]
        y = results[-1, 1]
        total_t = np.sum(t)
        ttnumbapath.append(total_t)
    ttnumbapath = np.array(ttnumbapath)
    ttmodpath = np.array(ttmodpath)
    ttcumulative_track = ct_results[0]
    ttcumulative_gu = ct_results_gu[:, 0]
    ttcumulative_cuda = ct_2[:, 0]
    np.testing.assert_allclose(ttnumbapath, ttmodpath, rtol = 1e-4, atol = 0.,)
    np.testing.assert_allclose(ttcumulative_track, ttmodpath, rtol = 1e-4, atol = 0.,)
    np.testing.assert_allclose(ttcumulative_gu, ttmodpath, rtol = 1e-4, atol = 0.,)
    np.testing.assert_allclose(ttnumbapath, ttmodpath, rtol = 1e-4, atol = 0.,)
    np.testing.assert_allclose(ttnumbapath, ttcumulative_cuda, rtol = 1e-4, atol = 0.,)