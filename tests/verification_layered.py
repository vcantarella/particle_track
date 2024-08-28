import numpy as np
import flopy


if __name__ == "__main__":
    # The model can be a 2D slice. It will have the same result as a 3D model.
    # The model will be a 2D slice of the 3D model. The 2D slice will be the xz plane
    ws = "MODFLOW_testmodels/ammer_cake_v7"
    name = "ammer_cake"
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name="mf6",
        version="mf6",
        sim_ws=ws,
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
    unique_facies = np.array([2, 4, 6, 7, 8, 9, 10, 11, 12, 21, 31])
    k_value_pair = { #hydraulic conductivity [m/s]
            2: 1e-5,
            4: 6e-6,
            6: 2e-6,
            7: 1e-6,
            8: 1e-7,
            9: 1e-7,
            10: 1e-7,
            11: 1.5e-5,
            12: 3e-4,
            21: 1e-8,
            31: 1e-8,
        }
    # TODO: fix porosity with the reference from Cora's paper
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
    phi = np.array(list(porosity_pairs.values()))
    TOC = np.array([2.5,5,8,18,24,34,1,2,1e-4, 0, 0])*1e-2 # %wt (proportion) total organic carbon
    C_OM = 37.5*1e-2 #%wt (proportion) carbon content in organic matter
    rho_om = 1350 #kg/m3 density of organic matter
    rho_ms = 2710 #kg/m3 density of calcite
    # /rho_s calculated based on expression from Ruehlmann et al. 2006
    rho_s = (TOC/C_OM/rho_om + (1-TOC/C_OM)/rho_ms)**-1 #kg/m3
    print(rho_s)
    M_C = 12.01*1e-3 #kg/mol
    C_content = TOC*rho_s*(1-phi)/M_C*(1/phi) #mol/m3
    C_content = C_content*1e-3 #mol/L
    print(C_content)
    reactivity_pairs = dict(zip(unique_facies, C_content)) #mol/m3

    vectorized_k = np.vectorize(k_value_pair.get)
    vectorized_r = np.vectorize(reactivity_pairs.get)
    vectorized_p = np.vectorize(porosity_pairs.get)
    counts = np.array([ 885711, 1578126,  872386, 2006518,  579647,  592348,  298093,
        797687,  117372, 2232505,  839607])
    # Proportions converted to model layers:
    proportions = counts / np.sum(counts)
    # reorder index to match stratigraphic sequence:
    order = np.array([31, 9, 8, 7,6,4,2,12,11,10,21])
    index = [np.where(unique_facies == i)[0][0] for i in order]
    corrs_facies = unique_facies[index]
    assert np.all(corrs_facies == order)
    proportions = proportions[index]
    contacts = 9*np.cumsum(proportions)
    top_botm = np.concatenate([np.array([0]),contacts])

    # Grid properties:
    Lx = 900  # problem lenght [m]
    Ly = 600  # problem width [m]
    H = 9  # aquifer height [m]
    delx = 1.5  # block size x direction
    dely = 20  # block size y direction
    #delz = 0.07  # block size z direction
    nlay = top_botm.shape[0] - 1  # number of layers
    ncol = int(Lx / delx)  # number of columns
    nrow = int(Ly / dely)  # number of layers

    facies_cake = np.flip(corrs_facies)[:, np.newaxis, np.newaxis]*np.ones((nlay, nrow, ncol))
    idomain = np.where(facies_cake == 21, 0, np.where(facies_cake == 31, 0, 1))
    # Flopy Discretizetion Objects (DIS)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        xorigin=0.0,
        yorigin=0.0,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delx,
        delc=dely,
        top=H,
        botm=np.flip(top_botm)[1:],
        idomain=idomain,
    )

    # Calculate the proportion of each facies considering the layers have different thicknesses
    layer_thicknesses = np.diff(top_botm)
    total_thickness = np.sum(layer_thicknesses)
    facies_proportions = layer_thicknesses / total_thickness
    act_facies = np.unique(facies_cake)
    # make a layered model where the layers represent the volumetric ration of the V6 model

    # Checking if our assignment meets the measured proportions:
    assert np.all(act_facies == np.sort(corrs_facies))
    assert np.allclose(facies_proportions, proportions)
    k_array = vectorized_k(facies_cake)
    r_array = vectorized_r(facies_cake)
    p_array = vectorized_p(facies_cake)
 

    # Node property flow
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,  # This we define the model as confined
        k=k_array,
        save_specific_discharge=True,
    )


    # Acessing the grid
    grid = gwf.modelgrid
    tdis = flopy.mf6.ModflowTdis(
        sim, pname="tdis", time_units="SECONDS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )
    ## Constant head
    chd_rec = []
    h = 12
    for lay in range(nlay):
        for row in range(nrow):
            # ((layer,row,col),head,iface)
            if idomain[lay, row, 0] == 1:
                chd_rec.append(((lay, row, 0), h, 1))
    h2 = h - ncol * delx * (2 / 750) #natural hydraulic gradient
    for lay in range(nlay):
        for row in range(nrow):
            # ((layer,row,col),head,iface)
            if idomain[lay, row, ncol-1] == 1:
                chd_rec.append(((lay, row, ncol - 1), h2, 2))
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        auxiliary=[("iface",)],
        stress_period_data=chd_rec,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    # Flopy initial Conditions
    start = h * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)
    # Output control and Solver
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
        # linear_acceleration="BICGSTAB",
        outer_maximum=10,
        inner_maximum=20000,
        outer_dvclose=1e-5,
        inner_dvclose=1e-6,
        rcloserecord=[1e-6, "STRICT"],
    )
    # Solving
    sim.write_simulation()
    sim.check()
    success, buff = sim.run_simulation()
    if not success:
        raise Exception("MODFLOW 6 did not terminate normally.")
    print(buff)
    ## Reading output files

    head = gwf.output.head()
    head_array = head.get_data()
    grid = gwf.modelgrid

    # Calculating travel times and reactivity:
    centers = grid.xyzcellcenters
    X = centers[0]
    Y = centers[1]
    Z = centers[2]
    X = np.broadcast_to(X, Z.shape)
    Y = np.broadcast_to(Y, Z.shape)
    # Extracting indexes:
    layers = np.arange(0, nlay)
    layers = layers[:, np.newaxis, np.newaxis]
    layers = layers * np.ones(Z.shape)
    rows = np.arange(0, nrow)
    rows = rows[np.newaxis, :, np.newaxis]
    rows = rows * np.ones(Z.shape)
    cols = np.arange(0, ncol)
    cols = cols[np.newaxis, np.newaxis, :]
    cols = cols * np.ones(Z.shape)
    ex_layers = np.ravel(layers[:,:,-1])
    ex_rows = np.ravel(rows[:,:,-1])
    ex_cols = np.ravel(cols[:,:,-1])
    ex_x = np.ravel(X[:,:,-1])
    ex_y = np.ravel(Y[:,:,-1])
    ex_z = np.ravel(Z[:,:,-1])
    outflow_points = np.column_stack((ex_layers, ex_rows, ex_cols)).astype(np.int64)
    # inflow nodes:
    in_layers = np.ravel(layers[:, :, 0])
    in_rows = np.ravel(rows[:, :, 0])
    in_cols = np.ravel(cols[:, :, 0])
    in_x = np.ravel(X[:, :, 0])
    in_y = np.ravel(Y[:, :, 0])
    in_z = np.ravel(Z[:, :, 0])
    inflow_points = np.column_stack((in_layers, in_rows, in_cols)).astype(np.int64)
    # Particle centers:


    
    ## Create MODPATH simulation

    inflow_lrc = [
        (lay, r, c) for lay, r, c in zip(inflow_points[:,0], inflow_points[:,1], inflow_points[:,2])
    ]
    outflow_lrc = [
        (lay, r, c)
        for lay, r, c in zip(outflow_points[:,0], outflow_points[:,1], outflow_points[:,2])
    ]

    inflow_nodes = grid.get_node(inflow_lrc)
    outflow_nodes = grid.get_node(outflow_lrc)

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
        exe_name="mp7",
        nodes=inflow_nodes,
    )

    # Create MPBAS package file:
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
        model_ws=ws,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
        exe_name="mp7",
        nodes=outflow_nodes,
    )

    # Create MPBAS package file:
    mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.3)

    # write modpath datasets
    mp.write_input()
    # run modpath
    mp.run_model()

