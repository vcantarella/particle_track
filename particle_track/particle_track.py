#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:35:06 2023

@author: vcant
"""

# Assumes model is unrotated.


import numpy as np
import flopy
from functools import partial
import numba
from numba import boolean, float64, int64
from multiprocessing import Pool
from .preprocessing import prepare_arrays


# creating the velocity function:

@numba.njit(nogil=True)
def velocity(coords, v0, gv, face_coords):
    """
    Calculates cell velocities at the starting location in the cell.

    Parameters
    ----------
    coords : ndarray
        [x,y,z] of the particle
    v0 : velocity at the lower face coordinate
        [vx0, vy0, vz0]: velocities of left, lower, and bottom faces
    gv : ndarray
        [dvdx, dvdy,dvdz]: linear velocity gradients for each cell
    face_coords : ndarray
        [x0,y0,z0]: coordinates for the left, lower and bottom faces

    Returns
    -------
    v : [vx,vy,vz]
    Current velocities for the particle at x,y,z

    """

    v = gv * (coords - face_coords) + v0

    return v


@numba.vectorize(nopython=True)
def exit_direction(v1, v2, v):
    """
    Define exit direction (lower or upper) based on current velocity of particle and the velocity at faces (run for each axis: x,y,z)
    Parameters
    ----------
    v1: velocity at lower face
    v2: velocity at upper face
    v: current velocity

    Returns
    index (-1: exit at lower face (negative axis direction),
            0: impossible exit at this axis
            1: exit at upper face (positive axis direction)
    -------

    """
    if (v1 >= 0.) & (v2 > 0.):
        r = 1
    elif (v1 < 0.) & (v2 <= 0.):
        r = -1
    elif (v1 >= 0.) & (v2 <= 0.):
        r = 0
    else:  # (v1 < 0) & (v2 > 0):
        if v > 0:
            r = 1
        elif v < 0:
            r = -1
        else:
            r = 0
    return r


@numba.vectorize([float64(int64, boolean, float64, float64, float64, float64, float64, float64, float64)],
                 nopython=True)
def reach_time(exit_ind, gradient_logic, v1, v2, v, gv, x, left_x, right_x):
    """
    Calculates the time to reach the exit faces at each axis.

    Parameters
    ----------
    exit_ind: index calulated from exit_direction
    gradient_logic: check if there is a velocity gradient (if true then the normal expression cannot be used)
    v1: velocity at the lower face
    v2: velocity at the upper face
    v: current particle velocity
    gv: velocity gradient in the cell
    x: current particle coordinates
    left_x: coordinates of lower left corner
    right_x: coordinates of upper right corner

    Returns
    -------
    dt: travel time at the current cell

    """
    if exit_ind == 0:
        tx = np.inf
    elif exit_ind == -1:
        if ~gradient_logic:
            tx = (-1) * (x - left_x) / v
        else:
            tx = np.log(v1 / v) / gv
    else:  # exit_ind == 1
        if ~ gradient_logic:
            tx = (right_x - x) / v
        else:
            tx = np.log(v2 / v) / gv
    return tx


@numba.vectorize([float64(int64, boolean, float64, float64, float64, float64, float64,
                          float64, float64, float64)], nopython=True)
def exit_location(exit_ind, gradient_logic, dt, v1, v2, v, gv, x, left_x, right_x):
    """
    Calculate the coordinates at the exit location in the cell
    Parameters
    ----------
    exit_ind: index calulated from exit_direction
    gradient_logic: check if there is a velocity gradient (if true then the normal expression cannot be used)
    dt: calculated travel time at the cell
    v1: velocity at the lower face
    v2: velocity at the upper face
    v: current particle velocity
    gv: velocity gradient in the cell
    x: current particle coordinates
    left_x: coordinates of lower left corner
    right_x: coordinates of upper right corner

    Returns
    -------
    coords: exit coordinate at each axis

    """
    if ~gradient_logic:
        x_new = x + v * dt
    elif np.abs(v) > 1e-20:
        if exit_ind == 1:

            x_new = left_x + (1 / gv) * (v * np.exp(gv * dt) - v1)

        elif exit_ind == -1:

            x_new = right_x + (1 / gv) * (v * np.exp(gv * dt) - v2)
        else:
            x_new = left_x + (1 / gv) * (v * np.exp(gv * dt) - v1)
    else:
        x_new = x

    return x_new


@numba.njit(nogil=True)
def numba_max_abs(d_array):
    res = []
    for i in range(d_array.shape[0]):
        array = np.max(np.abs(d_array[i, :]))
        res.append(array)

    return np.array(res)


@numba.njit(nogil=True)
def negative_index(ind_array):
    for i in range(ind_array.shape[0]):
        if ind_array[i] < 0:
            return True
    return False


@numba.njit(nogil=True)
def larger_index(test_array, reference_array):
    for i in range(test_array.shape[0]):
        if test_array[i] > reference_array[i]:
            return True
    return False


@numba.njit
def trajectory(initial_position, initial_cell, face_velocities, gvs, xedges, yedges, z_lf, z_uf, termination):
    """
    Calculates the trajectory of a particle until it leaves the domain or find a termination point.
    Parameters
    ----------
    initial_position
    initial_cell
    face_velocities
    gvs
    xedges
    yedges
    z_lf
    z_uf
    termination

    return
    inds: layer,row,col at each cell it passes
    ts: travel time at each cell
    xes: x positions
    yes: y position
    zes: z positions
    -------

    """
    # initializing:
    x = initial_position[0]
    y = initial_position[1]
    z = initial_position[2]
    layer = initial_cell[0]
    row = initial_cell[1]
    col = initial_cell[2]

    termination_criteria = True
    ts = []
    layers = [layer, ]
    rows = [row, ]
    cols = [col, ]
    xes = [x, ]
    yes = [y, ]
    zes = [z, ]
    coords = np.array([x, y, z])

    # While loop until termination:
    while termination_criteria:

        # coordinates at lower and upper faces:
        left_x = xedges[col]
        right_x = xedges[col + 1]
        low_y = yedges[row + 1]
        top_y = yedges[row]
        bt_z = z_lf[layer, row, col]
        up_z = z_uf[layer, row, col]

        coords_0 = np.array([left_x, low_y, bt_z])
        coords_1 = np.array([right_x, top_y, up_z])
        # gradients for the cell
        gvp = gvs[:, layer, row, col]

        # velocities at lower coordinate faces:
        v0 = face_velocities[np.array([0, 2, 4]), layer, row, col]
        # velocities at upper coordinate faces:
        v1 = face_velocities[np.array([1, 3, 5]), layer, row, col]

        # current velocities
        v = velocity(coords, v0, gvp, coords_0)

        # Where is it going:
        velocity_gradient_index = np.abs(v0 - v1) > 1e-10 * numba_max_abs(np.column_stack((v0, v1)))
        exit_direction_index = exit_direction(v0, v1, v)

        # Time to reach each end
        dt_array = reach_time(exit_direction_index, velocity_gradient_index, v0, v1, v, gvp, coords, coords_0, coords_1)

        # actual travel time:
        dt = np.min(dt_array)
        exit_point_loc = np.argmin(dt_array)

        if dt == np.inf:
            break

        exit_point = exit_location(exit_direction_index, velocity_gradient_index, dt, v0, v1, v, gvp, coords, coords_0,
                                   coords_1)

        if exit_point_loc == 0:
            col = col + exit_direction_index[0]
        if exit_point_loc == 1:
            row = row + (-exit_direction_index[1])
        if exit_point_loc == 2:
            layer = layer + (-exit_direction_index[2])

        # Adding to the dataset:
        layers.append(layer)
        rows.append(row)
        cols.append(col)
        ts.append(dt)
        xes.append(exit_point[0])
        yes.append(exit_point[1])
        zes.append(exit_point[2])

        ##termination criteria evaluation: whether the particle has reached a termination layer or out of the system

        has_negative_index = negative_index(np.array([layer, row, col]))
        over_index = larger_index(np.array([layer, row, col]), np.array(termination.shape))
        term_value = termination[layer, row, col]

        if ((term_value == 1) | has_negative_index | over_index):
            termination_criteria = False
        # new loop:
        coords = exit_point

    ts.append(0.)
    ts = np.array(ts)
    xes = np.array(xes)
    yes = np.array(yes)
    zes = np.array(zes)
    layers = np.array(layers)
    rows = np.array(rows)
    cols = np.array(cols)
    inds = np.column_stack((layers, rows, cols))
    return inds, ts, xes, yes, zes


def work(iterator_realization, face_velocities, gvs, xedges, yedges, z_lf, z_uf, termination):
    '''
    Runs the particle trajectory in a multiprocessing enviroment.
    The particle array information is stored in the iterator_realization variable
    Returns
    -------
    particle_traj : TYPE
        returns the particle trajectory

    '''

    j = iterator_realization[1]
    particle_array = iterator_realization[0]

    # Getting the coordinates and cell location from the particle array
    particle_starting_coords = particle_array[:3]
    particle_starting_cell = particle_array[3:].astype(np.int64)
    # Initializing array
    inds, ts, xes, yes, zes = trajectory(particle_starting_coords, particle_starting_cell,
                                         face_velocities, gvs, xedges, yedges, z_lf, z_uf, termination)
    jes = np.repeat(j, xes.shape[0])
    particle_traj = np.column_stack((jes, xes, yes, zes, ts))
    return particle_traj


def particle_track(gwfmodel: flopy.mf6.MFModel,
                   model_directory: str,
                   particles_starting_location: np.ndarray,
                   mode: str = 'forward',
                   processes: int = 4):
    iter_coords = iter(particles_starting_location)
    js = np.arange(particles_starting_location.shape[0])

    xedges, yedges, z_lf, z_uf, gvs, face_velocities, termination = prepare_arrays(gwfmodel, model_directory)

    if mode == 'backwards':
        face_velocities = (-1) * face_velocities
        gvs = (-1) * gvs

    good_work = partial(work, face_velocities=face_velocities, gvs=gvs,
                        xedges=xedges,
                        yedges=yedges, z_lf=z_lf, z_uf=z_uf,
                        termination=termination)

    with Pool(processes=processes) as pool:
        results = pool.map(good_work, zip(iter_coords, js))

    true_results = np.vstack(results)

    return true_results


if __name__ == 'main':
    print('TESTING')
    dirc = 'test1'
    sim = flopy.mf6.MFSimulation.load(sim_name='mfsim.nam', sim_ws=dirc, )
    gwf = sim.gwf[0]
    grid = gwf.modelgrid
    nlays = grid.nlay
    ncols = grid.ncol
    nrows = grid.nrow
    cell_centers = grid.xyzcellcenters
    head = gwf.output.head()
    head_array = head.get_data()
    X = np.ravel(np.tile(cell_centers[0], (nlays, 1, 1)))
    Y = np.ravel(np.tile(cell_centers[1], (nlays, 1, 1)))
    Z = cell_centers[2].flatten()
    layers = np.arange(0, nlays)
    layers = layers[:, np.newaxis, np.newaxis]
    layers = layers * np.ones(cell_centers[2].shape)
    rows = np.arange(0, ncols)
    rows = rows[np.newaxis, :, np.newaxis]
    rows = rows * np.ones(cell_centers[2].shape)
    cols = np.arange(0, ncols)
    cols = cols[np.newaxis, np.newaxis, :]
    cols = cols * np.ones(cell_centers[2].shape)
    layers = np.ravel(layers)
    rows = np.ravel(rows)
    cols = np.ravel(cols)

    particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))

    particle_results = particle_track(gwf, dirc, particle_centers, 'backwards')

    np.save('test1_particles.npy', particle_results)

    '''------------------------------------------------'''
    print('TESTING HYVR MODEL')
    dirc = 'hysim_model_test'
    sim = flopy.mf6.MFSimulation.load(sim_name='mfsim.nam', sim_ws=dirc, )
    gwf = sim.gwf[0]
    grid = gwf.modelgrid
    nlays = grid.nlay
    ncols = grid.ncol
    nrows = grid.nrow
    cell_centers = grid.xyzcellcenters
    head = gwf.output.head()
    head_array = head.get_data()
    X = np.ravel(np.tile(cell_centers[0], (nlays, 1, 1)))
    Y = np.ravel(np.tile(cell_centers[1], (nlays, 1, 1)))
    Z = cell_centers[2].flatten()
    layers = np.arange(0, nlays)
    layers = layers[:, np.newaxis, np.newaxis]
    layers = layers * np.ones(cell_centers[2].shape)
    rows = np.arange(0, ncols)
    rows = rows[np.newaxis, :, np.newaxis]
    rows = rows * np.ones(cell_centers[2].shape)
    cols = np.arange(0, ncols)
    cols = cols[np.newaxis, np.newaxis, :]
    cols = cols * np.ones(cell_centers[2].shape)
    layers = np.ravel(layers)
    rows = np.ravel(rows)
    cols = np.ravel(cols)

    particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))

    particle_results = particle_track(gwf, dirc, particle_centers, 'backwards')

    np.save('hysim_particles.npy', particle_results)
